"""
Paper Analyzer — iterative section-by-section analysis engine.

For each section of an ingested paper, this engine:
  1. Collects the leaf chunks that belong to that section (via heading metadata)
  2. Calls the LLM with a rich analysis prompt requesting:
       - Plain-English explanation
       - Mathematical analysis (LaTeX re-stating + expansion)
       - Physical / intuitive meaning of theorems/lemmas/proofs
       - Optional Python visualization snippet
  3. Emits an AnalysisEvent so the WebSocket can stream progress in real-time
  4. Accumulates all section analyses into a final Markdown document

Usage::

    analyzer = PaperAnalyzer(ai_engine=ai_engine, db=db, file_id=file_id)
    markdown = await analyzer.run(event_callback=my_async_callback)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalysisEvent:
    """Emitted for each step of the analysis pipeline."""
    type: str            # "start" | "section" | "complete" | "error"
    section_title: str   # Section currently being analysed  (empty for start/complete)
    content: str         # Markdown text for this section (or full doc on complete)
    section_index: int   # 0-based index of the current section
    total_sections: int  # Total number of sections to analyse
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "section_title": self.section_title,
            "content": self.content,
            "section_index": self.section_index,
            "total_sections": self.total_sections,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Analysis prompt templates
# ---------------------------------------------------------------------------

SECTION_ANALYSIS_PROMPT = """You are an expert scientific paper analyst with deep knowledge across physics, mathematics, computer science, and engineering. Analyze the following section of a research paper and produce a comprehensive, well-structured Markdown explanation.

## Section: {section_title}

### Content from the paper:
{chunk_text}

---

Produce your analysis following this exact structure. Use proper Markdown formatting with headers, bullet points, and LaTeX math ($$...$$ for display equations, $...$ for inline).

**CRITICAL FORMATTING RULES:**
- All Python code blocks MUST use ```python (never ```code or bare ```).
- All shell commands MUST use ```bash.
- All LaTeX snippets MUST use ```latex.
- Never use generic ```code fences.

## {section_title} — Analysis {section_index}

### 1. Plain-English Summary
Explain what this section accomplishes in clear, accessible language — as if briefing a brilliant colleague from a different subfield. Define any domain-specific jargon on first use. Aim for 3–5 sentences that capture *why* this section matters in the paper's overall narrative.

### 2. Mathematical Analysis
- Re-state every equation, theorem, lemma, or proposition in LaTeX.
- For each equation: name every variable/symbol and explain its role.
- If a proof is presented, decompose it into numbered logical steps and explain the reasoning behind each transition.
- Highlight any clever mathematical techniques or non-obvious tricks (e.g., change of variables, bounding arguments, symmetry exploitation).
- If no significant math appears, write: "This section is primarily descriptive; no formal mathematical content to analyze."

### 3. Physical / Intuitive Meaning
Explain the physical, geometric, or practical significance of the main results. What does the math *mean* concretely? Why should a practitioner care? Provide an analogy if one helps. Connect the result to real-world implications or engineering constraints when applicable.

### 4. Connections & Context
- How does this section build on or connect to other parts of the paper?
- Does it extend, generalize, or contradict prior work mentioned in the paper?
- What assumptions does it rely on, and what are their practical limitations?

### 5. Visualization Code
If there is an equation, distribution, function, scaling behavior, or data relationship that can be usefully visualized, provide a **self-contained** Python snippet using numpy + matplotlib. Requirements:
- Must run standalone (include all imports)
- Use clear axis labels, a descriptive title, and a legend if multiple curves are plotted
- Use `plt.tight_layout()` before `plt.show()`
- If truly no visualization is appropriate (e.g., purely qualitative discussion), write: "No visualization applicable for this section."

```python
# Your complete, runnable visualization code here
```

### 6. Key Takeaways
A concise bullet-point list (3–5 items) of the most important insights from this section. Each point should be a standalone sentence that a reader could scan independently.
"""

INTRO_PROMPT = """You are analyzing a scientific paper. Produce a structured preamble that orients the reader before diving into section-by-section analysis.

## Paper Title / Identifier:
{paper_name}

## File Summary (if available):
{summary}

**CRITICAL FORMATTING RULES:**
- All Python code blocks MUST use ```python (never ```code or bare ```).
- All shell commands MUST use ```bash.
- Never use generic ```code fences.

Produce the following sections in Markdown:

# Analysis of: {paper_name}

### What This Paper Is About
2–3 sentences summarizing the core problem, approach, and setting.

### Main Contributions
A bullet list of the paper's key contributions or novel results.

### Prerequisites & Background
What background knowledge (concepts, prior papers, mathematical tools) is most helpful for understanding this work? Keep it to 3–5 items with brief explanations.

### Reading Guide
Briefly describe how the paper is organized and what the reader should expect from each major part of the analysis that follows.
"""


# ---------------------------------------------------------------------------
# PaperAnalyzer
# ---------------------------------------------------------------------------

class PaperAnalyzer:
    """
    Iteratively analyses each section of an already-ingested research paper.

    Parameters
    ----------
    ai_engine : AIEngine
        The application-level AI engine (used for LLM calls via OpenRouter).
    db : DatabaseClient
        The application-level database client.
    file_id : str
        UUID of the file to analyse (must already be ingested).
    """

    def __init__(self, ai_engine, db, file_id: str):
        self.ai_engine = ai_engine
        self.db = db
        self.file_id = file_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_file_meta(self) -> Dict[str, Any]:
        try:
            resp = self.db.client.table("files").select("*").eq("id", self.file_id).execute()
            return resp.data[0] if resp.data else {}
        except Exception as e:
            logger.warning(f"Could not fetch file metadata: {e}")
            return {}

    def _collect_sections_with_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group leaf chunks by the deepest section heading in their metadata.

        Returns a list of ``{title, level, content}`` dicts, ordered by first
        appearance in the document.

        We intentionally use chunk ordering (chunk_index) so the document
        reading order is preserved.
        """
        # Sort by chunk_index (None / -1 for synthetic section chunks → last)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get("chunk_index") if c.get("chunk_index", -1) >= 0 else 99999,
        )

        # Build an ordered map: section_title -> [chunk_content]
        seen_order: List[str] = []
        section_map: Dict[str, Dict[str, Any]] = {}

        for chunk in sorted_chunks:
            meta = chunk.get("metadata") or {}
            headings: List[str] = meta.get("headings", [])

            # Skip purely synthetic section aggregation chunks (no leaf content)
            if meta.get("is_synthetic"):
                continue

            # Skip reference sections
            if chunk.get("is_reference"):
                continue

            # Determine section title: deepest heading, or "Introduction / Preamble"
            section_title = headings[-1] if headings else "Introduction / Preamble"

            if section_title not in section_map:
                seen_order.append(section_title)
                section_map[section_title] = {
                    "title": section_title,
                    "level": len(headings),
                    "chunks": [],
                }

            section_map[section_title]["chunks"].append(chunk.get("content", ""))

        # Build ordered list with concatenated text
        ordered_sections = []
        for title in seen_order:
            data = section_map[title]
            text = "\n\n".join(data["chunks"])
            ordered_sections.append({
                "title": title,
                "level": data["level"],
                "content": text,
            })

        return ordered_sections

    # ------------------------------------------------------------------
    # Core run loop
    # ------------------------------------------------------------------

    async def run(
        self,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """
        Run the analysis pipeline.

        Parameters
        ----------
        event_callback : callable, optional
            Called with each ``AnalysisEvent.to_dict()`` as it is produced.
            May be a regular function or a coroutine function.

        Returns
        -------
        str
            The final assembled Markdown document.
        """

        async def emit(event: AnalysisEvent):
            if event_callback is None:
                return
            try:
                if asyncio.iscoroutinefunction(event_callback):
                    await event_callback(event.to_dict())
                else:
                    event_callback(event.to_dict())
            except Exception as e:
                logger.debug(f"event_callback error: {e}")

        # 1. Load file metadata + all chunks
        file_meta = self._get_file_meta()
        paper_name = file_meta.get("name", "Unknown Paper")
        paper_summary = file_meta.get("summary", "")

        all_chunks = self.db.get_file_chunks(self.file_id)
        if not all_chunks:
            err_event = AnalysisEvent(
                type="error",
                section_title="",
                content="No chunks found for this file. Please re-ingest the document.",
                section_index=0,
                total_sections=0,
            )
            await emit(err_event)
            return "# Error\n\nNo chunks found for this file."

        # 2. Organise chunks into sections
        sections = self._collect_sections_with_chunks(all_chunks)
        total = len(sections)

        if total == 0:
            err_event = AnalysisEvent(
                type="error",
                section_title="",
                content="Could not detect any sections. The document may not have headings.",
                section_index=0,
                total_sections=0,
            )
            await emit(err_event)
            return "# Error\n\nNo sections detected."

        logger.info(f"PaperAnalyzer: {total} sections found in file {self.file_id}")

        # 3. Emit "start" event
        await emit(AnalysisEvent(
            type="start",
            section_title="",
            content=f"Starting analysis of **{paper_name}** — {total} sections detected.",
            section_index=0,
            total_sections=total,
            metadata={"paper_name": paper_name},
        ))

        # 4. Generate preamble
        preamble_md = ""
        try:
            preamble_prompt = INTRO_PROMPT.format(
                paper_name=paper_name,
                summary=paper_summary or "Not available.",
            )
            preamble_md = await self.ai_engine._openrouter_generate(preamble_prompt)
        except Exception as e:
            logger.warning(f"Could not generate preamble: {e}")
            preamble_md = f"# Analysis of: {paper_name}\n\n"

        # 5. Analyse each section sequentially to maintain reading order
        section_markdowns: List[str] = [preamble_md]

        for idx, section in enumerate(sections):
            title = section["title"]
            content_text = section["content"]

            # Trim very long sections to avoid context overflow  
            max_chars = 8000
            if len(content_text) > max_chars:
                content_text = content_text[:max_chars] + "\n\n[... content truncated for analysis ...]"

            logger.info(f"Analyzing section {idx+1}/{total}: {title}")

            try:
                prompt = SECTION_ANALYSIS_PROMPT.format(
                    section_title=title,
                    chunk_text=content_text,
                    section_index=idx + 1,
                )
                section_md = await self.ai_engine._openrouter_generate(prompt)
            except Exception as e:
                logger.error(f"Error analyzing section '{title}': {e}")
                section_md = f"## {title}\n\n*Analysis failed for this section: {e}*\n"

            section_markdowns.append(section_md)

            # Emit section event with the analysis produced
            await emit(AnalysisEvent(
                type="section",
                section_title=title,
                content=section_md,
                section_index=idx,
                total_sections=total,
                metadata={"section_level": section.get("level", 1)},
            ))

        # 6. Assemble final document
        separator = "\n\n---\n\n"
        final_markdown = separator.join(section_markdowns)

        # 7. Emit "complete" event
        await emit(AnalysisEvent(
            type="complete",
            section_title="",
            content=final_markdown,
            section_index=total - 1,
            total_sections=total,
            metadata={"paper_name": paper_name, "total_sections": total},
        ))

        logger.info(f"PaperAnalyzer complete for file {self.file_id}. "
                    f"Total Markdown length: {len(final_markdown)} chars")

        return final_markdown
