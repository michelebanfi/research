import streamlit as st
import asyncio
import os
from pathlib import Path
from src.ingestion import FileRouter, DoclingParser, ASTParser, FileType

def render_ingest_tab(db, ai):
    """
    Renders the Ingest Files tab content.
    """
    st.subheader("Ingest Documents & Code")
    
    # File browser
    files_col, upload_col = st.columns([1, 1])
    
    with files_col:
        st.markdown("### 📂 Current Files")
        files = db.get_project_files(st.session_state.selected_project_id)
        
        if files:
            for f in files:
                with st.expander(f"📄 {f['name']}"):
                    st.write(f"**Summary:** {f['summary'][:200]}..." if f.get('summary') and len(f['summary']) > 200 else f.get('summary', 'No summary'))
                    st.write(f"**Keywords:** {', '.join(f['metadata'].get('keywords', [])[:5])}")
                    st.caption(f"Processed: {f['processed_at']}")
                    
                    # Related files
                    related = db.get_related_files(f['id'])
                    if related:
                        st.write("**Related:**")
                        for rf in related[:3]:
                            st.caption(f"• {rf['name']} ({rf['shared_count']} shared)")
                    
                    if st.button("🗑️ Delete", key=f"del_{f['id']}"):
                        if db.delete_file(f['id']):
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error("Delete failed.")
        else:
            st.info("No files ingested yet.")
    
    with upload_col:
        st.markdown("### 📤 Upload New File")
        
        uploaded_file = st.file_uploader(
            "Upload Document or Code", 
            type=['pdf', 'md', 'txt', 'py', 'docx'],
            key="file_uploader"
        )
        
        if uploaded_file and st.button("🚀 Process & Ingest", key="ingest_btn"):
            process_upload(uploaded_file, db, ai)

def process_upload(uploaded_file, db, ai):
    with st.spinner("Processing..."):
        # Save to temp file
        upload_dir = Path("static/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        tmp_path = str(file_path)
        
        try:
            # 1. Route
            ftype = FileRouter.route(tmp_path)
            st.info(f"Detected file type: {ftype.value}")
            
            # 2. Parse
            chunks = []
            code_graph_data = {"nodes": [], "edges": []}
            
            if ftype == FileType.CODE:
                parser = ASTParser()
                parse_result = parser.parse(tmp_path)
                chunks = parse_result.get("chunks", [])
                code_graph_data = parse_result.get("graph_data", {"nodes": [], "edges": []})
            elif ftype == FileType.DOCUMENT:
                parser = DoclingParser()
                chunks = parser.parse(tmp_path)
            else:
                st.warning("Unsupported file type.")
            
            st.write(f"Parsed {len(chunks)} chunks.")
            
            # Preview Code Structure
            if ftype == FileType.CODE and code_graph_data["nodes"]:
                with st.expander("🔍 Code Structure Preview", expanded=True):
                    classes = [n for n in code_graph_data["nodes"] if n["type"] == "Class"]
                    methods = [n for n in code_graph_data["nodes"] if n["type"] == "Method"]
                    functions = [n for n in code_graph_data["nodes"] if n["type"] == "Function"]
                    
                    st.markdown(f"**Classes:** {len(classes)} | **Methods:** {len(methods)} | **Functions:** {len(functions)}")
            
            # Preview Document
            elif ftype == FileType.DOCUMENT and chunks:
                with st.expander("📄 Document Preview", expanded=True):
                    sample = chunks[0].get("content", "")[:800]
                    if len(chunks[0].get("content", "")) > 800:
                        sample += "..."
                    st.markdown(sample)
            
            # 3. AI Processing (Async & Parallel)
            st.text("Running AI Analysis (Summary, Graph, Embeddings)...")
            
            full_text = "\n".join([c['content'] for c in chunks])
            
            # Fix for event loop issues with Streamlit
            import nest_asyncio
            try:
                nest_asyncio.apply()
            except RuntimeError:
                pass
            
            async def run_ingestion_pipeline():
                # Create fresh async client
                import ollama
                async_client = ollama.AsyncClient()
                
                async def safe_summary():
                    try:
                        return await ai.generate_summary_async(full_text[:4000])
                    except Exception as e:
                        print(f"Summary error: {e}")
                        return "Summary generation failed."
                
                async def safe_graph():
                    try:
                        return await ai.extract_metadata_graph_async(full_text)
                    except Exception as e:
                        print(f"Graph error: {e}")
                        return {"nodes": [], "edges": []}
                
                async def safe_embedding(text):
                    try:
                        response = await async_client.embeddings(model=ai.embed_model, prompt=text)
                        return response["embedding"]
                    except Exception as e:
                        print(f"Embedding error: {e}")
                        return []
                
                # Run all tasks
                summary_result = await safe_summary()
                graph_result = await safe_graph()
                embedding_results = await asyncio.gather(
                    *[safe_embedding(c['content']) for c in chunks]
                )
                
                return summary_result, graph_result, embedding_results
            
            # Use existing event loop if available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                summary, graph_data, embeddings = loop.run_until_complete(run_ingestion_pipeline())
            except RuntimeError:
                summary, graph_data, embeddings = asyncio.run(run_ingestion_pipeline())
            
            # Validate embeddings
            valid_chunks = []
            for i, emb in enumerate(embeddings):
                if emb and len(emb) > 0:
                    chunks[i]['embedding'] = emb
                    valid_chunks.append(chunks[i])
                else:
                    st.warning(f"Chunk {i} had empty embedding, skipping.")
            
            chunks = valid_chunks
            st.write(f"Generated {len(chunks)} valid embeddings.")
            st.text_area("Summary", summary, height=100)
            
            # Graph Data
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            if code_graph_data["nodes"]:
                nodes.extend(code_graph_data["nodes"])
                edges.extend(code_graph_data["edges"])
            
            st.write(f"Extracted {len(nodes)} nodes and {len(edges)} edges.")
            keywords = [n['name'] for n in nodes]
            
            # 4. Storage
            st.text("Storing data...")
            
            file_meta = db.upload_file_metadata(
                project_id=st.session_state.selected_project_id,
                name=uploaded_file.name,
                path=str(file_path),
                summary=summary,
                metadata={"keywords": keywords}
            )
            
            if file_meta:
                file_id = file_meta['id']
                db.store_chunks(file_id, chunks)
                db.store_keywords(file_id, keywords)
                db.store_graph_data(file_id, nodes, edges)
                st.success("✅ Ingestion Complete!")
            else:
                st.error("Failed to upload file metadata.")
                
        except Exception as e:
            st.error(f"Error during ingestion: {e}")
            import traceback
            st.text(traceback.format_exc())
