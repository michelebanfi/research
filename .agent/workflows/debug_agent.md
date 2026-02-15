---
description: How to debug common agent issues
---

1. Check the logs in `logs/` directory for detailed error messages.
2. If Web Search fails with SSL/Protocol errors, modify `src/tools.py` to use `backend="lite"` in `DDGS()`.
3. If UI elements are missing (like scores), trace the data flow from `src/tools.py` -> `src/agent_graph.py` -> `src/ui/chat.py`.
4. Use `grep_search` to find where keys are being set or stripped.
5. Ensure `_sanitize_for_state` in `agent_graph.py` handles all data types (especially numpy).
6. Verify database RPCs in `src/database.py` return efficiently (avoid N+1).
