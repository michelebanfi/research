import streamlit as st
import json
import time
from typing import Dict, Any, List

def render_process_monitor():
    """
    Renders the Process Monitor tab content.
    Consumes events from st.session_state.agent_events
    """
    st.markdown("### 🧠 Live Process Monitor")
    
    if "agent_events" not in st.session_state:
        st.session_state.agent_events = []
        
    events = st.session_state.agent_events
    
    if not events:
        st.info("No active process. Start a chat to see the agent's reasoning.")
        return
    
    # Reverse order for latest first? Or Keep chronological?
    # Chronological is better for following reasoning.
    
    # Group events by iteration if possible?
    current_iteration = 0
    
    for i, event in enumerate(events):
        event_type = event.get("type", "unknown")
        content = event.get("content", "")
        metadata = event.get("metadata", {})
        timestamp = event.get("timestamp_str", "")
        
        # Iteration Header
        if event_type == "thought" and metadata.get("iteration") is not None:
             iter_num = metadata.get("iteration")
             if iter_num != current_iteration:
                 st.divider()
                 st.caption(f"🔄 **Iteration {iter_num}**")
                 current_iteration = iter_num
        
        # Render Event
        if event_type == "thought":
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(f"**Thought:** {content}")
                
        elif event_type == "tool":
            with st.chat_message("assistant", avatar="🔧"):
                st.markdown(f"**To Do:** {content}")
                if "argument" in metadata:
                    with st.expander("Tool Arguments"):
                        st.code(metadata["argument"], language="json")
                        
        elif event_type == "tool_result":
            with st.chat_message("assistant", avatar="✅"):
                st.markdown(f"**Tool Output:** {content}")
                if "preview" in metadata:
                     st.caption(f"Preview: {metadata['preview'][:200]}...")
                     
        elif event_type == "search":
            with st.chat_message("assistant", avatar="🔍"):
                st.markdown(f"**Search:** {content}")
                if "queries" in metadata:
                    st.json(metadata["queries"])
                if "count" in metadata:
                     st.caption(f"Found {metadata['count']} items.")
                     
        elif event_type == "result":
            st.success(f"🎉 **Result:** {content}")
            
        elif event_type == "error":
            st.error(f"❌ **Error:** {content}")
            
        else:
            st.text(f"[{timestamp}] {event_type}: {content}")

    # Auto-scroll? Streamlit doesn't support generic auto-scroll easily.
