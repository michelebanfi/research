import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config as AgraphConfig

def render_graph_tab(db):
    """
    Renders the Knowledge Graph Explorer tab content.
    """
    st.subheader("Knowledge Graph Explorer")
    
    st.caption("Visualizing the global knowledge graph for your project.")
    
    if st.button("🔄 Load/Refresh Graph", key="load_graph_btn"):
        with st.spinner("Loading graph data..."):
            graph_data_rows = db.get_project_graph(st.session_state.selected_project_id, limit=500)
            
            nodes = []
            edges = []
            added_nodes = set()
            
            for row in graph_data_rows:
                s_name = row['source_name']
                t_name = row['target_name']
                s_type = row['source_type']
                t_type = row['target_type']
                relation = row['edge_type']
                
                # Color by type
                type_colors = {
                    "Concept": "#FF6B6B",
                    "Tool": "#4ECDC4",
                    "System": "#45B7D1",
                    "Metric": "#96CEB4",
                    "Person": "#DDA0DD",
                }
                
                if s_name not in added_nodes:
                    color = type_colors.get(s_type, "#95A5A6")
                    nodes.append(Node(id=s_name, label=s_name, size=20, color=color))
                    added_nodes.add(s_name)
                
                if t_name not in added_nodes:
                    color = type_colors.get(t_type, "#95A5A6")
                    nodes.append(Node(id=t_name, label=t_name, size=20, color=color))
                    added_nodes.add(t_name)
                
                edges.append(Edge(source=s_name, target=t_name, label=relation))
            
            st.session_state.graph_nodes = nodes
            st.session_state.graph_edges = edges
    
    if st.session_state.graph_nodes:
        st.success(f"Showing {len(st.session_state.graph_nodes)} nodes and {len(st.session_state.graph_edges)} edges.")
        
        # Legend
        st.markdown("""
        **Legend:** 
        🔴 Concept | 
        🔵 Tool | 
        💙 System | 
        💚 Metric | 
        💜 Person
        """)
        
        config = AgraphConfig(
            width=900, 
            height=600, 
            directed=True, 
            nodeHighlightBehavior=True, 
            highlightColor="#F7A7A6",
            collapsible=True,
            physics=True
        )
        
        clicked_node = agraph(
            nodes=st.session_state.graph_nodes, 
            edges=st.session_state.graph_edges, 
            config=config
        )
        
        if clicked_node:
            st.info(f"Selected: **{clicked_node}**")
    else:
        st.info("Click 'Load/Refresh Graph' to visualize your knowledge base.")
