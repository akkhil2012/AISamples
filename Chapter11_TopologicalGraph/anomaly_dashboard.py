import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import random
from datetime import datetime
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AnomalyDetector:
    def __init__(self):
        self.subdomain_data = {
            "ServiceOne": {"nodes": 3, "connections": [(0, 1), (1, 2), (0, 2)]},
            "ServiceTwo": {"nodes": 3, "connections": [(0, 1), (1, 2)]},
            "ServiceThree": {"nodes": 3, "connections": [(0, 1), (1, 2), (2, 0)]}
        }

def create_d3_force_graph(subdomain, node_configs=None):
    """Create D3 force-directed graph"""
    detector = AnomalyDetector()
    graph_data = detector.subdomain_data[subdomain]
    
    nodes = []
    edges = []
    
    # Create nodes
    for i in range(graph_data["nodes"]):
        node = Node(
            id=str(i),
            label=f"Service{chr(65 + i)}",
            size=25,
            color="#4CAF50" if node_configs and i in node_configs else "#2196F3",
            font={"color": "white", "size": 14},
            title=f"Service{chr(65 + i)} - {'Configured' if node_configs and i in node_configs else 'Click to configure'}"
        )
        nodes.append(node)
    
    # Create edges
    for source, target in graph_data["connections"]:
        edge = Edge(source=str(source), target=str(target), width=2)
        edges.append(edge)
    
    # Configure the graph
    config = Config(
        width=800,
        height=600,
        directed=False,
        physics={
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
            },
            "minVelocity": 0.75,
            "solver": "barnesHut"
        },
        node={"highlightBehavior": True},
        highlightColor="#FFA500",
        collapsible=False,
        staticGraphWithDragAndDrop=False,
    )
    
    return nodes, edges, config

# Initialize session state
if 'selected_subdomain' not in st.session_state:
    st.session_state.selected_subdomain = None
if 'current_node' not in st.session_state:
    st.session_state.current_node = None
if 'node_configs' not in st.session_state:
    st.session_state.node_configs = {}
if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False

# Sidebar for subdomain selection
with st.sidebar:
    st.header("Dashboard Controls")
    subdomain = st.selectbox(
        "Select Subdomain",
        ["ServiceOne", "ServiceTwo", "ServiceThree"]
    )
    
    if st.button("Load Graph"):
        st.session_state.selected_subdomain = subdomain
        st.session_state.show_graph = True
        st.session_state.current_node = None
        st.rerun()

# Main content
st.title("Anomaly Detection Dashboard")

if st.session_state.get('show_graph') and st.session_state.selected_subdomain:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Topological Graph - {st.session_state.selected_subdomain}")
        
        # Create and display the D3 force-directed graph
        nodes, edges, config = create_d3_force_graph(
            st.session_state.selected_subdomain,
            st.session_state.node_configs
        )
        
        # Handle node selection
        selected_node = agraph(
            nodes=nodes,
            edges=edges,
            config=config
        )
        
        # Update session state when a node is selected
        if selected_node:
            st.session_state.current_node = int(selected_node)
            st.rerun()
    
    with col2:
        # This column will show the configuration panel when a node is selected
        if 'current_node' in st.session_state and st.session_state.current_node is not None:
            node_id = st.session_state.current_node
            
            # Create a container for the configuration panel
            with st.expander(f"⚙️ Node {node_id} Configuration", expanded=True):
                # Display node information
                node_info = {
                    "Node ID": node_id,
                    "Service Name": f"Service{chr(65 + node_id)}",
                    "Status": "Configured" if node_id in st.session_state.node_configs else "Not Configured"
                }
                
                st.json(node_info)
                
                # Configuration form
                with st.form(f"config_form_{node_id}"):
                    st.write("### Configuration Settings")
                    
                    # Example configuration options
                    threshold = st.slider(
                        "Anomaly Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        key=f"threshold_{node_id}"
                    )
                    
                    monitoring = st.checkbox(
                        "Enable Monitoring",
                        value=True,
                        key=f"monitoring_{node_id}"
                    )
                    
                    # Form actions
                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("💾 Save")
                    with col2:
                        if st.button("❌ Close"):
                            st.session_state.current_node = None
                            st.rerun()
                    
                    if submitted:
                        if node_id not in st.session_state.node_configs:
                            st.session_state.node_configs[node_id] = {}
                        
                        st.session_state.node_configs[node_id].update({
                            "threshold": threshold,
                            "monitoring_enabled": monitoring,
                            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        st.success("Configuration saved!")
                        # Update the node color to show it's configured
                        for node in nodes:
                            if node.id == str(node_id):
                                node.color = "#4CAF50"
                        
                        # Rerun to update the graph
                        st.rerun()
        else:
            st.info("ℹ️ Click on any node to configure it")
