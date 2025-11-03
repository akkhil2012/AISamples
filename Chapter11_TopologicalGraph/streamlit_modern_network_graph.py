
import streamlit as st
import pandas as pd
import numpy as np
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

# Set page configuration
st.set_page_config(page_title="Modern Network Graph with st-link-analysis", layout="wide")

# Title and description
st.title("🔗 Modern Network Visualization with st-link-analysis")
st.markdown("""
This example uses the `st-link-analysis` component, which provides advanced interactivity 
features like fullscreen mode, dynamic sidebars, and JSON export capabilities.
""")

# Sidebar for configuration
st.sidebar.header("Graph Settings")
layout_type = st.sidebar.selectbox("Layout Algorithm", ["cose", "circle", "grid", "breadthfirst", "concentric"])
show_labels = st.sidebar.checkbox("Show Node Labels", value=True)
enable_physics = st.sidebar.checkbox("Enable Physics", value=True)

# Sample data for network
def create_network_data():
    """Create sample network data"""
    # Define nodes with various properties
    nodes = [
        {"data": {"id": "1", "label": "User A", "type": "person", "department": "Engineering"}},
        {"data": {"id": "2", "label": "User B", "type": "person", "department": "Marketing"}},
        {"data": {"id": "3", "label": "User C", "type": "person", "department": "Sales"}},
        {"data": {"id": "4", "label": "Project Alpha", "type": "project", "status": "active"}},
        {"data": {"id": "5", "label": "Project Beta", "type": "project", "status": "planning"}},
        {"data": {"id": "6", "label": "Database", "type": "resource", "importance": "high"}},
        {"data": {"id": "7", "label": "API Gateway", "type": "resource", "importance": "medium"}},
        {"data": {"id": "8", "label": "Frontend App", "type": "resource", "importance": "high"}}
    ]

    # Define edges (relationships)
    edges = [
        {"data": {"id": "e1", "source": "1", "target": "4", "relationship": "assigned_to"}},
        {"data": {"id": "e2", "source": "2", "target": "4", "relationship": "collaborates"}},
        {"data": {"id": "e3", "source": "1", "target": "5", "relationship": "leads"}},
        {"data": {"id": "e4", "source": "3", "target": "5", "relationship": "supports"}},
        {"data": {"id": "e5", "source": "4", "target": "6", "relationship": "uses"}},
        {"data": {"id": "e6", "source": "4", "target": "7", "relationship": "depends_on"}},
        {"data": {"id": "e7", "source": "5", "target": "8", "relationship": "develops"}},
        {"data": {"id": "e8", "source": "8", "target": "7", "relationship": "connects_via"}},
        {"data": {"id": "e9", "source": "1", "target": "2", "relationship": "works_with"}},
        {"data": {"id": "e10", "source": "2", "target": "3", "relationship": "coordinates"}}
    ]

    return {"nodes": nodes, "edges": edges}

# Create network elements
elements = create_network_data()

# Define node styles based on node type
node_styles = [
    NodeStyle("person", "#FF6B6B", "person", size=30),
    NodeStyle("project", "#4ECDC4", "work", size=40),
    NodeStyle("resource", "#45B7D1", "storage", size=25)
]

# Define edge styles based on relationship type
edge_styles = [
    EdgeStyle("assigned_to", color="#FF9999", directed=True, label=show_labels),
    EdgeStyle("collaborates", color="#99CCFF", directed=False, label=show_labels),
    EdgeStyle("leads", color="#FF6666", directed=True, label=show_labels),
    EdgeStyle("supports", color="#99FF99", directed=True, label=show_labels),
    EdgeStyle("uses", color="#FFCC99", directed=True, label=show_labels),
    EdgeStyle("depends_on", color="#CC99FF", directed=True, label=show_labels),
    EdgeStyle("develops", color="#FFFF99", directed=True, label=show_labels),
    EdgeStyle("connects_via", color="#99FFFF", directed=True, label=show_labels),
    EdgeStyle("works_with", color="#FFB399", directed=False, label=show_labels),
    EdgeStyle("coordinates", color="#B3FF99", directed=False, label=show_labels)
]

# Display the graph
st.subheader("Interactive Network Visualization")

# Create the interactive graph
result = st_link_analysis(
    elements=elements,
    layout={"name": layout_type},
    node_styles=node_styles,
    edge_styles=edge_styles,
    key="network_graph"
)

# Display information about the selected elements
if result:
    st.subheader("Selected Elements")
    if result.get("nodes"):
        st.write("**Selected Nodes:**")
        for node_id in result["nodes"]:
            node_info = next((n for n in elements["nodes"] if n["data"]["id"] == node_id), None)
            if node_info:
                st.json(node_info["data"])

    if result.get("edges"):
        st.write("**Selected Edges:**")
        for edge_id in result["edges"]:
            edge_info = next((e for e in elements["edges"] if e["data"]["id"] == edge_id), None)
            if edge_info:
                st.json(edge_info["data"])

# Network statistics
st.subheader("📊 Network Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Nodes", len(elements["nodes"]))
with col2:
    st.metric("Edges", len(elements["edges"]))
with col3:
    # Calculate average connections per node
    avg_connections = len(elements["edges"]) * 2 / len(elements["nodes"])
    st.metric("Avg Connections", f"{avg_connections:.1f}")
with col4:
    # Calculate network density
    max_possible_edges = len(elements["nodes"]) * (len(elements["nodes"]) - 1) / 2
    density = len(elements["edges"]) / max_possible_edges
    st.metric("Density", f"{density:.2%}")

# Node type distribution
st.subheader("🏷️ Node Type Distribution")
node_types = {}
for node in elements["nodes"]:
    node_type = node["data"]["type"]
    node_types[node_type] = node_types.get(node_type, 0) + 1

# Display as a bar chart using built-in Streamlit
type_df = pd.DataFrame(list(node_types.items()), columns=["Type", "Count"])
st.bar_chart(type_df.set_index("Type"))

# Relationship analysis
st.subheader("🔗 Relationship Analysis")
relationship_counts = {}
for edge in elements["edges"]:
    rel_type = edge["data"]["relationship"]
    relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1

rel_df = pd.DataFrame(list(relationship_counts.items()), columns=["Relationship", "Count"])
st.dataframe(rel_df)

# Export network data
st.subheader("📤 Export Options")
col1, col2 = st.columns(2)

with col1:
    if st.button("Export Network as JSON"):
        st.download_button(
            label="Download JSON",
            data=st.json(elements),
            file_name="network_data.json",
            mime="application/json"
        )

with col2:
    # Convert to DataFrame for CSV export
    nodes_df = pd.json_normalize([node["data"] for node in elements["nodes"]])
    edges_df = pd.json_normalize([edge["data"] for edge in elements["edges"]])

    csv_data = f"NODES\n{nodes_df.to_csv(index=False)}\n\nEDGES\n{edges_df.to_csv(index=False)}"
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="network_data.csv",
        mime="text/csv"
    )

# Features showcase
with st.expander("🎯 Component Features"):
    st.markdown("""
    **st-link-analysis** provides several advanced features:

    - **Interactive Toolbar**: Fullscreen mode, JSON export, layout refresh
    - **Dynamic Sidebar**: View properties of selected nodes/edges
    - **Multiple Layouts**: Various algorithms for node positioning
    - **Material Icons**: Built-in icon support for better visualization
    - **Event Handling**: Node expansion, removal, and selection events
    - **Responsive Design**: Adapts to different screen sizes
    - **Export Capabilities**: JSON export with node positions
    """)

# Installation instructions
with st.expander("📦 Installation & Setup"):
    st.code("""
# Install required packages
pip install streamlit st-link-analysis pandas

# Run this app
streamlit run app.py
    """, language="bash")

with st.expander("💻 Basic Usage Example"):
    st.code("""
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

# Define network elements
elements = {
    "nodes": [
        {"data": {"id": "1", "label": "Node 1", "type": "person"}},
        {"data": {"id": "2", "label": "Node 2", "type": "project"}}
    ],
    "edges": [
        {"data": {"id": "e1", "source": "1", "target": "2", "relationship": "works_on"}}
    ]
}

# Define styles
node_styles = [NodeStyle("person", "#FF6B6B", "person")]
edge_styles = [EdgeStyle("works_on", color="#666666")]

# Create the visualization
result = st_link_analysis(
    elements=elements,
    node_styles=node_styles,
    edge_styles=edge_styles
)
    """, language="python")
