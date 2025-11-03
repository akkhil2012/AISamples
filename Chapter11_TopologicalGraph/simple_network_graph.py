
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Set page title
st.title("🌐 Simple D3.js Network Graph Example")
st.write("A minimal example showing how to create an interactive network graph in Streamlit.")

# Create nodes
nodes = [
    Node(id="A", label="Alice", size=25, color="#FF6B6B"),
    Node(id="B", label="Bob", size=30, color="#4ECDC4"), 
    Node(id="C", label="Charlie", size=20, color="#45B7D1"),
    Node(id="D", label="Diana", size=35, color="#96CEB4"),
    Node(id="E", label="Eve", size=28, color="#FFEAA7")
]

# Create edges (connections between nodes)
edges = [
    Edge(source="A", target="B", label="friends"),
    Edge(source="B", target="C", label="colleagues"),
    Edge(source="C", target="D", label="neighbors"),
    Edge(source="D", target="E", label="family"),
    Edge(source="E", target="A", label="friends")
]

# Configure the graph
config = Config(
    width=600, 
    height=400,
    directed=True,  # Show arrow directions
    physics=True,   # Enable node physics/movement
    hierarchical=False
)

# Display the graph
st.subheader("Interactive Network")
st.write("Click and drag the nodes to rearrange them!")

return_value = agraph(nodes=nodes, edges=edges, config=config)

# Show some basic statistics
st.write(f"**Nodes:** {len(nodes)}")
st.write(f"**Edges:** {len(edges)}")

# Show installation instructions
with st.expander("Installation Instructions"):
    st.code("pip install streamlit streamlit-agraph")
    st.write("Then run: `streamlit run your_app.py`")

# Show the selected node (if any)
if return_value:
    st.write("**Selected Node:**")
    st.json(return_value)
