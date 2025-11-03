import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import io
import base64
from datetime import datetime
import json
import random
import requests
from io import BytesIO

import os

# Configure Streamlit page
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend methods for anomaly detection
class AnomalyDetector:
    def __init__(self):
        self.subdomain_data = {
            "RelAPG": {"nodes": 3, "connections": [(0, 1), (1, 2), (0, 2)]},
            "IdentityAPG": {"nodes": 3, "connections": [(0, 1), (1, 2)]},
            "PreferenceAPG": {"nodes": 3, "connections": [(0, 1), (1, 2), (2, 0)]}
        }

    def detect_anomalies(self, subdomain, node_id, features_config):
        """
        Mock anomaly detection method
        Returns count of anomalies and feature values
        """
        # Simulate anomaly detection based on selected features and thresholds
        anomaly_count = random.randint(5, 50)
        feature_values = {}

        for feature, threshold in features_config.items():
            if feature == "ResponseTime":
                feature_values[feature] = round(random.uniform(100, 2000), 2)
            elif feature == "Client":
                feature_values[feature] = f"Client_{random.randint(1, 100)}"
            elif feature == "ErrorCode":
                feature_values[feature] = random.choice([200, 404, 500, 503])
            elif feature == "ErrorMessage":
                feature_values[feature] = random.choice([
                    "Connection timeout", "Invalid request", "Server error", "Not found"
                ])

        return {
            "anomaly_count": anomaly_count,
            "feature_values": feature_values,
            "threshold_config": features_config,
            "subdomain": subdomain,
            "node_id": node_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def generate_pdf_report(analysis_results):
    """Generate PDF report for anomaly analysis results"""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Anomaly Detection Report")

    # Metadata
    p.setFont("Helvetica", 12)
    y_position = height - 100

    p.drawString(50, y_position, f"Subdomain: {analysis_results['subdomain']}")
    y_position -= 20
    p.drawString(50, y_position, f"Node ID: {analysis_results['node_id']}")
    y_position -= 20
    p.drawString(50, y_position, f"Timestamp: {analysis_results['timestamp']}")
    y_position -= 20
    p.drawString(50, y_position, f"Anomaly Count: {analysis_results['anomaly_count']}")
    y_position -= 40

    # Feature values
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y_position, "Feature Analysis:")
    y_position -= 30

    p.setFont("Helvetica", 10)
    for feature, value in analysis_results['feature_values'].items():
        threshold = analysis_results['threshold_config'][feature]
        p.drawString(70, y_position, f"{feature}: {value} (Threshold: {threshold})")
        y_position -= 15

    p.save()
    buffer.seek(0)
    return buffer

def create_topological_graph(subdomain, node_configs=None):
    """Create interactive topological graph using plotly"""
    detector = AnomalyDetector()
    graph_data = detector.subdomain_data[subdomain]

    # Create networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["nodes"]))
    G.add_edges_from(graph_data["connections"])

    # Get node positions
    pos = nx.spring_layout(G, seed=42)

    # Create plotly figure
    fig = go.Figure()

    # Map node indices to service names (A, B, C, ...)
    service_names = ['Service' + chr(65 + i) for i in range(graph_data["nodes"])]

    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_names = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        service_name = service_names[node]
        
        # Check if node has been configured
        if node_configs and node in node_configs:
            node_text.append(f"{service_name}<br>Configured ✓")
            node_colors.append('lightgreen')
        else:
            node_text.append(f"{service_name}<br>Click to configure")
            node_colors.append('lightblue')
        node_names.append(service_name)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_names,
        textposition="middle center",
        hovertext=node_text,
        marker=dict(
            size=50,
            color=node_colors,
            line=dict(width=2, color='darkblue')
        )
    ))

    fig.update_layout(
        title=f"Topological Graph - {subdomain}",
        title_font_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Click on nodes to configure anomaly detection parameters",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig

# Initialize session state
if 'selected_subdomain' not in st.session_state:
    st.session_state.selected_subdomain = None
if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False
if 'node_configs' not in st.session_state:
    st.session_state.node_configs = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'current_node' not in st.session_state:
    st.session_state.current_node = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

# Main dashboard
st.title("🔍 Anomaly Detection Dashboard")
st.markdown("---")

# Sidebar for subdomain selection
with st.sidebar:
    st.header("Configuration")

    # SubDomain dropdown
    subdomain_options = ["RelAPG", "IdentityAPG", "PreferenceAPG"]
    selected_subdomain = st.selectbox(
        "SubDomain",
        options=["Select SubDomain..."] + subdomain_options,
        key="subdomain_selector"
    )

    if selected_subdomain != "Select SubDomain..." and selected_subdomain != st.session_state.selected_subdomain:
        st.session_state.selected_subdomain = selected_subdomain
        st.session_state.show_graph = True
        st.session_state.node_configs = {}
        st.session_state.analysis_results = {}
        st.rerun()

# Main content area
if st.session_state.show_graph and st.session_state.selected_subdomain:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Topological Graph - {st.session_state.selected_subdomain}")

        # Create and display graph
        fig = create_topological_graph(
            st.session_state.selected_subdomain, 
            st.session_state.node_configs
        )

        # Display graph with click handling
        selected_points = st.plotly_chart(fig, use_container_width=True, key="topology_graph")

        # Node selection buttons
        st.markdown("### Node Configuration")
        node_cols = st.columns(3)

        for i in range(3):
            with node_cols[i]:
                if st.button(f"Configure Node {i}", key=f"node_btn_{i}"):
                    st.session_state.current_node = i

    with col2:
        if st.session_state.current_node is not None:
            st.subheader("Features to determine anomalies")

            node_id = st.session_state.current_node

            # Primary feature selection
            st.markdown("**Select Features:**")
            primary_options = ["ResponseTime", "Client", "ErrorCode", "ErrorMessage"]
            selected_features = st.multiselect(
                "Primary Features",
                primary_options,
                key=f"features_{node_id}"
            )

            # Secondary threshold selection for each selected feature
            feature_config = {}
            if selected_features:
                st.markdown("**Set Thresholds:**")
                secondary_options = [0.01, 0.05, 0.10, 0.20]

                for feature in selected_features:
                    threshold = st.selectbox(
                        f"Threshold for {feature}",
                        secondary_options,
                        key=f"threshold_{feature}_{node_id}"
                    )
                    feature_config[feature] = threshold

            # CSV File Upload Section
            st.markdown("---")
            st.markdown("**📁 Upload Data File:**")
            
            # Add sample CSV download option
            sample_data = {
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
                'value': np.random.normal(100, 20, 100),
                'response_time': np.random.exponential(500, 100),
                'error_count': np.random.poisson(2, 100)
            }
            sample_df = pd.DataFrame(sample_data)
            sample_csv = sample_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_csv,
                file_name="sample_data.csv",
                mime="text/csv",
                key=f"sample_download_{node_id}"
            )
            st.caption("Download this sample file to see the expected CSV format")
            
            # Add format instructions
            with st.expander("📋 CSV Format Requirements"):
                st.markdown("""
                **Your CSV file should contain:**
                - **Headers**: First row should contain column names
                - **Numeric Data**: At least one column with numeric values for analysis
                - **No Empty Rows**: Remove any completely empty rows
                - **Proper Delimiters**: Use commas (,) as column separators
                
                **Example format:**
                ```
                timestamp,value,response_time,error_count
                2024-01-01 00:00:00,95.2,450.1,1
                2024-01-01 01:00:00,102.8,512.3,0
                ```
                """)
            
            uploaded_file = st.file_uploader(
                f"Choose a CSV file for Node {node_id}",
                type=['csv'],
                key=f"file_uploader_{node_id}"
            )
            
            if uploaded_file is not None:
                # Store uploaded file in session state
                st.session_state.uploaded_files[node_id] = uploaded_file
                
                # Read and preview the CSV data
                try:
                    # Check if file is empty
                    if uploaded_file.size == 0:
                        st.error("❌ The uploaded file is empty!")
                        st.session_state.uploaded_files[node_id] = None
                    else:
                        # Try to read the CSV file
                        df_uploaded = pd.read_csv(uploaded_file)
                        os.makedirs('data', exist_ok=True)
    
                        # Save the uploaded file
                        file_path = os.path.join('data', f'node_{node_id}_{uploaded_file.name}')
                        df_uploaded.to_csv(file_path, index=False)
                        st.success(f"✅ File saved to: {file_path}")

                        
                        # Check if DataFrame is empty or has no columns
                        if df_uploaded.empty or len(df_uploaded.columns) == 0:
                            st.error("❌ The CSV file contains no data or columns!")
                            st.session_state.uploaded_files[node_id] = None
                        else:
                            st.success(f"✅ File uploaded successfully! Shape: {df_uploaded.shape}")
                            
                            # Show data preview
                            st.markdown("**Data Preview:**")
                            st.dataframe(df_uploaded.head(), use_container_width=True)
                            
                            # Show column information
                            st.markdown("**Available Columns:**")
                            st.write(f"Columns: {list(df_uploaded.columns)}")
                            st.write(f"Rows: {len(df_uploaded)}")
                            
                            # Validate that there are numeric columns for analysis
                            numeric_columns = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
                            if not numeric_columns:
                                st.warning("⚠️ Warning: No numeric columns found. Analysis may not work properly.")
                            else:
                                st.info(f"📊 Numeric columns available for analysis: {numeric_columns}")
                    
                except pd.errors.EmptyDataError:
                    st.error("❌ The CSV file is empty or has no columns!")
                    st.session_state.uploaded_files[node_id] = None
                except pd.errors.ParserError as e:
                    st.error(f"❌ Error parsing CSV file: {str(e)}")
                    st.session_state.uploaded_files[node_id] = None
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {str(e)}")
                    st.session_state.uploaded_files[node_id] = None
            else:
                # Clear stored file if no file is uploaded
                if node_id in st.session_state.uploaded_files:
                    del st.session_state.uploaded_files[node_id]

            # Trigger button
            if st.button("🚀 Trigger Analysis", key=f"trigger_{node_id}"):
                # Check if file is uploaded
                if node_id not in st.session_state.uploaded_files or st.session_state.uploaded_files[node_id] is None:
                    st.error("❌ Please upload a CSV file before triggering analysis!")
                else:
                    print('before')
                    uploaded_file = os.path.join('data', 'node_0_test.csv')
                    #uploaded_file = os.path('node_0_test.csv')
                    #uploaded_file = st.session_state.uploaded_files[node_id]
                    print('uploaded_file--> ', uploaded_file)
                    print('after')
                    df = None
                    # Validate file again before processing
                    try:
                        # Check if file exists
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"File not found: {file_path}")
    
                        # Get file size
                        file_size = os.path.getsize(file_path)
                        print(f"File size: {file_size} bytes")
    
                        # Read file content
                        with open(file_path, 'r') as file:
                            # Read as text
                            content = file.read()
                            print("File content:", content[:200])  # Print first 200 chars
        
                        # Or read as pandas DataFrame if it's a CSV
                        file.seek(0)  # Reset file pointer
                        df = pd.read_csv(file)
                        print("DataFrame shape:", df.shape)
        
                    except Exception as e:
                        print(f"Error reading file: {e}")

                    
                    # Convert to CSV for API request
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    
                    # Prepare the API request
                    files = {
                        'file': ('node_data.csv', csv_data, 'text/csv')
                    }
                    
                    # Use selected features if available, otherwise use all numeric columns
                    if selected_features:
                        # Filter to only include selected features that exist in the uploaded data
                        available_features = [f for f in selected_features if f in df.columns]
                        if not available_features:
                            st.error("❌ None of the selected features exist in the uploaded data!")
                            st.stop()
                        params = {
                            'selected_features': available_features,
                            'contamination': 0.1,
                            'random_state': 42
                        }
                    else:
                        # Use all numeric columns if no features selected
                        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                        if not numeric_columns:
                            st.error("❌ No numeric columns found in the uploaded data!")
                            st.stop()
                        params = {
                            'selected_features': numeric_columns,
                            'contamination': 0.1,
                            'random_state': 42
                        }
                    
                    # Make the API request
                    try:
                        response = requests.post(
                            'http://localhost:8000/predict-csv',
                            files=files,
                            params=params
                        )
                        response.raise_for_status()
                        
                        # Process and display results
                        result = response.json()
                        st.success(f"Analysis complete! Found {result['n_anomalies']} anomalies.")
                        
                        # Display results in a DataFrame
                        df['prediction'] = ['Anomaly' if p == -1 else 'Normal' for p in result['predictions']]
                        df['anomaly_score'] = result['anomaly_scores']
                        
                        st.markdown("**Analysis Results:**")
                        st.dataframe(df, use_container_width=True)
                        
                        # Store results in session state
                        st.session_state.analysis_results[node_id] = {
                            "anomaly_count": result['n_anomalies'],
                            "feature_values": {col: df[col].mean() for col in df.columns if col in ['prediction', 'anomaly_score']},
                            "threshold_config": feature_config,
                            "subdomain": st.session_state.selected_subdomain,
                            "node_id": node_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # You can add more visualizations here using the results
                        
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error calling API: {str(e)}")

            # Display results if available
            if node_id in st.session_state.analysis_results:
                results = st.session_state.analysis_results[node_id]

                st.markdown("### Analysis Results")
                st.metric("Anomaly Count", results["anomaly_count"])

                st.markdown("**Feature Values:**")
                for feature, value in results["feature_values"].items():
                    st.text(f"{feature}: {value}")

                # PDF download
                if st.button(f"📄 Download PDF Report", key=f"pdf_{node_id}"):
                    pdf_buffer = generate_pdf_report(results)
                    st.download_button(
                        label="Download Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"anomaly_report_node_{node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key=f"download_{node_id}"
                    )

                # Continue/Abort flow
                st.markdown("### Flow Control")
                flow_col1, flow_col2 = st.columns(2)

                with flow_col1:
                    if st.button("➡️ Continue Flow", key=f"continue_{node_id}"):
                        next_node = (node_id + 1) % 3
                        st.session_state.current_node = next_node
                        st.info(f"Continuing to Node {next_node}")
                        st.rerun()

                with flow_col2:
                    if st.button("⛔ Abort Flow", key=f"abort_{node_id}"):
                        st.session_state.current_node = None
                        st.warning("Flow aborted")
                        st.rerun()

# Display summary
if st.session_state.node_configs:
    st.markdown("---")
    st.subheader("Configuration Summary")

    summary_data = []
    for node_id, config in st.session_state.node_configs.items():
        for feature, threshold in config.items():
            summary_data.append({
                "Node": f"Node {node_id}",
                "Feature": feature,
                "Threshold": threshold,
                "Status": "✅ Configured"
            })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Anomaly Detection Dashboard - Real-time Topological Analysis*")
