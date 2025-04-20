import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import io

# API endpoint
API_URL = "http://api:8000"

# The exact 70 standard feature names in the order expected by the model
STANDARD_FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min"
]

# Page configuration
st.set_page_config(
    page_title="Cyberattack Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling with ADDED SPACING BETWEEN TABS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;  /* INCREASED GAP BETWEEN TABS */
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;  /* INCREASED BORDER RADIUS */
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 15px;  /* ADDED PADDING */
        padding-right: 15px;  /* ADDED PADDING */
        margin-right: 5px;  /* ADDED MARGIN */
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    div.block-container {padding-top: 1rem;}
    .highlight-red {color: #ff4b4b; font-weight: bold;}
    .highlight-yellow {color: #ffa500; font-weight: bold;}
    .highlight-green {color: #00cc96; font-weight: bold;}
    .card {
        border-radius: 5px;
        background-color: white;
        padding: 20px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 16px;
        color: #555;
    }
    .flex-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .alert-high {background-color: #fee; border-left: 5px solid #ff4b4b;}
    .alert-medium {background-color: #fec; border-left: 5px solid #ffa500;}
    .alert-low {background-color: #efe; border-left: 5px solid #00cc96;}
    .tooltip {position: relative; display: inline-block;}
    .upload-section {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border: 1px dashed #ccc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize state for demo data if not exists
if 'events' not in st.session_state:
    st.session_state.events = []
    
if 'attack_stats' not in st.session_state:
    st.session_state.attack_stats = {
        'total': 0,
        'alert': 0,
        'monitor': 0,
        'ignore': 0,
        # Updated to include the 7 classes from the model
        'types': {'BENIGN': 0, 'Bot': 0, 'Brute Force': 0, 'DDoS': 0, 'DoS': 0, 'Port Scan': 0, 'Web Attack': 0}
    }
    
if 'history_data' not in st.session_state:
    # Generate some historical data for demo
    dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=30), 
                          end=datetime.datetime.now(), freq='D')
    history = []
    for date in dates:
        # Random data but with a consistent pattern
        if date.weekday() in [5, 6]:  # Weekend
            attack_count = random.randint(2, 8)
        else:
            attack_count = random.randint(5, 15)
            
        # Add some peaks
        if date.day in [10, 20]:
            attack_count += random.randint(10, 20)
            
        history.append({
            'date': date.strftime('%Y-%m-%d'),
            'total': attack_count + random.randint(20, 50),
            'attacks': attack_count,
            'dos': random.randint(0, attack_count // 3),
            'ddos': random.randint(0, attack_count // 4),
            'portscan': random.randint(0, attack_count // 2),
            'bruteforce': random.randint(0, attack_count // 3),
            'bot': random.randint(0, attack_count // 5),
            'webattack': random.randint(0, attack_count // 6)
        })
    
    st.session_state.history_data = pd.DataFrame(history)

# Initialize state for uploads - increased file size limit to 1GB
if 'uploaded_traffic_data' not in st.session_state:
    st.session_state.uploaded_traffic_data = None
    
if 'uploaded_alerts' not in st.session_state:
    st.session_state.uploaded_alerts = None
    
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None

# Helper functions
def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False

def make_prediction(network_data, alert_text=None):
    """
    Make a prediction using the API with correctly formatted data.
    Updated to work with /predict/raw endpoint.
    
    Args:
        network_data: Dictionary of network traffic parameters or list of features
        alert_text: Alert message text (optional)
        
    Returns:
        API response or None if error
    """
    # Determine if network_data is already in the right format (list of features)
    if isinstance(network_data, list):
        features = network_data
    else:
        # Convert dictionary to feature array in the right order
        features = []
        
        # Get base features
        for feature_name in STANDARD_FEATURE_NAMES:
            # Try several possible key formats for matching
            snake_case = feature_name.lower().replace(' ', '_').replace('/', '_')
            camel_case = ''.join(word.capitalize() for word in feature_name.split(' '))[0].lower() + ''.join(word.capitalize() for word in feature_name.split(' '))[1:].replace('/', '')
            
            # Check all possible keys
            possible_keys = [
                feature_name,
                feature_name.lower(),
                snake_case,
                camel_case
            ]
            
            # Find the value in the dictionary
            value = 0.0  # Default if not found
            for key in possible_keys:
                if key in network_data:
                    try:
                        value = float(network_data[key])
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Add the feature value
            features.append(value)
    
    # Format the request using just traffic_features
    payload = {
        "traffic_features": features
    }
    
    # Add alert message if provided
    if alert_text:
        payload["alert_message"] = alert_text
    
    try:
        # Use the raw endpoint which handles feature arrays
        response = requests.post(f"{API_URL}/predict/raw", json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            
            # Add is_attack field for dashboard compatibility if not present
            if "is_attack" not in result:
                result["is_attack"] = result["predicted_class"] != "BENIGN"
                
            return result
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None
    
# Replace the existing helper functions with these updated versions
def get_color_from_score(score):
    """Get color based on threat score"""
    try:
        score_val = float(score)
    except (ValueError, TypeError):
        score_val = 0
        
    if score_val >= 70:
        return "#ff4b4b"  # Red
    elif score_val >= 40:
        return "#ffa500"  # Orange
    else:
        return "#00cc96"  # Green

def format_threat_score(score):
    """Format threat score with appropriate styling"""
    try:
        score_val = float(score)
    except (ValueError, TypeError):
        score_val = 0
        
    if score_val >= 70:
        return f'<span class="highlight-red">{score_val:.1f}</span>'
    elif score_val >= 40:
        return f'<span class="highlight-yellow">{score_val:.1f}</span>'
    else:
        return f'<span class="highlight-green">{score_val:.1f}</span>'

def get_alert_class(score):
    """Get alert CSS class based on threat score"""
    try:
        score_val = float(score)
    except (ValueError, TypeError):
        score_val = 0
        
    if score_val >= 70:
        return "alert-high"
    elif score_val >= 40:
        return "alert-medium"
    else:
        return "alert-low"

def add_event(prediction_result, source_ip=None):
    """Add an event to the session state"""
    if not prediction_result:
        return
    
    # Generate a random IP if none provided
    if not source_ip:
        source_ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    # Update stats
    st.session_state.attack_stats['total'] += 1
    
    action = prediction_result.get('recommended_action', 'Ignore')
    st.session_state.attack_stats[action.lower()] += 1
    
    attack_type = prediction_result.get('predicted_class', 'Unknown')
    if attack_type in st.session_state.attack_stats['types']:
        st.session_state.attack_stats['types'][attack_type] += 1
    
    # Add to events
    st.session_state.events.insert(0, {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'source_ip': source_ip,
        'predicted_class': attack_type,
        'threat_score': prediction_result.get('threat_score', 0),
        'action': action,
        'is_attack': prediction_result.get('is_attack', False)
    })
    
    # Keep only last 100 events
    if len(st.session_state.events) > 100:
        st.session_state.events = st.session_state.events[:100]

def process_traffic_csv(df):
    """Process traffic data CSV and return standardized data"""
    try:
        # Check if dataframe is valid
        if df is None or df.empty:
            st.error("Empty or invalid CSV file")
            return None
            
        # Try to clean column names (remove extra spaces, lowercase)
        df.columns = df.columns.str.strip()
        
        # Map columns to expected features
        mapped_data = []
        
        for _, row in df.iterrows():
            # Create a feature array matching STANDARD_FEATURE_NAMES
            features = []
            for feature_name in STANDARD_FEATURE_NAMES:
                feature_value = 0.0  # Default
                
                # Try different naming conventions
                possible_names = [
                    feature_name,
                    feature_name.lower(),
                    feature_name.replace(' ', '_').lower(),
                    feature_name.replace(' ', '').lower(),
                    feature_name.replace(' ', '-').lower()
                ]
                
                for name in possible_names:
                    if name in df.columns:
                        feature_value = float(row[name]) if not pd.isna(row[name]) else 0.0
                        break
                        
                features.append(feature_value)
                
            mapped_data.append(features)
            
        return mapped_data
    
    except Exception as e:
        st.error(f"Error processing traffic CSV: {str(e)}")
        return None

# Function to parse CSV text input
def parse_csv_input(csv_text):
    """
    Parse comma-separated values into a dictionary of 70 features.
    Show an error if the length is not exactly 70.
    """
    # Define the 70 keys (feature names)
    keys = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
        "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
        "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
        "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
        "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
        "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
        "Avg Bwd Segment Size", "Fwd Header Length.1", "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
        "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
        "Idle Std", "Idle Max", "Idle Min"
    ]
    
    try:
        values = csv_text.strip().split(',')

        # Check for exactly 70 features
        if len(values) != 70:
            st.error(f"Expected exactly 70 values, but got {len(values)}.")
            return None
        
        network_data = {}
        for i, key in enumerate(keys):
            try:
                value_str = values[i].strip()
                network_data[key] = float(value_str) if value_str else 0.0
            except ValueError:
                st.error(f"Error converting value for {key}: {values[i]}")
                network_data[key] = 0.0

        return network_data
    
    except Exception as e:
        st.error(f"Error parsing CSV input: {str(e)}")
        return None
def parse_csv_input(csv_text):
    """
    Parse comma-separated values into a dictionary of 70 features.
    Show an error if the length is not exactly 70.
    """
    # Define the 70 keys (feature names)
    keys = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
        "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
        "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
        "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
        "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
        "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
        "Avg Bwd Segment Size", "Fwd Header Length.1", "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
        "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
        "Idle Std", "Idle Max", "Idle Min"
    ]
    
    try:
        values = csv_text.strip().split(',')

        # Check for exactly 70 features
        if len(values) != 70:
            st.error(f"Expected exactly 70 values, but got {len(values)}.")
            return None
        
        network_data = {}
        for i, key in enumerate(keys):
            try:
                value_str = values[i].strip()
                network_data[key] = float(value_str) if value_str else 0.0
            except ValueError:
                st.error(f"Error converting value for {key}: {values[i]}")
                network_data[key] = 0.0

        return network_data
    
    except Exception as e:
        st.error(f"Error parsing CSV input: {str(e)}")
        return None


# Create a title and subtitle
st.title("🛡️ Multimodal Cyberattack Detection Dashboard")
st.markdown("Real-time monitoring and visualization of network security threats")

# Check if API is available
api_available = check_api_health()
if not api_available:
    st.warning("⚠️ API is not available. Dashboard is running in demo mode.")

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Threat Analysis", "Historical Data", "Simulation", "Batch Processing"])

# Tab 1: Overview Dashboard
with tab1:
    # Top row with key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alerts = st.session_state.attack_stats['alert']
        st.markdown(f"""
        <div class="metric-card" style="background-color: #ffebee;">
            <div class="metric-value">{total_alerts}</div>
            <div class="metric-label">Critical Alerts</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        monitor_count = st.session_state.attack_stats['monitor']
        st.markdown(f"""
        <div class="metric-card" style="background-color: #fff8e1;">
            <div class="metric-value">{monitor_count}</div>
            <div class="metric-label">Warnings</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        benign_count = st.session_state.attack_stats['types']['BENIGN']
        st.markdown(f"""
        <div class="metric-card" style="background-color: #e8f5e9;">
            <div class="metric-value">{benign_count}</div>
            <div class="metric-label">Benign Traffic</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        total_inspected = st.session_state.attack_stats['total']
        st.markdown(f"""
        <div class="metric-card" style="background-color: #e3f2fd;">
            <div class="metric-value">{total_inspected}</div>
            <div class="metric-label">Total Inspected</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row with recent events and distribution
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Recent Security Events</h3>
        """, unsafe_allow_html=True)
        
        # Display recent events
        if st.session_state.events:
            for i, event in enumerate(st.session_state.events[:5]):
                alert_class = get_alert_class(event['threat_score'])
                st.markdown(f"""
                <div class="card {alert_class}" style="margin: 5px 0; padding: 10px;">
                    <small>{event['timestamp']}</small>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>Source:</strong> {event['source_ip']} | 
                            <strong>Type:</strong> {event['predicted_class']}
                        </div>
                        <div>
                            <strong>Score:</strong> {format_threat_score(event['threat_score'])} | 
                            <strong>Action:</strong> {event['action']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<p>No events recorded yet.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Attack Type Distribution</h3>
        """, unsafe_allow_html=True)
        
        # Create a dataframe for the chart
        attack_types = []
        attack_counts = []
        for attack_type, count in st.session_state.attack_stats['types'].items():
            if attack_type != "BENIGN" and count > 0:  # Exclude benign from this chart
                attack_types.append(attack_type)
                attack_counts.append(count)
        
        if attack_types:
            attack_df = pd.DataFrame({
                'Attack Type': attack_types,
                'Count': attack_counts
            })
            
            # Sort by count
            attack_df = attack_df.sort_values('Count', ascending=False)
            
            # Create attack distribution pie chart
            fig = px.pie(
                attack_df, 
                values='Count', 
                names='Attack Type',
                color='Attack Type',
                color_discrete_map={
                    'DoS': '#FF6384',
                    'DDoS': '#FF9F40',
                    'Port Scan': '#36A2EB',
                    'Brute Force': '#FFCE56',
                    'Bot': '#4BC0C0',
                    'Web Attack': '#9966FF'
                },
                hole=0.3
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<p>No attack data available.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Third row with traffic and threat visualization
    st.markdown("""
    <div class="card">
        <h3>Live Network Traffic & Threat Levels</h3>
    """, unsafe_allow_html=True)
    
    # Create a dataframe with events grouped by hour
    if st.session_state.events:
        # Get the last 24 hours in hourly intervals
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=24)
        hourly_intervals = pd.date_range(start=start_time, end=end_time, freq='H')
        
        # Initialize dataframe
        traffic_data = pd.DataFrame({
            'time': hourly_intervals,
            'total': np.random.normal(loc=50, scale=10, size=len(hourly_intervals)),
            'attack': np.random.normal(loc=5, scale=3, size=len(hourly_intervals))
        })
        
        # Add some spikes for visualization
        for i in range(len(traffic_data)):
            if i % 6 == 0:  # Every 6 hours
                traffic_data.loc[i, 'attack'] *= 2
                traffic_data.loc[i, 'total'] *= 1.2
        
        # Create a time series chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=traffic_data['time'], 
                y=traffic_data['total'],
                name="Total Traffic",
                line=dict(color='#4285F4', width=2),
                fill='tozeroy',
                fillcolor='rgba(66, 133, 244, 0.1)'
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=traffic_data['time'], 
                y=traffic_data['attack'],
                name="Suspicious Traffic",
                line=dict(color='#EA4335', width=2),
                fill='tozeroy',
                fillcolor='rgba(234, 67, 53, 0.1)'
            ),
            secondary_y=True,
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Time")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Total Traffic (GB)", secondary_y=False)
        fig.update_yaxes(title_text="Suspicious Traffic (GB)", secondary_y=True)
        
        # Update layout
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<p>No traffic data available.</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Detailed Threat Analysis
with tab2:
    st.markdown("""
    <div class="card">
        <h3>Detailed Threat Analysis</h3>
    """, unsafe_allow_html=True)
    
    # Create threat level distribution
    threat_levels = [0] * 3  # Low, Medium, High
    
    for event in st.session_state.events:
        score = event['threat_score']
        if score >= 70:
            threat_levels[2] += 1  # High
        elif score >= 40:
            threat_levels[1] += 1  # Medium
        else:
            threat_levels[0] += 1  # Low
    
    # Create a horizontal stacked bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['Threat<br>Level'],
        x=[threat_levels[0]],
        name='Low',
        orientation='h',
        marker=dict(color='#00cc96'),
        hovertemplate="Low Threats: %{x}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=['Threat<br>Level'],
        x=[threat_levels[1]],
        name='Medium',
        orientation='h',
        marker=dict(color='#ffa500'),
        hovertemplate="Medium Threats: %{x}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=['Threat<br>Level'],
        x=[threat_levels[2]],
        name='High',
        orientation='h',
        marker=dict(color='#ff4b4b'),
        hovertemplate="High Threats: %{x}<extra></extra>"
    ))
    
    fig.update_layout(
        barmode='stack',
        height=150,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(title='Number of Events'),
        yaxis=dict(title='')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threat_filter = st.selectbox(
            "Filter by Threat Level",
            options=["All", "High", "Medium", "Low"]
        )
    
    with col2:
        action_filter = st.selectbox(
            "Filter by Recommended Action",
            options=["All", "Alert", "Monitor", "Ignore"]
        )
    
    with col3:
        attack_type_filter = st.selectbox(
            "Filter by Attack Type",
            options=["All"] + list(st.session_state.attack_stats['types'].keys())
        )
    
    # Filter events based on selections
    filtered_events = st.session_state.events
    
    if threat_filter != "All":
        if threat_filter == "High":
            filtered_events = [e for e in filtered_events if e['threat_score'] >= 70]
        elif threat_filter == "Medium":
            filtered_events = [e for e in filtered_events if 40 <= e['threat_score'] < 70]
        elif threat_filter == "Low":
            filtered_events = [e for e in filtered_events if e['threat_score'] < 40]
    
    if threat_filter != "All":
        if threat_filter == "High":
            filtered_events = [e for e in filtered_events if e['threat_score'] >= 70]
        elif threat_filter == "Medium":
            filtered_events = [e for e in filtered_events if 40 <= e['threat_score'] < 70]
        elif threat_filter == "Low":
            filtered_events = [e for e in filtered_events if e['threat_score'] < 40]
    
    if action_filter != "All":
        filtered_events = [e for e in filtered_events if e['action'] == action_filter]
    
    if attack_type_filter != "All":
        filtered_events = [e for e in filtered_events if e['predicted_class'] == attack_type_filter]
    
    # Display events table
    if filtered_events:
        events_df = pd.DataFrame(filtered_events)
        
        # Style the dataframe
        def color_threat_score(val):
            color = get_color_from_score(val)
            return f'background-color: {color}; color: white;'
        
        styled_df = events_df.style.applymap(
            color_threat_score, 
            subset=['threat_score']
        )
        
        st.dataframe(
            styled_df,
            column_order=["timestamp", "source_ip", "predicted_class", "threat_score", "action", "is_attack"],
            column_config={
                "timestamp": st.column_config.TextColumn("Time"),
                "source_ip": st.column_config.TextColumn("Source IP"),
                "predicted_class": st.column_config.TextColumn("Attack Type"),
                "threat_score": st.column_config.NumberColumn("Threat Score", format="%.1f"),
                "action": st.column_config.TextColumn("Action"),
                "is_attack": st.column_config.CheckboxColumn("Is Attack")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No events match the selected filters.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Attack Type Breakdown</h3>
        """, unsafe_allow_html=True)
        
        # Create a horizontal bar chart of attack types
        attack_types = []
        attack_counts = []
        for attack_type, count in st.session_state.attack_stats['types'].items():
            attack_types.append(attack_type)
            attack_counts.append(count)
        
        if attack_types:
            attack_df = pd.DataFrame({
                'Attack Type': attack_types,
                'Count': attack_counts
            }).sort_values('Count', ascending=True)
            
            colors = ['#4285F4' if t == 'BENIGN' else '#EA4335' for t in attack_df['Attack Type']]
            
            fig = px.bar(
                attack_df, 
                y='Attack Type',
                x='Count',
                orientation='h',
                color='Attack Type',
                color_discrete_map={
                    'BENIGN': '#4285F4',
                    'DoS': '#EA4335',
                    'DDoS': '#FF9F40',
                    'Port Scan': '#FBBC05',
                    'Brute Force': '#34A853',
                    'Bot': '#4BC0C0',
                    'Web Attack': '#9966FF'
                }
            )
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<p>No attack data available.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Threat Score Distribution</h3>
        """, unsafe_allow_html=True)
        
        if st.session_state.events:
            # Extract threat scores
            threat_scores = [event['threat_score'] for event in st.session_state.events]
            
            # Create histogram
            fig = px.histogram(
                x=threat_scores,
                nbins=20,
                labels={'x': 'Threat Score'},
                color_discrete_sequence=['#4285F4']
            )
            
            # Add vertical lines for threat thresholds
            fig.add_vline(x=40, line_dash="dash", line_color="#FBBC05", 
                          annotation_text="Monitor Threshold", annotation_position="top right")
            fig.add_vline(x=70, line_dash="dash", line_color="#EA4335", 
                          annotation_text="Alert Threshold", annotation_position="top right")
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<p>No threat data available.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Historical Data
with tab3:
    history_df = st.session_state.history_data
    
    st.markdown("""
    <div class="card">
        <h3>Historical Attack Trends</h3>
    """, unsafe_allow_html=True)
    
    # Create line chart of historical attacks
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['total'],
        name='Total Traffic',
        line=dict(color='#4285F4', width=2),
        fill='tozeroy',
        fillcolor='rgba(66, 133, 244, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['attacks'],
        name='Attacks',
        line=dict(color='#EA4335', width=2)
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(title='Date'),
        yaxis=dict(title='Count')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Attack type breakdown over time
    st.markdown("""
    <div class="card">
        <h3>Attack Type Breakdown Over Time</h3>
    """, unsafe_allow_html=True)
    
    # Create stacked area chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['dos'],
        name='DoS Attacks',
        mode='lines',
        line=dict(width=0.5, color='#EA4335'),
        stackgroup='one',
        fillcolor='rgba(234, 67, 53, 0.6)'
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['ddos'],
        name='DDoS Attacks',
        mode='lines',
        line=dict(width=0.5, color='#FF9F40'),
        stackgroup='one',
        fillcolor='rgba(255, 159, 64, 0.6)'
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['portscan'],
        name='Port Scans',
        mode='lines',
        line=dict(width=0.5, color='#FBBC05'),
        stackgroup='one',
        fillcolor='rgba(251, 188, 5, 0.6)'
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['bruteforce'],
        name='Brute Force',
        mode='lines',
        line=dict(width=0.5, color='#34A853'),
        stackgroup='one',
        fillcolor='rgba(52, 168, 83, 0.6)'
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['bot'],
        name='Bot Attacks',
        mode='lines',
        line=dict(width=0.5, color='#4BC0C0'),
        stackgroup='one',
        fillcolor='rgba(75, 192, 192, 0.6)'
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['webattack'],
        name='Web Attacks',
        mode='lines',
        line=dict(width=0.5, color='#9966FF'),
        stackgroup='one',
        fillcolor='rgba(153, 102, 255, 0.6)'
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(title='Date'),
        yaxis=dict(title='Attack Count')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 4: Simulation
with tab4:
    st.markdown("""
    <div class="card">
        <h3>Test Attack Detection</h3>
        <p>Use this simulator to test the detection capabilities against different attack types.</p>
    """, unsafe_allow_html=True)
    
    # API status check
    if not api_available:
        st.warning("⚠️ API is not available. Simulation will use synthetic data.")
    
    
    # Predefined alert messages for each attack type
    alert_messages = {
        "DoS Attack": "ALERT: DoS slowloris from 172.16.0.1 to 192.168.10.50 on port 80 using protocol TCP",
        "DDoS Attack": "ALERT: DDoS from 172.16.0.1 to 192.168.10.50 on port 80 using protocol TCP",
        "Port Scan": "ALERT: PortScan from 172.16.0.1 to 192.168.10.50 on port 80 using protocol TCP",
        "Brute Force Attack": "ALERT: Web Attack Brute Force from 172.16.0.1 to 192.168.10.50 on port 80.0 using protocol UNKNOWN",
        "Bot Attack": "ALERT: Bot from 192.168.10.12 to 52.6.13.28 on port 8080 using protocol TCP",
        "Web Attack": "ALERT: Web Attack Sql Injection from 172.16.0.1 to 192.168.10.50 on port 80.0 using protocol UNKNOWN",
        "Benign Traffic": "ALERT: BENIGN from 192.168.10.15 to 172.217.3.110 on port 443 using protocol TCP"
    }
    

    
    # Show network traffic parameters - make them editable
    st.subheader("Input Methods")
    
    # Let user choose between form and CSV input
    input_method = st.radio(
        "Select input method",
        options=["CSV Input"]
    )
    
    if input_method == "CSV Input":
        st.subheader("Enter Comma-Separated Feature Values")
        csv_input = st.text_area(
            "Enter comma-separated feature values:",
            height=100,
            help="Enter values in order: destination_port, flow_duration, total_fwd_packets, etc."
        )
        
        # Example data to show
        st.markdown("""
        <div style="font-size: 0.8em; color: #666;">
        Example: 80,80.0,2802948,4,24,0,1618279.32,934316,40,60,44,3,44,44,1,0,0,0,0,0
        </div>
        """, unsafe_allow_html=True)
        
        if csv_input:
            # Parse the CSV input
            parsed_data = parse_csv_input(csv_input)
            if parsed_data is not None:
                network_data = parsed_data
                st.success("Successfully parsed input data")
    else:
        # Show form interface
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        

    
    # Alert message input
    st.subheader("Alert Message")
    
    # First let the user select from predefined alerts or custom
    alert_selection = st.radio(
        "Select alert type",
        options=["Predefined Alert", "Custom Alert"]
    )
    
    if alert_selection == "Predefined Alert":
        # Let them select from the predefined alerts for each attack type
        alert_message = st.selectbox(
            "Select predefined alert message",
            options=list(alert_messages.values()),
            index=list(alert_messages.keys()).index(attack_type)
        )
    else:
        # Custom alert input
        alert_message = st.text_area(
            "Custom alert message", 
            value=alert_messages[attack_type], 
            height=100
        )
    
    # Create columns for submit button and results
    submit_col, reset_col = st.columns([3, 1])
    
    with submit_col:
        if st.button("Run Detection Analysis", use_container_width=True):
            with st.spinner("Analyzing traffic pattern..."):
                # Generate prediction
                if api_available:
                    result = make_prediction(network_data, alert_message)
                    
                    if result:

                        threat_score = float(result.get('threat_score', 0))
                        if threat_score >= 70:
                                result['recommended_action'] = "Alert"
                        elif threat_score >= 40:
                                result['recommended_action'] = "Monitor"
                        else:
                                result['recommended_action'] = "Ignore"
                        # Update dashboard with new prediction
                        add_event(result, source_ip="192.168.1.100")
                        
                        # Display result in a nice format
                        st.markdown("""
                        <div class="card">
                            <h3>Detection Results</h3>
                        """, unsafe_allow_html=True)

                        alert_class = get_alert_class(result['threat_score'])
                        st.markdown(f"""
                        <div class="card {alert_class}" style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-size: 20px; font-weight: bold;">
                                    {result['predicted_class']}
                                </div>
                                <div>
                                    <span style="font-size: 18px;">
                                        Threat Score: {format_threat_score(result['threat_score'])}
                                    </span>
                                </div>
                            </div>
                            <div style="margin-top: 10px;">
                                <strong>Recommended Action:</strong> {result['recommended_action']}
                            </div>
                            <div style="margin-top: 5px;">
                                <strong>Is Attack:</strong> {"Yes" if result['is_attack'] else "No"}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add additional details
# Add additional details
                        st.markdown("<h4>Detection Details</h4>", unsafe_allow_html=True)
                       
                       # Create expandable sections for more details
                        with st.expander("View Raw API Response"):
                           st.json(result)
                           
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                   # Generate synthetic result based on the 7 attack classes
                   predicted_class = ""
                   if attack_type == "Benign Traffic":
                       threat_score = random.uniform(5, 35)
                       is_attack = False
                       predicted_class = "BENIGN"
                       action = "Ignore"
                   else:
                       # Map the selection to the actual class name
                       attack_map = {
                           "DoS Attack": "DoS",
                           "DDoS Attack": "DDoS",
                           "Port Scan": "Port Scan", 
                           "Brute Force Attack": "Brute Force",
                           "Bot Attack": "Bot",
                           "Web Attack": "Web Attack"
                       }
                       predicted_class = attack_map[attack_type]
                       
                       if attack_type in ["Port Scan", "Brute Force Attack", "Bot Attack"]:
                           threat_score = random.uniform(40, 65)
                           action = "Monitor"
                       else:
                           threat_score = random.uniform(70, 95)
                           action = "Alert"
                       is_attack = True
                   
                   synthetic_result = {
                       "predicted_class": predicted_class,
                       "threat_score": threat_score,
                       "is_attack": is_attack,
                       "recommended_action": action
                   }
                   
                   # Update dashboard with synthetic prediction
                   add_event(synthetic_result, source_ip="192.168.1.100")
                   
                   # Display synthetic result
                   st.markdown("""
                   <div class="card">
                       <h3>Detection Results (Simulated)</h3>
                   """, unsafe_allow_html=True)
                   
                   alert_class = get_alert_class(threat_score)
                   st.markdown(f"""
                   <div class="card {alert_class}" style="margin-top: 10px;">
                       <div style="display: flex; justify-content: space-between; align-items: center;">
                           <div style="font-size: 20px; font-weight: bold;">
                               {predicted_class}
                           </div>
                           <div>
                               <span style="font-size: 18px;">
                                   Threat Score: {format_threat_score(threat_score)}
                               </span>
                           </div>
                       </div>
                       <div style="margin-top: 10px;">
                           <strong>Recommended Action:</strong> {action}
                       </div>
                       <div style="margin-top: 5px;">
                           <strong>Is Attack:</strong> {"Yes" if is_attack else "No"}
                       </div>
                       <div style="margin-top: 10px; font-style: italic; color: #888;">
                           Note: This is simulated data as the API is not available.
                       </div>
                   </div>
                   """, unsafe_allow_html=True)
                   
                   st.markdown("</div>", unsafe_allow_html=True)
   
    with reset_col:
       if st.button("Reset", use_container_width=True):
           # This will reset the form to defaults through Streamlit's automatic re-run
           pass
   
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 5: Batch Processing with CSV Uploads
with tab5:
   st.markdown("""
   <div class="card">
       <h3>Batch Processing with CSV Files</h3>
       <p>Upload network traffic data and alert messages to analyze multiple records at once.</p>
   </div>
   """, unsafe_allow_html=True)
   
   # API status check
   if not api_available:
       st.warning("⚠️ API is not available. CSV processing requires a connection to the API.")
   
   # Create two columns for traffic and alert uploads
   col1, col2 = st.columns(2)
   
   with col1:
       st.markdown("""
       <div class="upload-section">
           <h4>Upload Traffic Data CSV</h4>
           <p>CSV file should contain network traffic features.</p>
       </div>
       """, unsafe_allow_html=True)
       
       # Increased the maximum upload size to 1GB (1024 MB)
       traffic_file = st.file_uploader("Upload traffic CSV file", type=["csv"], key="traffic_csv")
       
       if traffic_file is not None:
           try:
               # Load the CSV data
               traffic_df = pd.read_csv(traffic_file)
               
               # Display sample of the uploaded data
               st.write("Preview of uploaded traffic data:")
               st.dataframe(traffic_df.head())
               
               # Save to session state
               st.session_state.uploaded_traffic_data = traffic_df
               
               # Show summary
               st.success(f"Successfully loaded traffic data: {traffic_df.shape[0]} rows, {traffic_df.shape[1]} columns")
               
               # Show column info
               st.write("Columns in the traffic data:")
               
               # Check for expected columns
               expected_columns = [col.lower().replace(' ', '_') for col in STANDARD_FEATURE_NAMES]
               found_columns = []
               missing_columns = []
               
               for col in expected_columns:
                   variants = [
                       col,
                       col.replace('_', ''),
                       col.replace('_', ' '),
                       col.replace('_', '-')
                   ]
                   
                   if any(v in map(str.lower, traffic_df.columns) for v in variants):
                       found_columns.append(col)
                   else:
                       missing_columns.append(col)
               
               if missing_columns:
                   st.warning(f"Missing {len(missing_columns)} expected columns. The API will use default values for these.")
                   with st.expander("View missing columns"):
                       st.write(missing_columns)
               
           except Exception as e:
               st.error(f"Error reading traffic CSV file: {str(e)}")
               st.session_state.uploaded_traffic_data = None
   
   with col2:
       st.markdown("""
       <div class="upload-section">
           <h4>Upload Alert Messages CSV </h4>
           <p>CSV file should contain alert messages to analyze along with traffic data.</p>
       </div>
       """, unsafe_allow_html=True)
       
       # Increased the maximum upload size to 1GB (1024 MB)
       alert_file = st.file_uploader("Upload alerts CSV file", type=["csv"], key="alerts_csv")
       
       if alert_file is not None:
           try:
               # Load the CSV data
               alert_df = pd.read_csv(alert_file)
               
               # Display sample of the uploaded data
               st.write("Preview of uploaded alert data:")
               st.dataframe(alert_df.head())
               
               # Save to session state
               st.session_state.uploaded_alerts = alert_df
               
               # Show summary
               st.success(f"Successfully loaded alert data: {alert_df.shape[0]} rows, {alert_df.shape[1]} columns")
               
               # Check for alert message column
               alert_column = None
               for col in alert_df.columns:
                   if 'alert' in col.lower() or 'message' in col.lower():
                       alert_column = col
                       break
               
               if alert_column:
                   st.info(f"Found alert messages in column: {alert_column}")
               else:
                   st.warning("Could not identify alert message column. Please select it during processing.")
               
           except Exception as e:
               st.error(f"Error reading alerts CSV file: {str(e)}")
               st.session_state.uploaded_alerts = None
   
   # Process uploaded files
   st.markdown("""
   <div class="card">
       <h4>Process Uploaded Files</h4>
       <p>Process your uploaded CSV files for batch detection analysis.</p>
   </div>
   """, unsafe_allow_html=True)
   
   # Create options for processing
   if st.session_state.uploaded_traffic_data is not None:
       # Only show process button if API is available
       if api_available:
           # Options for processing
           use_alerts = st.checkbox("Process with alerts", value=st.session_state.uploaded_alerts is not None)
           
           if use_alerts and st.session_state.uploaded_alerts is not None:
               # If using alerts, need to select the alert column
               alert_column = st.selectbox(
                   "Select the column containing alert messages",
                   options=st.session_state.uploaded_alerts.columns,
                   index=0
               )
               
               # Option to merge datasets or process separately
               if st.session_state.uploaded_traffic_data.shape[0] != st.session_state.uploaded_alerts.shape[0]:
                   st.warning(f"Traffic data has {st.session_state.uploaded_traffic_data.shape[0]} rows but alert data has {st.session_state.uploaded_alerts.shape[0]} rows. The data will be matched by index.")
           
           # Add button to process data
           if st.button("Process Data", type="primary"):
               with st.spinner("Processing uploaded data..."):
                   # Process the traffic data
                   traffic_features = process_traffic_csv(st.session_state.uploaded_traffic_data)
                   
                   if traffic_features:
                       results = []
                       total_rows = len(traffic_features)
                       progress_bar = st.progress(0)
                       
                       for i, feature_row in enumerate(traffic_features):
                           # Get alert message if available
                           alert_message = None
                           if use_alerts and st.session_state.uploaded_alerts is not None:
                               # Get alert for this row if index exists
                               if i < len(st.session_state.uploaded_alerts):
                                   alert_message = st.session_state.uploaded_alerts.iloc[i][alert_column]
                           
                           # Make prediction
                           prediction = make_prediction(feature_row, alert_message)
                           
                           if prediction:
                               # Use source IP from traffic data if available
                               source_ip = None
                               if 'source_ip' in st.session_state.uploaded_traffic_data.columns:
                                   source_ip = str(st.session_state.uploaded_traffic_data.iloc[i]['source_ip'])
                               elif 'src_ip' in st.session_state.uploaded_traffic_data.columns:
                                   source_ip = str(st.session_state.uploaded_traffic_data.iloc[i]['src_ip'])
                               
                               # Add to events for dashboard tracking
                               add_event(prediction, source_ip)
                               
                               # Add to results for this batch
                               results.append({
                                   'row': i+1,
                                   'predicted_class': prediction.get('predicted_class', 'Unknown'),
                                   'threat_score': prediction.get('threat_score', 0),
                                   'recommended_action': prediction.get('recommended_action', 'Unknown'),
                                   'is_attack': prediction.get('is_attack', False),
                                   'model1_confidence': prediction.get('model1_confidence', 0),
                                   'model2_confidence': prediction.get('model2_confidence', 0),
                                   'source_ip': source_ip
                               })
                           
                           # Update progress
                           progress_bar.progress((i + 1) / total_rows)
                       
                       # Store results
                       st.session_state.processed_results = pd.DataFrame(results)
                       
                       # Show success message
                       st.success(f"Successfully processed {len(results)} records!")
                   else:
                       st.error("Failed to process traffic data. Please check the format.")
       else:
           st.error("Cannot process data without API connection. Please ensure the API is running.")
   else:
       st.info("Please upload traffic data to process.")
   
   # Display results if available
   if st.session_state.processed_results is not None and not st.session_state.processed_results.empty:
       st.markdown("""
       <div class="card">
           <h4>Processing Results</h4>
       </div>
       """, unsafe_allow_html=True)
       
       # Show summary statistics
       st.subheader("Summary Statistics")
       
       # Count by class
       class_counts = st.session_state.processed_results['predicted_class'].value_counts()
       
       # Count by action
       action_counts = st.session_state.processed_results['recommended_action'].value_counts()
       
       # Create columns for statistics
       col1, col2 = st.columns(2)
       
       with col1:
           st.write("Detections by Class:")
           st.dataframe(class_counts)
           
           # Create a pie chart of classes
           fig = px.pie(
               values=class_counts.values,
               names=class_counts.index,
               title="Distribution of Detected Classes"
           )
           st.plotly_chart(fig, use_container_width=True)
           
       with col2:
           st.write("Detections by Recommended Action:")
           st.dataframe(action_counts)
           
           # Create a pie chart of actions
           fig = px.pie(
               values=action_counts.values,
               names=action_counts.index,
               title="Distribution of Recommended Actions",
               color=action_counts.index,
               color_discrete_map={
                   'Alert': '#FF6384',
                   'Monitor': '#FFCE56',
                   'Ignore': '#36A2EB'
               }
           )
           st.plotly_chart(fig, use_container_width=True)
       
       # Show threat score distribution
       st.subheader("Threat Score Distribution")
       
       # Create histogram of threat scores
       fig = px.histogram(
           st.session_state.processed_results,
           x="threat_score",
           nbins=20,
           title="Distribution of Threat Scores",
           color_discrete_sequence=['#4285F4']
       )
       
       # Add vertical lines for thresholds
       fig.add_vline(x=40, line_dash="dash", line_color="#FFCE56", 
                   annotation_text="Monitor Threshold")
       fig.add_vline(x=70, line_dash="dash", line_color="#FF6384", 
                   annotation_text="Alert Threshold")
       
       st.plotly_chart(fig, use_container_width=True)
       
       # Show data table
       st.subheader("All Results")
       
       # Display the results dataframe
       st.dataframe(
           st.session_state.processed_results,
           use_container_width=True,
           column_config={
               "row": st.column_config.NumberColumn("Row #"),
               "predicted_class": st.column_config.TextColumn("Detected Class"),
               "threat_score": st.column_config.NumberColumn("Threat Score", format="%.1f"),
               "recommended_action": st.column_config.TextColumn("Action"),
               "is_attack": st.column_config.CheckboxColumn("Is Attack"),
               "model1_confidence": st.column_config.NumberColumn("Model 1 Confidence", format="%.2f"),
               "model2_confidence": st.column_config.NumberColumn("Model 2 Confidence", format="%.2f"),
               "source_ip": st.column_config.TextColumn("Source IP")
           }
       )
       
       # Download button for results
       csv_data = st.session_state.processed_results.to_csv(index=False)
       csv_bytes = csv_data.encode()
       
       st.download_button(
           label="Download Results as CSV",
           data=csv_bytes,
           file_name="cyberattack_detection_results.csv",
           mime="text/csv",
       )

# Update function for auto-refresh
if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
   st.rerun()


# Add sidebar content
with st.sidebar:
   st.image("https://raw.githubusercontent.com/TomSchimansky/streamlit-components/master/android-chrome-512x512.png", width=100)
   st.markdown("## Dashboard Controls")
   
   # Simple settings
   st.markdown("### Settings")
   show_benign = st.checkbox("Show Benign Traffic", value=True)
   
   # Theme selection
   st.markdown("### Theme")
   theme = st.radio("", options=["Light", "Dark"], horizontal=True)
   
   # Refresh rate
   st.markdown("### Auto-Refresh")
   refresh_rate = st.selectbox("Interval", options=["Off", "30 seconds", "1 minute", "5 minutes", "15 minutes"])
   
   # API connection
   st.markdown("### API Connection")
   api_status = "✅ Connected" if api_available else "❌ Not Connected"
   st.markdown(f"**Status:** {api_status}")
   
   if not api_available:
       api_url_input = st.text_input("API URL", value=API_URL)
       if st.button("Connect") and api_url_input != API_URL:
           # Update the global API URL
           API_URL = api_url_input
           st.rerun()

   
   # Dashboard info
   st.markdown("---")
   st.markdown("### Dashboard Info")
   st.markdown("Version: 1.0.0")
   st.markdown("Last Updated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #888;">
   <p>Multimodal Cyberattack Detection System • © 2025</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh logic (if enabled)
if refresh_rate != "Off":
   seconds = {"30 seconds": 30, "1 minute": 60, "5 minutes": 300, "15 minutes": 900}
   time_to_wait = seconds[refresh_rate]
   
   # Show countdown
   countdown_placeholder = st.empty()
   
   # Add one random event every refresh to simulate activity
   #if api_available:
    #   generate_sample_event()
