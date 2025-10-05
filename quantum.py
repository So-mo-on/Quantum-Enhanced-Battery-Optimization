import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Set page config
st.set_page_config(page_title="RayQ Battery Optimizer", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; font-size: 72px;'>RayQ</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Quantum-Enhanced Battery Optimization</h3>", unsafe_allow_html=True)

# Initialize session state
if 'computation_done' not in st.session_state:
    st.session_state.computation_done = False

# Sidebar inputs
st.sidebar.header("Configuration")
st.sidebar.button("Upload", use_container_width=True)

N = st.sidebar.number_input("Number of Locations (N)", min_value=5, max_value=50, value=14, step=1)
budget = st.sidebar.number_input("Battery Budget", min_value=1, max_value=20, value=5, step=1)
T = st.sidebar.number_input("Time Periods ", min_value=24, max_value=168, value=24, step=24)

st.sidebar.header("Battery Parameters")
P_max = st.sidebar.number_input("Max Power (kW)", min_value=10.0, max_value=200.0, value=50.0, step=10.0)
C_cap = st.sidebar.number_input("Capacity (kWh)", min_value=500.0, max_value=5000.0, value=2000.0, step=100.0)

# Run Optimization Button
if st.sidebar.button("Run Optimization", type="primary", use_container_width=True):
    st.session_state.computation_done = True
    st.warning("🚀 Buy the premium version to run the optimization!")

# Display results placeholder
if st.session_state.computation_done:
    st.markdown("---")
    st.header("Optimization Results")
    st.info("⚡ Optimization is locked in the free version. Upgrade to premium to see results.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by Quantum Computing & Optimization</p>", unsafe_allow_html=True)
