import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.image import imread
import networkx as nx

# Import from qubo.py
from qubo import hybrid_qaoa, plotsoc

# Set page config
st.set_page_config(page_title="RayQ Battery Optimizer", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; font-size: 72px;'>RayQ</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Quantum-Enhanced Battery Optimization</h3>", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'N' not in st.session_state:
    st.session_state.N = 14
if 'T' not in st.session_state:
    st.session_state.T = 24
if 'computation_done' not in st.session_state:
    st.session_state.computation_done = False
if 'show_plot' not in st.session_state:
    st.session_state.show_plot = False

# Sidebar inputs
st.sidebar.header("Configuration")

# Upload button (non-functional)
st.sidebar.button("Upload", use_container_width=True)

N = st.sidebar.number_input("Number of Locations (N)", min_value=5, max_value=50, value=14, step=1)
budget = st.sidebar.number_input("Battery Budget", min_value=1, max_value=20, value=5, step=1)
T = st.sidebar.number_input("Time Periods ", min_value=24, max_value=168, value=24, step=24)

# Parameters
st.sidebar.header("Battery Parameters")
P_max = st.sidebar.number_input("Max Power (kW)", min_value=10.0, max_value=200.0, value=50.0, step=10.0)
C_cap = st.sidebar.number_input("Capacity (kWh)", min_value=500.0, max_value=5000.0, value=2000.0, step=100.0)

# Run Optimization Button
if st.sidebar.button("Run Optimization", type="primary", use_container_width=True):
    with st.spinner("Running hybrid QAOA optimization..."):
        # Problem setup
        Delta_t = 1.0

        # Initial SoC - generate based on N
        np.random.seed(42)
        S0 = list(np.random.uniform(0.1, 150.0, N))

        # Cost over 24 time periods (repeating pattern for longer periods)
        C_t_base = [4.0, 3.5, 3.0, 3.0, 3.5, 4.5,
                    6.0, 7.0, 8.0, 8.5, 8.0, 7.5,
                    7.0, 6.5, 6.0, 6.5, 7.0, 8.0,
                    9.0, 9.5, 9.0, 8.0, 6.5, 5.0]
        C_t = C_t_base * (T // 24)

        # Load generation
        L_it = np.zeros((N, T))
        for i in range(N):
            base_load = np.random.uniform(0.5, 2.0)
            for t in range(T):
                hour_of_day = t % 24
                time_factor = 1.0 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12) if 6 <= hour_of_day <= 22 else 0.5
                L_it[i, t] = base_load * time_factor + np.random.uniform(-0.1, 0.1)

        # Run optimization
        res = hybrid_qaoa(C_t, L_it, Delta_t, P_max, C_cap, S0, budget)

        # Store results
        st.session_state.results = res
        st.session_state.N = N
        st.session_state.T = T
        st.session_state.computation_done = True
        st.session_state.show_plot = False
        st.success("Optimization complete!")

# Display results
if st.session_state.computation_done and st.session_state.results is not None:
    res = st.session_state.results

    st.markdown("---")
    st.header("Optimization Results")

    # Metrics
    col1, col3 = st.columns(2)
    with col1:
        st.metric("Batteries Installed", sum(res['z']))

    with col3:
        battery_locations = [i for i, z in enumerate(res['z']) if z == 1]
        st.metric("Battery Locations", str(battery_locations))

    # Show battery placement
    st.subheader("Battery Placement")
    st.write(f"Binary decision vector: {res['z']}")

    # Next button to show plot (OUTSIDE the spinner)
    if st.button("Next: Show Visualization", type="primary", use_container_width=True):
        st.session_state.show_plot = True

    # Display plot if flag is set
    if st.session_state.show_plot:
        st.markdown("---")
        st.header("State of Charge Animation")

        battery_locations = [i for i, z in enumerate(res['z']) if z == 1]
        image_path = 'city_map.jpg'

        with st.spinner("Generating animation video..."):
            # Generate the full animation with smaller figure size
            # Temporarily modify plotsoc or pass smaller dimensions
            fig_ani, ani = plotsoc(st.session_state.N, min(24, st.session_state.T), battery_locations,
                                   res['SoC_opt'], image_path)
            plt.close(fig_ani)

        st.success("Animation generated successfully!")

        # Display the video with custom width
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            video_file = open('graph_timeseries.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()

else:
    st.info("Configure parameters in the sidebar and click 'Run Optimization' to start")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by Quantum Computing & Optimization</p>",
            unsafe_allow_html=True)