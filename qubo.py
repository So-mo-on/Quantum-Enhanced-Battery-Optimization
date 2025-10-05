"""
Hybrid QAOA + classical optimizer for mixed discrete-continuous Hamiltonian.

- Binary variables: z_i (install battery or not)
- Continuous variables: p_{i,t} (power), SoC_{i,t} (state of charge)
- Hamiltonian: same as your cost function + penalty terms

Requires:
    pip install qiskit==0.45.3 qiskit-aer==0.13.3 qiskit-algorithms==0.2.2 qiskit-optimization==0.6.0 cvxpy
"""

import numpy as np
import cvxpy as cp
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from itertools import product

# -----------------------------
# Classical optimization function
# -----------------------------
def classical_opt_p_soc(z, C_t, L_it, Delta_t, P_max, C_cap, S0):
    """
    Solve for continuous p_{i,t} and SoC_{i,t} for a given z vector using QP
    """
    N = len(z)
    T = len(C_t)

    # Variables
    p = cp.Variable((N,T))
    SoC = cp.Variable((N,T))

    constraints = []

    for i in range(N):
        if z[i] == 0:
            constraints += [p[i,:] == 0, SoC[i,:] == 0]
        else:
            for t in range(T):
                constraints += [p[i,t] <= P_max, p[i,t] >= -P_max]
                constraints += [SoC[i,t] >= 0, SoC[i,t] <= C_cap]
            # SoC dynamics
            for t in range(T-1):
                constraints += [SoC[i,t+1] == SoC[i,t] - p[i,t]*Delta_t]
            # Initial SoC
            constraints += [SoC[i,0] == S0[i]]

    # Objective
    obj = 0
    for t in range(T):
        obj += C_t[t] * cp.sum(L_it[:,t] - p[:,t]) * Delta_t
    obj += cp.sum([100*z[i] for i in range(N)])  # example fix cost; adjust as needed

    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(verbose=False)

    # Return optimal energy and p, SoC
    return problem.value, p.value, SoC.value

# -----------------------------
# Function to build QUBO for z only (dummy linear cost + budget penalty)
# -----------------------------
def build_z_qubo(N, budget, fix_cost=100, mu_budget=1e4):
    """
    Minimal QUBO just for QAOA: linear cost + budget penalty
    """
    Q = np.zeros((N,N))
    linear = np.array([fix_cost]*N, dtype=float)
    const = 0.0
    # budget penalty: mu*(sum_i z_i - budget)^2
    linear += -2*mu_budget*budget*np.ones(N)
    for i in range(N):
        for j in range(i+1,N):
            Q[i,j] += 2*mu_budget
    const += mu_budget*budget**2
    return Q, linear, const

def qubo_to_qp(Q, linear, const):
    N = len(linear)
    qp = QuadraticProgram()
    for i in range(N):
        qp.binary_var(f"z{i}")
    linear_dict = {f"z{i}": float(linear[i]) for i in range(N)}
    quadratic_dict = {}
    for i in range(N):
        for j in range(i+1,N):
            val = float(Q[i,j])
            if abs(val)>1e-12:
                quadratic_dict[(f"z{i}", f"z{j}")] = val
    qp.minimize(linear=linear_dict, quadratic=quadratic_dict, constant=float(const))
    return qp

# -----------------------------
# Solve hybrid problem
# -----------------------------
def hybrid_qaoa(C_t, L_it, Delta_t, P_max, C_cap, S0, budget):
    N = len(S0)
    # Build QUBO for z
    Q, linear, const = build_z_qubo(N, budget)
    qp = qubo_to_qp(Q, linear, const)

    # Setup QAOA with new API
    backend = AerSimulator()
    sampler = BackendSampler(backend=backend, options={"shots": 1024, "seed_simulator": 123})
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=100), reps=1)
    meo = MinimumEigenOptimizer(min_eigen_solver=qaoa)
    result = meo.solve(qp)

    # Extract z candidate
    z_candidate = [int(round(x)) for x in result.x]

    # Solve classical p, SoC optimization for this z
    energy, p_opt, SoC_opt = classical_opt_p_soc(z_candidate, C_t, L_it, Delta_t, P_max, C_cap, S0)

    return {
        "z": z_candidate,
        "energy": energy,
        "p_opt": p_opt,
        "SoC_opt": SoC_opt
    }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Problem dimensions
    T = 24*365  # Changed from 3 to 24
    N = 14  # Changed from 4 to 14
    Delta_t = 1.0
    P_max = 50.0
    C_cap = 2000.0
    budget = 5  # Increased budget for more batteries

    # Initial SoC for all 14 locations
    S0 = [53.5, 0.1, 4.5, 135.0, 5.5, 6.0, 6.5, 117.0, 17.5, 28.0, 4.2, 5.8, 6.3, 7.2]

    # Cost over 24 time periods (example: varying electricity prices)
    C_t = [4.0, 3.5, 3.0, 3.0, 3.5, 4.5,  # Hours 0-5 (night/early morning)
           6.0, 7.0, 8.0, 8.5, 8.0, 7.5,  # Hours 6-11 (morning peak)
           7.0, 6.5, 6.0, 6.5, 7.0, 8.0,  # Hours 12-17 (afternoon)
           9.0, 9.5, 9.0, 8.0, 6.5, 5.0]  # Hours 18-23 (evening peak)

    # Load for 14 locations over 24 time periods
    # Random loads with daily patterns (higher during day, lower at night)
    np.random.seed(42)  # For reproducibility
    L_it = np.zeros((N, T))
    for i in range(N):
        base_load = np.random.uniform(0.5, 2.0)  # Base load varies by location
        for t in range(T):
            # Add time-of-day variation (higher load during 6-22, lower at night)
            time_factor = 1.0 + 0.5 * np.sin((t - 6) * np.pi / 12) if 6 <= t <= 22 else 0.5
            L_it[i, t] = base_load * time_factor + np.random.uniform(-0.1, 0.1)

    # print(f"Problem setup: N={N} locations, T={T} time periods, Budget={budget} batteries")
    # print(f"Total load shape: {L_it.shape}")
    #
    # # Run hybrid solver
    # print("\nRunning hybrid QAOA solver...")
    res = hybrid_qaoa(C_t, L_it, Delta_t, P_max, C_cap, S0, budget)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Optimal battery placement (z): {res['z']}")
    print(f"Number of batteries installed: {sum(res['z'])}")
    print(f"Battery locations: {[i for i, z in enumerate(res['z']) if z == 1]}")
    print(f"\nTotal Energy Cost: {res['energy']:.2f}")

    print(f"\nOptimal power dispatch (p) shape: {res['p_opt'].shape}")
    print("First 3 time periods for installed batteries:")
    for i, z in enumerate(res['z']):
        if z == 1:
            print(f"  Battery {i}: {res['p_opt'][i, :3]}")

    print(f"\nOptimal State of Charge (SoC) shape: {res['SoC_opt'].shape}")
    print("First 3 time periods for installed batteries:")
    battery_locations = [i for i, z in enumerate(res['z']) if z == 1]
    # for i, z in enumerate(res['z']):
    #     if z == 1:
    #         print(f"  Battery {i}: {res['SoC_opt'][i, :3]}")
# print(res['z'])
# import matplotlib.pyplot as plt
# for i in range(14):
#     plt.plot(res['SoC_opt'][i]*res['z'][i])
#     plt.show()

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.image import imread


def plotsoc(n, timesteps, chosen_nodes, time_series_ros, background_image_path=None):
    """
    Create an animated network graph with optional background image.

    Parameters:
    -----------
    n : int
        Number of nodes in the graph
    timesteps : int
        Number of time steps in the animation
    chosen_nodes : list
        List of node indices that have time series data
    time_series_ros : dict or numpy array
        Dictionary mapping node indices to their time series arrays,
        OR a 2D numpy array of shape (n, timesteps)
    background_image_path : str, optional
        Path to local image file (e.g., 'my_image.jpg', 'path/to/photo.png')
    """
    # 1. Create a simple graph
    G = nx.erdos_renyi_graph(n, 0.2, seed=42)

    # 2. Convert numpy array to dict if necessary
    if isinstance(time_series_ros, np.ndarray):
        time_series_ros = {i: time_series_ros[i, :] for i in range(time_series_ros.shape[0])}

    # 3. Choose m nodes randomly and assign time series
    time_series = {node: time_series_ros[node] for node in chosen_nodes}

    # 3. Assign positions for plotting
    pos = nx.spring_layout(G, seed=42)

    # 4. Node color setup
    cmap = cm.YlOrRd
    # Find actual min/max of your data
    all_values = [time_series[node][t] for node in chosen_nodes for t in range(timesteps)]
    vmin, vmax = np.min(all_values), np.max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax)

    def get_node_colors(t):
        colors = []
        for node in G.nodes():
            if node in chosen_nodes:
                val = time_series[node][t]
                colors.append(cmap(norm(val)))
            else:
                colors.append(cmap(0.0))
        return colors

    # 5. Load background image from local file if provided
    bg_img = None
    if background_image_path:
        try:
            bg_img = imread(background_image_path)
            print(f"✓ Background image loaded successfully from: {background_image_path}")
            print(f"  Image shape: {bg_img.shape}")
            print(f"  Image dtype: {bg_img.dtype}")
        except FileNotFoundError:
            print(f"✗ Error: Could not find image file at '{background_image_path}'")
            print("  Please check the file path and try again.")
            print("  Continuing without background image...")
        except Exception as e:
            print(f"✗ Error loading image: {e}")
            print("  Continuing without background image...")

    # 6. Create animation
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display background image ONCE with better settings
    bg_artist = None
    if bg_img is not None:
        # Get position boundaries from the layout
        x_vals = [pos[node][0] for node in G.nodes()]
        y_vals = [pos[node][1] for node in G.nodes()]
        x_min, x_max = min(x_vals) - 0.1, max(x_vals) + 0.1
        y_min, y_max = min(y_vals) - 0.1, max(y_vals) + 0.1

        # Display the background image
        bg_artist = ax.imshow(bg_img, extent=[x_min, x_max, y_min, y_max],
                              aspect='auto', alpha=0.7, zorder=0)
        print(f"✓ Background image displayed with extent: [{x_min:.2f}, {x_max:.2f}, {y_min:.2f}, {y_max:.2f}]")

        # Set axis limits to match the extent
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax.axis('off')

    # Add a ScalarMappable for the colorbar ONCE
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('State of Charge')

    # Store references to graph elements to update them
    graph_elements = {'nodes': None, 'edges': None, 'labels': None}

    def update(frame):
        # Remove only the graph elements, not the background
        if graph_elements['nodes']:
            graph_elements['nodes'].remove()
        if graph_elements['edges']:
            graph_elements['edges'].remove()
        if graph_elements['labels']:
            for label in graph_elements['labels'].values():
                label.remove()

        node_colors = get_node_colors(frame)

        # Draw graph and store references
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
        edges = nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        labels = nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        # Set zorder after creation
        nodes.set_zorder(2)
        if edges:
            edges.set_zorder(1)

        graph_elements['nodes'] = nodes
        graph_elements['edges'] = edges
        graph_elements['labels'] = labels

        # Make sure background stays in place
        if bg_artist is not None:
            bg_artist.set_zorder(0)

    print(f"\nCreating animation with {timesteps} frames...")
    ani = FuncAnimation(fig, update, frames=timesteps, interval=300)

    # Save as video (requires ffmpeg)
    print("Saving animation to 'graph_timeseries.mp4'...")
    ani.save('graph_timeseries.mp4', writer='ffmpeg', dpi=200)
    print("✓ Animation saved successfully!")

    plt.show(block=True)

    return fig, ani




    # Example usage with diagnostics:
if __name__ == "__main__":
    import os
    # # # Check if file exists first
    image_path = 'city_map.jpg'  # <-- Change this to your file path

    if os.path.exists(image_path):
        print(f"✓ Found image file: {image_path}")
    else:
        print(f"✗ Image file not found: {image_path}")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Files in current directory: {os.listdir('lib')[:10]}")  # Show first 10 files

    plotsoc(N, 24, battery_locations, res['SoC_opt'], image_path)
