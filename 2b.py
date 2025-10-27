import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

# --- Configuration ---
# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Network Parameters
N = 1500  # Number of nodes for both R and S
P_R = 0.5  # Edge probability for Random Network R
M_SF = 3   # Edges to attach for Scale-Free Network S (Barabási-Albert)
STEPS = 100 # Number of removal steps

# --- Core Functions (Adapted from 2a.ipynb) ---

def generate_random_network(n, p):
    """Generate Erdős-Rényi random graph G(n,p)."""
    # Use fast_gnp_random_graph for dense graphs like this, but gnp_random_graph is also fine
    return nx.gnp_random_graph(n, p, seed=SEED)

def generate_scale_free_network(n, m):
    """Generate Barabási-Albert scale-free graph BA(n,m)."""
    return nx.barabasi_albert_graph(n, m, seed=SEED)

def analyze_robustness(G, attack_type='targeted', steps=100):
    """Analyze robustness under node removal."""
    G_copy = G.copy()
    N_initial = G.number_of_nodes()
    
    # Check for empty graph or single node
    if N_initial <= 1:
        return pd.DataFrame({
            'fraction_removed': [0.0, 1.0],
            'giant_component_size': [1.0, 0.0]
        })
        
    frac_removed = [0.0]
    
    # Calculate initial LCC size
    if nx.is_connected(G_copy):
        gc_size = [1.0]
    else:
        gc_size = [len(max(nx.connected_components(G_copy), key=len)) / N_initial]

    nodes_per_step = max(1, N_initial // steps)
    
    for _ in range(steps):
        current_nodes = G_copy.number_of_nodes()
        if current_nodes == 0:
            break

        # Determine nodes to remove
        if attack_type == 'targeted':
            # Remove highest-degree nodes
            try:
                degrees = dict(G_copy.degree())
                to_remove = sorted(degrees, key=degrees.get, reverse=True)[:nodes_per_step]
            except:
                # Fallback if degree calculation fails (shouldn't happen with nx.Graph)
                to_remove = random.sample(list(G_copy.nodes()), nodes_per_step)
        else:
            # Remove random nodes
            to_remove = random.sample(list(G_copy.nodes()), min(nodes_per_step, current_nodes))

        G_copy.remove_nodes_from(to_remove)

        curr = G_copy.number_of_nodes()
        frac_removed.append((N_initial - curr) / N_initial)
        
        if curr == 0:
            gc_size.append(0.0)
        else:
            try:
                # Find the largest connected component (LCC)
                lcc_nodes = max(nx.connected_components(G_copy), key=len)
                gc_size.append(len(lcc_nodes) / N_initial)
            except ValueError:
                 # This happens if there are no components (e.g., only isolated nodes)
                gc_size.append(0.0)
                
    return pd.DataFrame({
        'fraction_removed': frac_removed,
        'giant_component_size': gc_size
    })

def report_attack(df, network_name, attack_type):
    """Print report for one attack type."""
    threshold_idx = df[df['giant_component_size'] < 0.01].index
    threshold_frac = df.loc[threshold_idx[0], 'fraction_removed'] if not threshold_idx.empty else 1.0

    print(f"\n=== {network_name} - {attack_type.upper()} ATTACK REPORT ===")
    print(f"Percolation threshold: {threshold_frac:.3f} "
          f"({threshold_frac * 100:.1f}% nodes removed)")
    
    return threshold_frac

def plot_robustness_comparison(results_dfs):
    """Plot the robustness curves for all four scenarios."""
    plt.figure(figsize=(10, 6))
    
    # Colors and styles for clear differentiation
    styles = {
        'R_targeted': {'color': 'red', 'linestyle': '-', 'label': 'R (Random) - Targeted'},
        'R_random':   {'color': 'red', 'linestyle': '--', 'label': 'R (Random) - Random'},
        'S_targeted': {'color': 'blue', 'linestyle': '-', 'label': 'S (Scale-Free) - Targeted'},
        'S_random':   {'color': 'blue', 'linestyle': '--', 'label': 'S (Scale-Free) - Random'}
    }
    
    for key, df in results_dfs.items():
        plt.plot(df['fraction_removed'], df['giant_component_size'], 
                 **styles[key], linewidth=2)

    plt.xlabel('Fraction of Nodes Removed ($f$)')
    plt.ylabel('Largest Connected Component Size ($S$) / N')
    plt.title(f'Robustness Comparison: Random (R) vs. Scale-Free (S) Network ($N={N}$)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.4)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('robustness_comparison.png', dpi=300)
    plt.show()

def visualize_sf_network(G, node_size=40, alpha=0.5, figsize=(10,10)):
    """Visualize the Scale-Free network, highlighting hubs by node size."""
    
    # 1. Determine node size based on degree to highlight hubs
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    
    # Normalize degree for node size (scaling factor * degree)
    # Scale from a base size (e.g., 20) up to a max size (e.g., 500)
    node_sizes = [20 + (d / max_degree) * 480 for d in degrees.values()]
    
    # 2. Visualization
    plt.figure(figsize=figsize)
    # Using Kamada-Kawai layout often produces a better visual separation of clusters
    # and peripheral nodes, which is good for SF networks.
    pos = nx.kamada_kawai_layout(G)
    
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        alpha=alpha, 
        node_color='blue',  # Use a different color than the Random Network (red)
        label='Nodes (Size scaled by Degree)'
    )
    
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    
    plt.title(f"Visualization of Scale-Free Network S (BA Model: N={G.number_of_nodes()}, m={M_SF})", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('scale_free_network_visualization.png', dpi=300)
    plt.show()
# --- Main Execution ---

if __name__ == "__main__":
    print(f"--- Generating Networks (N={N}) ---")
    
    # Generate Random Network R (Erdős-Rényi)
    R = generate_random_network(N, P_R)
    print(f"Network R (Erdős-Rényi G({N}, {P_R})) generated. Avg Degree: {R.number_of_edges() * 2 / N:.2f}")

    # Generate Scale-Free Network S (Barabási-Albert)
    S = generate_scale_free_network(N, M_SF)
    print(f"Network S (Barabási-Albert BA({N}, {M_SF})) generated. Avg Degree: {S.number_of_edges() * 2 / N:.2f}")
    visualize_sf_network(S)

    # --- Analyze Robustness ---
    
    results = {}
    
    # 1. R - Targeted Attack
    results['R_targeted'] = analyze_robustness(R, attack_type='targeted', steps=STEPS)
    report_attack(results['R_targeted'], 'R', 'targeted')

    # 2. R - Random Failure
    results['R_random'] = analyze_robustness(R, attack_type='random', steps=STEPS)
    report_attack(results['R_random'], 'R', 'random')
    
    # 3. S - Targeted Attack (Vulnerability)
    results['S_targeted'] = analyze_robustness(S, attack_type='targeted', steps=STEPS)
    sf_targeted_threshold = report_attack(results['S_targeted'], 'S', 'targeted')

    # 4. S - Random Failure (Robustness)
    results['S_random'] = analyze_robustness(S, attack_type='random', steps=STEPS)
    sf_random_threshold = report_attack(results['S_random'], 'S', 'random')

    # --- Plot Comparison ---
    plot_robustness_comparison(results)
    
    # --- Final Report Summary ---
    print("\n" + "="*50)
    print("      ROBUSTNESS COMPARATIVE SUMMARY")
    print("="*50)
    print(f"Network R (Dense Random, Avg Deg {R.number_of_edges() * 2 / N:.2f}):")
    print(f"  - Targeted Threshold: {report_attack(results['R_targeted'], 'R', 'targeted'):.3f}")
    print(f"  - Random Threshold:   {report_attack(results['R_random'], 'R', 'random'):.3f}")
    print("\n" + "-"*50)
    print(f"Network S (Scale-Free, Avg Deg {S.number_of_edges() * 2 / N:.2f}):")
    print(f"  - Targeted Threshold: {sf_targeted_threshold:.3f}  <- HIGH VULNERABILITY")
    print(f"  - Random Threshold:   {sf_random_threshold:.3f}  <- HIGH ROBUSTNESS")
    print("="*50)