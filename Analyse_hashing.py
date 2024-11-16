# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:23:23 2024

@author: admin
"""
from Hashing import *
def test_collision_rate(tilings, tile_c_values, hash_space_size, samples):
    """
    Test collision rates for given tilings, c-values, and hash space size.

    tilings: List of tiling grids.
    tile_c_values: List of dictionaries storing c-values for tiles in each tiling.
    hash_space_size: Size of the hash space.
    samples: List of states to test.
    """
    hasher = FeatureHasherWithIncrementalTiles(tilings, hash_space_size)

    # Create hash values for all samples
    hashes = [hasher.hash_state(sample) for sample in samples]

    # Count collisions
    unique_hashes = len(set(hashes))
    collision_count = len(hashes) - unique_hashes
    collision_rate = collision_count / len(hashes)*100

    return collision_rate

def generate_collision_surface(low, high, max_tilings, max_hash_space, spacing=None):
    """
    Generate a surface of collision rates for varying hash space and tiling counts.

    low: Lower bounds for (x, y).
    high: Upper bounds for (x, y).
    z_range: Ranges for z1 and z2 as tuples (z1_min, z1_max) and (z2_min, z2_max).
    max_tilings: Maximum number of tilings to test.
    max_hash_space: Maximum hash space size to test.
    spacing: Spacing between tested points for (x, y) (None to test all integers).
    sample_count: Number of random samples for z1 and z2.
    """
    collision_rates = []
    tiling_counts = range(1, max_tilings + 1)
    hash_space_sizes = np.linspace(10000, max_hash_space, num=10, dtype=int)
    if spacing is None:
        x_vals = np.arange(low[0], high[0] + 1)
        y_vals = np.arange(low[1], high[1] + 1)
    else:
        x_vals = np.arange(low[0], high[0] + 1, spacing)
        y_vals = np.arange(low[1], high[1] + 1, spacing)

    # Create a grid for (x, y)
    X, Y = np.meshgrid(x_vals, y_vals)
    xy_samples = np.column_stack([X.ravel(), Y.ravel()])

    # Generate random z1 and z2 for each (x, y) sample
    samples = [(x, y,np.random.uniform(low[2], high[2]), np.random.uniform(low[3], high[3]))
               for x, y in xy_samples]

    for num_tilings in tiling_counts:
        row = []
        for hash_space_size in hash_space_sizes:
            # Create tilings with offsets
            tiling_specs = [((648, 648,648,648), (i * 0.1, i * 0.2,i*0.3,i*0.4)) for i in range(num_tilings)]
            tilings = create_tilings(low, high, tiling_specs)

            # Initialize tile c-values (empty for this test)
            tile_c_values = [{} for _ in tilings]

            # Test collision rate
            collision_rate = test_collision_rate(tilings, tile_c_values, hash_space_size, samples)
            row.append(collision_rate)
        collision_rates.append(row)

    return np.array(tiling_counts), hash_space_sizes, np.array(collision_rates)

def generate_memory_surface(low, high, max_tilings, max_hash_space):
    """
    Generate a surface of memory usage for varying hash space and tiling counts.

    low: Lower bounds for (x, y).
    high: Upper bounds for (x, y).
    max_tilings: Maximum number of tilings to test.
    max_hash_space: Maximum hash space size to test.
    """
    memory_usages = []
    tiling_counts = range(1, max_tilings + 1)
    hash_space_sizes = np.linspace(10000, max_hash_space, num=10, dtype=int)

    for num_tilings in tiling_counts:
        row = []
        for hash_space_size in hash_space_sizes:
            # Create tilings with offsets
            tiling_specs = [((648, 648,648,648), (i * 0.1, i * 0.2,i*0.3,i*0.4)) for i in range(num_tilings)]

            tilings = create_tilings(low, high, tiling_specs)

            # Initialize tile c-values (empty for this test)
            tile_c_values = [{} for _ in tilings]

            # Calculate memory usage
            memory_usage = calculate_memory_usage(tilings, tile_c_values, hash_space_size)
            row.append(memory_usage)
        memory_usages.append(row)

    return np.array(tiling_counts), hash_space_sizes, np.array(memory_usages)

def plot_surface(x, y, z, title, xlabel, ylabel, zlabel):
    """
    Plot a 3D surface with enhanced visibility for axis labels and ticks.

    x: X-axis data (e.g., tiling counts).
    y: Y-axis data (e.g., hash space sizes).
    z: Z-axis data (collision rates or memory usage).
    title: Title of the plot.
    xlabel: Label for the X-axis.
    ylabel: Label for the Y-axis.
    zlabel: Label for the Z-axis.
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(y, x)
    surf = ax.plot_surface(X, Y, z, cmap='viridis', edgecolor='k', alpha=0.8)

    # Title and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=15)
    ax.set_zlabel(zlabel, fontsize=14, labelpad=10)

    # Customize tick sizes
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    # Add a color bar for the surface
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.ax.tick_params(labelsize=12)

    plt.show()


tiling_counts, hash_space_sizes, collision_rates = generate_collision_surface(low, high, 10, 1e5, 648)
plot_surface(
    tiling_counts, hash_space_sizes, collision_rates,
    title="Collision Rate vs. Hash Space and Tilings",
    xlabel="Hash Space Size",
    ylabel="Number of Tilings",
    zlabel="Collision Rate (%)"
)

# Generate memory surface
tiling_counts, hash_space_sizes, memory_usages = generate_memory_surface(low, high, 10, 1e5)
plot_surface(
    tiling_counts, hash_space_sizes, memory_usages,
    title="Memory Usage vs. Hash Space and Tilings",
    xlabel="Hash Space Size",
    ylabel="Number of Tilings",
    zlabel="Memory Usage (MB)"
)




