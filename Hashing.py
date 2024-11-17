# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:18:21 2024

@author: admin
"""

import numpy as np
import hashlib
import sys
import matplotlib.pyplot as plt


def calculate_memory_usage(tilings, tile_c_values, hash_space_size):
    """
    Calculate the RAM memory used by the tilings, c-values, and hash space.

    tilings: List of tiling grids.
    tile_c_values: List of dictionaries storing c-values for tiles in each tiling.
    hash_space_size: Size of the hash space.
    """
    # Memory used by tilings (each tiling grid as a numpy array)
    tilings_memory = sum(sys.getsizeof(tiling) for tiling in tilings)

    # Memory used by c-values dictionaries and their stored values
    c_values_memory = sum(
        sys.getsizeof(tile_dict) +
        sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in tile_dict.items())
        for tile_dict in tile_c_values
    )

    # Memory used by hash space size (as an integer, negligible but included for completeness)
    hash_space_memory = sys.getsizeof(hash_space_size)

    # Total memory in bytes
    total_memory_bytes = tilings_memory + c_values_memory + hash_space_memory

    # Convert to megabytes (MB)
    total_memory_mb = total_memory_bytes / (1024 ** 2)
    return total_memory_mb



def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space."""
    low = np.array(low)
    high = np.array(high)
    step_sizes = (high - low) / bins
    
    grid = np.array([np.zeros(bins[dim] - 1) for dim in range(len(low))])
    for nbin in range(len(bins)):
        for nstep in range(1, bins[nbin]):
            grid[nbin][nstep - 1] = nstep * step_sizes[nbin] + low[nbin] + offsets[nbin]
    return grid

def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications."""
    return [create_tiling_grid(low, high, spec[0], spec[1]) for spec in tiling_specs]

def discretize(sample, grid):
    """Discretize a sample as per given grid."""
    return tuple(int(np.digitize(sample[dim], grid[dim])) for dim in range(len(sample)))

def tile_encode(sample, tilings):
    """Encode given sample using tile-coding."""
    return [discretize(sample, tiling) for tiling in tilings]

class FeatureHasherWithIncrementalTiles:
    def __init__(self, tilings, hash_space_size, seed=42):
        """
        Initialize the feature hasher with tilings and hash space.

        tilings: List of tiling grids.
        hash_space_size: Size of the hash space.
        seed: Random seed for consistent hashing.
        """
        self.tilings = tilings
        self.hash_space_size = hash_space_size
        self.seed = seed

        # Incremental storage for tile-specific c-values
        # Each tiling has a dictionary storing c-values for its tiles
        self.tile_c_values = [{} for _ in tilings]

    def update_c_value(self, tiling_index, tile_index, update):
        """
        Incrementally update the stored c-value for a specific tile.

        tiling_index: Index of the tiling.
        tile_index: Index of the tile within the tiling.
        increment: Value to add to the current c-value for the tile.
        """
        if tile_index not in self.tile_c_values[tiling_index]:
            self.tile_c_values[tiling_index][tile_index] = 0
        self.tile_c_values[tiling_index][tile_index] = update

    def hash_state(self, sample):
        """
        Compute feature hash for a given sample using stored c-values.

        sample: Input state (x, y, m, e).
        """
        tile_indices = tile_encode(sample, self.tilings)
        # print(tile_indices)

        # Collect c-values from all intersecting tiles
        combined_c_values = []
        for tiling_index, tile_index in enumerate(tile_indices):
            c_value = self.tile_c_values[tiling_index].get(tile_index, 0)
            combined_c_values.append((tile_index, c_value))

        # Combine all tile indices and c-values into a single key
        key = str(combined_c_values) + str(self.seed)

        # Hash the combined key into the hash space
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16) % self.hash_space_size
        return hash_value


# Define tiling parameters
low = [0, 0,0,0]  # Lower bounds for (x, y)
high = [21600, 10800,8,1]  # Upper bounds for (x, y)
tiling_specs = [(tuple(648 for _ in range(4)), (-0.08, -0.06, -0.04, -0.02)),
                (tuple(648 for _ in range(4)), (0.02, 0.0, -0.02, -0.04)),
                (tuple(648 for _ in range(4)), (-0.06, -0.04, 0.0, -0.06))]
tilings = create_tilings(low, high, tiling_specs)

# Initialize the feature hasher
hash_space_size = 100000  # Size of hash space
hasher = FeatureHasherWithIncrementalTiles(tilings, hash_space_size)

# Update c-values incrementally
hasher.update_c_value(0, (2, 3), 1.)  # Increment tile (2, 3) in tiling 0 by 1.5
hasher.update_c_value(1, (2, 3), 0.5)  # Increment tile (2, 3) in tiling 1 by 2.0

# Test with a sample state (x, y, z1, z2)
sample_state = (2, 3, 1.0, 1.0)
hashed_values = hasher.hash_state(sample_state)

print("Hashed Values:", hashed_values)
memory_usage_mb = calculate_memory_usage(tilings, hasher.tile_c_values, hash_space_size)
print(f"Estimated RAM usage: {memory_usage_mb:.3f} MB")


