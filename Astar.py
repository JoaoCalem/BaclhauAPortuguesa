# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:43:52 2024

@author: admin
"""

import math
import random
import numpy as np
import heapq
from adaptivempc import *

class AStar:
    def __init__(self, possible_actions, coverage, square_size=600):
        self.coverage = coverage  # Coverage dictionary
        self.possible_actions = possible_actions
        self.transition_count = 0
        self.square_size = square_size

    def search(self, start_state):
        """Perform A* search starting from a given state."""
        # Priority queue for A* (min-heap)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_state, []))  # (f, g, state, path)

        visited = set()

        while open_set:
            f, g, current_state, path = heapq.heappop(open_set)

            if current_state in visited:
                continue
            visited.add(current_state)

            # Check if we've reached the goal
            if self.goal_condition():
                return path, g

            # Expand the current node
            for action in self.possible_actions:
                next_state = self.find_next_center(current_state, action)
                if next_state in visited:
                    continue

                new_g = g + self.cost_function(current_state, action)
                h = self.heuristic_function(next_state)
                heapq.heappush(open_set, (new_g + h, new_g, next_state, path + [action]))

        return [], float("inf")  # If the goal was not reached

    def cost_function(self, state, action):
        """Calculate the cost for a given state and action."""
        x, y, vx, vy, m, e, mode,steps_passed = state
        return steps_passed

    def heuristic_function(self, state):
        """Estimate the remaining cost to the goal."""
        x, y, vx, vy, m, e, mode,steps_passed = state
        
        heur = len([i for i in self.coverage.values() if i==0]) *2*107
        return heur 

    def find_next_center(self, current_state, action):
        """Predict the next center state based on the current state and action."""
        x, y, vx, vy, m, e, mode,steps_passed = current_state
        if (action == 1 and mode < 3) or (action == 0 and mode == 4):
            key, val = self.get_idx_coverage(x, y)
            temp = list(self.coverage)
            try:
                next_pos = temp[temp.index(key) + 1]
            except (ValueError, IndexError):
                next_pos = temp[0]
        else:
            x = x + vx * 60 * 3 if x + vx * 60 * 3 <= 21600 else x + vx * 60 * 3 - 21600
            y = y + vy * 60 * 3 if y + vy * 60 * 3 <= 10800 else y + vy * 60 * 3 - 10800
            key, val = self.get_idx_coverage(x, y)
            temp = list(self.coverage)
            try:
                next_pos = temp[temp.index(key) + 1]
            except (ValueError, IndexError):
                next_pos = temp[0]
            mode = 0 if mode >= 3 else 4

        steps_passed = (next_pos[0] - x) / vx if next_pos[0] - x >= 0 else (next_pos[0] + 21600 - x) / vx
        if action == 0:
            e += steps_passed * 0.2
        else:
            e -= steps_passed * 0.2
            m += 1
            self.coverage[next_pos] = 1
        e = min(max(e, 0), 1)
        next_pos = np.round(next_pos)
        return next_pos[0], next_pos[1], vx, vy, m, e, mode, steps_passed

    def get_idx_coverage(self, x, y):
        """Return idx of coverage list based on x and y."""
        res_key, res_val = min(self.coverage.items(), key=lambda v: np.sqrt((x - v[0][0]) ** 2 + (y - v[0][1]) ** 2))
        return res_key, res_val

    def goal_condition(self):
        """Goal is achieved when the sum of all coverage values equals 1."""
        return sum(self.coverage.values()) / len(self.coverage) == 1
x0 = 500
y0=250
intersections = calculate_border_intersections(domain_width, domain_height, x0, y0, v_x, v_y, proximity_threshold)


m_distance = find_distance_between_trajectories(intersections)
midpoint = find_midpoint_of_trajectory(intersections)

closest_value = map_to_closest_value(m_distance/2, [500, 400, 300])
trajectory = bundle_segments(intersections[1:])
# Step 3: Create a grid centered at the midpoint with squares twice the mapped size
square_size = 2 * closest_value
centers,efective_length,direction_vector = place_squares_trajectory(trajectory,590, declive)

coverage={}
for center in centers:
    coverage[(center[0] , center[1])]=0
# Possible actions
possible_actions = [0, 1, 2, 3, 4, 5]

# Run MCTS
start_state = (500, 250,4.35,5.49, 0, 1, 7,0)


astar_centers = AStar(possible_actions, coverage)
optimal_path, total_cost = astar_centers.search(start_state)

print("Optimal Path:", optimal_path)
print("Total Cost:", total_cost)


