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
from tqdm import tqdm
import sys

class AStar:
    def __init__(self, possible_actions, coverage, square_size=600):
        self.coverage = coverage  # Coverage dictionary
        self.possible_actions = possible_actions
        self.transition_count = 0
        self.square_size = square_size

    def search(self, start_state,efective_length,direction_vector):
        """Perform A* search starting from a given state."""
        # Priority queue for A* (min-heap)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_state, []))  # (f, g, state, path)

        visited = set()

        for i in range(10000):
            try:
                f, g, current_state, path = heapq.heappop(open_set)

                if current_state in visited:
                    continue
                visited.add(current_state)

                # Check if we've reached the goal
                if self.goal_condition(current_state):
                    return path, g

                # Expand the current node
                for action in self.possible_actions:
                    next_state, extra_cost = self.find_next_center(current_state, action,efective_length,direction_vector)
                    if next_state in visited:
                        continue

                    g = self.cost_function(next_state, action, extra_cost)
                    h = self.heuristic_function(next_state)
                    heapq.heappush(open_set, (g + h, g, next_state, path + [action]))
                print([(i[-1],i[0],i[1]) for i in open_set[:1]])
                # breakpoint()
            except KeyboardInterrupt:
                print([(i[-1],i[0],i[1]) for i in open_set[:1]])
                inp = input('Continue: y or n\n')
                if inp=='n':
                    sys.exit()

        print([(i[-1],i[0],i[1]) for i in open_set[:1]])
        return [], float("inf")  # If the goal was not reached
    def cost_function(self, state, action, extra_cost):
        """Calculate the cost for a given state and action."""
        key, coverage, e, mode,steps_passed = state
        return steps_passed + extra_cost

    def heuristic_function(self, state):
        """Estimate the remaining cost to the goal."""
        key, coverage, e, mode,steps_passed = state
        
        heur = len([i for i in coverage if i==0]) *2
        return heur 

    def find_next_center(self, current_state, action,efective_length,direction_vector,v_x = 4.35,v_y = 5.49):
        """Predict the next center state based on the current state and action."""
        key, coverage, e, mode,steps_passed = current_state
        extra_cost = 0
        time_square = efective_length*direction_vector[0]*2/4.35
        if (action == 1 and mode < 3) or (action == 0 and mode == 4):
            n_squares = 1
            e_dif = time_square * 0.2/100
            e += e_dif if action==0 else -e_dif
        elif mode!=5:
            n_squares = 2
            mode = 0 if action == 1 else 4
            e += (time_square*2-60*3) *0.2/100
        else:
            n_squares = math.ceil(20*60/time_square)
            mode = 0 if action == 1 else 4
            if action==4:
                e += (time_square*n_squares-20*3) *0.2/100 + 20*60*0.05/100
        x = key[0] + v_x * time_square*n_squares if key[0] + v_x * time_square*n_squares <= 21600 else key[0] + v_x * time_square*n_squares - 21600
        y = key[1] + v_y * time_square*n_squares if key[1] + v_y * time_square*n_squares <= 10800 else key[1] + v_y * time_square*n_squares - 10800
        key, _ = self.get_idx_coverage(x, y)
        
        
        if e<0.01:
            mode=5
            e=0.01
            extra_cost+=np.inf
        elif action == 1:
            coverage = list(coverage)
            coverage[[*self.coverage].index(key)] = 1
            coverage = tuple(coverage)
        e = min(max(e, 0), 1)
        steps_passed += n_squares
        return (key,coverage, e, mode,steps_passed), extra_cost

    def get_idx_coverage(self, x, y):
        """Return idx of coverage list based on x and y."""
        res_key, res_val = min(self.coverage.items(), key=lambda v: np.sqrt((x - v[0][0]) ** 2 + (y - v[0][1]) ** 2))
        return res_key, res_val

    def goal_condition(self, state):
        """Goal is achieved when the sum of all coverage values equals 1."""
        key, coverage, e, mode,steps_passed = state
        coverage = list(coverage)
        return sum(coverage) / len(coverage) == 1
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
n_squares = len(coverage)
# Possible actions
possible_actions = [0, 1]

# Run MCTS
start_state = (centers[0],tuple([*coverage.values()]), 1, 7, 0)


astar_centers = AStar(possible_actions, coverage)
optimal_path, total_cost = astar_centers.search(start_state,efective_length,direction_vector)

print("Optimal Path:", optimal_path)
print("Total Cost:", total_cost)


