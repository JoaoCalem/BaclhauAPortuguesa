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
    def __init__(self, coverage, square_size=600, max_iterations=1000):
        self.coverage = coverage  # Coverage dictionary
        self.transition_count = 0
        self.square_size = square_size
        self.max_iterations=max_iterations
        self.possible_actions=[0,1]
    def search(self, start_state):
        """Perform Anytime A* search starting from a given state."""
        # Priority queue for A* (min-heap)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_state, []))  # (f, g, state, path)

        visited = set()
        best_solution = ([], float("inf"))  # Store best path and cost

        for _ in range(self.max_iterations):
            if not open_set:
                break
            print(open_set)
            f, g, current_state, path = heapq.heappop(open_set)

            if current_state in visited:
                continue
            visited.add(current_state)

            # Check if we've reached the goal
            if self.goal_condition(current_state):
                if g < best_solution[1]:
                    best_solution = (path, g)
                continue

            # Expand the current node
            for action in self.possible_actions:
                next_state = self.find_next_center(current_state, action)
                if next_state in visited:
                    continue

                new_g = g + self.cost_function(current_state, action)
                h = self.heuristic_function(next_state)
                heapq.heappush(open_set, (new_g + h, new_g, next_state, path + [action]))

        return best_solution  # Return the best solution found

    def cost_function(self, state, action):
        """Calculate the cost for a given state and action."""
        x, y, vx, vy, m, e, mode,steps_passed,coverage = state
        return 1- sum(coverage) / len(coverage)

    def heuristic_function(self, state):
        """Estimate the remaining cost to the goal."""
        x, y, vx, vy, m, e, mode,steps_passed,coverage = state
        heur = max(len(coverage)-m,0)
        return heur 

    def find_next_center(self, current_state, action):
        """Predict the next center state based on the current state and action."""
        x, y, vx, vy, m, e, mode,steps_passed,coverage= current_state
        if (action == 1 and mode < 3) or (action == 0 and mode == 4):
            key, val = self.get_idx_coverage(x, y)
            temp = list(self.coverage)
            try:
                idx= temp.index(key) + 1
                next_pos = temp[idx]
            except (ValueError, IndexError):
                idx=0
                next_pos = temp[idx]
        elif mode == 5:
            x = x+ 20*60*vx if  x+ 20*60*vx<= 21600 else  x+ 20*60*vx - 21600
            y = y + 20*60*vy if y + 20*60*vy<= 10800 else y + 20*60*vy - 10800
            key, val = self.get_idx_coverage(x, y)
            temp = list(self.coverage)
            try:
                idx= temp.index(key) + 1
                next_pos = temp[idx]
            except (ValueError, IndexError):
                idx=0
                next_pos = temp[idx]
            mode = 0 if action==1 else 4
            e = 20*60*0.05
        else:
            x = x + vx * 60 * 3 if x + vx * 60 * 3 <= 21600 else x + vx * 60 * 3 - 21600
            y = y + vy * 60 * 3 if y + vy * 60 * 3 <= 10800 else y + vy * 60 * 3 - 10800
            key, val = self.get_idx_coverage(x, y)
            temp = list(self.coverage)
            try:
                idx= temp.index(key) + 1
                next_pos = temp[idx]
            except (ValueError, IndexError):
                idx=0
                next_pos = temp[idx]
            mode = 0 if mode >= 3 else 4
        steps_passed = (next_pos[0] - x) / vx if next_pos[0] - x >= 0 else (next_pos[0] + 21600 - x) / vx
        if action == 0:
            e += steps_passed * 0.2
        else:
            e -= steps_passed * 0.2
            m += 1
            list(coverage)[idx] = 1
            coverage = tuple(coverage)
        if e<0:
            backtrack = -e/0.2
            x = next_pos[0]-backtrack*vx if next_pos[0]-backtrack*vx>0 else 21600- next_pos[0]-backtrack*vx
            y = next_pos[1]-backtrack*vy if next_pos[1]-backtrack*vy>0 else 10800- next_pos[1]-backtrack*vy
            next_pos = (x,y)
            mode = 5
            steps_passed -=backtrack
        else:
            e = min(max(e, 0), 100)    
        next_pos = np.round(next_pos)
        return next_pos[0], next_pos[1], vx, vy, m, e, mode, steps_passed, coverage

    def get_idx_coverage(self, x, y):
        """Return idx of coverage list based on x and y."""
        res_key, res_val = min(self.coverage.items(), key=lambda v: np.sqrt((x - v[0][0]) ** 2 + (y - v[0][1]) ** 2))
        return res_key, res_val

    def goal_condition(self,current_state):
        """Goal is achieved when the sum of all coverage values equals 1."""
        x, y, vx, vy, m, e, mode,steps_passed,coverage= current_state
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


start_state = (500, 250,4.35,5.49, 0, 100, 7,0,tuple(coverage.values()))
# print(centers)

astar_centers = AStar(coverage)
optimal_path, total_cost = astar_centers.search(start_state)

print("Optimal Path:", optimal_path)
print("Total Cost:", total_cost)