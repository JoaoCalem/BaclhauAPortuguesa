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

        while open_set:
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
                    alpha = 1
                    g = self.cost_function(next_state, action, extra_cost)
                    h = alpha * self.heuristic_function(next_state)
                    heapq.heappush(open_set, (g + h, g, next_state, path + [action]))
                MAX_DEPTH = 6
                path_same_length = len(path) + 1 - MAX_DEPTH
                open_set = [
                    node for node in open_set if path_same_length<=0 or node[3][:path_same_length] == path[:path_same_length]
                ]
                heapq.heapify(open_set)  # Rebuild the heap after pruning
                print([(round((i[0]-i[1])/2), round(i[-2][2]*100), i[1]) for i in open_set[:1]])
                # if (open_set[0][0]-open_set[0][1])/2 < 336:
                #     temp = len(open_set[0][-1])-5
                #     print([(i[-1][temp:],i[2][2],(i[0]-i[1])/2) for i in open_set])
                #     breakpoint()
            except KeyboardInterrupt:
                print([(i[-1],(i[0]-i[1])/2) for i in open_set[:1]])
                # x_data = []
                # y_data = []
                # fig, ax = plt.subplots()
                # ax.set_title('Dynamic x, y Positions')
                # ax.set_xlabel('X Position')
                # ax.set_ylabel('Y Position')
                # for i,v in enumerate(self.coverage.keys()):
                #     if open_set[0][2][1][i] == 0:
                #         print(v)
                #         x_data.append(v[0])
                #         y_data.append(v[1])
                #         ax.clear()
                #         ax.plot(x_data, y_data, marker='o', linestyle='', color='b')
                #         ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
                #         ax.set_ylim(min(y_data) - 1, max(y_data) + 1)
                #         plt.draw()
                #         plt.pause(0.001)
                
                inp = input('Continue: y or n\n')
                if inp=='n':
                    sys.exit()

        return open_set[0][-1], open_set[0][1]
        return [], float("inf")  # If the goal was not reached
    def cost_function(self, state, action, extra_cost):
        """Calculate the cost for a given state and action."""
        key, coverage, e, mode,steps_passed = state
        return steps_passed + extra_cost

    def heuristic_function(self, state):
        """Estimate the remaining cost to the goal."""
        key, coverage, e, mode,steps_passed = state
        extra_cost = 0
        heur = len([i for i in coverage if i==0]) * 2
        extra_cost -= 0.01*e
        return heur + extra_cost

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
        
        new_index = [*self.coverage].index(key)+n_squares
        if new_index >= len(self.coverage):
            new_index += -len(self.coverage)
        key = [*self.coverage][new_index]
        
        if e<0.70:
            mode=5
            e=0.70
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
# plot_squares(trajectory,centers,590,domain_width, domain_height,efective_length,direction_vector)

coverage={}
for center in centers:
    coverage[(max(min((center[0],21600)),0) , max(min((center[1],10800)),0))]=0
n_squares = len(coverage)
# Possible actions
possible_actions = [0, 1]

# Run MCTS
start_state = ([*coverage][0],tuple([*coverage.values()]), 1, 7, 0)


astar_centers = AStar(possible_actions, coverage)
optimal_path, total_cost = astar_centers.search(start_state,efective_length,direction_vector)

print("Optimal Path:", optimal_path)
print("Total Cost:", total_cost)


