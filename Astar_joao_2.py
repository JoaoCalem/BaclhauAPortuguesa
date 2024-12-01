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
import time

class AStar:
    def __init__(self, possible_actions, coverage, trajectory, square_size=600):
        self.coverage = coverage  # Coverage dictionary
        self.possible_actions = possible_actions
        self.transition_count = 0
        self.square_size = square_size
        self.trajectory = trajectory

    def search(self, start_state,tol):
        """Perform A* search starting from a given state."""
        # Priority queue for A* (min-heap)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_state, []))  # (f, g, state, path)

        visited = set()
        break_temp = False
        self.start = start_state
        
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
                    next_state, extra_cost = self.find_next_center(current_state, action, tol)
                    if next_state in visited:
                        continue

                    g = self.cost_function(next_state, action, extra_cost)
                    alpha = 2
                    h = alpha*self.heuristic_function(next_state)
                    heapq.heappush(open_set, (g + h, g, next_state, path + [action]))
                MAX_DEPTH = 3
                path_same_length = len(path) - MAX_DEPTH
                open_set = [
                    node for node in open_set if path_same_length<=0 or node[3][:path_same_length] == path[:path_same_length]
                ]
                heapq.heapify(open_set)  # Rebuild the heap after pruning
                print([(round((i[0]-i[1])/(2*alpha)), round(i[-2][2]*100), i[1]) for i in open_set[:1]])
                # if open_set[0][2][0] == (np.float64(11725.201456466048), np.float64(10320.42666574682)):
                #     break_temp = True
                # if 1==1 or break_temp:
                #     temp = max(len(open_set[0][-1])-10,0)
                #     print(temp,[(i[-1][temp:],len([c for c in i[2][1] if c==0]),np.round(i[0],1),np.round(i[1],1),100*round(i[2][2],2)) for i in open_set])
                #     breakpoint()
            except KeyboardInterrupt:
                temp = len(open_set[0][-1])-10
                print([(i[-1][temp:],len([c for c in i[2][1] if c==0]),100*round(i[2][2],2)) for i in open_set])
      
                inp = input('Continue: y or n or p (plot)\n')
                if inp=='n':
                    sys.exit()
                elif inp=='p':
                    x_data = []
                    y_data = []
                    fig, ax = plt.subplots()
                    ax.set_title('Dynamic x, y Positions')
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    for i,v in enumerate(self.coverage.keys()):
                        if open_set[0][2][1][i] == 0:
                            print(v)
                            x_data.append(v[0])
                            y_data.append(v[1])
                            ax.clear()
                            ax.plot(x_data, y_data, marker='o', linestyle='', color='b')
                            ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
                            ax.set_ylim(min(y_data) - 1, max(y_data) + 1)
                            plt.draw()
                            plt.pause(0.001)
                    inp = input('Continue: y or n\n')
                    if inp=='n':
                        sys.exit()                      
        return open_set[0][-1], open_set[0][1]
        return [], float("inf")  # If the goal was not reached
    
    def create_plan(self, path, tol, start_pos, sim_speed, v_x = 4.35,v_y = 5.49):
        """Calculate action plan with positions based on action list"""
        self.plan = []
        state = self.start
        start_time = None
        for action in path:
            key, _, _, mode, _ = state
            if (action == 1 and mode < 3) or (action == 0 and mode == 4):
                travel_time, key = self.single_square_travel(key,v_x)
                if not self.plan[-1][2]:
                    self.plan[-1][2] = start_time
                    start_time+=travel_time/sim_speed
            else:
                if mode!=5:
                    total_time = 3*60
                    tracking_time = total_time
                else:
                    total_time = 20*60
                tracking_time = total_time
                next_key = key
                while tracking_time > 0:
                    travel_time, next_key = self.single_square_travel(next_key,v_x)
                    tracking_time -= travel_time
                if action==1:
                    pos = (key[0]-(tracking_time+tol)*v_x,key[1]-(tracking_time+tol)*v_y)
                else:
                    pos = key
                    if len(self.plan)>0 and self.plan[-1][0] == (pos[0],pos[1]):
                        del self.plan[-1]
                    else:
                        action = 3
                if not start_time:
                    y_dif = pos[1]-start_pos
                    y_dif = y_dif if y_dif>0 else y_dif+10800
                    start_time = time.time() + (y_dif/v_x+tol)/sim_speed
                self.plan.append([(pos[0],pos[1]),action,start_time])
                start_time += (total_time-tracking_time)/sim_speed
                key = next_key
            mode = 0 if action == 1 else 4
            if action==1:
                self.plan.append([(key[0],key[1]),2,None])
            state = (key, _, _, mode, _)
            
    def cost_function(self, state, action, extra_cost):
        """Calculate the cost for a given state and action."""
        key, coverage, e, mode, steps_passed = state
        return steps_passed + extra_cost

    def heuristic_function(self, state):
        """Estimate the remaining cost to the goal."""
        key, coverage, e, mode,steps_passed = state
        extra_cost = 0
        heur = len([i for i in coverage if i==0]) * 2
        return heur + extra_cost

    def find_next_center(self, current_state, action,tol,v_x = 4.35,v_y = 5.49):
        """Predict the next center state based on the current state and action."""
        key, coverage, e, mode, steps_passed = current_state
        extra_cost = 0
        if action == 1 and e>0.9:
            extra_cost -= 0.01
        travel_distance = 0
        key1 = key
        # if key == (np.float64(13165.06040070777), np.float64(294.869333307052)) and action==1:
        #     breakpoint()
        if (action == 1 and mode < 3) or (action == 0 and mode == 4):
            travel_time, key = self.single_square_travel(key,v_x)
            travel_distance += travel_time*v_y/self.square_size
            e_dif = travel_time * 0.2/100
            e += e_dif if action==0 else -e_dif
        elif mode!=5:
            total_time = 3*60
            while total_time > 0:
                travel_time, key = self.single_square_travel(key,v_x)
                travel_distance += travel_time*v_y/self.square_size
                total_time -= travel_time
            e += -(total_time+tol)*0.2/100
            if action == 1:
                e -= tol*0.2/100
            mode = 0 if action == 1 else 4
        else:
            total_time = 20*60
            while total_time > 0:
                travel_time, key = self.single_square_travel(key,v_x)
                travel_distance += travel_time*v_y/self.square_size
                total_time -= travel_time
            mode = 0 if action == 1 else 4
            if action==0:
                e += -total_time*0.2/100
            else:
                e += total_time*0.2/100
        
        if e<0.3:
            extra_cost+=np.inf
        elif action == 1:
            coverage = list(coverage)
            coverage[[*self.coverage].index(key)] = 1
            coverage = tuple(coverage)
        e = min(max(e, 0), 1)
        # extra_cost += (1-e)/0.22
        steps_passed += travel_distance
        return (key,coverage, e, mode, steps_passed), extra_cost

    def get_idx_coverage(self, x, y):
        """Return idx of coverage list based on x and y."""
        res_key, res_val = min(self.coverage.items(), key=lambda v: np.sqrt((x - v[0][0]) ** 2 + (y - v[0][1]) ** 2))
        return res_key, res_val
    
    def get_next_idx(self, x, y, vx, vy):
        line = self.closest_line([x,y])
        possible = self.points_within_margin(self.coverage, line, margin=1)
        larger = [key for key in possible if key[0]>x]
        if len(larger)>1:
            return larger[1]
        if larger:
            return possible[0]
        return possible[1]

    def goal_condition(self, state):
        """Goal is achieved when the sum of all coverage values equals 1."""
        key, coverage, e, mode,steps_passed = state
        coverage = list(coverage)
        return sum(coverage) / len(coverage) == 1
    
    def single_square_travel(self, key,v_x):
        new_index = [*self.coverage].index(key)+1
        if new_index == len(self.coverage):
            new_index = 0
        next_key = [*self.coverage][new_index]
        travel_dist = next_key[0]-key[0]
        output = (travel_dist if travel_dist>0 else travel_dist+21600)/v_x
        # if output < 107:
        #     breakpoint()
        return output, next_key
    
    def closest_line(self, point):
        x0, y0 = point
        min_distance = float('inf')
        closest = None

        for line in self.trajectory:
            x1, y1, x2, y2 = line

            # Compute coefficients of the line equation
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2

            # Compute the perpendicular distance
            distance = abs(a * x0 + b * y0 + c) / math.sqrt(a**2 + b**2)

            # Update the closest line
            if distance < min_distance:
                min_distance = distance
                closest = line

        return closest
    
    def points_within_margin(self, points, line, margin=1):
        x1, y1, x2, y2 = line
        filtered_points = []

        # Compute coefficients of the line equation
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        denominator = math.sqrt(a**2 + b**2)  # Precompute for efficiency

        for x, y in points:
            # Perpendicular distance from point to the line
            distance = abs(a * x + b * y + c) / denominator

            # Check if the point is within the margin
            if distance <= margin:
                # Check if the point projection is within the line segment bounds
                # Parametric t for projection
                dx, dy = x2 - x1, y2 - y1
                length_squared = dx**2 + dy**2
                if length_squared != 0:  # Avoid division by zero
                    t = ((x - x1) * dx + (y - y1) * dy) / length_squared
                    if 0 <= t <= 1:
                        filtered_points.append((x, y))

        return filtered_points

if __name__ == '__main__':
        
    x0 = 500
    y0=250
    intersections = calculate_border_intersections(domain_width, domain_height, x0, y0, v_x, v_y, proximity_threshold)

    m_distance = find_distance_between_trajectories(intersections)
    midpoint = find_midpoint_of_trajectory(intersections)

    closest_value = map_to_closest_value(m_distance/2, [500, 400, 300])
    trajectory = bundle_segments(intersections[1:])
    # Step 3: Create a grid centered at the midpoint with squares twice the mapped size
    square_size = 2 * closest_value
    square_size = 590
    centers,efective_length,direction_vector = place_squares_trajectory(trajectory,square_size, declive)
    # plot_squares(trajectory,centers,590,domain_width, domain_height,efective_length,direction_vector)

    coverage={}
    for center in centers:
        coverage[(center[0] ,center[1])]=0
    n_squares = len(coverage)
    # Possible actions
    possible_actions = [0, 1]

    # Run MCTS
    start_state = ([*coverage][0],tuple([*coverage.values()]), 1, 7, 0)


    astar_centers = AStar(possible_actions, coverage, square_size)
    optimal_path, total_cost = astar_centers.search(start_state,efective_length,direction_vector)

    print("Optimal Path:", optimal_path)
    print("Total Cost:", total_cost)


