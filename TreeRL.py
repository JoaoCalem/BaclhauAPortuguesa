# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:40:18 2024

@author: admin
"""

import numpy as np
import random
from collections import defaultdict
from adaptivempc import *

class Environment:
    def __init__(self, coverage, square_size=600):
        self.coverage = coverage
        self.square_size = square_size

    def find_next_center(self, current_state, action, tol, v_x=4.35, v_y=5.49):
        """Simulate the transition as in A*."""
        key, coverage, e, mode, steps_passed = current_state
        extra_cost = 0
        travel_distance = 0

        if action == 1 and e > 0.9:
            extra_cost -= 0.01

        if (action == 1 and mode < 3) or (action == 0 and mode == 4):
            travel_time, key = self.single_square_travel(key, v_x)
            travel_distance += travel_time * v_y / self.square_size
            e_dif = travel_time * 0.2 / 100
            e += e_dif if action == 0 else -e_dif
        elif mode != 5:
            total_time = 3 * 60
            while total_time > 0:
                travel_time, key = self.single_square_travel(key, v_x)
                travel_distance += travel_time * v_y / self.square_size
                total_time -= travel_time
            e += -(total_time + tol) * 0.2 / 100
            if action == 1:
                e -= tol * 0.2 / 100
            mode = 0 if action == 1 else 4
        else:
            total_time = 20 * 60
            while total_time > 0:
                travel_time, key = self.single_square_travel(key, v_x)
                travel_distance += travel_time * v_y / self.square_size
                total_time -= travel_time
            mode = 0 if action == 1 else 4
            if action == 0:
                e += -total_time * 0.2 / 100
            else:
                e += total_time * 0.2 / 100

        if e < 0.1:
            extra_cost += np.inf
        elif action == 1:
            coverage = list(coverage)
            coverage[[*self.coverage].index(key)] = 1
            coverage = tuple(coverage)
        e = min(max(e, 0), 1)
        steps_passed += travel_distance
        return (key, coverage, e, mode, steps_passed), extra_cost

    def single_square_travel(self, key, v_x):
        """Simulate travel to the next square."""
        new_index = [*self.coverage].index(key) + 1
        if new_index == len(self.coverage):
            new_index = 0
        next_key = [*self.coverage][new_index]
        travel_dist = next_key[0] - key[0]
        output = (travel_dist if travel_dist > 0 else travel_dist + 21600) / v_x
        return output, next_key

    def get_possible_actions(self, state):
        """Return possible actions for a given state."""
        return [0, 1]

    def is_terminal(self, state):
        """Check if the goal condition is satisfied."""
        _, coverage, e, _, _ = state
        return e<0.01

    def reward(self, state):
        """Compute the reward for the given state."""
        _, coverage, _, _, steps = state
        efficiency_penalty = -0.1 * steps
        return sum(coverage) / len(coverage)-efficiency_penalty


class MCTS:
    def __init__(self, env, exploration_weight=math.sqrt(2)):
        self.env = env
        self.exploration_weight = exploration_weight
        self.Q = defaultdict(float)  # Q(s, a): Total reward for state-action pair
        self.N = defaultdict(int)    # N(s, a): Visit count for state-action pair
        self.Ns = defaultdict(int)   # N(s): Visit count for state
        self.children = dict()       # Children of a state

    def select_action(self, state):
        """Select the best action based on UCT."""
        if state not in self.children:
            return None  # Expand if no children exist

        def uct(action):
            
            if (state, action) not in self.N:
                return float('inf')  # Encourage exploration
            next_state, _ = self.env.find_next_center(state, action, tol=0.01)
            return (
                self.Q[(state, action)] / self.N[(state, action)] +
                self.exploration_weight * math.sqrt(math.log(self.Ns[state]) / self.N[(state, action)])+self.heuristic_score(next_state)
            )

        return max(self.children[state], key=uct)
    def heuristic_score(self, state):
        """Calculate heuristic score for a state."""
        key, coverage, energy, mode, steps_passed = state
         # Example energy decay
    
        # Heuristic: Favor high energy and coverage
        score = -(len(coverage)-sum(coverage))/len(coverage) # Weight coverage heavily
   
        return score
    def expand(self, state):
        """Expand the node by generating its children."""
        if state in self.children:
            return  # Already expanded

        self.children[state] = self.env.get_possible_actions(state)

    def simulate(self, state, depth, tol):
        """Simulate a rollout from the given state."""
        if self.env.is_terminal(state) or depth == 0:
            return self.env.reward(state) if depth==0 else -100

        action = random.choice(self.env.get_possible_actions(state))
        next_state, _ = self.env.find_next_center(state, action, tol=tol)
        return self.simulate(next_state, depth - 1, tol)

    def backpropagate(self, path, reward):
        """Backpropagate the result of the simulation."""
        for state, action in reversed(path):
            self.Q[(state, action)] += reward
            self.N[(state, action)] += 1
            self.Ns[state] += 1

    def search(self, initial_state, num_simulations, depth, tol):
        """Run MCTS starting from the initial state."""
        for _ in range(num_simulations):
            state = initial_state
            path = []

            # Selection
            while state in self.children and self.children[state]:
                action = self.select_action(state)
                path.append((state, action))
                state, extra_cost = self.env.find_next_center(state, action, tol=tol)

            # Expansion
            self.expand(state)

            # Simulation
            reward = self.simulate(state, depth, tol)

            # Backpropagation
            self.backpropagate(path, reward)

        # Return the best action from the root
        if initial_state not in self.children:
            return None
        return max(self.children[initial_state], key=lambda a: self.Q[(initial_state, a)] / self.N[(initial_state, a)])
    def get_best_sequence(self, initial_state, tol):
        """Retrieve the best sequence of actions and states based on Q-values."""
        state = initial_state
        best_sequence = []

        while not self.env.is_terminal(state):
            # Check if the state has children
            if state not in self.children:
                break

            # Select the action with the highest Q-value
            best_action = max(
                self.children[state],
                key=lambda action: self.Q[(state, action)] / max(1, self.N[(state, action)])
            )

            # Append the best action and state to the sequence
            _, coverage, e, _, steps = state
            reward = sum(coverage)/len(coverage)
            best_sequence.append((best_action,reward,e))

            # Transition to the next state based on the selected action
            state, _ = self.env.find_next_center(state, best_action, tol=tol)

        return best_sequence,steps
domain_width = 21600
domain_height = 10800
x0 = 500
y0=250
v_x = 4.35  # Velocity in x-direction
v_y = 5.49  # Velocity in y-direction
height = 590
declive = v_y/v_x
angle = math.atan(declive)
proximity_threshold = 100
intersections = calculate_border_intersections(domain_width, domain_height, x0, y0, v_x, v_y, proximity_threshold, height)
# Step 2: Plot the trajectory intersections
# plot_intersections(domain_width, domain_height, intersections)


m_distance = find_distance_between_trajectories(intersections)

intersections = calculate_border_intersections(domain_width, domain_height, x0+m_distance, y0, v_x, v_y, proximity_threshold, height)
m_distance = find_distance_between_trajectories(intersections)

midpoint = find_midpoint_of_trajectory(intersections)

closest_value = map_to_closest_value(m_distance/2, [500, 400, 300])
trajectory = bundle_segments(intersections[1:])
# Step 3: Create a grid centered at the midpoint with squares twice the mapped size
square_size = 2 * closest_value
centers = place_squares_trajectory(trajectory,height, declive, domain_height)
# plot_squares(trajectory,centers,590,domain_width, domain_height,efective_length,direction_vector)

coverage={}
for center in centers:
    coverage[(center[0] ,center[1])]=0
# env = CoverageEnv(coverage)
env = Environment(coverage)
mcts = MCTS(env)

# Define the initial state as per A*'s structure
initial_state = ([*coverage][0], tuple(coverage.values()), 1, 7, 0)  # Example initial state

# Run MCTS
##### best action deve chegar caso se faça forward em cada estado########
best_action = mcts.search(initial_state, num_simulations=500, depth=100, tol=0.01)
best_sequence,steps= mcts.get_best_sequence(initial_state,tol=0.01)
print(best_sequence)
print(steps)