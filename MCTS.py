# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:01:40 2024

@author: admin
"""
import math
import random
import numpy as np
from adaptivempc import *
class Node:
    def __init__(self, index, parent=None, action=None):
        self.index = index  # Index along the trajectory
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, possible_actions):
        return len(self.children) == len(possible_actions)

    def best_child(self, exploration_weight=1.0):
        """Returns the child with the highest UCB1 value."""
        return max(
            self.children,
            key=lambda child: child.value / (child.visits + 1e-5) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-5))
        )

    def add_child(self, index, action):
        """Adds a child node to the current node."""
        child = Node(index=index, parent=self, action=action)
        self.children.append(child)
        return child



class MCTS:
    def __init__(self, possible_actions,coverage,square_size=600, max_iterations=1000):
        self.coverage=coverage  # External reward calculation function
        self.possible_actions = possible_actions
        self.max_iterations = max_iterations
        self.root = None  # Root node will be set dynamically

    def search(self, start_state):
        """Perform the search starting from a given state."""
        # Initialize the root node with the given state
        self.root = Node(index=0)
        self.root.state = start_state  # Store the starting state at the root
        for _ in range(self.max_iterations):
            node = self.select(self.root)
            if not node.is_fully_expanded(self.possible_actions):
                node = self.expand(node)
            reward = self.simulate(node)
            print(reward)
            self.backpropagate(node, reward)
        return self.root.best_child(exploration_weight=0.0)

    def select(self, node):
        """Select the best node to explore based on UCT."""
        while node.children:
            node = node.best_child()
        print(node.state)
        return node

    def expand(self, node):
        """Expand a node by adding a new child with an untried action."""
        tried_actions = [child.action for child in node.children]
        untried_actions = [action for action in self.possible_actions if action not in tried_actions]
        if not untried_actions:
            return node  # No new expansions possible
        action = random.choice(untried_actions)
        # Calculate the new state externally using the forecast function
        new_state = self.forecast_function(node.state, action)
        child = node.add_child(index=node.index + 1, action=action)
        child.state = new_state  # Store the forecasted state in the child node
        return child

    def simulate(self, node):
        """Perform a random playout from the node."""
        total_reward = 0
        current_state = node.state
        for a in range(len(self.possible_actions)):
            action = self.possible_actions[a]
            current_state = self.forecast_function(current_state, action)
            # print(current_state)
            total_reward += self.reward_function(current_state, action)
        return total_reward

    def backpropagate(self, node, reward):
        """Backpropagate the reward up the tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def reward_function(self,state, action):
        """Calculate the reward for a given state and action."""      
        x, y,vx,vy, m, e, mode, coverage, close = state
        reward =0
        if close and coverage==0 and action ==5:
             if mode==0:
                reward+=100
        reward-=1-e
        if action ==5 and mode>=3:
            reward-=1
        if mode==5:
            reward-=1
        if mode<3 and action==5 and not close:
            reward-=1
        return reward

    
    def get_idx_coverage(self,x,y) -> int:
        """ 
        Return idx of covergae list based on x and y.
        Return None if x and y not around center of square
        """
        res_key, res_val = min(self.coverage.items(), key=lambda v: np.sqrt((x-v[0][0])**2+(y-v[0][1])**2))
        return res_key
    
    def step(self,current_state,action,steps=1):
        """update for n steps (seconds)"""
        x, y, vx, vy, m, e, mode, coverage, close = current_state
        x =x + vx * steps if x + vx * steps<= 21600 else 0
        y = y+  vy * steps if y + vy * steps<= 10800 else 0
        x = int(np.round(x))
        y = int(np.round(y))
        idx = self.get_idx_coverage(x,y)
        dist =np.sqrt((x-idx[0])**2+(y-idx[1])**2)
        if dist<5:
            close=True
            print('ole')
        if mode<3 and (action<3 or action==5):
            e -= 0.2*steps
        if mode ==5:
            e += 0.05*steps
        elif mode==4 and action>3:
            e += 0.2*steps
        if mode<3 and action ==5 and close:
            if close:
                coverage=1
                self.coverage[idx] = 1
            m+=1
        e = min(max(e,0),1)
        if e<0.1:
            mode = 5
        else:
            mode = action if action<5 else mode
        
        return (x,y,vx,vy,m,e,mode,coverage,close)
        
    # Example external forecast function
    def forecast_function(self,current_state, action):
        """Predict the next state based on the current state and action."""
        x, y, vx, vy, m, e, mode, coverage, close = current_state
        # Example logic to compute the next state
        no_transition = any([
            action==5,
            mode<3 and action<3, mode==action
            ])
        transition = False if no_transition else True
        if mode == 6 and action<5:
            new_state = self.step(current_state,action,20*60)
        if transition:
            new_state = self.step(current_state,action,3*60)
        else:
            new_state = self.step(current_state,action)
        # print(new_state)
        return new_state

# Example reward function

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
start_state = (500, 250,4.35,5.49, 0, 1, 7, 0,False)

# Run MCTS starting from the new state
mcts = MCTS(possible_actions,coverage, max_iterations=1000)
best_node = mcts.search(start_state)

# Retrieve the best action sequence
best_actions = []
node = best_node
while node.parent:
    best_actions.append(node.action)
    node = node.parent
best_actions.reverse()

print("Best action sequence from the given state:", best_actions)


