# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:01:40 2024

@author: admin
"""
import math
import random
import numpy as np
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
    def __init__(self, forecast_function, reward_function, possible_actions, max_iterations=1000):
        self.forecast_function = forecast_function  # External state forecasting function
        self.reward_function = reward_function  # External reward calculation function
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
            self.backpropagate(node, reward)
        return self.root.best_child(exploration_weight=0.0)

    def select(self, node):
        """Select the best node to explore based on UCT."""
        while node.children:
            node = node.best_child()
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
        for _ in range(len(self.possible_actions)):
            action = random.choice(self.possible_actions)
            current_state = self.forecast_function(current_state, action)
            total_reward += self.reward_function(current_state, action)
        return total_reward

    def backpropagate(self, node, reward):
        """Backpropagate the reward up the tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

def get_idx_coverage(x,y) -> int:
    pass

def step(current_state,action,steps=1):
    """update for n steps (seconds)"""
    x, y, vx, vy, m, e, mode, coverage = current_state
    x += vx * steps
    y += vy * steps
    x = int(np.round(x))
    y = int(np.round(y))
    if mode<3 and (action<3 or action==5):
        e -= 0.2*steps
    elif mode==4 and action>3:
        e += 0.2*steps
    if mode<3 and action ==5:
        coverage[get_idx_coverage(x,y)] = 1
    mode = action if action<5 else mode
    return current_state
    
# Example external forecast function
def forecast_function(current_state, action):
    """Predict the next state based on the current state and action."""
    x, y, vx, vy, m, e, mode, coverage = current_state
    # Example logic to compute the next state
    no_transition = any([
        action==5,
        mode==action,
        mode<3 and action<3
        ])
    transition = False if no_transition else True
    if mode == 6 and action<5:
        new_state = step(current_state,action,20*60)
    if transition:
        new_state = step(current_state,action,3*60)
    else:
        new_state = step(current_state,action)
    return new_state

# Example reward function
def reward_function(state, action):
    """Calculate the reward for a given state and action."""
    x, y,vx,vy, m, e, mode, coverage = state
    return -((x - 10000)**2 + (y - 5000)**2) + action


# Possible actions
possible_actions = [1, 2, 3, 4, 5, 6]

# Run MCTS
coverage_size = 10 #ToDO
coverage = np.zeros([coverage_size])
start_state = (500, 250,4.35,5.49, 80, 90, 1, coverage)

# Run MCTS starting from the new state
mcts = MCTS(forecast_function, reward_function, possible_actions, max_iterations=500)
best_node = mcts.search(start_state)

# Retrieve the best action sequence
best_actions = []
node = best_node
while node.parent:
    best_actions.append(node.action)
    node = node.parent
best_actions.reverse()

print("Best action sequence from the given state:", best_actions)


