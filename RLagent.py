# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:56:07 2024

@author: admin
"""
import numpy as np


class NStepQLearning:
    def __init__(self, qfunction, n, alpha=0.1, gamma=0.99, lambda1=1, lambda2=1, lambda3=1e-3):
        """
        Initialize the n-step Q-learning trainer with Beta-based Thompson Sampling for exploration.

        Parameters:
        -----------
        qfunction : np.ndarray
            The Q-function to update, stored as a 2D array (states x actions).
        n : int
            Number of steps for n-step updates.
        alpha : float
            Learning rate (0 < alpha <= 1).
        gamma : float
            Discount factor (0 <= gamma <= 1).
        """
        self.qfunction = qfunction
        self.n = n
        self.alpha = alpha
        self.gamma = gamma

        # Buffers for storing (state, action, reward)
        self.buffer = []
        self.state = None
        self.action = None
        self.reward = None
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Beta distribution parameters (successes and failures for each action)
        self.beta_params = np.ones((qfunction.shape[0], qfunction.shape[1], 2))  # [successes, failures]

    def select_action(self, state):
        """
        Select an action using Thompson Sampling with Beta distribution.

        Parameters:
        -----------
        state : int
            Hashed state representation.

        Returns:
        --------
        action : int
            Selected action.
        """
        # Sample from Beta distribution for each action in the current state
        samples = [
            np.random.beta(self.beta_params[state, a, 0], self.beta_params[state, a, 1])
            for a in range(self.qfunction.shape[1])
        ]
        # Select the action with the highest sampled value
        return int(np.argmax(samples))

    def half_step(self, state):
        """
        Select an action and prepare for the next transition.

        Parameters:
        -----------
        state : int
            Hashed state representation.

        Returns:
        --------
        action : int
            Selected action.
        """
        action = self.select_action(state)
        self.state = state
        self.action = action
        return action

    def update_reward(self, status, action, deltacoverage, done):
        """
        Calculate and update the reward based on the agent's status.

        Parameters:
        -----------
        status : dict
            Current state information (battery, images taken, etc.).
        action : int
            Selected action.
        deltacoverage : float
            Change in coverage area.
        done : bool
            Whether the episode is terminated.

        Returns:
        --------
        reward : float
            Computed reward.
        """
        if done:
            reward = -100
        else:
            reward = 0
            if status['state'] != "acquisition" and action == 5:
                reward -= 1
            elif action == 5:
                reward += self.lambda1 * deltacoverage
            reward -= self.lambda2 * (1 - status["battery"] / 100)
            reward -= self.lambda3 * status["images_taken"]

        self.reward = reward
        return reward

    def process_transition(self, next_state, done):
        """
        Process a single transition in real-time.

        Parameters:
        -----------
        next_state : int
            Hashed next state representation.
        done : bool
            Whether the episode is terminated.

        Returns:
        --------
        next_action : int
            Action to take in the next state.
        """
        # Store the transition in the buffer
        self.buffer.append((self.state, self.action, self.reward))

        # Perform update if the buffer contains n steps
        if len(self.buffer) == self.n:
            # Compute the n-step return
            G = sum(self.gamma ** i * self.buffer[i][2] for i in range(self.n))  # Discounted rewards

            if not done:
                # Add the estimated value of the next state
                G += self.gamma ** self.n * max(self.qfunction[next_state, :])

            # Update Q-value for the first state-action pair in the buffer
            state_to_update, action_to_update, reward_to_update = self.buffer[0]
            current_q_value = self.qfunction[state_to_update, action_to_update]
            self.qfunction[state_to_update, action_to_update] += self.alpha * (G - current_q_value)

            # Update Beta distribution parameters
            if reward_to_update > 0:
                self.beta_params[state_to_update, action_to_update, 0] += 1  # Increment successes
            else:
                self.beta_params[state_to_update, action_to_update, 1] += 1  # Increment failures

            # Remove the first element from the buffer
            self.buffer.pop(0)

        if done:
            self.buffer = []  # Clear the buffer if the episode ends

        # Select the next action
        next_action = self.select_action(next_state) if not done else None
        return next_action



# def calculate_reward(image_taken, deltacoverage,energy, lambda1, lambda2,lambda3):
#     return lambda1*deltacoverage -lambda2*image_taken +lambda3*energy
    



