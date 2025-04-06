import numpy as np
import random
from collections import deque
import math

class SimpleAgent:
    """A simplified RL agent that uses Q-learning with enhancements"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # Larger memory for better learning
        self.learning_rate = learning_rate  # Learning rate
        self.gamma = gamma                  # Higher discount factor for long-term rewards
        self.epsilon = 1.0                  # Exploration rate
        self.epsilon_min = 0.05             # Higher minimum exploration
        self.epsilon_decay = 0.998          # Slower decay for better exploration
        self.q_table = {}
        
        # Additional parameters for better learning
        self.learning_rate_decay = 0.9999  # Learning rate decay
        self.min_learning_rate = 0.0001    # Minimum learning rate
        self.priority_memory = []          # For prioritized experience replay
        self.priority_alpha = 0.6          # Priority exponent
        self.rnd_weight = 0.1              # Weight for random network distillation 
        
    def _get_state_key(self, state):
        """Convert state array to a hashable key with variable precision"""
        # Use higher precision for position and angle, lower for sensors
        rounded_state = []
        for i, val in enumerate(state):
            if i < 8:  # Sensor readings - lower precision
                rounded_state.append(round(val, 1))
            else:  # Position and angle - higher precision
                rounded_state.append(round(val, 2))
        return tuple(rounded_state)
        
    def act(self, state):
        """Choose an action based on epsilon-greedy policy with adaptive exploration"""
        state_key = self._get_state_key(state)
        
        # Calculate adaptive epsilon based on state novelty
        local_epsilon = self.epsilon
        if state_key not in self.q_table:
            # Increase exploration for novel states
            local_epsilon = min(1.0, self.epsilon * 1.5)
        
        # Explore: choose random action with probability epsilon
        if np.random.rand() <= local_epsilon:
            # Occasionally favor forward movement for better exploration
            if np.random.rand() < 0.3:  # 30% of exploration will be forward
                return 0  # Forward action
            else:
                return random.randrange(self.action_size)
            
        # Exploit: choose best action from Q-table
        if state_key in self.q_table:
            # Add small noise to q-values to break ties randomly
            q_values = self.q_table[state_key] + np.random.normal(0, 0.01, self.action_size)
            return np.argmax(q_values)
        else:
            # If state not in Q-table, initialize it with optimistic initial values
            self.q_table[state_key] = np.ones(self.action_size) * 0.1  # Optimistic initialization
            return random.randrange(self.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory with priority"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Calculate TD error for prioritized replay
        td_error = abs(reward + (0 if done else self.gamma * self._get_max_q(next_state_key)) - self._get_q(state_key, action))
        
        # Store experience with priority
        self.memory.append((state, action, reward, next_state, done, td_error))
        
    def _get_q(self, state_key, action):
        """Get Q-value safely with initialization if needed"""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(self.action_size) * 0.1
        return self.q_table[state_key][action]
    
    def _get_max_q(self, state_key):
        """Get maximum Q-value safely with initialization if needed"""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(self.action_size) * 0.1
        return np.max(self.q_table[state_key])
    
    def replay(self, batch_size):
        """Train the agent by replaying experiences with enhancements"""
        if len(self.memory) < batch_size:
            return
            
        # Sort experiences by TD error (higher error → higher priority)
        # We don't sort the entire memory to avoid expensive operations
        sorted_memory = sorted(list(self.memory), key=lambda x: x[5], reverse=True)[:batch_size*4]
        
        # Prioritized experience replay (sample more important experiences more often)
        # But still keep some randomness for stability
        if np.random.random() < 0.7 and len(sorted_memory) >= batch_size:
            # Use prioritized samples 70% of the time
            minibatch = sorted_memory[:batch_size]
        else:
            # Random sampling 30% of the time
            minibatch = random.sample(self.memory, batch_size)
        
        # Calculate current LR with decay
        current_lr = max(self.min_learning_rate, self.learning_rate)
        
        for state, action, reward, next_state, done, _ in minibatch:
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)
            
            # Initialize state in Q-table if not exists
            if state_key not in self.q_table:
                self.q_table[state_key] = np.ones(self.action_size) * 0.1
                
            # Initialize next_state in Q-table if not exists
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.ones(self.action_size) * 0.1
            
            # Q-learning update with double Q-learning principle
            if done:
                target = reward
            else:
                # Double Q-learning: first find best action from current Q-values
                best_action = np.argmax(self.q_table[next_state_key])
                # Then use that action's Q-value for the target
                target = reward + self.gamma * self.q_table[next_state_key][best_action]
            
            # Calculate TD error for monitoring
            td_error = target - self.q_table[state_key][action]
            
            # Update Q-value with adaptive learning rate
            # Smaller updates for more frequently visited states
            state_visit_count = np.sum(self.q_table[state_key] > 0)
            adaptive_lr = current_lr / (1 + 0.1 * state_visit_count)
            
            self.q_table[state_key][action] += adaptive_lr * td_error
            
            # Ensure Q-values don't explode
            self.q_table[state_key][action] = min(10.0, max(-5.0, self.q_table[state_key][action]))
        
        # Decay epsilon for next round
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Decay learning rate
        self.learning_rate *= self.learning_rate_decay
        
    def save_model(self, filepath):
        """Save Q-table to file"""
        np.save(filepath, self.q_table)
        
    def load_model(self, filepath):
        """Load Q-table from file"""
        self.q_table = np.load(filepath, allow_pickle=True).item()
        
    def get_exploration_rate(self):
        """Return current exploration rate for display"""
        return self.epsilon