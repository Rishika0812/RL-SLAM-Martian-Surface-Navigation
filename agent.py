import numpy as np
import random
from collections import deque
import math

class SimpleAgent:
    """A simplified RL agent that uses Q-learning with enhancements"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Higher minimum exploration
        self.epsilon_decay = 0.97  # Slower decay
        
        # Double Q-learning tables
        self.q_table_1 = {}  # First Q-table
        self.q_table_2 = {}  # Second Q-table
        
        # Enhanced learning parameters
        self.learning_rate_decay = 0.99995
        self.min_learning_rate = 0.00001
        self.priority_memory = []
        self.priority_alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.max_priority = 1.0
        
        # State discretization parameters
        self.position_bins = 30  # Increased position resolution
        self.angle_bins = 24    # Increased angle resolution
        self.sensor_bins = 15   # Increased sensor resolution
        self.map_size = 20
        
        # Exploration tracking
        self.episodes_seen = 0
        self.visited_states = set()  # Track visited states
        self.consecutive_revisits = 0  # Track repeated states
        self.exploration_bonus = 0.5   # Bonus for exploring new states
        
    def _get_state_key(self, state):
        """Convert state array to a hashable key with improved discretization"""
        rounded_state = []
        for i, val in enumerate(state):
            if i < 8:  # Sensor readings
                # Discretize sensor readings (normalized to [0,1])
                normalized_val = val / self.map_size
                bin_index = int(normalized_val * self.sensor_bins)
                rounded_state.append(min(bin_index, self.sensor_bins - 1))
            elif i < 10:  # Position
                # Discretize position (normalized to [0,1])
                normalized_val = val / self.map_size
                bin_index = int(normalized_val * self.position_bins)
                rounded_state.append(min(bin_index, self.position_bins - 1))
            else:  # Angle
                # Discretize angle (normalized to [0,2π])
                angle = (val + np.pi) % (2 * np.pi)  # Normalize to [0, 2π]
                normalized_val = angle / (2 * np.pi)
                bin_index = int(normalized_val * self.angle_bins)
                rounded_state.append(min(bin_index, self.angle_bins - 1))
        return tuple(rounded_state)
        
    def act(self, state):
        """Action selection using average of both Q-tables"""
        state_key = self._get_state_key(state)
        
        if np.random.rand() <= self.epsilon:
            # Exploration logic remains the same
            sensor_readings = state[:8]
            left_space = np.mean(sensor_readings[:4])
            right_space = np.mean(sensor_readings[4:])
            front_space = np.mean(sensor_readings[3:5])
            
            if front_space > 0.7 * self.map_size:
                return 0
            if max(left_space, right_space) > 0.3 * self.map_size:
                return 1 if left_space > right_space else 2
            return np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
        
        # Exploitation using average of both Q-tables
        if state_key in self.q_table_1 and state_key in self.q_table_2:
            q_values_1 = np.array(self.q_table_1[state_key])
            q_values_2 = np.array(self.q_table_2[state_key])
            average_q = (q_values_1 + q_values_2) / 2
            return np.argmax(average_q)
        
        # Initialize new state in both tables
        if state_key not in self.q_table_1:
            self.q_table_1[state_key] = np.ones(self.action_size) * 0.1
        if state_key not in self.q_table_2:
            self.q_table_2[state_key] = np.ones(self.action_size) * 0.1
        return 0
            
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory with priority"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Calculate TD error for priority
        if state_key not in self.q_table_1:
            self.q_table_1[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table_1:
            self.q_table_1[next_state_key] = np.zeros(self.action_size)
            
        current_q = self.q_table_1[state_key][action]
        next_q = np.max(self.q_table_1[next_state_key]) if not done else 0
        td_error = abs(reward + self.gamma * next_q - current_q)
        
        # Store experience with priority
        priority = (td_error + 1e-6) ** self.priority_alpha
        self.priority_memory.append((state_key, action, reward, next_state_key, done, priority))
        
        # Keep memory size limited
        if len(self.priority_memory) > 10000:
            self.priority_memory.pop(0)
            
    def _get_q(self, state_key, action):
        """Get Q-value safely with initialization if needed"""
        if state_key not in self.q_table_1:
            self.q_table_1[state_key] = np.ones(self.action_size) * 0.1
        return self.q_table_1[state_key][action]
    
    def _get_max_q(self, state_key):
        """Get maximum Q-value safely with initialization if needed"""
        if state_key not in self.q_table_1:
            self.q_table_1[state_key] = np.ones(self.action_size) * 0.1
        return np.max(self.q_table_1[state_key])
    
    def replay(self, batch_size):
        """Double Q-learning update"""
        if len(self.priority_memory) < batch_size:
            return
            
        priorities = np.array([x[5] for x in self.priority_memory])
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.priority_memory), batch_size, p=probs)
        batch = [self.priority_memory[i] for i in indices]
        weights = (len(self.priority_memory) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        for i, (state_key, action, reward, next_state_key, done, _) in enumerate(batch):
            # Initialize states if needed
            for table in [self.q_table_1, self.q_table_2]:
                if state_key not in table:
                    table[state_key] = np.zeros(self.action_size)
                if next_state_key not in table:
                    table[next_state_key] = np.zeros(self.action_size)
            
            # Randomly choose which Q-table to update
            if np.random.rand() < 0.5:
                # Update Q-table 1 using Q-table 2 for next state value
                current_q = self.q_table_1[state_key][action]
                best_action = np.argmax(self.q_table_1[next_state_key])
                next_q = self.q_table_2[next_state_key][best_action] if not done else 0
                target = reward + self.gamma * next_q
                self.q_table_1[state_key][action] += self.learning_rate * weights[i] * (target - current_q)
            else:
                # Update Q-table 2 using Q-table 1 for next state value
                current_q = self.q_table_2[state_key][action]
                best_action = np.argmax(self.q_table_2[next_state_key])
                next_q = self.q_table_1[next_state_key][best_action] if not done else 0
                target = reward + self.gamma * next_q
                self.q_table_2[state_key][action] += self.learning_rate * weights[i] * (target - current_q)
            
            # Update priorities
            td_error = abs(target - current_q)
            self.priority_memory[indices[i]] = (state_key, action, reward, next_state_key, done,
                                              (td_error + 1e-6) ** self.priority_alpha)
        
        # Update parameters
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.episodes_seen += 1
        if self.episodes_seen > 10:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if len(self.visited_states) > 100:
            self.learning_rate = max(self.min_learning_rate, 
                                   self.learning_rate * self.learning_rate_decay)
        
    def save_model(self, filepath):
        """Save Q-table to file"""
        np.save(filepath, self.q_table_1)
        
    def load_model(self, filepath):
        """Load Q-table from file"""
        self.q_table_1 = np.load(filepath, allow_pickle=True).item()
        self.q_table_2 = self.q_table_1.copy()
        
    def get_exploration_rate(self):
        """Return current exploration rate for display"""
        return self.epsilon