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
        
        self.stuck_threshold = 5  # Number of steps to consider agent as stuck
        self.position_history = deque(maxlen=10)  # Store recent positions
        self.corner_penalty = -0.5  # Penalty for staying in corners
        self.movement_bonus = 0.2   # Bonus for moving to new areas
        
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
        
    def _is_in_corner(self, state):
        """Detect if agent is in a corner or against a wall"""
        sensor_readings = state[:8]
        left_sensors = sensor_readings[:4]
        right_sensors = sensor_readings[4:]
        
        # Check for corner patterns in sensor readings
        left_blocked = np.mean(left_sensors) < 0.3 * self.map_size
        right_blocked = np.mean(right_sensors) < 0.3 * self.map_size
        front_blocked = np.mean(sensor_readings[3:5]) < 0.3 * self.map_size
        
        return (left_blocked and front_blocked) or (right_blocked and front_blocked)
    
    def _is_stuck(self):
        """Check if agent is stuck by analyzing recent positions"""
        if len(self.position_history) < self.stuck_threshold:
            return False
            
        recent_positions = np.array(self.position_history)
        max_distance = np.max(np.linalg.norm(recent_positions - recent_positions[0], axis=1))
        return max_distance < 0.5  # If haven't moved more than 0.5 units
        
    def act(self, state):
        """Enhanced action selection with corner escape behavior"""
        state_key = self._get_state_key(state)
        
        # Store position for stuck detection
        position = state[8:10]  # Extract position from state
        self.position_history.append(position)
        
        # Check if stuck in corner or against wall
        in_corner = self._is_in_corner(state)
        is_stuck = self._is_stuck()
        
        # Increase exploration if stuck or in corner
        local_epsilon = self.epsilon
        if in_corner or is_stuck:
            local_epsilon = min(1.0, self.epsilon * 2)  # Double exploration rate
        
        if np.random.rand() <= local_epsilon:
            sensor_readings = state[:8]
            
            # If stuck or in corner, prioritize escape
            if in_corner or is_stuck:
                # Find the direction with most open space
                left_space = np.mean(sensor_readings[:4])
                right_space = np.mean(sensor_readings[4:])
                front_space = np.mean(sensor_readings[3:5])
                
                # Choose direction with most space
                spaces = [front_space, left_space, right_space]
                max_space_idx = np.argmax(spaces)
                
                if max_space_idx == 0 and front_space > 0.3 * self.map_size:
                    return 0  # Move forward if enough space
                elif max_space_idx == 1:
                    return 1  # Turn left
                else:
                    return 2  # Turn right
            
            # Normal exploration strategy
            if np.mean(sensor_readings) > 0.7 * self.map_size:  # In open space
                return 0  # Prefer forward movement
            elif np.random.rand() < 0.7:  # 70% chance of intelligent choice
                left_space = np.mean(sensor_readings[:4])
                right_space = np.mean(sensor_readings[4:])
                return 1 if left_space > right_space else 2
            else:
                return np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
        
        # Exploitation with corner avoidance
        if state_key in self.q_table_1 and state_key in self.q_table_2:
            q_values_1 = np.array(self.q_table_1[state_key])
            q_values_2 = np.array(self.q_table_2[state_key])
            average_q = (q_values_1 + q_values_2) / 2
            
            # Apply penalties/bonuses based on predicted next states
            for action in range(self.action_size):
                next_state = self._predict_next_state(state, action)
                if self._is_in_corner(next_state):
                    average_q[action] += self.corner_penalty
                elif not any(np.array_equal(self._predict_next_state(state, action)[8:10], 
                           pos) for pos in self.position_history):
                    average_q[action] += self.movement_bonus
            
            return np.argmax(average_q)
        
        return 0  # Default to forward movement for new states
        
    def _predict_next_state(self, state, action):
        """Predict next state based on action"""
        next_state = np.array(state)
        if action == 0:  # Forward
            # Update position based on current angle
            angle = state[10]
            next_state[8] += 0.1 * np.cos(angle)  # x position
            next_state[9] += 0.1 * np.sin(angle)  # y position
        elif action == 1:  # Left
            next_state[10] = (next_state[10] + 0.2) % (2 * np.pi)  # Update angle
        else:  # Right
            next_state[10] = (next_state[10] - 0.2) % (2 * np.pi)  # Update angle
        return next_state
        
    def remember(self, state, action, reward, next_state, done):
        """Enhanced memory with corner penalties"""
        # Add corner penalty to reward if applicable
        if self._is_in_corner(state):
            reward += self.corner_penalty
        
        # Add movement bonus if exploring new area
        if not any(np.array_equal(next_state[8:10], pos) for pos in self.position_history):
            reward += self.movement_bonus
        
        # Store experience with modified reward
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
        
        # Store in priority memory
        self.priority_memory.append((state_key, action, reward, next_state_key, done, 
                                   (td_error + 1e-6) ** self.priority_alpha))
        
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