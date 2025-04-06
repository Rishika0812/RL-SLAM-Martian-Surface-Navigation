import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import cv2

class SimpleRoverEnv(gym.Env):
    """A simplified rover environment for SLAM simulation"""
    
    def __init__(self, map_size=20, render_scale=20, render_mode=None):
        super(SimpleRoverEnv, self).__init__()
        
        # Environment parameters
        self.map_size = map_size
        self.render_scale = render_scale
        self.max_steps = 200
        self.current_step = 0
        self.render_mode = render_mode
        
        # Action space: move forward, turn left, turn right
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 8 sensor readings + position (x,y) + orientation
        self.observation_space = spaces.Box(
            low=0, high=self.map_size, 
            shape=(11,), dtype=np.float32
        )
        
        # Initialize state
        self.rover_pos = np.array([map_size/2, map_size/2])
        self.rover_angle = 0  # Angle in radians
        self.obstacles = self._generate_obstacles(5)  # Generate 5 obstacles
        self.map = np.zeros((map_size, map_size))  # SLAM map
        self.visited_map = np.zeros((map_size, map_size))  # Track visited cells
        
        # For tracking rover path
        self.path_history = [self.rover_pos.copy()]
        
        # For better performance
        self.sensor_noise = 0.1  # Add small noise to sensor readings
        self.movement_noise = 0.05  # Add small noise to movements
        
    def _generate_obstacles(self, num_obstacles):
        """Generate random obstacles with better distribution"""
        obstacles = []
        min_distance = 4  # Minimum distance between obstacles
        
        for _ in range(num_obstacles):
            while True:
                x = np.random.uniform(2, self.map_size-2)
                y = np.random.uniform(2, self.map_size-2)
                size = np.random.uniform(0.5, 1.5)
                
                # Ensure obstacles aren't at the starting position
                if np.linalg.norm(np.array([x, y]) - self.rover_pos) > 4:
                    # Check distance from other obstacles to avoid overlap
                    valid_position = True
                    for (ox, oy, _) in obstacles:
                        if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < min_distance:
                            valid_position = False
                            break
                    
                    if valid_position:
                        obstacles.append((x, y, size))
                        break
                        
        return obstacles
    
    def _get_sensor_readings(self):
        """Get sensor readings in 8 directions for better coverage"""
        readings = np.ones(8) * self.map_size  # Initialize with max distance
        
        # Directions: 8 directions at 45-degree intervals
        angles = [self.rover_angle + i * np.pi/4 for i in range(8)]
        directions = [(np.cos(angle), np.sin(angle)) for angle in angles]
        
        # Maximum sensor range
        max_range = self.map_size
        
        # Check distance to obstacles in each direction
        for i, (dx, dy) in enumerate(directions):
            for step in np.arange(0.5, max_range, 0.5):  # More precise steps
                x = self.rover_pos[0] + step * dx
                y = self.rover_pos[1] + step * dy
                
                # Check if out of bounds
                if x < 0 or x >= self.map_size or y < 0 or y >= self.map_size:
                    readings[i] = step
                    break
                
                # Check if hit obstacle
                for (ox, oy, size) in self.obstacles:
                    if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < size:
                        # Add small noise to sensor readings for realism
                        noise = np.random.normal(0, self.sensor_noise)
                        readings[i] = step + noise
                        readings[i] = max(0.1, readings[i])  # Ensure positive readings
                        
                        # Update SLAM map with probability
                        map_x, map_y = int(x), int(y)
                        if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                            # More definitive obstacle marking (1.0 instead of previous fixed value)
                            self.map[map_x, map_y] = 1.0
                        break
        
        return readings
    
    def step(self, action):
        """Take a step in the environment with improved physics"""
        self.current_step += 1
        
        # Initialize collision variable for all action paths
        collision = False
        
        # Add small random noise to actions for more realism
        if np.random.random() < 0.1:  # 10% chance of action noise
            action = np.random.randint(0, 3)
        
        # Execute action
        if action == 0:  # Move forward
            move_distance = 0.5
            # Add small noise to movement
            noise_x = np.random.normal(0, self.movement_noise)
            noise_y = np.random.normal(0, self.movement_noise)
            
            new_x = self.rover_pos[0] + move_distance * np.cos(self.rover_angle) + noise_x
            new_y = self.rover_pos[1] + move_distance * np.sin(self.rover_angle) + noise_y
            
            # Check for collisions before moving with better collision detection
            collision = False
            for (ox, oy, size) in self.obstacles:
                # Use a slightly larger collision radius for safety
                safety_margin = 0.2
                if np.linalg.norm(np.array([new_x, new_y]) - np.array([ox, oy])) < (size + safety_margin):
                    collision = True
                    break
            
            # Only move if no collision and within bounds
            if not collision and 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                self.rover_pos = np.array([new_x, new_y])
                
        elif action == 1:  # Turn left
            # Smoother turning with slight randomness
            turn_amount = np.pi/6 + np.random.normal(0, 0.02)
            self.rover_angle -= turn_amount
            
        elif action == 2:  # Turn right
            # Smoother turning with slight randomness
            turn_amount = np.pi/6 + np.random.normal(0, 0.02)
            self.rover_angle += turn_amount
        
        # Normalize angle to [-π, π]
        self.rover_angle = ((self.rover_angle + np.pi) % (2 * np.pi)) - np.pi
        
        # Add current position to path history
        self.path_history.append(self.rover_pos.copy())
        
        # Update visited map at current position and nearby cells
        x, y = int(self.rover_pos[0]), int(self.rover_pos[1])
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    # Mark as visited with weight based on distance
                    dist = np.sqrt(dx**2 + dy**2)
                    weight = max(0, 1 - 0.3 * dist)
                    if self.visited_map[nx, ny] < weight:
                        self.visited_map[nx, ny] = weight
        
        # Get sensor readings
        readings = self._get_sensor_readings()
        
        # Create observation
        obs = np.concatenate([readings, self.rover_pos, [self.rover_angle]])
        
        # Improved reward calculation
        reward = 0
        
        # Base reward for surviving
        reward += 0.01
        
        # Reward for exploration - count newly visited cells
        new_cells_visited = np.sum(self.visited_map > 0) - np.sum(self.visited_map > 0.1)
        reward += 0.1 * new_cells_visited
        
        # Movement reward - encourage forward movement rather than spinning
        if action == 0 and not collision:  # Successfully moved forward
            reward += 0.05
        
        # Obstacle discovery reward
        obstacle_cells = np.sum(self.map > 0.9)
        reward += 0.01 * obstacle_cells
        
        # Penalty for staying still - encourages exploration
        if len(self.path_history) > 5:
            recent_positions = np.array(self.path_history[-5:])
            if np.max(np.std(recent_positions, axis=0)) < 0.2:  # Not moving much
                reward -= 0.1
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Check for collisions - should rarely happen due to improved collision detection
        for (ox, oy, size) in self.obstacles:
            if np.linalg.norm(self.rover_pos - np.array([ox, oy])) < size:
                reward = -2.0  # Stronger penalty for collisions
                done = True
                break
        
        # Additional info for debugging
        info = {
            "obstacle_count": len(self.obstacles),
            "explored_percentage": np.sum(self.visited_map > 0) / (self.map_size * self.map_size),
            "collision": collision
        }
        
        return obs, reward, done, info
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment with improved initialization"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.rover_pos = np.array([self.map_size/2, self.map_size/2])
        self.rover_angle = np.random.uniform(-np.pi, np.pi)  # Random initial orientation
        self.obstacles = self._generate_obstacles(5)
        self.map = np.zeros((self.map_size, self.map_size))
        self.visited_map = np.zeros((self.map_size, self.map_size))
        self.path_history = [self.rover_pos.copy()]
        
        # Mark initial position as visited
        x, y = int(self.rover_pos[0]), int(self.rover_pos[1])
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            self.visited_map[x, y] = 1.0
        
        # Get initial observation
        readings = self._get_sensor_readings()
        return np.concatenate([readings, self.rover_pos, [self.rover_angle]])
    
    def render_opencv(self):
        """Render the environment using OpenCV with improved visualization in dark mode"""
        # Create a dark background image
        img_size = self.map_size * self.render_scale
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 20  # Dark background
        
        # Draw grid lines
        for i in range(0, img_size + 1, self.render_scale):
            cv2.line(img, (0, i), (img_size, i), (50, 50, 50), 1)  # Darker grid lines
            cv2.line(img, (i, 0), (i, img_size), (50, 50, 50), 1)
        
        # Draw visited areas on the map with gradient
        for x in range(self.map_size):
            for y in range(self.map_size):
                if self.visited_map[x, y] > 0:
                    x_px = int(x * self.render_scale)
                    y_px = int(y * self.render_scale)
                    intensity = int(self.visited_map[x, y] * 150)
                    cv2.circle(img, (x_px, y_px), 3, (0, intensity, intensity), -1)  # Teal color for visited areas
        
        # Draw obstacles detected in SLAM map
        for x in range(self.map_size):
            for y in range(self.map_size):
                if self.map[x, y] > 0.8:  # Detected obstacle
                    x_px = int(x * self.render_scale)
                    y_px = int(y * self.render_scale)
                    cv2.circle(img, (x_px, y_px), 4, (0, 0, 200), -1)  # Red for detected obstacles
        
        # Draw actual obstacles with transparency
        for (x, y, size) in self.obstacles:
            x_px = int(x * self.render_scale)
            y_px = int(y * self.render_scale)
            size_px = int(size * self.render_scale)
            overlay = img.copy()
            cv2.circle(overlay, (x_px, y_px), size_px, (100, 100, 100), -1)  # Gray for actual obstacles
            # Apply transparency
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw path history with gradient color to show direction
        for i in range(1, len(self.path_history)):
            start = self.path_history[i-1]
            end = self.path_history[i]
            start_px = (int(start[0] * self.render_scale), int(start[1] * self.render_scale))
            end_px = (int(end[0] * self.render_scale), int(end[1] * self.render_scale))
            
            # Calculate color based on recency (newer paths are brighter)
            progress = i / len(self.path_history)
            green = int(50 + 150 * progress)
            cv2.line(img, start_px, end_px, (0, green, 50), 2)  # Brighter green for newer paths
        
        # Draw rover with clearer orientation indicator
        rover_x = int(self.rover_pos[0] * self.render_scale)
        rover_y = int(self.rover_pos[1] * self.render_scale)
        
        # Draw rover body
        cv2.circle(img, (rover_x, rover_y), int(0.5 * self.render_scale), (0, 50, 255), -1)  # Orange for rover
        
        # Draw rover direction with arrow
        direction_x = int(rover_x + np.cos(self.rover_angle) * self.render_scale)
        direction_y = int(rover_y + np.sin(self.rover_angle) * self.render_scale)
        cv2.arrowedLine(img, (rover_x, rover_y), (direction_x, direction_y), (180, 180, 255), 2, tipLength=0.3)  # Yellow for direction
        
        # Draw sensor lines
        angles = [self.rover_angle + i * np.pi/4 for i in range(8)]
        readings = self._get_sensor_readings()
        
        for i, angle in enumerate(angles):
            # Get endpoint based on sensor reading
            endpoint_x = int(rover_x + np.cos(angle) * readings[i] * self.render_scale)
            endpoint_y = int(rover_y + np.sin(angle) * readings[i] * self.render_scale)
            
            # Draw sensor line
            cv2.line(img, (rover_x, rover_y), (endpoint_x, endpoint_y), (100, 100, 200), 1, cv2.LINE_AA)  # Light purple for sensors
        
        # Add text for step count and metrics with better visibility on dark background
        cv2.putText(img, f"Step: {self.current_step}/{self.max_steps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)  # Light gray text
        
        explored = np.sum(self.visited_map > 0) / (self.map_size * self.map_size) * 100
        cv2.putText(img, f"Explored: {explored:.1f}%", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                   
        return img
    
    def render(self):
        """Render the environment"""
        mode = self.render_mode
        
        if mode == "opencv" or mode is None:
            return self.render_opencv()
        
        # Default matplotlib rendering with dark mode
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#121212')
        ax.set_facecolor('#121212')
        
        # Draw visited map
        visited_map_display = self.visited_map.copy().T
        ax.imshow(visited_map_display, origin='lower', cmap='viridis', 
                 extent=[0, self.map_size, 0, self.map_size], alpha=0.3)
        
        # Draw SLAM map (detected obstacles)
        slam_map_display = self.map.copy().T
        ax.imshow(slam_map_display, origin='lower', cmap='plasma', 
                 extent=[0, self.map_size, 0, self.map_size], alpha=0.5)
        
        # Draw obstacles
        for (x, y, size) in self.obstacles:
            circle = plt.Circle((x, y), size, color='gray', alpha=0.6)
            ax.add_patch(circle)
        
        # Draw rover
        circle = plt.Circle((self.rover_pos[0], self.rover_pos[1]), 0.5, color='orange')
        ax.add_patch(circle)
        
        # Draw rover direction
        ax.arrow(self.rover_pos[0], self.rover_pos[1], 
                np.cos(self.rover_angle), np.sin(self.rover_angle), 
                head_width=0.3, head_length=0.3, fc='yellow', ec='yellow')
        
        # Draw path
        path = np.array(self.path_history)
        ax.plot(path[:, 0], path[:, 1], color='#00ff99', linestyle='-', alpha=0.7)
        
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.grid(True, alpha=0.2, color='gray')
        ax.set_title(f'Step: {self.current_step}/{self.max_steps} - Explored: {np.sum(self.visited_map > 0)/(self.map_size*self.map_size)*100:.1f}%', color='white')
        
        # Set labels with light color
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.tick_params(colors='white')
        
        return fig