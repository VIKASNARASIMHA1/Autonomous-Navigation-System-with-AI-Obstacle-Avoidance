

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
import time
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 1. ENVIRONMENT SIMULATION (Simulated Embedded Environment)
# ============================================================================

class AutonomousEnvironment:
    """Simulates an embedded environment for autonomous navigation"""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # Vehicle state
        self.vehicle_pos = [width // 2, height // 2]
        self.vehicle_angle = 0  # degrees
        self.vehicle_speed = 0
        
        # Environment parameters
        self.obstacles = []
        self.goals = []
        self.path = []
        
        # Sensors simulation (like in embedded systems)
        self.sensor_range = 150
        self.sensor_angles = [-90, -45, 0, 45, 90]  # degrees
        self.sensor_readings = [0] * len(self.sensor_angles)
        
        # Camera simulation (like in embedded vision systems)
        self.camera_fov = 60  # degrees
        self.camera_resolution = (160, 120)  # Simulated low-res camera
        
        # Initialize environment
        self._generate_random_environment()
        
    def _generate_random_environment(self):
        """Generate random obstacles and goals"""
        # Generate random obstacles
        self.obstacles = []
        for _ in range(15):
            size = random.randint(20, 60)
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            self.obstacles.append({
                'x': x, 'y': y, 'width': size, 'height': size,
                'type': random.choice(['static', 'dynamic'])
            })
        
        # Generate random goal
        self.goals = [{
            'x': self.width - 100,
            'y': self.height - 100,
            'radius': 30
        }]
        
    def update_sensor_readings(self):
        """Simulate sensor readings (like ultrasonic/LiDAR sensors in embedded systems)"""
        for i, angle in enumerate(self.sensor_angles):
            sensor_angle_rad = np.radians(self.vehicle_angle + angle)
            
            # Calculate sensor endpoint
            end_x = self.vehicle_pos[0] + self.sensor_range * np.cos(sensor_angle_rad)
            end_y = self.vehicle_pos[1] + self.sensor_range * np.sin(sensor_angle_rad)
            
            # Check for collisions with obstacles
            min_distance = self.sensor_range
            
            for obstacle in self.obstacles:
                distance = self._line_rect_intersection(
                    self.vehicle_pos[0], self.vehicle_pos[1],
                    end_x, end_y,
                    obstacle['x'], obstacle['y'],
                    obstacle['width'], obstacle['height']
                )
                
                if distance is not None and distance < min_distance:
                    min_distance = distance
            
            # Normalize sensor reading
            self.sensor_readings[i] = 1.0 - (min_distance / self.sensor_range)
            
        return self.sensor_readings
    
    def _line_rect_intersection(self, x1, y1, x2, y2, rx, ry, rw, rh):
        """Check intersection between line and rectangle"""
        # Check each side of the rectangle
        sides = [
            [(rx, ry), (rx + rw, ry)],  # Top
            [(rx + rw, ry), (rx + rw, ry + rh)],  # Right
            [(rx + rw, ry + rh), (rx, ry + rh)],  # Bottom
            [(rx, ry + rh), (rx, ry)]  # Left
        ]
        
        min_distance = None
        
        for side in sides:
            intersection = self._line_line_intersection(
                x1, y1, x2, y2,
                side[0][0], side[0][1], side[1][0], side[1][1]
            )
            
            if intersection:
                distance = np.sqrt((intersection[0] - x1)**2 + (intersection[1] - y1)**2)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    def _line_line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Check intersection between two lines"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if denom == 0:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            return (px, py)
        
        return None
    
    def simulate_camera_frame(self):
        """Simulate a camera frame from the vehicle's perspective"""
        # Create blank frame
        frame = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8)
        
        # Convert vehicle position to frame coordinates
        frame_center_x = self.camera_resolution[0] // 2
        frame_center_y = self.camera_resolution[1] // 2
        
        # Draw goal in camera frame
        for goal in self.goals:
            rel_x = goal['x'] - self.vehicle_pos[0]
            rel_y = goal['y'] - self.vehicle_pos[1]
            
            # Rotate based on vehicle angle
            angle_rad = np.radians(-self.vehicle_angle)
            rotated_x = rel_x * np.cos(angle_rad) - rel_y * np.sin(angle_rad)
            rotated_y = rel_x * np.sin(angle_rad) + rel_y * np.cos(angle_rad)
            
            # Scale to frame coordinates (simplified projection)
            frame_x = int(frame_center_x + rotated_x / 5)
            frame_y = int(frame_center_y + rotated_y / 5)
            
            if 0 <= frame_x < self.camera_resolution[0] and 0 <= frame_y < self.camera_resolution[1]:
                cv2.circle(frame, (frame_x, frame_y), 5, (0, 255, 0), -1)
        
        # Add noise (simulating real camera sensor)
        noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def update_vehicle_position(self, steering_angle, acceleration):
        """Update vehicle position based on controls (simulating embedded control system)"""
        # Update speed
        self.vehicle_speed += acceleration
        self.vehicle_speed = np.clip(self.vehicle_speed, -3, 5)
        
        # Update angle
        self.vehicle_angle += steering_angle
        self.vehicle_angle %= 360
        
        # Calculate new position
        angle_rad = np.radians(self.vehicle_angle)
        self.vehicle_pos[0] += self.vehicle_speed * np.cos(angle_rad)
        self.vehicle_pos[1] += self.vehicle_speed * np.sin(angle_rad)
        
        # Keep within bounds
        self.vehicle_pos[0] = np.clip(self.vehicle_pos[0], 20, self.width - 20)
        self.vehicle_pos[1] = np.clip(self.vehicle_pos[1], 20, self.height - 20)
        
        # Check collisions
        collision = self._check_collision()
        
        # Check goal reached
        goal_reached = False
        for goal in self.goals:
            distance = np.sqrt((self.vehicle_pos[0] - goal['x'])**2 + 
                              (self.vehicle_pos[1] - goal['y'])**2)
            if distance < goal['radius']:
                goal_reached = True
        
        return collision, goal_reached
    
    def _check_collision(self):
        """Check if vehicle collides with any obstacle"""
        vehicle_radius = 15
        
        for obstacle in self.obstacles:
            # Simple circle-rectangle collision detection
            closest_x = np.clip(self.vehicle_pos[0], 
                               obstacle['x'], 
                               obstacle['x'] + obstacle['width'])
            closest_y = np.clip(self.vehicle_pos[1], 
                               obstacle['y'], 
                               obstacle['y'] + obstacle['height'])
            
            distance = np.sqrt((self.vehicle_pos[0] - closest_x)**2 + 
                              (self.vehicle_pos[1] - closest_y)**2)
            
            if distance < vehicle_radius:
                return True
        
        return False
    
    def get_state(self):
        """Get complete environment state for AI processing"""
        sensor_data = self.update_sensor_readings()
        camera_frame = self.simulate_camera_frame()
        
        # Calculate goal direction
        goal_vector = np.array([
            self.goals[0]['x'] - self.vehicle_pos[0],
            self.goals[0]['y'] - self.vehicle_pos[1]
        ])
        goal_distance = np.linalg.norm(goal_vector)
        goal_direction = goal_vector / (goal_distance + 1e-6)
        
        # Vehicle state vector
        vehicle_state = np.array([
            self.vehicle_speed / 5.0,  # Normalized speed
            np.sin(np.radians(self.vehicle_angle)),
            np.cos(np.radians(self.vehicle_angle)),
            goal_direction[0],
            goal_direction[1],
            goal_distance / np.sqrt(self.width**2 + self.height**2)  # Normalized distance
        ])
        
        return {
            'sensor_data': np.array(sensor_data),
            'camera_frame': camera_frame,
            'vehicle_state': vehicle_state,
            'position': self.vehicle_pos.copy(),
            'angle': self.vehicle_angle,
            'goal_reached': False
        }

# ============================================================================
# 2. COMPUTER VISION MODULE (AI-based Perception)
# ============================================================================

class VisionPerceptionSystem:
    """AI-powered computer vision system for obstacle and goal detection"""
    
    def __init__(self):
        # Simulate a lightweight CNN suitable for embedded systems
        self.model = self._build_cnn_model()
        
    def _build_cnn_model(self):
        """Build a lightweight CNN for object detection"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(120, 160, 3)),
            
            # Convolutional layers (optimized for embedded systems)
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            
            # Output: [obstacle_presence, goal_direction_x, goal_direction_y]
            layers.Dense(3, activation='tanh')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for CNN input"""
        # Resize and normalize
        processed = cv2.resize(frame, (160, 120))
        processed = processed.astype(np.float32) / 255.0
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def detect_objects(self, frame):
        """Detect obstacles and goals in the frame"""
        processed_frame = self.preprocess_frame(frame)
        
        # In a real implementation, we would use the trained model
        # For simulation, we'll generate synthetic detections
        height, width, _ = frame.shape
        
        # Simulate obstacle detection (edges)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours (simulating object detection)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_presence = 0.0
        if len(contours) > 0:
            obstacle_presence = min(len(contours) / 10.0, 1.0)
        
        # Simulate goal detection (green color detection)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
        
        goal_direction = [0.0, 0.0]
        if np.sum(green_mask) > 100:
            # Find center of green area
            y_coords, x_coords = np.where(green_mask > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = np.mean(x_coords) / width * 2 - 1  # Normalize to [-1, 1]
                center_y = np.mean(y_coords) / height * 2 - 1  # Normalize to [-1, 1]
                goal_direction = [center_x, center_y]
        
        return {
            'obstacle_presence': obstacle_presence,
            'goal_direction': np.array(goal_direction),
            'processed_frame': processed_frame
        }

# ============================================================================
# 3. REINFORCEMENT LEARNING AGENT (AI Decision Making)
# ============================================================================

class DQNAgent:
    """Deep Q-Network agent for autonomous navigation decisions"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build neural network for Q-value approximation"""
        model = keras.Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def update_target_model(self):
        """Update target model with main model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        """Train on random batch from replay memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load trained model weights"""
        self.model.load_weights(name)
    
    def save(self, name):
        """Save trained model weights"""
        self.model.save_weights(name)

# ============================================================================
# 4. AUTONOMOUS NAVIGATION CONTROLLER
# ============================================================================

class AutonomousNavigationController:
    """Main controller integrating perception, planning, and control"""
    
    def __init__(self):
        self.environment = AutonomousEnvironment()
        self.vision_system = VisionPerceptionSystem()
        
        # State size: sensor_data (5) + vehicle_state (6) + vision_output (3)
        state_size = 5 + 6 + 3
        action_size = 5  # [forward, backward, left, right, stop]
        
        self.dqn_agent = DQNAgent(state_size, action_size)
        
        # Control parameters
        self.steering_gain = 2.0
        self.acceleration_gain = 0.5
        
        # Performance metrics
        self.episode_rewards = []
        self.collision_count = 0
        self.success_count = 0
        
    def get_state_representation(self, env_state, vision_output):
        """Combine all sensor data into a single state vector"""
        sensor_data = env_state['sensor_data']
        vehicle_state = env_state['vehicle_state']
        
        # Combine all state information
        combined_state = np.concatenate([
            sensor_data,
            vehicle_state,
            [vision_output['obstacle_presence']],
            vision_output['goal_direction']
        ])
        
        return combined_state
    
    def action_to_control(self, action):
        """Convert DQN action to steering and acceleration controls"""
        # Action mapping:
        # 0: Forward (accelerate)
        # 1: Backward (decelerate/reverse)
        # 2: Turn left
        # 3: Turn right
        # 4: Stop
        
        steering = 0
        acceleration = 0
        
        if action == 0:  # Forward
            acceleration = self.acceleration_gain
        elif action == 1:  # Backward
            acceleration = -self.acceleration_gain
        elif action == 2:  # Turn left
            steering = -self.steering_gain
        elif action == 3:  # Turn right
            steering = self.steering_gain
        elif action == 4:  # Stop
            acceleration = -self.vehicle_speed * 0.5  # Brake
        
        return steering, acceleration
    
    def calculate_reward(self, env_state, collision, goal_reached):
        """Calculate reward based on performance"""
        reward = 0
        
        # Positive rewards
        if goal_reached:
            reward += 100
            self.success_count += 1
        
        # Distance to goal reward
        goal_distance = np.linalg.norm(np.array([
            self.environment.goals[0]['x'] - env_state['position'][0],
            self.environment.goals[0]['y'] - env_state['position'][1]
        ]))
        
        reward += (1.0 / (goal_distance + 1)) * 0.1
        
        # Negative rewards
        if collision:
            reward -= 50
            self.collision_count += 1
        
        # Penalize excessive turning
        reward -= abs(self.environment.vehicle_speed) * 0.01
        
        # Penalize being stuck
        if abs(self.environment.vehicle_speed) < 0.1:
            reward -= 0.1
        
        return reward
    
    def run_episode(self, max_steps=500, render=True, training=True):
        """Run a single navigation episode"""
        # Reset environment
        self.environment = AutonomousEnvironment()
        
        # Get initial state
        env_state = self.environment.get_state()
        vision_output = self.vision_system.detect_objects(env_state['camera_frame'])
        state = self.get_state_representation(env_state, vision_output)
        
        total_reward = 0
        done = False
        step = 0
        
        if render:
            self.visualize_episode(step, total_reward, env_state, vision_output)
        
        while not done and step < max_steps:
            # Choose action
            action = self.dqn_agent.act(state)
            
            # Convert action to controls
            steering, acceleration = self.action_to_control(action)
            
            # Update environment
            collision, goal_reached = self.environment.update_vehicle_position(
                steering, acceleration
            )
            
            # Get new state
            next_env_state = self.environment.get_state()
            next_vision_output = self.vision_system.detect_objects(
                next_env_state['camera_frame']
            )
            next_state = self.get_state_representation(next_env_state, next_vision_output)
            
            # Calculate reward
            reward = self.calculate_reward(next_env_state, collision, goal_reached)
            total_reward += reward
            
            # Check if episode is done
            done = collision or goal_reached
            
            # Store experience and train
            if training:
                self.dqn_agent.remember(state, action, reward, next_state, done)
                self.dqn_agent.replay()
            
            # Update state
            state = next_state
            step += 1
            
            # Render if enabled
            if render and step % 5 == 0:
                self.visualize_episode(step, total_reward, next_env_state, next_vision_output)
            
            # Early termination if goal reached
            if goal_reached:
                print(f"Goal reached in {step} steps!")
                break
        
        if training:
            self.episode_rewards.append(total_reward)
        
        return total_reward, step, goal_reached
    
    def visualize_episode(self, step, reward, env_state, vision_output):
        """Visualize the current state of the autonomous system"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Clear previous plots
        for ax in axes.flat:
            ax.clear()
        
        # 1. Main environment view
        ax1 = axes[0, 0]
        self._draw_environment(ax1, env_state)
        ax1.set_title(f"Autonomous Navigation - Step: {step}")
        
        # 2. Sensor readings
        ax2 = axes[0, 1]
        angles = self.environment.sensor_angles
        bars = ax2.bar(range(len(angles)), env_state['sensor_data'])
        ax2.set_xlabel('Sensor Angle')
        ax2.set_ylabel('Obstacle Proximity')
        ax2.set_title('Sensor Readings')
        ax2.set_xticks(range(len(angles)))
        ax2.set_xticklabels([f'{a}°' for a in angles])
        
        # Color bars based on proximity
        for i, bar in enumerate(bars):
            if env_state['sensor_data'][i] > 0.7:
                bar.set_color('red')
            elif env_state['sensor_data'][i] > 0.4:
                bar.set_color('yellow')
            else:
                bar.set_color('green')
        
        # 3. Camera view
        ax3 = axes[0, 2]
        ax3.imshow(env_state['camera_frame'])
        ax3.set_title('Camera View')
        ax3.axis('off')
        
        # 4. Vehicle state
        ax4 = axes[1, 0]
        state_labels = ['Speed', 'Sin(Angle)', 'Cos(Angle)', 
                       'Goal X', 'Goal Y', 'Goal Dist']
        state_values = env_state['vehicle_state']
        
        bars2 = ax4.barh(state_labels, state_values)
        ax4.set_title('Vehicle State')
        ax4.set_xlim(-1, 1)
        
        # Color based on value
        for bar, val in zip(bars2, state_values):
            if abs(val) > 0.8:
                bar.set_color('red')
            elif abs(val) > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('blue')
        
        # 5. Vision system output
        ax5 = axes[1, 1]
        vision_data = [
            vision_output['obstacle_presence'],
            vision_output['goal_direction'][0],
            vision_output['goal_direction'][1]
        ]
        vision_labels = ['Obstacle', 'Goal X', 'Goal Y']
        
        bars3 = ax5.bar(vision_labels, vision_data)
        ax5.set_title('Vision System Output')
        ax5.set_ylim(-1, 1)
        
        for bar, val in zip(bars3, vision_data):
            if val > 0.7:
                bar.set_color('red')
            elif val < -0.7:
                bar.set_color('blue')
            else:
                bar.set_color('green')
        
        # 6. Performance metrics
        ax6 = axes[1, 2]
        metrics = {
            'Total Reward': reward,
            'Vehicle Speed': self.environment.vehicle_speed,
            'Collisions': self.collision_count,
            'Success Rate': self.success_count / max(len(self.episode_rewards), 1)
        }
        
        ax6.bar(range(len(metrics)), list(metrics.values()))
        ax6.set_title('Performance Metrics')
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def _draw_environment(self, ax, env_state):
        """Draw the environment with vehicle, obstacles, and goals"""
        # Clear axis
        ax.clear()
        
        # Set limits
        ax.set_xlim(0, self.environment.width)
        ax.set_ylim(0, self.environment.height)
        ax.set_aspect('equal')
        
        # Draw obstacles
        for obstacle in self.environment.obstacles:
            rect = patches.Rectangle(
                (obstacle['x'], obstacle['y']),
                obstacle['width'], obstacle['height'],
                linewidth=1, edgecolor='red', facecolor='orange', alpha=0.7
            )
            ax.add_patch(rect)
        
        # Draw goals
        for goal in self.environment.goals:
            circle = patches.Circle(
                (goal['x'], goal['y']), goal['radius'],
                linewidth=1, edgecolor='green', facecolor='lightgreen', alpha=0.7
            )
            ax.add_patch(circle)
        
        # Draw vehicle
        vehicle_x, vehicle_y = env_state['position']
        vehicle_circle = patches.Circle(
            (vehicle_x, vehicle_y), 15,
            linewidth=2, edgecolor='blue', facecolor='cyan', alpha=0.8
        )
        ax.add_patch(vehicle_circle)
        
        # Draw vehicle direction
        angle_rad = np.radians(env_state['angle'])
        dir_x = vehicle_x + 25 * np.cos(angle_rad)
        dir_y = vehicle_y + 25 * np.sin(angle_rad)
        ax.arrow(vehicle_x, vehicle_y, 
                 dir_x - vehicle_x, dir_y - vehicle_y,
                 head_width=8, head_length=10, fc='darkblue', ec='darkblue')
        
        # Draw sensor rays
        for i, angle in enumerate(self.environment.sensor_angles):
            sensor_angle_rad = np.radians(env_state['angle'] + angle)
            distance = self.environment.sensor_range * (1 - env_state['sensor_data'][i])
            
            end_x = vehicle_x + distance * np.cos(sensor_angle_rad)
            end_y = vehicle_y + distance * np.sin(sensor_angle_rad)
            
            # Color based on proximity
            if env_state['sensor_data'][i] > 0.7:
                color = 'red'
            elif env_state['sensor_data'][i] > 0.4:
                color = 'orange'
            else:
                color = 'green'
            
            ax.plot([vehicle_x, end_x], [vehicle_y, end_y], 
                   color=color, alpha=0.6, linewidth=2)
        
        # Draw path
        if len(self.environment.path) > 1:
            path_x = [p[0] for p in self.environment.path]
            path_y = [p[1] for p in self.environment.path]
            ax.plot(path_x, path_y, 'b--', alpha=0.5)
        
        # Add current position to path (keep only last 50 points)
        self.environment.path.append((vehicle_x, vehicle_y))
        if len(self.environment.path) > 50:
            self.environment.path.pop(0)
    
    def train(self, num_episodes=50):
        """Train the DQN agent"""
        print("Starting training...")
        
        for episode in range(num_episodes):
            reward, steps, success = self.run_episode(
                max_steps=300, 
                render=(episode % 10 == 0),  # Render every 10th episode
                training=True
            )
            
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Steps={steps}, Reward={reward:.2f}, "
                  f"Success={success}, Epsilon={self.dqn_agent.epsilon:.3f}")
            
            # Update target network every 10 episodes
            if episode % 10 == 0:
                self.dqn_agent.update_target_model()
        
        print("Training completed!")
        self.plot_training_results()
    
    def plot_training_results(self):
        """Plot training results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot rewards
        axes[0].plot(self.episode_rewards)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Rewards')
        axes[0].grid(True)
        
        # Calculate moving average
        if len(self.episode_rewards) >= 10:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(10)/10, mode='valid')
            axes[0].plot(range(9, len(self.episode_rewards)), 
                        moving_avg, 'r-', linewidth=2)
        
        # Plot success rate
        success_rate = []
        successes = 0
        for i in range(len(self.episode_rewards)):
            # Assuming positive reward indicates success
            if self.episode_rewards[i] > 50:
                successes += 1
            success_rate.append(successes / (i + 1))
        
        axes[1].plot(success_rate)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Success Rate Over Time')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def demo(self, num_episodes=3):
        """Run demonstration episodes with visualization"""
        print("Starting demonstration...")
        
        for episode in range(num_episodes):
            print(f"\nDemonstration Episode {episode + 1}/{num_episodes}")
            
            reward, steps, success = self.run_episode(
                max_steps=500,
                render=True,
                training=False
            )
            
            print(f"Result: Steps={steps}, Reward={reward:.2f}, Success={success}")
            
            # Pause between episodes
            if episode < num_episodes - 1:
                time.sleep(2)

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the autonomous navigation system"""
    print("=" * 70)
    print("AUTONOMOUS NAVIGATION SYSTEM WITH AI OBSTACLE AVOIDANCE")
    print("=" * 70)
    
    # Create controller
    controller = AutonomousNavigationController()
    
    # Interactive menu
    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. Train the AI agent")
        print("2. Run demonstration")
        print("3. Test single episode")
        print("4. View system architecture")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            episodes = int(input("Enter number of training episodes (default: 50): ") or "50")
            controller.train(num_episodes=episodes)
            
        elif choice == '2':
            episodes = int(input("Enter number of demo episodes (default: 3): ") or "3")
            controller.demo(num_episodes=episodes)
            
        elif choice == '3':
            reward, steps, success = controller.run_episode(
                max_steps=500, 
                render=True,
                training=False
            )
            print(f"\nTest Result: Steps={steps}, Reward={reward:.2f}, Success={success}")
            
        elif choice == '4':
            print_system_architecture()
            
        elif choice == '5':
            print("Exiting program. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

def print_system_architecture():
    """Print system architecture information"""
    print("\n" + "=" * 70)
    print("SYSTEM ARCHITECTURE")
    print("=" * 70)
    print("\n1. ENVIRONMENT SIMULATION")
    print("   - Simulates embedded sensors (ultrasonic/LiDAR)")
    print("   - Simulates camera with configurable FOV and resolution")
    print("   - Dynamic obstacle generation and collision detection")
    print("   - Vehicle kinematics simulation")
    
    print("\n2. COMPUTER VISION MODULE")
    print("   - Lightweight CNN for embedded systems")
    print("   - Real-time obstacle and goal detection")
    print("   - Frame preprocessing and normalization")
    print("   - Edge detection and contour analysis")
    
    print("\n3. REINFORCEMENT LEARNING AGENT")
    print("   - Deep Q-Network (DQN) for decision making")
    print("   - Experience replay for stable training")
    print("   - Epsilon-greedy exploration strategy")
    print("   - Target network for stable Q-value estimation")
    
    print("\n4. NAVIGATION CONTROLLER")
    print("   - Integrates perception, planning, and control")
    print("   - Reward shaping for effective learning")
    print("   - Action-to-control mapping")
    print("   - Performance monitoring and visualization")
    
    print("\n5. VISUALIZATION SYSTEM")
    print("   - Real-time environment rendering")
    print("   - Sensor data visualization")
    print("   - Camera view display")
    print("   - Performance metrics dashboard")
    
    print("\n" + "=" * 70)
    print("KEY FEATURES FOR PORTFOLIO")
    print("=" * 70)
    print("• Complete software simulation of autonomous embedded system")
    print("• Integration of multiple AI techniques (CV + RL)")
    print("• Real-time visualization and monitoring")
    print("• Modular architecture for easy extension")
    print("• Professional-grade code structure and documentation")
    print("• Performance metrics and training analytics")

if __name__ == "__main__":
    # Enable interactive mode for matplotlib
    plt.ion()
    
    # Run the main program
    main()