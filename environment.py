import pygame
import math
import numpy as np
import random

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (15, 15, 20)
DRONE_COLOR = (50, 200, 255)
GOAL_COLOR = (50, 255, 50)
OBSTACLE_COLOR = (100, 100, 110)
RAY_COLOR = (50, 100, 255, 100) # Semi-transparent
COLLISION_COLOR = (255, 50, 50)

class DroneSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("UAV RL Navigation Simulator - Phase 2")
        self.clock = pygame.time.Clock()
        
        # Sensor Configuration
        self.num_rays = 9
        self.ray_spread = math.pi # 180 degrees field of view
        self.max_ray_length = 150
        
        self.reset()
        
    def _generate_obstacles(self):
        """Generates random rectangular obstacles."""
        self.obstacles = []
        for _ in range(6):
            w = random.randint(50, 150)
            h = random.randint(50, 150)
            x = random.randint(100, WIDTH - 100 - w)
            y = random.randint(100, HEIGHT - 100 - h)
            self.obstacles.append(pygame.Rect(x, y, w, h))

    def reset(self):
        """Resets the environment for a new episode."""
        self.drone_x = 50.0
        self.drone_y = HEIGHT // 2
        self.angle = 0.0 
        self.speed = 0.0
        self.max_speed = 5.0
        
        # Fixed goal position for now
        self.goal_x = WIDTH - 50.0
        self.goal_y = HEIGHT // 2
        
        self._generate_obstacles()
        self.sensor_distances = [self.max_ray_length] * self.num_rays
        self.ray_hit_points = []
        
        return self._get_state()
        
    def step(self, action):
        """Takes an action and returns (state, reward, done)."""
        if action == 0:
            self.speed = min(self.speed + 0.5, self.max_speed)
        elif action == 1:
            self.angle -= 0.1
        elif action == 2:
            self.angle += 0.1
        elif action == 3:
            self.speed = max(self.speed - 0.2, 0)
            
        self.drone_x += self.speed * math.cos(self.angle)
        self.drone_y += self.speed * math.sin(self.angle)
        
        self._cast_rays()
        
        reward = -0.1 # Small penalty every step to encourage speed
        done = False
        
        # 1. Wall Collision Penalty
        if self.drone_x <= 0 or self.drone_x >= WIDTH or self.drone_y <= 0 or self.drone_y >= HEIGHT:
            reward = -100
            done = True
            
        # 2. Obstacle Collision Penalty
        # Create a tiny bounding box around the drone's center for collision checking
        drone_rect = pygame.Rect(self.drone_x - 8, self.drone_y - 8, 16, 16)
        for obs in self.obstacles:
            if drone_rect.colliderect(obs):
                reward = -100
                done = True
                
        # 3. Goal Reached Reward
        dist_to_goal = math.hypot(self.goal_x - self.drone_x, self.goal_y - self.drone_y)
        if dist_to_goal < 20: # If within 20 pixels of the goal center
            reward = 100
            done = True
            
        return self._get_state(), reward, done
        
    def _cast_rays(self):
        """Projects rays and calculates distance to nearest obstacles."""
        self.sensor_distances = []
        self.ray_hit_points = []
        
        start_angle = self.angle - (self.ray_spread / 2)
        angle_step = self.ray_spread / (self.num_rays - 1)
        
        for i in range(self.num_rays):
            current_angle = start_angle + (i * angle_step)
            end_x = self.drone_x + self.max_ray_length * math.cos(current_angle)
            end_y = self.drone_y + self.max_ray_length * math.sin(current_angle)
            
            closest_dist = self.max_ray_length
            closest_point = (end_x, end_y)
            
            # Check intersection with all obstacles
            for obs in self.obstacles:
                # clipline returns the overlapping line segment if it intersects the rect
                clip = obs.clipline((self.drone_x, self.drone_y), (end_x, end_y))
                if clip:
                    # clip[0] is the point where the ray enters the rectangle
                    hit_x, hit_y = clip[0]
                    dist = math.hypot(hit_x - self.drone_x, hit_y - self.drone_y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = (hit_x, hit_y)
            
            # Check intersection with screen borders
            border_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
            clip = border_rect.clipline((self.drone_x, self.drone_y), (end_x, end_y))
            if clip:
                hit_x, hit_y = clip[1] # clip[1] because clip[0] is the drone's center (inside rect)
                dist = math.hypot(hit_x - self.drone_x, hit_y - self.drone_y)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = (hit_x, hit_y)
            
            self.sensor_distances.append(closest_dist)
            self.ray_hit_points.append(closest_point)

    def _get_state(self):
        """Returns the current state vector for the RL agent."""
        # Normalize distances to a 0.0 - 1.0 range for the neural network
        normalized_sensors = [d / self.max_ray_length for d in self.sensor_distances]
        
        # Calculate distance and angle to goal
        dist_to_goal = math.hypot(self.goal_x - self.drone_x, self.goal_y - self.drone_y)
        angle_to_goal = math.atan2(self.goal_y - self.drone_y, self.goal_x - self.drone_x)
        relative_angle = angle_to_goal - self.angle
        
        # Keep relative angle between -pi and pi
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
        
        state = normalized_sensors + [dist_to_goal / WIDTH, relative_angle / math.pi]
        return np.array(state, dtype=np.float32)
        
    def render(self):
        self.screen.fill(BLACK)
        
        # Draw Goal
        pygame.draw.circle(self.screen, GOAL_COLOR, (int(self.goal_x), int(self.goal_y)), 15)
        
        # Draw Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, obs)
            
        # Draw Sensor Rays
        drone_pos = (int(self.drone_x), int(self.drone_y))
        for pt in self.ray_hit_points:
            # Draw the faint blue beam
            pygame.draw.line(self.screen, RAY_COLOR, drone_pos, (int(pt[0]), int(pt[1])), 1)
            # Draw the red dot at the collision point
            pygame.draw.circle(self.screen, COLLISION_COLOR, (int(pt[0]), int(pt[1])), 3)
        
        # Draw Drone
        pygame.draw.circle(self.screen, DRONE_COLOR, drone_pos, 10)
        
        # Draw Heading Line
        end_x = self.drone_x + 20 * math.cos(self.angle)
        end_y = self.drone_y + 20 * math.sin(self.angle)
        pygame.draw.line(self.screen, WHITE, drone_pos, (int(end_x), int(end_y)), 2)
        
        pygame.display.flip()
        self.clock.tick(FPS)

if __name__ == "__main__":
    env = DroneSim()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Press R to reset and generate new obstacles
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
        
        keys = pygame.key.get_pressed()
        action = 3 
        if keys[pygame.K_w]: action = 0
        if keys[pygame.K_a]: action = 1
        if keys[pygame.K_d]: action = 2
        
        env.step(action)
        env.render()
        
    pygame.quit()