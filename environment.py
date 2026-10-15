import pygame
import math
import numpy as np
import random

# Constants
WIDTH, HEIGHT = 650, 850
FPS = 60
WHITE = (255, 255, 255)
BLACK = (15, 15, 20)
DRONE_COLOR = (50, 200, 255)
GOAL_COLOR = (50, 255, 50)
OBSTACLE_COLOR = (100, 100, 110)
RAY_COLOR = (50, 100, 255, 100) # Semi-transparent
COLLISION_COLOR = (255, 50, 50)

class DroneSim:
    def __init__(self, render_mode=True, num_rays=16, ray_spread=2 * math.pi):
        self.render_mode = render_mode
        self.num_rays = num_rays
        self.ray_spread = ray_spread # 360 degrees field of view (2 * pi)
        self.max_ray_length = 150
        
        self.screen = None
        self.clock = None
        self.frame_count = 0 # Animation ticker
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("UAV RL Navigation Simulator - 360° Perception")
            self.clock = pygame.time.Clock()
        
        self.reset()
        
    def _generate_obstacles(self):
        """Generates random rectangular obstacles with guaranteed clearance at spawn and goal."""
        self.obstacles = []
        spawn_rect = pygame.Rect(20, (HEIGHT // 2) - 60, 110, 120)
        goal_rect = pygame.Rect(WIDTH - 130, (HEIGHT // 2) - 60, 110, 120)

        attempts = 0
        while len(self.obstacles) < 5 and attempts < 100:
            attempts += 1
            w = random.randint(60, 130)
            h = random.randint(60, 130)
            x = random.randint(110, WIDTH - 110 - w)
            y = random.randint(80, HEIGHT - 80 - h)
            obs = pygame.Rect(x, y, w, h)
            
            # Avoid placing obstacles directly inside spawn or goal clearance zones
            if obs.colliderect(spawn_rect) or obs.colliderect(goal_rect):
                continue
                
            self.obstacles.append(obs)

    def reset(self):
        """Resets the environment for a new episode."""
        self.drone_x = 50.0
        self.drone_y = HEIGHT // 2
        self.angle = 0.0 
        self.speed = 0.0
        self.max_speed = 5.0
        self.frame_count = 0
        
        # Fixed goal position for now
        self.goal_x = WIDTH - 50.0
        self.goal_y = HEIGHT // 2
        
        self._generate_obstacles()
        self.sensor_distances = [self.max_ray_length] * self.num_rays
        self.ray_hit_points = []
        self._cast_rays()
        
        return self._get_state()
        
    def step(self, action):
        """
        Takes an action and returns (state, reward, done).
        Supports both continuous SAC actions [throttle, steer] in [-1, 1]
        and legacy discrete integer actions (0, 1, 2, 3).
        """
        self.frame_count += 1
        
        # 1. Calculate distance before movement
        prev_dist = math.hypot(self.goal_x - self.drone_x, self.goal_y - self.drone_y)

        # Handle continuous vs discrete action
        if isinstance(action, (list, tuple, np.ndarray)):
            # Continuous action [throttle, steer] in [-1.0, 1.0]
            throttle = float(np.clip(action[0], -1.0, 1.0))
            steer = float(np.clip(action[1], -1.0, 1.0))

            # Continuous Throttle: positive accelerates, negative brakes
            if throttle > 0:
                self.speed = min(self.speed + throttle * 0.4, self.max_speed)
            else:
                self.speed = max(self.speed + throttle * 0.3, 0.0)

            # Continuous Steering: smooth angular turn rate
            self.angle += steer * 0.12
        else:
            # Legacy Discrete action execution (for keyboard testing)
            if action == 0:
                self.speed = min(self.speed + 0.5, self.max_speed)
            elif action == 1:
                self.angle -= 0.1
            elif action == 2:
                self.angle += 0.1
            elif action == 3:
                self.speed = max(self.speed - 0.2, 0.0)
            
        # Update drone position
        self.drone_x += self.speed * math.cos(self.angle)
        self.drone_y += self.speed * math.sin(self.angle)
        
        self._cast_rays()
        
        # 2. Distance after movement
        curr_dist = math.hypot(self.goal_x - self.drone_x, self.goal_y - self.drone_y)
        
        # Base Step Penalty (-0.05) + Reward for moving closer to the goal
        progress_reward = (prev_dist - curr_dist) * 0.15
        reward = progress_reward - 0.05
        done = False

        # 3. Obstacle Proximity / Danger Zone Penalty (Repulsive field)
        # Prevents the agent from hugging obstacles and forces safe margins
        min_sensor = min(self.sensor_distances) if self.sensor_distances else self.max_ray_length
        danger_threshold = 50.0  # Safe clearance distance in pixels
        if min_sensor < danger_threshold:
            # Quadratic penalty that increases sharply as distance approaches 0
            proximity_penalty = -2.0 * ((danger_threshold - min_sensor) / danger_threshold) ** 2
            reward += proximity_penalty
        
        # 4. Wall Collision Penalty (accounting for drone radius 10)
        drone_radius = 10
        if (self.drone_x <= drone_radius or self.drone_x >= WIDTH - drone_radius or 
            self.drone_y <= drone_radius or self.drone_y >= HEIGHT - drone_radius):
            reward = -100.0
            done = True
            
        # 5. Obstacle Collision Penalty (20x20 bounding box)
        drone_rect = pygame.Rect(self.drone_x - 10, self.drone_y - 10, 20, 20)
        for obs in self.obstacles:
            if drone_rect.colliderect(obs):
                reward = -100.0
                done = True
                break
                
        # 6. Goal Reached Reward (distance < radius_goal + radius_drone)
        if curr_dist < 25: 
            reward = 150.0
            done = True
            
        return self._get_state(), reward, done

        
    def _cast_rays(self):
        """Projects rays and calculates distance to nearest obstacles."""
        self.sensor_distances = []
        self.ray_hit_points = []
        
        # If full 360-degree FOV, divide evenly by num_rays so ray 0 and last ray don't duplicate
        if math.isclose(self.ray_spread, 2 * math.pi, rel_tol=1e-3):
            angle_step = self.ray_spread / self.num_rays
            start_angle = self.angle  # Ray 0 points directly in drone heading direction
        else:
            angle_step = self.ray_spread / (self.num_rays - 1)
            start_angle = self.angle - (self.ray_spread / 2)
        
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
        """
        Returns the current state vector for the RL agent.
        State features (dim = num_rays + 3):
          - normalized LiDAR distances: [d_0, ..., d_{num_rays-1}] in [0, 1]
          - normalized distance to goal: dist / WIDTH in [0, 1]
          - normalized relative heading to goal: angle / pi in [-1, 1]
          - normalized drone speed: speed / max_speed in [0, 1] (Markov momentum!)
        """
        # Normalize distances to a 0.0 - 1.0 range for the neural network
        normalized_sensors = [d / self.max_ray_length for d in self.sensor_distances]
        
        # Calculate distance and angle to goal
        dist_to_goal = math.hypot(self.goal_x - self.drone_x, self.goal_y - self.drone_y)
        angle_to_goal = math.atan2(self.goal_y - self.drone_y, self.goal_x - self.drone_x)
        relative_angle = angle_to_goal - self.angle
        
        # Keep relative angle between -pi and pi
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
        
        # Speed feature solves the POMDP (partially observable) momentum issue
        normalized_speed = self.speed / self.max_speed

        state = normalized_sensors + [dist_to_goal / WIDTH, relative_angle / math.pi, normalized_speed]
        return np.array(state, dtype=np.float32)

    # =========================================================================
    #  RENDERING — Enhanced Visuals
    # =========================================================================

    def _draw_goal(self):
        """Draws an animated pulsing beacon with concentric rings and rotating crosshair."""
        gx, gy = int(self.goal_x), int(self.goal_y)
        t = self.frame_count * 0.06  # Animation phase
        
        # Outer pulsing glow halo (expands and contracts)
        pulse = abs(math.sin(t)) * 0.5 + 0.5  # 0.5 to 1.0
        glow_radius = int(22 + pulse * 10)
        glow_alpha = int(30 + pulse * 45)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (50, 255, 120, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (gx - glow_radius, gy - glow_radius))

        # Concentric rings (3 rings, staggered pulse)
        for ring_i in range(3):
            ring_phase = t + ring_i * 0.8
            ring_pulse = abs(math.sin(ring_phase)) * 0.5 + 0.5
            r = int(8 + ring_i * 6 + ring_pulse * 3)
            alpha = int(180 - ring_i * 50)
            green_val = min(255, 180 + int(ring_pulse * 75))
            ring_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (40, green_val, 100, alpha), (r + 1, r + 1), r, 2)
            self.screen.blit(ring_surf, (gx - r - 1, gy - r - 1))

        # Solid bright core
        pygame.draw.circle(self.screen, (80, 255, 140), (gx, gy), 6)
        pygame.draw.circle(self.screen, (180, 255, 210), (gx, gy), 3)
        
        # Rotating crosshair arms
        cross_len = 18
        cross_angle = t * 0.7  # Slow spin
        for arm in range(4):
            a = cross_angle + arm * (math.pi / 2)
            ex = gx + int(cross_len * math.cos(a))
            ey = gy + int(cross_len * math.sin(a))
            ix = gx + int(7 * math.cos(a))
            iy = gy + int(7 * math.sin(a))
            pygame.draw.line(self.screen, (60, 230, 130), (ix, iy), (ex, ey), 1)
            # Small tick at the end of each arm
            pygame.draw.circle(self.screen, (120, 255, 160), (ex, ey), 2)

    def _draw_obstacle(self, obs):
        """Draws a single obstacle with 3D beveling, metallic surface, and hazard accents."""
        x, y, w, h = obs.x, obs.y, obs.width, obs.height
        
        # Shadow layer (offset dark rect behind)
        shadow_offset = 4
        pygame.draw.rect(self.screen, (8, 8, 12), 
                         pygame.Rect(x + shadow_offset, y + shadow_offset, w, h))
        
        # Main body — dark metallic fill
        pygame.draw.rect(self.screen, (65, 68, 78), obs)
        
        # Inner panel slightly lighter (inset by 3px) for a beveled "plate" look
        if w > 10 and h > 10:
            inner = pygame.Rect(x + 3, y + 3, w - 6, h - 6)
            pygame.draw.rect(self.screen, (80, 85, 95), inner)
            
            # Subtle horizontal scan lines across the plate for a metallic texture
            for sy in range(inner.top + 3, inner.bottom, 6):
                pygame.draw.line(self.screen, (72, 75, 86), (inner.left, sy), (inner.right, sy), 1)
        
        # Top bevel highlight (light edge on top and left)
        pygame.draw.line(self.screen, (120, 125, 140), (x, y), (x + w - 1, y), 2)          # Top
        pygame.draw.line(self.screen, (110, 115, 130), (x, y), (x, y + h - 1), 2)          # Left
        # Bottom/right shadow edge (darker)
        pygame.draw.line(self.screen, (35, 38, 45), (x, y + h - 1), (x + w - 1, y + h - 1), 2)  # Bottom
        pygame.draw.line(self.screen, (40, 42, 50), (x + w - 1, y), (x + w - 1, y + h - 1), 2)  # Right
        
        # Hazard stripe corners (diagonal yellow-black warning dashes)
        stripe_len = min(14, w // 3, h // 3)
        if stripe_len >= 5:
            # Top-left corner hazard stripes
            for s in range(0, stripe_len, 4):
                c = (210, 190, 40) if (s // 4) % 2 == 0 else (50, 50, 55)
                pygame.draw.line(self.screen, c, (x + s, y), (x, y + s), 2)
            # Bottom-right corner hazard stripes
            for s in range(0, stripe_len, 4):
                c = (210, 190, 40) if (s // 4) % 2 == 0 else (50, 50, 55)
                pygame.draw.line(self.screen, c, (x + w - s, y + h), (x + w, y + h - s), 2)
        
        # Small red "warning LED" dot on the top-right
        if w > 18 and h > 18:
            led_pulse = abs(math.sin(self.frame_count * 0.08 + x * 0.1))
            led_color = (int(180 + led_pulse * 75), int(30 + led_pulse * 20), 30)
            pygame.draw.circle(self.screen, led_color, (x + w - 8, y + 8), 3)
            # Tiny highlight on the LED
            pygame.draw.circle(self.screen, (255, 200, 200), (x + w - 9, y + 7), 1)

    def _draw_drone(self):
        """Draws a high-visibility Arctic White & Electric Neon quadcopter with illuminated motor pods and spinning blades."""
        drone = pygame.Surface((100, 100), pygame.SRCALPHA)

        # High-visibility color palette
        body_white = (245, 250, 255)
        body_rim = (190, 205, 225)
        visor_dark = (30, 36, 48)
        arm_alloy = (205, 215, 230)
        arm_highlight = (255, 255, 255)
        motor_housing = (32, 36, 44)
        motor_cap = (220, 230, 245)
        blade_neon = (0, 240, 255)       # Electric cyan (bright and distinct)
        camera_lens = (0, 255, 170)      # High-vis neon mint-green for heading perception

        # Four diagonal reinforced arms (bright alloy with pure white highlight)
        arms = [
            ((35, 35), (18, 18)),
            ((65, 35), (82, 18)),
            ((35, 65), (18, 82)),
            ((65, 65), (82, 82))
        ]

        for start, end in arms:
            pygame.draw.line(drone, arm_alloy, start, end, 5)
            pygame.draw.line(drone, arm_highlight, start, end, 2)

        # Four motor housings with LEDs and dynamic spinning neon blades
        motors = [
            (18, 18, (0, 240, 255)),    # Front-left: cyan LED
            (82, 18, (0, 240, 255)),    # Front-right: cyan LED
            (18, 82, (255, 80, 80)),    # Rear-left: red LED
            (82, 82, (255, 80, 80))     # Rear-right: red LED
        ]
        spin_speed = 0.4 + self.speed * 0.2
        blade_phase = self.frame_count * spin_speed

        for i, (mx, my, led_color) in enumerate(motors):
            # Motor pod base
            pygame.draw.circle(drone, motor_housing, (mx, my), 7)
            pygame.draw.circle(drone, arm_alloy, (mx, my), 7, 1)

            # Spinning high-vis neon blades
            angle_offset = blade_phase + i * (math.pi / 2)
            bx = 13 * math.cos(angle_offset)
            by = 13 * math.sin(angle_offset)

            # Vibrant propeller blade
            pygame.draw.line(
                drone, blade_neon,
                (int(mx - bx), int(my - by)),
                (int(mx + bx), int(my + by)), 3
            )
            # Propeller core highlight
            pygame.draw.circle(drone, motor_cap, (mx, my), 3)
            # Navigation LED beacon on each motor
            pygame.draw.circle(drone, led_color, (mx, my), 2)

        # Main drone fuselage body (Clean Arctic White)
        pygame.draw.ellipse(drone, body_white, (33, 33, 34, 34))
        pygame.draw.ellipse(drone, body_rim, (33, 33, 34, 34), 2)

        # Cockpit canopy / avionics bay
        pygame.draw.ellipse(drone, visor_dark, (37, 37, 26, 26))

        # Center status beacon
        pulse = abs(math.sin(self.frame_count * 0.12))
        core_cyan = (int(0 + pulse * 40), int(210 + pulse * 45), 255)
        pygame.draw.circle(drone, core_cyan, (50, 50), 4)
        pygame.draw.circle(drone, (255, 255, 255), (50, 50), 2)

        # Front-facing camera / perception sensor (facing UP in local surface coords)
        pygame.draw.circle(drone, body_white, (50, 37), 5)
        pygame.draw.circle(drone, camera_lens, (50, 37), 3)
        pygame.draw.circle(drone, (255, 255, 255), (50, 36), 1)

        # Heading rotation & scaling (scale increased to 0.48 for crisp visibility)
        rot_angle = -math.degrees(self.angle) - 90
        scale = 0.48

        rotated = pygame.transform.rotozoom(drone, rot_angle, scale)
        rect = rotated.get_rect(center=(int(self.drone_x), int(self.drone_y)))

        # Soft luminous halo around drone for pop against black background
        halo_radius = int(22 * scale * 2)
        halo_surf = pygame.Surface((halo_radius * 2, halo_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(halo_surf, (0, 200, 255, 35), (halo_radius, halo_radius), halo_radius)
        self.screen.blit(halo_surf, (int(self.drone_x - halo_radius), int(self.drone_y - halo_radius)))

        self.screen.blit(rotated, rect)

    def _draw_rays(self):
        """Draws the LiDAR sensor rays with hit point indicators."""
        drone_pos = (int(self.drone_x), int(self.drone_y))
        for i, pt in enumerate(self.ray_hit_points):
            hit_pt = (int(pt[0]), int(pt[1]))
            dist = self.sensor_distances[i] if i < len(self.sensor_distances) else self.max_ray_length
            proximity = 1.0 - (dist / self.max_ray_length)  # 0=far, 1=close
            
            if i == 0:
                # Forward-facing ray: brighter cyan
                beam_color = (100, 230, 255)
                beam_width = 2
            else:
                # Rays that detect close obstacles glow more orange/red
                r = int(50 + proximity * 180)
                g = int(100 - proximity * 60)
                b = int(255 - proximity * 180)
                beam_color = (r, g, b)
                beam_width = 1
            
            pygame.draw.line(self.screen, beam_color, drone_pos, hit_pt, beam_width)
            
            # Hit point indicator: red dot, bigger when object is closer
            dot_size = int(2 + proximity * 3)
            hit_r = min(255, int(150 + proximity * 105))
            pygame.draw.circle(self.screen, (hit_r, 50, 50), hit_pt, dot_size)

    def render(self):
        """Main render pipeline: background, obstacles, goal, rays, drone."""
        if self.screen is None:
            return
            
        self.screen.fill(BLACK)
        
        # 1. Goal beacon (drawn under everything so glow doesn't overlap drone)
        self._draw_goal()
        
        # 2. Obstacles with enhanced styling
        for obs in self.obstacles:
            self._draw_obstacle(obs)
            
        # 3. LiDAR sensor rays
        self._draw_rays()
        
        # 4. Quadcopter drone (drawn last, on top)
        self._draw_drone()

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
        action = None
        if keys[pygame.K_w]: action = 0
        elif keys[pygame.K_a]: action = 1
        elif keys[pygame.K_d]: action = 2
        elif keys[pygame.K_s]: action = 3
        
        if action is not None:
            env.step(action)
        else:
            env.step(3) # Coast/decelerate if no key is pressed
        env.render()
        pygame.display.flip()
        env.clock.tick(FPS)
        
    pygame.quit()