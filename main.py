import pygame
import os
import numpy as np
from environment import DroneSim
from dqn_agent import DQNAgent

# Hyperparameters
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
MODEL_PATH = "drone_memory.pth"

def draw_dashboard(env, font, episode, step, reward, epsilon):
    """Draws the UI metrics and the reward/penalty meter."""
    # Semi-transparent background for UI
    ui_surface = pygame.Surface((250, 160))
    ui_surface.set_alpha(200)
    ui_surface.fill((20, 20, 25))
    env.screen.blit(ui_surface, (10, 10))

    # Text Metrics
    metrics = [
        f"Episode: {episode} / {MAX_EPISODES}",
        f"Steps: {step} / {MAX_STEPS}",
        f"Epsilon (Randomness): {epsilon:.2f}",
        f"Current Reward: {reward:.1f}"
    ]
    
    for i, text in enumerate(metrics):
        surface = font.render(text, True, (255, 255, 255))
        env.screen.blit(surface, (20, 20 + (i * 25)))

    # Reward/Penalty Meter Visuals
    meter_y = 135
    center_x = 130
    pygame.draw.line(env.screen, (100, 100, 100), (30, meter_y), (230, meter_y), 4) # Base line
    pygame.draw.line(env.screen, (255, 255, 255), (center_x, meter_y - 8), (center_x, meter_y + 8), 2) # Center zero marker
    
    # Calculate bar size (scale reward down visually to fit the 100px width limit per side)
    bar_length = min(abs(reward), 100) 
    
    if reward > 0:
        # Green bar pointing right
        pygame.draw.rect(env.screen, (50, 255, 50), (center_x, meter_y - 4, bar_length, 8))
    elif reward < 0:
        # Red bar pointing left
        pygame.draw.rect(env.screen, (255, 50, 50), (center_x - bar_length, meter_y - 4, bar_length, 8))

def main():
    env = DroneSim()
    
    state_size = env.num_rays + 2 
    action_size = 4 
    
    agent = DQNAgent(state_size, action_size)
    
    # Induce Memory: Check if a trained model already exists
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print("Successfully loaded previous drone memory!")
        # Since it already knows how to fly, we reduce the random exploration rate significantly
        epsilon = 0.20 
    else:
        print("No previous memory found. Starting from scratch.")
        epsilon = EPSILON_START
        
    font = pygame.font.SysFont("Consolas", 16)
    
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        
        for step in range(MAX_STEPS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Save before exiting manually
                    agent.save(MODEL_PATH)
                    pygame.quit()
                    return
            
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step(BATCH_SIZE)
            
            state = next_state
            total_reward += reward
            
            env.render()
            draw_dashboard(env, font, episode, step, total_reward, epsilon)
            pygame.display.flip()
            
            if done:
                break
                
        # Save memory automatically every 10 episodes
        if episode % 10 == 0:
            agent.save(MODEL_PATH)
            
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        print(f"Episode {episode} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    main()