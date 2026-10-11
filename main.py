import pygame
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

def main():
    env = DroneSim()
    
    # 9 sensors + 2 goal parameters (dist, angle) = 11 state features
    state_size = env.num_rays + 2 
    action_size = 4 # thrust, left, right, coast
    
    agent = DQNAgent(state_size, action_size)
    
    epsilon = EPSILON_START
    font = pygame.font.SysFont("Arial", 18)
    
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        
        for step in range(MAX_STEPS):
            # 1. Handle Pygame events (allows quitting during training)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # 2. Select and execute action
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            
            # 3. Store experience in memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # 4. Train the network
            agent.train_step(BATCH_SIZE)
            
            state = next_state
            total_reward += reward
            
            # 5. Render the environment
            env.render()
            
            # Overlay Training Metrics on the screen
            metrics = [
                f"Episode: {episode}/{MAX_EPISODES}",
                f"Step: {step}/{MAX_STEPS}",
                f"Reward: {total_reward:.1f}",
                f"Epsilon: {epsilon:.3f}"
            ]
            
            for i, text in enumerate(metrics):
                surface = font.render(text, True, (255, 255, 255))
                env.screen.blit(surface, (10, 10 + (i * 25)))
                
            pygame.display.flip()
            
            if done:
                break
                
        # Decay epsilon after each episode
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        print(f"Episode {episode} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    main()