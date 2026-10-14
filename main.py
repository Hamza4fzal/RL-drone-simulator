import pygame
import os
import sys
import time
import argparse
import numpy as np
from collections import deque
from environment import DroneSim
from dqn_agent import DQNAgent

# Hyperparameters
MAX_STEPS = 500
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_MIN = 0.01  # Minimum exploration rate (1% random, 99% learned policy)
EPSILON_DECAY = 0.995
MODEL_PATH = "drone_memory.pth"

# Training Modes
MODE_NORMAL = "NORMAL"          # 60 FPS cap, standard real-time rendering
MODE_TURBO = "TURBO"            # Uncapped FPS, render every frame (~10x-15x speedup)
MODE_ULTRA = "ULTRA TURBO"      # Uncapped FPS, render 1 in every N frames (~25x-50x speedup)
MODE_HEADLESS = "HEADLESS"      # Windowed headless, maximum compute speed (~50x-100x speedup)

MODE_CYCLE = [MODE_NORMAL, MODE_TURBO, MODE_ULTRA]

def draw_dashboard(env, font, small_font, episode, step, reward, epsilon, mode, steps_per_sec, win_rate):
    """Draws the UI metrics, speed indicator, and reward/penalty meter."""
    if env.screen is None:
        return

    # Semi-transparent background for UI
    ui_width = 270
    ui_height = 215
    ui_surface = pygame.Surface((ui_width, ui_height))
    ui_surface.set_alpha(215)
    ui_surface.fill((18, 20, 28))
    env.screen.blit(ui_surface, (10, 10))

    # Mode Badge Color
    mode_colors = {
        MODE_NORMAL: (80, 200, 255),
        MODE_TURBO: (255, 200, 50),
        MODE_ULTRA: (255, 120, 50),
        MODE_HEADLESS: (255, 80, 180)
    }
    mode_color = mode_colors.get(mode, (255, 255, 255))
    
    speed_mult = steps_per_sec / 60.0 if steps_per_sec > 0 else 1.0
    
    metrics = [
        (f"MODE: {mode}", mode_color),
        (f"Speed: {steps_per_sec:.0f} s/s ({speed_mult:.1f}x)", (150, 255, 150)),
        (f"Episode: {episode} (Continuous)", (240, 240, 240)),
        (f"Steps: {step} / {MAX_STEPS}", (240, 240, 240)),
        (f"Epsilon: {epsilon:.3f}", (240, 240, 240)),
        (f"Reward: {reward:.1f}", (240, 240, 240)),
        (f"Win Rate: {win_rate:.0f}% (Recent)", (200, 230, 255))
    ]
    
    for i, (text, color) in enumerate(metrics):
        surface = font.render(text, True, color)
        env.screen.blit(surface, (20, 18 + (i * 20)))

    # Reward/Penalty Meter Visuals
    meter_y = 168
    center_x = 140
    pygame.draw.line(env.screen, (80, 80, 95), (40, meter_y), (240, meter_y), 4) # Base line
    pygame.draw.line(env.screen, (255, 255, 255), (center_x, meter_y - 7), (center_x, meter_y + 7), 2) # Center zero
    
    bar_length = min(abs(reward), 95)
    if reward > 0:
        pygame.draw.rect(env.screen, (50, 255, 80), (center_x, meter_y - 4, bar_length, 8))
    elif reward < 0:
        pygame.draw.rect(env.screen, (255, 60, 60), (center_x - bar_length, meter_y - 4, bar_length, 8))

    # Controls guide footer
    guide_text = "[T] Turbo  [H] Headless  [Space] Pause"
    guide_surface = small_font.render(guide_text, True, (160, 160, 180))
    env.screen.blit(guide_surface, (20, 188))

def draw_headless_screen(env, font, small_font, episode, step, epsilon, steps_per_sec, win_rate, paused=False):
    """Draws a high-contrast HUD screen when running in windowed headless mode."""
    if env.screen is None:
        return
        
    env.screen.fill((12, 14, 18))
    
    speed_mult = steps_per_sec / 60.0 if steps_per_sec > 0 else 1.0
    status_str = "PAUSED" if paused else "ACTIVE (MAX THROUGHPUT)"
    status_color = (255, 100, 100) if paused else (80, 255, 120)
    
    box_rect = pygame.Rect(75, 220, 500, 380)
    pygame.draw.rect(env.screen, (22, 26, 36), box_rect, border_radius=12)
    pygame.draw.rect(env.screen, (60, 75, 105), box_rect, width=2, border_radius=12)
    
    title = font.render("=== HEADLESS ULTRA-SPEED TRAINING ===", True, (255, 200, 80))
    env.screen.blit(title, (box_rect.x + 40, box_rect.y + 25))
    
    lines = [
        (f"Status: {status_str}", status_color),
        (f"Throughput: {steps_per_sec:.0f} steps/sec ({speed_mult:.1f}x Realtime)", (100, 230, 255)),
        (f"Current Episode: {episode}", (240, 240, 240)),
        (f"Step: {step} / {MAX_STEPS}", (240, 240, 240)),
        (f"Epsilon: {epsilon:.3f}", (240, 240, 240)),
        (f"Recent Success Rate: {win_rate:.0f}%", (255, 220, 120)),
        ("", (0, 0, 0)),
        ("Hotkeys:", (180, 180, 180)),
        ("  [H] Exit Headless & return to live visual mode", (200, 200, 200)),
        ("  [T] Switch directly to Turbo visual mode", (200, 200, 200)),
        ("  [Space] Pause / Resume simulation", (200, 200, 200)),
        ("  [S] Save model checkpoint", (200, 200, 200)),
        ("  [Q] / [ESC] Save checkpoint & exit cleanly", (200, 200, 200))
    ]
    
    y = box_rect.y + 65
    for text, color in lines:
        if not text:
            y += 8
            continue
        surf = small_font.render(text, True, color)
        env.screen.blit(surf, (box_rect.x + 35, y))
        y += 20
        
    pygame.display.flip()

def parse_args():
    parser = argparse.ArgumentParser(description="UAV RL Navigation Simulator - DQN Training")
    parser.add_argument("--headless", action="store_true", help="Run windowless headless training without GUI display")
    parser.add_argument("--turbo", action="store_true", help="Start directly in Turbo mode")
    parser.add_argument("--rays", type=int, default=16, help="Number of LiDAR sensor rays covering 360 degrees (default: 16)")
    parser.add_argument("--render-interval", type=int, default=15, help="Render every N steps in Ultra Turbo mode (default: 15)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Mode selection
    if args.headless:
        # True windowless headless
        render_mode = False
        mode = MODE_HEADLESS
    elif args.turbo:
        render_mode = True
        mode = MODE_TURBO
    else:
        render_mode = True
        mode = MODE_NORMAL

    # Initialize Environment with 360-degree LiDAR
    env = DroneSim(render_mode=render_mode, num_rays=args.rays)
    
    state_size = env.num_rays + 2 
    action_size = 4 
    
    agent = DQNAgent(state_size, action_size)
    
    start_episode = 0
    if os.path.exists(MODEL_PATH):
        start_episode, epsilon = agent.load(MODEL_PATH)
        print(f"[Memory] Resuming from Episode {start_episode} (Epsilon: {epsilon:.3f})")
    else:
        print("[Memory] No previous memory found. Starting training from scratch.")
        epsilon = EPSILON_START
        
    font = None
    small_font = None
    if render_mode:
        font = pygame.font.SysFont("Consolas", 15, bold=True)
        small_font = pygame.font.SysFont("Consolas", 13)
        
    current_episode = start_episode
    paused = False
    render_interval = max(1, args.render_interval)
    
    # Speed & Success tracking metrics
    total_steps = 0
    last_metric_time = time.time()
    last_metric_steps = 0
    steps_per_sec = 60.0
    recent_outcomes = deque(maxlen=30) # Stores 1 for goal reach, 0 for crash/timeout

    print("==================================================================")
    print(f" UAV RL Simulator Started | State Size: {state_size} (360° LiDAR)")
    print(" Controls: [T] Cycle Turbo  [H] Headless  [Space] Pause  [S] Save")
    print("==================================================================")

    try:
        episode = start_episode
        while True:
            episode += 1
            current_episode = episode
            state = env.reset()
            total_reward = 0
            episode_success = False
            
            for step in range(MAX_STEPS):
                total_steps += 1
                
                # Check for keyboard inputs & window events
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            agent.save(MODEL_PATH, episode, epsilon)
                            pygame.quit()
                            print(f"[Exit] Checkpoint saved at Episode {episode}. Goodbye!")
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                paused = not paused
                                print(f"[Control] Simulation {'PAUSED' if paused else 'RESUMED'}")
                            elif event.key == pygame.K_t:
                                # Cycle speed modes
                                if mode == MODE_HEADLESS:
                                    mode = MODE_TURBO
                                else:
                                    idx = MODE_CYCLE.index(mode)
                                    mode = MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]
                                print(f"[Mode] Switched to {mode}")
                            elif event.key == pygame.K_h:
                                # Toggle headless mode
                                mode = MODE_NORMAL if mode == MODE_HEADLESS else MODE_HEADLESS
                                print(f"[Mode] Headless mode: {'ENABLED' if mode == MODE_HEADLESS else 'DISABLED'}")
                            elif event.key == pygame.K_s:
                                agent.save(MODEL_PATH, episode, epsilon)
                                print(f"[Manual Save] Checkpoint saved at Episode {episode}.")
                            elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                                agent.save(MODEL_PATH, episode, epsilon)
                                pygame.quit()
                                print(f"[Exit] Clean exit requested. Model saved at Episode {episode}.")
                                return

                # Handle Pause loop
                while paused:
                    if render_mode:
                        if mode == MODE_HEADLESS:
                            draw_headless_screen(env, font, small_font, episode, step, epsilon, steps_per_sec, 
                                                 (sum(recent_outcomes) / len(recent_outcomes) * 100) if recent_outcomes else 0, paused=True)
                        else:
                            env.render()
                            win_rate = (sum(recent_outcomes) / len(recent_outcomes) * 100) if recent_outcomes else 0
                            draw_dashboard(env, font, small_font, episode, step, total_reward, epsilon, "PAUSED", 0, win_rate)
                            pygame.display.flip()
                        
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                agent.save(MODEL_PATH, episode, epsilon)
                                pygame.quit()
                                return
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    paused = False
                                    print("[Control] Resumed!")
                                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                                    agent.save(MODEL_PATH, episode, epsilon)
                                    pygame.quit()
                                    return
                        env.clock.tick(15)
                    else:
                        time.sleep(0.1)

                # RL Decision & Environment Step
                action = agent.act(state, epsilon)
                next_state, reward, done = env.step(action)
                
                agent.memory.push(state, action, reward, next_state, done)
                agent.train_step(BATCH_SIZE)
                
                state = next_state
                total_reward += reward
                
                if reward >= 90:
                    episode_success = True

                # Compute dynamic throughput metrics every 0.4 seconds
                now = time.time()
                elapsed = now - last_metric_time
                if elapsed >= 0.4:
                    steps_per_sec = (total_steps - last_metric_steps) / elapsed
                    last_metric_time = now
                    last_metric_steps = total_steps

                # Rendering dispatch based on active mode
                if render_mode:
                    win_rate = (sum(recent_outcomes) / len(recent_outcomes) * 100) if recent_outcomes else 0
                    
                    if mode == MODE_NORMAL:
                        env.render()
                        draw_dashboard(env, font, small_font, episode, step, total_reward, epsilon, mode, steps_per_sec, win_rate)
                        pygame.display.flip()
                        env.clock.tick(60) # Capped at 60 FPS
                        
                    elif mode == MODE_TURBO:
                        env.render()
                        draw_dashboard(env, font, small_font, episode, step, total_reward, epsilon, mode, steps_per_sec, win_rate)
                        pygame.display.flip()
                        # Uncapped or minimal tick to yield CPU slice
                        env.clock.tick(0)
                        
                    elif mode == MODE_ULTRA:
                        # Render only 1 out of every N steps (or at episode end) for massive speedup
                        if step % render_interval == 0 or done:
                            env.render()
                            draw_dashboard(env, font, small_font, episode, step, total_reward, epsilon, mode, steps_per_sec, win_rate)
                            pygame.display.flip()
                        env.clock.tick(0)
                        
                    elif mode == MODE_HEADLESS:
                        # Refresh headless HUD screen periodically without ray/obstacle drawing
                        if step % 25 == 0 or done:
                            draw_headless_screen(env, font, small_font, episode, step, epsilon, steps_per_sec, win_rate, paused=False)
                        env.clock.tick(0)

                if done:
                    break
            
            recent_outcomes.append(1 if episode_success else 0)
            win_rate = (sum(recent_outcomes) / len(recent_outcomes) * 100) if recent_outcomes else 0

            # Automatic checkpoint save every 10 episodes
            if episode % 10 == 0:
                agent.save(MODEL_PATH, episode, epsilon)
                
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            speed_mult = steps_per_sec / 60.0 if steps_per_sec > 0 else 1.0
            print(f"Ep {episode:4d} | Rew: {total_reward:6.1f} | Steps: {step+1:3d} | Win: {win_rate:3.0f}% | "
                  f"Eps: {epsilon:.3f} | Speed: {steps_per_sec:5.0f} s/s ({speed_mult:4.1f}x) [{mode}]")

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Training paused by user (Ctrl+C). Saving memory at Episode {current_episode}...")
        agent.save(MODEL_PATH, current_episode, epsilon)
        if render_mode:
            pygame.quit()

if __name__ == "__main__":
    main()