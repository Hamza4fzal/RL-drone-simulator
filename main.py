import pygame
import os
import sys
import time
import argparse
import numpy as np
from collections import deque
from environment import DroneSim, WIDTH, HEIGHT
from sac_agent import SACAgent

# Hyperparameters
MAX_STEPS = 500
BATCH_SIZE = 64
WARMUP_STEPS = 600           # Initial steps collecting uniform random actions for buffer diversity
MODEL_PATH = "sac_drone_memory.pth"

# Training Modes
MODE_NORMAL = "NORMAL"          # 60 FPS cap, standard real-time rendering
MODE_TURBO = "TURBO"            # Uncapped FPS, render every frame (~10x-15x speedup)
MODE_ULTRA = "ULTRA TURBO"      # Uncapped FPS, render 1 in every N frames (~25x-50x speedup)
MODE_HEADLESS = "HEADLESS"      # Windowed headless, maximum compute speed (~50x-100x speedup)

MODE_CYCLE = [MODE_NORMAL, MODE_TURBO, MODE_ULTRA]

def draw_dashboard(env, font, episode, step, reward, alpha, action, mode, steps_per_sec, win_rate, total_goals=0, critic_loss=0.0):
    """Draws the UI metrics, continuous action readouts, and SAC entropy temperature."""
    if env.screen is None:
        return

    # Semi-transparent background for UI
    ui_width = 300
    ui_height = 222
    ui_surface = pygame.Surface((ui_width, ui_height))
    ui_surface.set_alpha(225)
    ui_surface.fill((16, 18, 26))
    env.screen.blit(ui_surface, (10, 10))
    pygame.draw.rect(env.screen, (55, 68, 95), pygame.Rect(10, 10, ui_width, ui_height), width=1, border_radius=6)

    # Mode Badge Color
    mode_colors = {
        MODE_NORMAL: (80, 200, 255),
        MODE_TURBO: (255, 200, 50),
        MODE_ULTRA: (255, 120, 50),
        MODE_HEADLESS: (255, 80, 180)
    }
    mode_color = mode_colors.get(mode, (255, 255, 255))
    
    speed_mult = steps_per_sec / 60.0 if steps_per_sec > 0 else 1.0
    throttle_val = action[0] if action is not None else 0.0
    steer_val = action[1] if action is not None else 0.0
    overall_win_rate = (total_goals / episode * 100) if episode > 0 else 0.0
    
    metrics = [
        (f"ALGO: Soft Actor-Critic (SAC)", (255, 205, 80)),
        (f"MODE: {mode}", mode_color),
        (f"Speed: {steps_per_sec:.0f} s/s ({speed_mult:.1f}x)", (150, 255, 150)),
        (f"Episode: {episode} | Step: {step}/{MAX_STEPS}", (240, 240, 240)),
        (f"Entropy Temp (Alpha): {alpha:.4f}", (140, 210, 255)),
        (f"Action: [Thr: {throttle_val:+.2f}, Str: {steer_val:+.2f}]", (255, 240, 130)),
        (f"Goals Reached: {total_goals} / {episode} ({overall_win_rate:.1f}%)", (80, 255, 160)),
        (f"Recent (Last 30): {win_rate:.0f}% | Loss: {critic_loss:.2f}", (200, 230, 255))
    ]
    
    for i, (text, color) in enumerate(metrics):
        surface = font.render(text, True, color)
        env.screen.blit(surface, (20, 14 + (i * 20)))

    # Reward/Penalty Meter Visuals
    meter_y = 202
    center_x = 150
    pygame.draw.line(env.screen, (80, 80, 95), (50, meter_y), (250, meter_y), 4) # Base line
    pygame.draw.line(env.screen, (255, 255, 255), (center_x, meter_y - 7), (center_x, meter_y + 7), 2) # Center zero
    
    bar_length = min(abs(reward), 95)
    if reward > 0:
        pygame.draw.rect(env.screen, (50, 255, 80), (center_x, meter_y - 4, bar_length, 8))
    elif reward < 0:
        pygame.draw.rect(env.screen, (255, 60, 60), (center_x - bar_length, meter_y - 4, bar_length, 8))

def draw_controls_bar(env, small_font, tiny_font):
    """Draws a dedicated controls and shortcuts reference dock on the screen."""
    if env.screen is None:
        return
        
    bar_width = WIDTH - 20 # 630 px
    bar_height = 118
    bar_x = 10
    bar_y = HEIGHT - bar_height - 10 # 850 - 118 - 10 = 722
    
    panel = pygame.Surface((bar_width, bar_height))
    panel.set_alpha(225)
    panel.fill((16, 18, 26))
    env.screen.blit(panel, (bar_x, bar_y))
    
    # Outer frame border
    pygame.draw.rect(env.screen, (55, 70, 100), pygame.Rect(bar_x, bar_y, bar_width, bar_height), width=1, border_radius=6)
    
    # Header
    title = small_font.render("SAC SIMULATOR CONTROLS & SHORTCUTS", True, (255, 205, 80))
    env.screen.blit(title, (bar_x + 14, bar_y + 7))
    
    shortcuts = [
        ("[T]", "Cycle visual speed modes (NORMAL -> TURBO -> ULTRA TURBO).", (255, 200, 50)),
        ("[H]", "Toggle HEADLESS accelerated mode on and off instantly without freezing the window.", (255, 90, 180)),
        ("[SPACE]", "Pause and resume training.", (100, 240, 255)),
        ("[S]", "Manually save the SAC policy checkpoint to sac_drone_memory.pth.", (130, 255, 130)),
        ("[Q] / [ESC]", "Save and cleanly exit.", (255, 120, 120))
    ]
    
    y = bar_y + 26
    for key_tag, desc, tag_color in shortcuts:
        k_surf = tiny_font.render(key_tag, True, tag_color)
        d_surf = tiny_font.render(f": {desc}", True, (225, 225, 235))
        env.screen.blit(k_surf, (bar_x + 14, y))
        env.screen.blit(d_surf, (bar_x + 105, y))
        y += 17

def draw_headless_screen(env, font, small_font, tiny_font, episode, step, alpha, steps_per_sec, win_rate, total_goals=0, paused=False):
    """Draws a high-contrast HUD screen when running in windowed headless mode."""
    if env.screen is None:
        return
        
    env.screen.fill((12, 14, 18))
    
    speed_mult = steps_per_sec / 60.0 if steps_per_sec > 0 else 1.0
    status_str = "PAUSED" if paused else "ACTIVE (MAX THROUGHPUT)"
    status_color = (255, 100, 100) if paused else (80, 255, 120)
    overall_win_rate = (total_goals / episode * 100) if episode > 0 else 0.0
    
    box_rect = pygame.Rect(35, 160, 580, 480)
    pygame.draw.rect(env.screen, (20, 24, 34), box_rect, border_radius=12)
    pygame.draw.rect(env.screen, (60, 75, 105), box_rect, width=2, border_radius=12)
    
    title = font.render("=== SAC HEADLESS ULTRA-SPEED TRAINING ===", True, (255, 200, 80))
    env.screen.blit(title, (box_rect.x + 40, box_rect.y + 22))
    
    metrics = [
        (f"Status: {status_str}", status_color),
        (f"Throughput: {steps_per_sec:.0f} steps/sec ({speed_mult:.1f}x Realtime)", (100, 230, 255)),
        (f"Algorithm: Soft Actor-Critic (Continuous 2D Control)", (255, 220, 120)),
        (f"Current Episode: {episode} | Step: {step} / {MAX_STEPS}", (240, 240, 240)),
        (f"Total Goals Reached: {total_goals} / {episode} ({overall_win_rate:.1f}%)", (80, 255, 160)),
        (f"Recent Success Rate: {win_rate:.0f}% (Last 30)", (255, 220, 120)),
        (f"Entropy Temperature (Alpha): {alpha:.4f}", (140, 210, 255)),
    ]
    
    y = box_rect.y + 58
    for text, color in metrics:
        surf = small_font.render(text, True, color)
        env.screen.blit(surf, (box_rect.x + 35, y))
        y += 22
        
    y += 10
    head_surf = small_font.render("CONTROLS & SHORTCUTS:", True, (255, 205, 80))
    env.screen.blit(head_surf, (box_rect.x + 35, y))
    y += 26
    
    shortcuts = [
        ("[T]", "Cycle visual speed modes (NORMAL -> TURBO -> ULTRA TURBO).", (255, 200, 50)),
        ("[H]", "Toggle HEADLESS accelerated mode on and off instantly without freezing the window.", (255, 90, 180)),
        ("[SPACE]", "Pause and resume training.", (100, 240, 255)),
        ("[S]", "Manually save the SAC policy checkpoint to sac_drone_memory.pth.", (130, 255, 130)),
        ("[Q] / [ESC]", "Save and cleanly exit.", (255, 120, 120))
    ]
    
    for key_tag, desc, tag_color in shortcuts:
        k_surf = tiny_font.render(key_tag, True, tag_color)
        d_surf = tiny_font.render(f": {desc}", True, (220, 225, 235))
        env.screen.blit(k_surf, (box_rect.x + 35, y))
        env.screen.blit(d_surf, (box_rect.x + 130, y))
        y += 22
        
    pygame.display.flip()

def parse_args():
    parser = argparse.ArgumentParser(description="UAV RL Navigation Simulator - Soft Actor-Critic (SAC)")
    parser.add_argument("--headless", action="store_true", help="Run windowless headless training without GUI display")
    parser.add_argument("--turbo", action="store_true", help="Start directly in Turbo mode")
    parser.add_argument("--rays", type=int, default=16, help="Number of LiDAR sensor rays covering 360 degrees (default: 16)")
    parser.add_argument("--render-interval", type=int, default=15, help="Render every N steps in Ultra Turbo mode (default: 15)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Mode selection
    if args.headless:
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
    
    # State features = 16 rays + [dist_to_goal, relative_angle, speed] = 19
    state_size = env.num_rays + 3
    # Continuous action space: [throttle, steering] in [-1.0, 1.0]
    action_size = 2 
    
    # Soft Actor-Critic Agent
    agent = SACAgent(
        state_dim=state_size, 
        action_dim=action_size, 
        lr=3e-4, 
        gamma=0.99, 
        tau=0.005, 
        auto_entropy=True
    )
    
    start_episode = 0
    total_goals = 0
    if os.path.exists(MODEL_PATH):
        start_episode, total_goals = agent.load(MODEL_PATH)
        print(f"[Memory] Resuming SAC policy from Episode {start_episode} (Goals: {total_goals}, Alpha: {agent.alpha:.4f})")
    else:
        print("[Memory] No previous SAC memory found. Starting training from scratch.")
        
    font = None
    small_font = None
    tiny_font = None
    if render_mode:
        font = pygame.font.SysFont("Consolas", 15, bold=True)
        small_font = pygame.font.SysFont("Consolas", 13)
        tiny_font = pygame.font.SysFont("Consolas", 11)
        
    current_episode = start_episode
    paused = False
    render_interval = max(1, args.render_interval)
    
    # Speed & Success tracking metrics
    total_steps = 0
    last_metric_time = time.time()
    last_metric_steps = 0
    steps_per_sec = 60.0
    recent_outcomes = deque(maxlen=30)
    current_critic_loss = 0.0

    print("==================================================================")
    print(f" UAV SAC Simulator Started | State: {state_size} | Action: {action_size}D Cont.")
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
            last_action = np.zeros(action_size, dtype=np.float32)
            
            for step in range(MAX_STEPS):
                total_steps += 1
                
                # Check for keyboard inputs & window events
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            agent.save(MODEL_PATH, episode, total_goals)
                            pygame.quit()
                            print(f"[Exit] SAC Checkpoint saved at Episode {episode} (Goals: {total_goals}). Goodbye!")
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                paused = not paused
                                print(f"[Control] Simulation {'PAUSED' if paused else 'RESUMED'}")
                            elif event.key == pygame.K_t:
                                if mode == MODE_HEADLESS:
                                    mode = MODE_TURBO
                                else:
                                    idx = MODE_CYCLE.index(mode)
                                    mode = MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]
                                print(f"[Mode] Switched to {mode}")
                            elif event.key == pygame.K_h:
                                mode = MODE_NORMAL if mode == MODE_HEADLESS else MODE_HEADLESS
                                print(f"[Mode] Headless mode: {'ENABLED' if mode == MODE_HEADLESS else 'DISABLED'}")
                            elif event.key == pygame.K_s:
                                agent.save(MODEL_PATH, episode, total_goals)
                                print(f"[Manual Save] SAC Checkpoint saved at Episode {episode} (Goals: {total_goals}).")
                            elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                                agent.save(MODEL_PATH, episode, total_goals)
                                pygame.quit()
                                print(f"[Exit] Clean exit requested. SAC model saved at Episode {episode} (Goals: {total_goals}).")
                                return

                # Handle Pause loop
                while paused:
                    if render_mode:
                        win_rate = (sum(recent_outcomes) / len(recent_outcomes) * 100) if recent_outcomes else 0
                        if mode == MODE_HEADLESS:
                            draw_headless_screen(env, font, small_font, tiny_font, episode, step, agent.alpha, 
                                                 steps_per_sec, win_rate, total_goals=total_goals, paused=True)
                        else:
                            env.render()
                            draw_dashboard(env, font, episode, step, total_reward, agent.alpha, last_action, 
                                           "PAUSED", 0, win_rate, total_goals=total_goals, critic_loss=current_critic_loss)
                            draw_controls_bar(env, small_font, tiny_font)
                            pygame.display.flip()
                        
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                agent.save(MODEL_PATH, episode, total_goals)
                                pygame.quit()
                                return
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    paused = False
                                    print("[Control] Resumed!")
                                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                                    agent.save(MODEL_PATH, episode, total_goals)
                                    pygame.quit()
                                    return
                        env.clock.tick(15)
                    else:
                        time.sleep(0.1)

                # Action selection: Warmup with uniform random actions, then sample from SAC stochastic policy
                if total_steps < WARMUP_STEPS and start_episode == 0:
                    action = np.random.uniform(-1.0, 1.0, size=(action_size,)).astype(np.float32)
                else:
                    action = agent.select_action(state, evaluate=False)

                last_action = action
                next_state, reward, done = env.step(action)
                
                # Store transition in continuous replay buffer
                agent.memory.push(state, action, reward, next_state, done)
                
                # Gradient update step
                metrics = agent.train_step(BATCH_SIZE)
                if metrics:
                    current_critic_loss = metrics["critic_loss"]
                
                state = next_state
                total_reward += reward
                
                if reward >= 90:
                    episode_success = True

                # Dynamic throughput metrics
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
                        draw_dashboard(env, font, episode, step, total_reward, agent.alpha, last_action, 
                                       mode, steps_per_sec, win_rate, total_goals=total_goals, critic_loss=current_critic_loss)
                        draw_controls_bar(env, small_font, tiny_font)
                        pygame.display.flip()
                        env.clock.tick(60)
                        
                    elif mode == MODE_TURBO:
                        env.render()
                        draw_dashboard(env, font, episode, step, total_reward, agent.alpha, last_action, 
                                       mode, steps_per_sec, win_rate, total_goals=total_goals, critic_loss=current_critic_loss)
                        draw_controls_bar(env, small_font, tiny_font)
                        pygame.display.flip()
                        env.clock.tick(0)
                        
                    elif mode == MODE_ULTRA:
                        if step % render_interval == 0 or done:
                            env.render()
                            draw_dashboard(env, font, episode, step, total_reward, agent.alpha, last_action, 
                                           mode, steps_per_sec, win_rate, total_goals=total_goals, critic_loss=current_critic_loss)
                            draw_controls_bar(env, small_font, tiny_font)
                            pygame.display.flip()
                        env.clock.tick(0)
                        
                    elif mode == MODE_HEADLESS:
                        if step % 25 == 0 or done:
                            draw_headless_screen(env, font, small_font, tiny_font, episode, step, agent.alpha, 
                                                 steps_per_sec, win_rate, total_goals=total_goals, paused=False)
                        env.clock.tick(0)

                if done:
                    break
            
            if episode_success:
                total_goals += 1

            recent_outcomes.append(1 if episode_success else 0)
            win_rate = (sum(recent_outcomes) / len(recent_outcomes) * 100) if recent_outcomes else 0
            overall_win_rate = (total_goals / episode * 100) if episode > 0 else 0.0

            # Automatic checkpoint save every 10 episodes
            if episode % 10 == 0:
                agent.save(MODEL_PATH, episode, total_goals)
                
            speed_mult = steps_per_sec / 60.0 if steps_per_sec > 0 else 1.0
            print(f"Ep {episode:4d} | Rew: {total_reward:6.1f} | Goals: {total_goals}/{episode} ({overall_win_rate:4.1f}%) | "
                  f"Win30: {win_rate:3.0f}% | Alpha: {agent.alpha:.4f} | Speed: {steps_per_sec:5.0f} s/s ({speed_mult:4.1f}x) [{mode}]")

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Training paused by user (Ctrl+C). Saving SAC memory at Episode {current_episode} (Goals: {total_goals})...")
        agent.save(MODEL_PATH, current_episode, total_goals)
        if render_mode:
            pygame.quit()

if __name__ == "__main__":
    main()