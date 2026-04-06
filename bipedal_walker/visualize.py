"""
visualize.py - Pygame-based visualization for the bipedal walker.

Watch your walker strut (or faceplant) in real-time!

Usage:
    # random actions (no trained model)
    python visualize.py

    # with a trained model
    python visualize.py --model checkpoints/ppo_bipedal.pt

Controls:
    ESC / close window  - quit
    SPACE               - pause / unpause
    R                   - reset episode
"""

import argparse

import numpy as np
import pygame
import torch

from config import CONFIG
from environment import BipedalWalkerEnv
from agent import ActorCritic


# ── colors ──────────────────────────────────────────────────────────────
SKY_COLOR = (135, 206, 235)         # sky blue
GROUND_COLOR = (86, 125, 70)        # earthy green
GROUND_LINE_COLOR = (60, 90, 50)    # darker edge on top
GRID_COLOR = (70, 110, 60)          # tick marks

TORSO_COLOR = (70, 100, 160)        # muted blue
THIGH_COLOR = (90, 130, 180)        # slightly lighter
SHIN_COLOR = (110, 150, 200)        # lighter still
FOOT_COLOR = (130, 170, 220)        # lightest blue (lighter than shin)
JOINT_COLOR = (220, 200, 60)        # yellow-ish dots

HUD_BG_COLOR = (0, 0, 0, 140)      # semi-transparent black
HUD_TEXT_COLOR = (255, 255, 255)


# ── rendering helpers ───────────────────────────────────────────────────

def world_to_screen(wx, wy, camera_x, screen_w):
    """
    Convert world coords to screen coords.

    The physics engine already uses y-down (matching pygame) so we
    only need to offset x for the horizontal camera follow.
    """
    sx = wx - camera_x + screen_w // 2
    sy = wy
    return int(sx), int(sy)


def draw_body_parts(screen, body_positions, camera_x, screen_w):
    """Draw ragdoll segments as colored polygons + joint circles."""
    colors = {
        "torso": TORSO_COLOR,
        "left_thigh": THIGH_COLOR,
        "right_thigh": THIGH_COLOR,
        "left_shin": SHIN_COLOR,
        "right_shin": SHIN_COLOR,
        "left_foot": FOOT_COLOR,
        "right_foot": FOOT_COLOR,
    }

    # draw back-to-front so torso is on top
    draw_order = [
        "left_foot", "right_foot",
        "left_shin", "right_shin",
        "left_thigh", "right_thigh",
        "torso",
    ]
    for name in draw_order:
        if name not in body_positions:
            continue

        verts = [
            world_to_screen(vx, vy, camera_x, screen_w)
            for vx, vy in body_positions[name]["vertices"]
        ]
        pygame.draw.polygon(screen, colors[name], verts)
        pygame.draw.polygon(screen, (40, 40, 40), verts, 2)  # outline

    # joint circles at the top edge of each limb segment.
    # in the box vertex order [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)],
    # indices 0 and 1 are the "top" edge (connects to parent body).
    for name in ["left_thigh", "right_thigh", "left_shin", "right_shin"]:
        if name not in body_positions:
            continue
        v = body_positions[name]["vertices"]
        jx = (v[0][0] + v[1][0]) / 2
        jy = (v[0][1] + v[1][1]) / 2
        sx, sy = world_to_screen(jx, jy, camera_x, screen_w)
        pygame.draw.circle(screen, JOINT_COLOR, (sx, sy), 4)
        pygame.draw.circle(screen, (40, 40, 40), (sx, sy), 4, 1)

    # ankle joint circles at the bottom edge of each shin.
    # in the vertex order [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)],
    # indices 2 and 3 are the bottom edge (connects to foot via ankle).
    for name in ["left_shin", "right_shin"]:
        if name not in body_positions:
            continue
        v = body_positions[name]["vertices"]
        jx = (v[2][0] + v[3][0]) / 2
        jy = (v[2][1] + v[3][1]) / 2
        sx, sy = world_to_screen(jx, jy, camera_x, screen_w)
        pygame.draw.circle(screen, JOINT_COLOR, (sx, sy), 4)
        pygame.draw.circle(screen, (40, 40, 40), (sx, sy), 4, 1)


def draw_ground(screen, camera_x, screen_w, screen_h, ground_y, small_font):
    """Draw the ground surface with tick-mark grid so you can see movement."""
    gy = int(ground_y)

    # filled ground area
    pygame.draw.rect(screen, GROUND_COLOR, (0, gy, screen_w, screen_h - gy))
    # top edge line
    pygame.draw.line(screen, GROUND_LINE_COLOR, (0, gy), (screen_w, gy), 3)

    # vertical tick marks every 100px (world units)
    left_wx = camera_x - screen_w // 2
    right_wx = camera_x + screen_w // 2
    start = int(left_wx // 100) * 100

    for wx in range(start, int(right_wx) + 100, 100):
        sx = int(wx - camera_x + screen_w // 2)
        pygame.draw.line(screen, GRID_COLOR, (sx, gy), (sx, gy + 15), 2)
        # distance label every 200px
        if wx % 200 == 0:
            label = small_font.render(str(wx), True, GRID_COLOR)
            screen.blit(label, (sx - label.get_width() // 2, gy + 17))


def draw_hud(screen, font, episode_reward, step_count, torso_x,
             using_model, paused):
    """Heads-up display in the top-left corner."""
    lines = [
        f"Reward: {episode_reward:.1f}",
        f"Steps: {step_count}",
        f"Pos X: {torso_x:.0f}",
        f"Mode: {'trained model' if using_model else 'random actions'}",
    ]
    if paused:
        lines.append("** PAUSED **")

    pad = 8
    line_h = font.get_height() + 2
    box_w = 220
    box_h = pad * 2 + line_h * len(lines)

    # semi-transparent background box
    hud_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    hud_surf.fill(HUD_BG_COLOR)
    screen.blit(hud_surf, (10, 10))

    for i, line in enumerate(lines):
        text = font.render(line, True, HUD_TEXT_COLOR)
        screen.blit(text, (10 + pad, 10 + pad + i * line_h))


# ── main visualization loop ────────────────────────────────────────────

def run_visualization(model_path=None):
    """Set up pygame + environment and run the render loop."""
    pygame.init()

    screen_w = CONFIG["viewport_width"]
    screen_h = CONFIG["viewport_height"]
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(
        "Bipedal Walker" + (" (trained)" if model_path else " (random)")
    )

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    small_font = pygame.font.SysFont(None, 18)

    # set up environment
    env = BipedalWalkerEnv(CONFIG)

    # optionally load a trained model
    using_model = False
    agent = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            hidden_size = checkpoint["hidden_size"]
            state_dict = checkpoint["model_state_dict"]
        else:
            # backward compat: old checkpoints are bare state_dicts
            hidden_size = CONFIG["hidden_size"]
            state_dict = checkpoint
        agent = ActorCritic(
            env.observation_size, env.action_size,
            hidden_size=hidden_size,
        ).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()
        using_model = True
        print(f"Loaded model from: {model_path} (hidden_size={hidden_size})")
    else:
        print("No model specified -- using random actions")

    # smooth camera: lerp toward the torso each frame
    camera_x = CONFIG["torso_start_x"]
    camera_smoothing = 0.08  # lower = smoother / slower follow

    # episode state
    obs = env.reset()
    episode_reward = 0.0
    step_count = 0
    paused = False
    running = True

    print("Controls: ESC=quit, SPACE=pause, R=reset")

    while running:
        # ── events ─────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    episode_reward = 0.0
                    step_count = 0
                    camera_x = CONFIG["torso_start_x"]
                    print("Episode reset (manual)")

        # ── simulation step (skip when paused) ─────────────────────
        if not paused:
            # pick an action
            if using_model:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, _, _ = agent.get_action(obs_t)
                action_np = action.cpu().numpy()
            else:
                action_np = np.random.uniform(
                    -1, 1, size=env.action_size
                ).astype(np.float32)

            obs, reward, done, info = env.step(action_np)
            episode_reward += reward
            step_count += 1

            if done:
                torso_x = info["torso_x"]
                distance = torso_x - CONFIG["torso_start_x"]
                print(
                    f"Episode done!  reward={episode_reward:.1f}, "
                    f"steps={step_count}, distance={distance:.0f}px"
                )
                # brief freeze so you can see the final pose
                pygame.time.wait(500)

                obs = env.reset()
                episode_reward = 0.0
                step_count = 0
                camera_x = CONFIG["torso_start_x"]

        # ── camera follow ──────────────────────────────────────────
        body_pos = env.get_body_positions()
        target_x = body_pos["torso"]["position"][0]
        camera_x += (target_x - camera_x) * camera_smoothing

        # ── draw everything ────────────────────────────────────────
        screen.fill(SKY_COLOR)
        draw_ground(
            screen, camera_x, screen_w, screen_h,
            CONFIG["ground_y"], small_font,
        )
        draw_body_parts(screen, body_pos, camera_x, screen_w)
        draw_hud(
            screen, font, episode_reward, step_count,
            body_pos["torso"]["position"][0], using_model, paused,
        )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Bye!")


# ── entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch the bipedal walker in action",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="path to a trained model checkpoint (e.g. checkpoints/ppo_bipedal.pt)",
    )
    args = parser.parse_args()
    run_visualization(model_path=args.model)
