"""
visualize.py - Pygame-based visualization for the bipedal walker.

Watch your walker strut (or faceplant) in real-time!

Usage:
    # random actions (no trained model)
    python visualize.py

    # with a trained model
    python visualize.py --model checkpoints/ppo_bipedal.pt

    # start in fullscreen
    python visualize.py --fullscreen

Controls:
    ESC / close window  - quit
    SPACE               - pause / unpause
    R                   - reset episode
    F11                 - toggle fullscreen
    0                   - switch to random actions (no model)
    1-5                 - switch to short / medium / long / xlong / max model
                          (ignored if the checkpoint file doesn't exist)
"""

import argparse
import math
import os

import numpy as np
import pygame
import torch

from config import CONFIG
from environment import BipedalWalkerEnv
from agent import ActorCritic


# ── colors ──────────────────────────────────────────────────────────────
SKY_COLOR = (135, 206, 255)         # sky blue
GROUND_COLOR = (60, 60, 60)
GROUND_LINE_COLOR = (40, 40, 40)    # darker edge on top
GRID_COLOR = (250, 190, 25)          # tick marks

TORSO_COLOR = (190, 190, 190)
THIGH_COLOR = (230, 230, 230)
SHIN_COLOR = (230, 230, 230)
FOOT_COLOR = (190, 190, 190)
JOINT_COLOR = (20, 20, 20)

HUD_BG_COLOR = (0, 0, 0, 140)      # semi-transparent black
HUD_TEXT_COLOR = (255, 255, 255)

MOUNTAIN_COLOR = (70, 120, 90)
PARALLAX_SPEED = 0.1              # mountains scroll at this fraction of ground speed


# ── mountains (perlin noise background) ──────────────────────────────

def _perlin_hash(i, seed=0):
    """Pseudo-random gradient at integer position."""
    h = ((i * 127 + seed * 311) ^ 0xB5297A4D) & 0xFFFFFFFF
    h = (((h >> 13) ^ h) * 0x68E31DA4) & 0xFFFFFFFF
    return (h & 0xFFFF) / 32768.0 - 1.0


def _perlin_1d(x, seed=0):
    """1D Perlin noise returning a value in roughly [-0.5, 0.5]."""
    x0 = int(math.floor(x))
    t = x - x0
    t = t * t * t * (t * (t * 6 - 15) + 10)  # quintic smoothstep
    d0 = _perlin_hash(x0, seed) * (x - x0)
    d1 = _perlin_hash(x0 + 1, seed) * (x - x0 - 1)
    return d0 + t * (d1 - d0)


def _mountain_height(wx):
    """Fractal Perlin noise for the mountain profile. Returns ~[-1, 1]."""
    total = 0.0
    amp, freq = 1.0, 0.002
    for octave in range(4):
        total += _perlin_1d(wx * freq, seed=octave) * amp
        amp *= 0.5
        freq *= 2.0
    return total


def draw_mountains(screen, camera_x, screen_w, ground_y):
    """Draw a parallax mountain silhouette between the horizon and the ground."""
    offset = camera_x * PARALLAX_SPEED
    points = []
    for sx in range(0, screen_w + 1, 2):
        wx = sx + offset - screen_w // 2
        my = ground_y - 100 - _mountain_height(wx) * 50
        points.append((sx, int(my)))
    points.append((screen_w, int(ground_y)))
    points.append((0, int(ground_y)))
    pygame.draw.polygon(screen, MOUNTAIN_COLOR, points)


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
        pygame.draw.aalines(screen, (40, 40, 40), True, verts)

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
             mode_label, paused):
    """Heads-up display in the top-left corner."""
    lines = [
        f"Reward: {episode_reward:.1f}",
        f"Steps: {step_count}",
        f"Pos X: {torso_x:.0f}",
        f"Mode: {mode_label}",
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


# ── model switching ────────────────────────────────────────────────────
# number keys 0-5 map to random / short / medium / long / xlong / max.
# profile=None means "no model -- use random actions".
PROFILE_KEYS = {
    pygame.K_0: None,
    pygame.K_1: "short",
    pygame.K_2: "medium",
    pygame.K_3: "long",
    pygame.K_4: "xlong",
    pygame.K_5: "max",
}


def _profile_path(profile):
    """Checkpoint path for a given training profile name."""
    return f"checkpoints/ppo_bipedal_{profile}.pt"


def _load_agent(model_path, env, device):
    """Load an ActorCritic from a checkpoint file. Returns (agent, hidden_size)."""
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
    return agent, hidden_size


# ── main visualization loop ────────────────────────────────────────────

def run_visualization(model_path=None, fullscreen=False):
    """Set up pygame + environment and run the render loop."""
    pygame.init()

    screen_w = CONFIG["viewport_width"]
    screen_h = CONFIG["viewport_height"]
    # SCALED keeps the logical draw surface at viewport_width x viewport_height
    # and GPU-scales it to the window/monitor while preserving aspect ratio
    # (letterboxed). This lets fullscreen work without touching world coords,
    # ground_y, HUD positioning, or any of the drawing code.
    flags = pygame.SCALED | (pygame.FULLSCREEN if fullscreen else 0)
    screen = pygame.display.set_mode((screen_w, screen_h), flags)
    pygame.display.set_caption(
        "Bipedal Walker" + (" (trained)" if model_path else " (random)")
    )

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    small_font = pygame.font.SysFont(None, 18)

    # set up environment
    env = BipedalWalkerEnv(CONFIG)

    # optionally load a trained model
    agent = None
    mode_label = "random"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path:
        agent, hidden_size = _load_agent(model_path, env, device)
        # label by profile name if the path matches a known profile,
        # otherwise fall back to the checkpoint filename stem
        mode_label = next(
            (p for p in PROFILE_KEYS.values()
             if p is not None and _profile_path(p) == model_path),
            os.path.splitext(os.path.basename(model_path))[0],
        )
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

    print(
        "Controls: ESC=quit, SPACE=pause, R=reset, F11=toggle fullscreen, "
        "0=random, 1-5=short/medium/long/xlong/max"
    )

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
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif event.key in PROFILE_KEYS:
                    profile = PROFILE_KEYS[event.key]
                    if profile is None:
                        agent = None
                        mode_label = "random"
                        print("Switched to random actions")
                    else:
                        path = _profile_path(profile)
                        if not os.path.exists(path):
                            print(
                                f"Cannot switch to '{profile}': "
                                f"{path} not found"
                            )
                        else:
                            agent, hidden_size = _load_agent(
                                path, env, device,
                            )
                            mode_label = profile
                            print(
                                f"Switched to '{profile}' model "
                                f"(hidden_size={hidden_size})"
                            )

        # ── simulation step (skip when paused) ─────────────────────
        if not paused:
            # pick an action
            if agent is not None:
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

            # only reset on actual falls -- ignore the env's step-count cap
            # so the walker can keep going indefinitely during visualization.
            # (the env still emits done at max_episode_steps, but we don't
            # care here; press R to reset manually if you want.)
            if info["fell"]:
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
        draw_mountains(screen, camera_x, screen_w, CONFIG["ground_y"])
        draw_ground(
            screen, camera_x, screen_w, screen_h,
            CONFIG["ground_y"], small_font,
        )
        draw_body_parts(screen, body_pos, camera_x, screen_w)
        draw_hud(
            screen, font, episode_reward, step_count,
            body_pos["torso"]["position"][0], mode_label, paused,
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
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="start in fullscreen mode (toggle with F11)",
    )
    args = parser.parse_args()
    run_visualization(model_path=args.model, fullscreen=args.fullscreen)
