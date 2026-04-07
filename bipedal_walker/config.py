"""
Central config for the bipedal walker project.

Every single hyperparameter lives here in one flat dict so there's
exactly one place to tweak things. Import CONFIG from this file
anywhere you need a value — don't hardcode numbers elsewhere.
"""

import math

CONFIG = {
    # ── environment / physics ───────────────────────────────────────
    # pymunk runs at a fixed timestep; 1/60 is standard 60-fps physics
    "physics_dt": 1.0 / 60.0,
    "gravity": 980.0,              # pixels/sec², points downward
    "ground_y": 350.0,             # y position of the ground surface
    "max_episode_steps": 1000,     # auto-reset after this many steps
    "viewport_width": 800,         # render window width  (pixels)
    "viewport_height": 400,        # render window height (pixels)

    # ── ragdoll body (7 segments: torso + 2 thighs + 2 shins + 2 feet) ─
    # torso
    "torso_width": 30.0,
    "torso_height": 60.0,
    "torso_mass": 10.0,
    "torso_start_x": 200.0,       # initial x position
    "torso_start_y": 200.0,       # initial y position (above ground)

    # upper legs (thighs)
    "upper_leg_length": 45.0,
    "upper_leg_width": 12.0,
    "upper_leg_mass": 3.0,

    # lower legs (shins)
    "lower_leg_length": 40.0,
    "lower_leg_width": 10.0,
    "lower_leg_mass": 2.0,

    # feet (wider than tall, attached at top-left corner via ankle)
    "foot_width": 22.0,
    "foot_height": 10.0,
    "foot_mass": 1.0,

    # joint limits (radians) — these feel reasonable for a walking gait
    "hip_min_angle": math.radians(-90),
    "hip_max_angle": math.radians(90),
    "knee_min_angle": math.radians(0),      # straight leg (no hyperextension)
    "knee_max_angle": math.radians(135),   # flexion only (shin behind thigh)

    # ankle joint limits (agent faces right)
    # dorsiflexion (toes up): negative relative angle
    # plantarflexion (toes down/push-off): positive relative angle
    "ankle_min_angle": math.radians(-30),
    "ankle_max_angle": math.radians(45),

    # motors — per-joint forces using biomechanical ratios
    # (hip muscles ~2x stronger than ankle muscles in humans)
    "hip_motor_force":   500_000.0,   # hip: full force (largest muscles)
    "knee_motor_force":  350_000.0,   # knee: ~70% of hip
    "ankle_motor_force": 250_000.0,   # ankle: ~50% of hip

    # ── PPO hyperparameters (shared across profiles) ─────────────────
    "learning_rate": 3e-4,         # initial LR; cosine-annealed to 0
    "gamma": 0.99,                 # discount factor
    "gae_lambda": 0.95,            # GAE lambda for advantage estimation
    "clip_epsilon": 0.2,           # PPO clipping range
    "vf_coef": 0.5,                # value function loss weight
    "max_grad_norm": 0.5,          # gradient clipping

    # ── PPO hyperparameters (profile-dependent) ───────────────────
    # defaults below are "short" profile values for quick testing.
    # see PROFILES dict and get_config() below for profile overrides.
    "total_timesteps": 500_000,    # short: 500K (~15 min)
    "hidden_size": 64,             # short: 64 neurons per hidden layer
    "rollout_steps": 2048,         # short: 2048 steps per rollout
    "minibatch_size": 64,          # short: 64 samples per minibatch
    "ppo_epochs": 10,              # short: 10 epochs per update
    "entropy_coef": 0.01,          # short: standard exploration bonus

    # ── reward shaping ──────────────────────────────────────────────
    # 4 components: forward velocity, alive bonus, energy penalty, fall penalty.
    # no explicit uprightness reward — balance emerges from foot contacts
    # and the devastating fall penalty.
    # target ordering: walk >> stand >> fall

    "forward_reward_scale": 5.0,   # velocity reward scale
    "velocity_bonus_scale": 1.0,   # superlinear speed bonus: v*(1+1.0*|v|)
    "alive_bonus": 2.0,            # per-step survival bonus, compounds over episode
                                   # (full-episode total: ~3000)
    "energy_penalty_scale": 0.1,   # penalizes motor vibration; max ~400/episode
    "fall_penalty": -5000.0,       # ~1.7x full-episode alive bonus — must dominate
                                   # short-episode forward reward to punish diving

    # ── logging / saving (profile-dependent) ─────────────────────────
    "log_interval": 5,             # short: every 5 updates
    "save_path": "checkpoints/ppo_bipedal_short.pt",

    # ── ground surface ──────────────────────────────────────────────
    # just flat ground for now; later could swap to "perlin" or whatever
    "surface_type": "flat",
    "surface_ground_y": 350.0,     # redundant with ground_y for now, but
                                   # keeps ground-gen params self-contained
}

# ── Training profiles ──────────────────────────────────────────────────
# 5 tiers with logarithmic spacing to map the full learning curve.
# medium+ all use h=256 so performance differences isolate training
# length as the sole variable.  short uses h=64 because 500K steps
# can't train a 145K-param network.
#
# CONFIG defaults are the "short" values.  Only parameters that need to
# differ are listed in each profile.
#
#  Parameter       | Short  | Medium | Long   | XLong  | Max
#  ────────────────┼────────┼────────┼────────┼────────┼────────
#  total_timesteps | 500K   | 3M     | 10M    | 25M    | 50M
#  hidden_size     | 64     | 256    | 256    | 256    | 256
#  rollout_steps   | 2048   | 4096   | 8192   | 8192   | 8192
#  minibatch_size  | 64     | 128    | 256    | 256    | 256
#  ppo_epochs      | 10     | 8      | 6      | 5      | 5
#  entropy_coef    | 0.01   | 0.005  | 0.005  | 0.01   | 0.01
#  log_interval    | 5      | 10     | 10     | 25     | 50
#  ~wall clock     | 15 min | 2.3 hr | 7.6 hr | 19 hr  | 38 hr

PROFILES = {
    "short": {},
    "medium": {
        "total_timesteps": 3_000_000,
        "hidden_size": 256,
        "rollout_steps": 4096,
        "minibatch_size": 128,
        "ppo_epochs": 8,
        "entropy_coef": 0.005,
        "log_interval": 10,
        "save_path": "checkpoints/ppo_bipedal_medium.pt",
    },
    "long": {
        "total_timesteps": 10_000_000,
        "hidden_size": 256,
        "rollout_steps": 8192,
        "minibatch_size": 256,
        "ppo_epochs": 6,
        "entropy_coef": 0.005,
        "log_interval": 10,
        "save_path": "checkpoints/ppo_bipedal_long.pt",
    },
    "xlong": {
        "total_timesteps": 25_000_000,
        "hidden_size": 256,
        "rollout_steps": 8192,
        "minibatch_size": 256,
        "ppo_epochs": 5,
        "entropy_coef": 0.01,
        "log_interval": 25,
        "save_path": "checkpoints/ppo_bipedal_xlong.pt",
    },
    "max": {
        "total_timesteps": 50_000_000,
        "hidden_size": 256,
        "rollout_steps": 8192,
        "minibatch_size": 256,
        "ppo_epochs": 5,
        "entropy_coef": 0.01,
        "log_interval": 50,
        "save_path": "checkpoints/ppo_bipedal_max.pt",
    },
}


def get_config(profile="short"):
    """Return CONFIG merged with the chosen profile's overrides."""
    config = dict(CONFIG)
    config.update(PROFILES[profile])
    config["profile"] = profile
    return config
