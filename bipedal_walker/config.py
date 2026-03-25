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

    # ── ragdoll body (5 segments: torso + 2 upper legs + 2 lower legs) ─
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

    # joint limits (radians) — these feel reasonable for a walking gait
    "hip_min_angle": math.radians(-90),
    "hip_max_angle": math.radians(90),
    "knee_min_angle": math.radians(-135),   # knees only bend backward
    "knee_max_angle": math.radians(0),

    # motors
    "motor_max_force": 5000.0,     # max force the joint motors can apply
    "motor_max_torque": 5000.0,

    # ── PPO hyperparameters ─────────────────────────────────────────
    # pretty standard PPO defaults, mostly from Schulman et al.
    "learning_rate": 3e-4,
    "gamma": 0.99,                 # discount factor
    "gae_lambda": 0.95,            # GAE lambda for advantage estimation
    "clip_epsilon": 0.2,           # PPO clipping range
    "ppo_epochs": 10,              # optimization epochs per rollout
    "minibatch_size": 64,
    "rollout_steps": 2048,         # steps collected before each update
    "vf_coef": 0.5,                # value function loss weight
    "entropy_coef": 0.01,          # entropy bonus to encourage exploration
    "max_grad_norm": 0.5,          # gradient clipping
    "total_timesteps": 500_000,    # total training budget (bump this up later)
    "hidden_size": 64,             # neurons per hidden layer in actor/critic

    # ── reward shaping ──────────────────────────────────────────────
    # tweak these to get the walker to actually walk instead of flailing
    "forward_reward_scale": 1.0,   # reward for moving to the right
    "alive_bonus": 0.1,            # small reward just for not falling
    "energy_penalty_scale": 0.001, # penalize huge joint torques
    "fall_penalty": -10.0,         # ouch
    "upright_reward_scale": 0.5,   # reward for keeping the torso vertical

    # ── logging / saving ────────────────────────────────────────────
    "log_interval": 5,             # print stats every N policy updates
    "save_path": "checkpoints/ppo_bipedal.pt",

    # ── ground surface ──────────────────────────────────────────────
    # just flat ground for now; later could swap to "perlin" or whatever
    "surface_type": "flat",
    "surface_ground_y": 350.0,     # redundant with ground_y for now, but
                                   # keeps ground-gen params self-contained
}
