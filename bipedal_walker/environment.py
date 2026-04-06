"""
environment.py - Gym-style environment wrapper for the bipedal walker.

This sits between the PPO agent and the raw physics simulation.  The agent
calls reset() to start an episode, then step(action) in a loop until it
gets done=True.  We handle reward computation, episode termination, and
packaging everything into the (obs, reward, done, info) tuple that RL
algorithms expect.

No actual gym dependency -- we just follow the same interface pattern.
"""

import numpy as np

from config import CONFIG
from physics import PhysicsWorld


class BipedalWalkerEnv:
    """
    Wraps PhysicsWorld to give a standard RL environment interface.

    Observation: 19 floats from the ragdoll's state vector
    Action:      6 floats in [-1, 1], one per joint motor
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.world = PhysicsWorld(config)

        # observation is 19 floats (see physics.py RagdollBody.get_state)
        self.observation_size = 19
        # 6 motors: left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
        self.action_size = 6

        # episode tracking
        self.step_count = 0
        self.prev_torso_x = config["torso_start_x"]

    def reset(self):
        """
        Reset the physics world and all tracking variables.
        Returns the initial observation as a numpy array.
        """
        obs = self.world.reset()
        self.step_count = 0
        self.prev_torso_x = self.config["torso_start_x"]
        return obs

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action: numpy array of 6 floats, will be clipped to [-1, 1]

        Returns:
            obs:    numpy array of 19 floats (new state)
            reward: float (total reward for this step)
            done:   bool (episode over?)
            info:   dict with debugging goodies
        """
        # clip actions so the agent can't go crazy
        action = np.clip(action, -1.0, 1.0)

        # advance the simulation
        obs = self.world.step(action)

        # figure out what happened
        torso_x = self.world.ragdoll.get_torso_x()
        torso_angle = self.world.ragdoll.get_torso_angle()
        fallen = self.world.ragdoll.is_fallen()

        # ── reward computation ───────────────────────────────────────
        # 4 components: forward velocity, alive bonus, energy penalty, fall penalty.
        # no explicit uprightness reward — balance emerges from foot contacts
        # and the devastating fall penalty.

        # forward velocity (pixels per physics step)
        forward_velocity = torso_x - self.prev_torso_x

        # superlinear velocity scaling: v * (1 + bonus * |v|)
        # makes running more rewarding than walking — doubling speed
        # more than doubles the reward.
        vbs = self.config["velocity_bonus_scale"]
        scaled_velocity = forward_velocity * (
            1.0 + vbs * abs(forward_velocity)
        )

        forward_reward = (
            scaled_velocity * self.config["forward_reward_scale"]
        )

        # compounding survival bonus: later steps worth more.
        # step 0 -> 1.0x, step 999 -> ~2.0x.  full-episode total: ~1500.
        # makes the last 500 steps worth more than the first 500.
        max_steps = self.config["max_episode_steps"]
        alive_bonus = self.config["alive_bonus"] * (
            1.0 + self.step_count / max_steps
        )

        # energy penalty
        energy_penalty = (
            -self.config["energy_penalty_scale"] * float(np.sum(action ** 2))
        )

        # fall penalty
        fall_penalty = self.config["fall_penalty"] if fallen else 0.0

        total_reward = (
            forward_reward + alive_bonus + energy_penalty + fall_penalty
        )

        # ── episode termination ──────────────────────────────────────
        self.step_count += 1
        done = fallen or self.step_count >= self.config["max_episode_steps"]

        # save torso x for next step's velocity calculation
        self.prev_torso_x = torso_x

        # ── info dict for debugging / logging ────────────────────────
        info = {
            "torso_x": torso_x,
            "torso_angle": torso_angle,
            "fell": fallen,
            "episode_steps": self.step_count,
            "reward_breakdown": {
                "forward_reward": forward_reward,
                "alive_bonus": alive_bonus,
                "energy_penalty": energy_penalty,
                "fall_penalty": fall_penalty,
            },
        }

        return obs, total_reward, done, info

    def get_body_positions(self):
        """Passthrough to physics world for rendering."""
        return self.world.get_body_positions()
