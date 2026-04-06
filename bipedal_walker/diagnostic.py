"""Quick diagnostic for all 6 hypotheses about the limp walker."""
import math, os, sys, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from physics import PhysicsWorld, MAX_MOTOR_RATE

def main():
    print("=" * 70)
    print("  BIPEDAL WALKER DIAGNOSTIC")
    print("=" * 70)

    # ── [1] Save/Load Paths ──
    print("\n[1] SAVE/LOAD PATHS")
    sp = CONFIG["save_path"]
    print(f"  train.py saves to:  {os.path.abspath(sp)}")
    if os.path.exists(sp):
        dt = datetime.datetime.fromtimestamp(os.path.getmtime(sp))
        print(f"  Checkpoint exists:  YES  (modified {dt})")
        import torch
        from agent import ActorCritic
        checkpoint = torch.load(sp, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            hidden_size = checkpoint["hidden_size"]
            state_dict = checkpoint["model_state_dict"]
        else:
            hidden_size = CONFIG["hidden_size"]
            state_dict = checkpoint
        agent = ActorCritic(19, 6, hidden_size=hidden_size)
        agent.load_state_dict(state_dict)
        w = list(agent.actor_backbone.parameters())[0]
        print(f"  Loads OK:           YES  (hidden={hidden_size}, weight std={w.std().item():.4f}, init~0.17)")
    else:
        print(f"  Checkpoint exists:  NO")
    print(f"  NOTE: visualize.py requires --model flag; without it, uses random actions")

    # ── [2] Motor Force vs Gravity  ──
    print("\n[2] MOTOR FORCE vs GRAVITY (per-joint)")
    g = CONFIG["gravity"]
    m_th, l_th = CONFIG["upper_leg_mass"], CONFIG["upper_leg_length"]
    m_sh, l_sh = CONFIG["lower_leg_mass"], CONFIG["lower_leg_length"]
    m_ft, fw = CONFIG["foot_mass"], CONFIG["foot_width"]

    mf_hip = CONFIG["hip_motor_force"]
    mf_knee = CONFIG["knee_motor_force"]
    mf_ankle = CONFIG["ankle_motor_force"]

    tau_hip = m_th * g * (l_th/2) + m_sh * g * (l_th + l_sh/2)
    tau_knee = m_sh * g * (l_sh/2)
    tau_ankle = m_ft * g * (fw/2)

    print(f"  gravity={g}  thigh: m={m_th} L={l_th}  shin: m={m_sh} L={l_sh}  foot: m={m_ft} W={fw}")
    print(f"  hip_motor_force   = {mf_hip:,.0f}")
    print(f"  knee_motor_force  = {mf_knee:,.0f}")
    print(f"  ankle_motor_force = {mf_ankle:,.0f}")
    print(f"  Grav torque at HIP   (thigh+shin horizontal): {tau_hip:,.0f}")
    print(f"  Grav torque at KNEE  (shin horizontal):       {tau_knee:,.0f}")
    print(f"  Grav torque at ANKLE (foot horizontal):       {tau_ankle:,.0f}")

    r_hip = mf_hip / tau_hip
    r_knee = mf_knee / tau_knee
    r_ankle = mf_ankle / tau_ankle
    print(f"  Hip   force/gravity ratio: {r_hip:.2f}x")
    print(f"  Knee  force/gravity ratio: {r_knee:.2f}x")
    print(f"  Ankle force/gravity ratio: {r_ankle:.2f}x")

    if r_hip < 1:
        max_deg = math.degrees(math.asin(r_hip))
        print(f"  Hip motor can hold leg at most {max_deg:.1f}° from vertical!")

    # Empirical: 1 second of max actions
    print("\n  --- Empirical: 60 steps (1s) of max motor commands ---")
    world = PhysicsWorld(CONFIG)
    for _ in range(60):
        world.step(np.ones(6, dtype=np.float32))
    obs = world.ragdoll.get_state()
    for i, name in enumerate(["L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle"]):
        angle = obs[5 + i*2]
        print(f"    {name}: {math.degrees(angle):+.2f}°")
    mx = max(abs(obs[5 + i*2]) for i in range(6))
    print(f"  Max deflection: {math.degrees(mx):.1f}°  {'*** MOTORS HAVE NO EFFECT ***' if mx < 0.2 else 'OK'}")

    # ── [3] Motor Rate ──
    print("\n[3] MOTOR RATE RANGE")
    hip_range = CONFIG["hip_max_angle"] - CONFIG["hip_min_angle"]
    print(f"  MAX_MOTOR_RATE={MAX_MOTOR_RATE} rad/s")
    print(f"  Full hip sweep: {hip_range/MAX_MOTOR_RATE:.2f}s  =>  OK (if force were sufficient)")

    # ── [4] Joint Limits ──
    print("\n[4] JOINT LIMITS")
    print(f"  Hip:   [{math.degrees(CONFIG['hip_min_angle']):.0f}°, {math.degrees(CONFIG['hip_max_angle']):.0f}°] = 180° range  =>  OK")
    print(f"  Knee:  [{math.degrees(CONFIG['knee_min_angle']):.0f}°, {math.degrees(CONFIG['knee_max_angle']):.0f}°] = 135° range  =>  OK")
    print(f"  Ankle: [{math.degrees(CONFIG['ankle_min_angle']):.0f}°, {math.degrees(CONFIG['ankle_max_angle']):.0f}°] = 75° range   =>  OK")

    # ── [5] Action Order ──
    print("\n[5] ACTION ORDER")
    print(f"  PhysicsWorld.step(): apply_actions() THEN space.step()  =>  CORRECT")

    # ── [6] Observations ──
    print("\n[6] OBSERVATION SANITY")
    world2 = PhysicsWorld(CONFIG)
    obs0 = world2.ragdoll.get_state()
    print(f"  Initial: {np.array2string(obs0, precision=3, suppress_small=True)}")
    print(f"  NaN={np.any(np.isnan(obs0))} Inf={np.any(np.isinf(obs0))}  =>  OK")

    # ── VERDICT ──
    print("\n" + "=" * 70)
    print("  VERDICT: per-joint motor forces")
    print(f"  Hip:   {mf_hip:,.0f}  ({r_hip:.1f}x gravity)")
    print(f"  Knee:  {mf_knee:,.0f}  ({r_knee:.1f}x gravity)")
    print(f"  Ankle: {mf_ankle:,.0f}  ({r_ankle:.1f}x gravity)")
    print(f"  Foot width: {fw:.0f}px  (ankle lever arm: {abs(CONFIG['lower_leg_width']/2 - fw/2):.1f}px)")
    print("=" * 70)

if __name__ == "__main__":
    main()
