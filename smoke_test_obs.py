#!/usr/bin/env python3
"""
smoke_test_obs.py
-----------------
Verifies that the policy checkpoint accepts the 85-D observation vector
constructed by rl_env_node.py WITHOUT needing a live robot.

Run:
    python3 smoke_test_obs.py BASELINE
    python3 smoke_test_obs.py SPD
    python3 smoke_test_obs.py BASELINE --checkpoint checkpoints/BASELINE_best_success_model.zip
"""
import sys, os, argparse
import numpy as np

GRIP_SITE_OFFSET = np.array([0.7018, -0.0067, 0.0865, 0.7071], dtype=np.float32)

OBS_SEGMENTS = [
    ("joint_pos",               7),
    ("cos(joint_pos)",          7),
    ("sin(joint_pos)",          7),
    ("joint_vel",               7),
    ("joint_acc",               7),
    ("eef_pos (eraser tip)",    3),
    ("eef_quat (flange)",       4),
    ("eef_quat_site (grip)",    4),
    ("robot0_contact",          1),
    ("marker0_pos",             3), ("marker0_wiped", 1), ("gripper_to_marker0", 3),
    ("marker1_pos",             3), ("marker1_wiped", 1), ("gripper_to_marker1", 3),
    ("marker2_pos",             3), ("marker2_wiped", 1), ("gripper_to_marker2", 3),
    ("marker3_pos",             3), ("marker3_wiped", 1), ("gripper_to_marker3", 3),
    ("marker4_pos",             3), ("marker4_wiped", 1), ("gripper_to_marker4", 3),
    ("gripper_to_active_wp",    3),
]
EXPECTED_DIM = sum(d for _, d in OBS_SEGMENTS)


def build_fake_obs():
    from scipy.spatial.transform import Rotation as R
    joint_pos = np.random.uniform(-2.0, 2.0, 7).astype(np.float32)
    joint_vel = np.random.uniform(-0.5, 0.5, 7).astype(np.float32)
    joint_acc = np.zeros(7, dtype=np.float32)
    eef_pos   = np.array([0.55, 0.0, 0.10], dtype=np.float32)
    eef_quat  = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    r_eef  = R.from_quat(eef_quat)
    r_site = R.from_quat(GRIP_SITE_OFFSET)
    eef_quat_site = (r_eef * r_site).as_quat().astype(np.float32)
    eef_pos_eraser = eef_pos + r_eef.apply([0.0, 0.0, 0.185])
    robot0_contact = np.array([0.0], dtype=np.float32)
    marker_positions = [
        np.array([0.60, -0.12, 0.02], dtype=np.float32),
        np.array([0.60,  0.00, 0.02], dtype=np.float32),
        np.array([0.60,  0.12, 0.02], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    ]
    marker_wiped = [
        np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]
    g2m = [mp - eef_pos_eraser for mp in marker_positions]
    active_idx = next(i for i, w in enumerate(marker_wiped) if w[0] < 0.5)
    obs_list = [joint_pos, np.cos(joint_pos), np.sin(joint_pos),
                joint_vel, joint_acc, eef_pos_eraser, eef_quat, eef_quat_site, robot0_contact]
    for i in range(5):
        obs_list += [marker_positions[i], marker_wiped[i], g2m[i]]
    obs_list.append(g2m[active_idx].copy())
    return np.concatenate(obs_list).astype(np.float32), active_idx


def print_layout(obs):
    print(f"\n{'─'*65}")
    print(f"{'Segment':<32} {'Slice':>10}  Values")
    print(f"{'─'*65}")
    offset = 0
    for name, dim in OBS_SEGMENTS:
        vals = obs[offset: offset + dim]
        val_str = "  ".join(f"{v:+.3f}" for v in vals)
        print(f"{name:<32} [{offset:3d}:{offset+dim:3d}]  {val_str}")
        offset += dim
    print(f"{'─'*65}")
    print(f"Total: {offset}-D\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="BASELINE")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            f"checkpoints/SAC_WIPE_ICRA_{args.config}/best_model",
            f"checkpoints/{args.config}_best_success_model.zip",
        ]
        for c in candidates:
            if os.path.exists(c) or os.path.exists(c + ".zip"):
                ckpt_path = c; break
        if ckpt_path is None:
            print(f"[ERROR] Cannot find checkpoint for '{args.config}'.")
            print(f"  Tried: {candidates}")
            sys.exit(1)

    print(f"\n[SMOKE] Loading: {ckpt_path}")
    try:
        from stable_baselines3 import SAC
        model = SAC.load(ckpt_path, device='cpu')
    except Exception as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    pol_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[0]
    print(f"[SMOKE] Policy obs_dim={pol_dim}  act_dim={act_dim}")
    print(f"[CHECK] Expected obs_dim={EXPECTED_DIM}")

    if pol_dim == EXPECTED_DIM:
        print(f"  ✓ MATCH — {EXPECTED_DIM}-D")
    else:
        print(f"  ✗ MISMATCH — policy={pol_dim} vs built={EXPECTED_DIM}")
        sys.exit(1)

    obs, active_idx = build_fake_obs()
    print(f"[SMOKE] Fake obs built: shape={obs.shape}  active_marker={active_idx}")
    print_layout(obs)

    action, _ = model.predict(obs, deterministic=True)
    print(f"[SMOKE] Action: shape={action.shape}  values={np.round(action, 4)}")
    if np.all(np.abs(action) <= 1.05):
        print("  ✓ Action values within expected range")
    else:
        print("  ⚠ Some action values outside [-1,1] — check policy scaling")

    print("\n[SMOKE] ✓ Passed. Policy ready for deployment.\n")

if __name__ == "__main__":
    main()
