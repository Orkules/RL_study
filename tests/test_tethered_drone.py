from pathlib import Path
import numpy as np
import torch

from rl.ppo_agent import ActorCritic
from sim.tethered_drone_env import TetheredDroneEnv


def test_env_spaces_and_telemetry():
    env = TetheredDroneEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (6,)
    assert env.action_space.shape == (3,)
    assert len(info["wire_tensions_n"]) == 4
    assert len(info["lugs_world"]) == 4
    assert info["com_world"][2] == 0.45  # suspended off the ground


def test_fin_slipstream_rejects_yaw_spin():
    env = TetheredDroneEnv(terminate_on_wire_limit=False)
    env.reset(
        seed=7,
        options={"initial_angles_deg": [0.0, 0.0, 0.0], "initial_omega": [0.0, 0.0, 6.0]},
    )
    initial_yaw_rate = abs(env.omega[2])
    for _ in range(150):
        # High motor speed generates strong slipstream over the 4 anti-spin fins
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.25], dtype=np.float32))
    final_yaw_rate = abs(env.omega[2])
    assert info["fin_airflow_ms"] > 10.0
    assert final_yaw_rate < initial_yaw_rate * 0.35


def test_trained_checkpoint_stabilizes_from_tilt():
    ckpt_random = Path("checkpoints/stage_0_random.pt")
    ckpt_trained = Path("checkpoints/stage_4_stabilized.pt")
    assert ckpt_random.exists() and ckpt_trained.exists()

    def run_rollout(ckpt_path: Path) -> float:
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_dim=64)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        env = TetheredDroneEnv(terminate_on_wire_limit=False)
        obs, _ = env.reset(
            seed=123,
            options={
                "initial_angles_deg": [22.0, -18.0, 12.0],
                "initial_omega": [1.2, -1.0, 0.8],
            },
        )
        info = {}
        for _ in range(400):
            action = model.act_deterministic(obs)
            obs, _, _, _, info = env.step(action)
        return float(info["tilt_deg"])

    tilt_untrained = run_rollout(ckpt_random)
    tilt_stabilized = run_rollout(ckpt_trained)
    assert tilt_untrained > 30.0
    assert tilt_stabilized < 2.0
