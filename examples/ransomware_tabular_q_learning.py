from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_ransomgame.envs
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_ransomgame.agents.training_agents.q_learning.tabular_q_learning.ransom_tabular_q_agent import (
    RansomTabularQAgent,
)

SCRIPT_DIR = Path(__file__).parent


def create_artefact_dirs(output_dir: Path, random_seed: int) -> None:
    for subdir in [
        output_dir / "results" / "data" / str(random_seed),
        output_dir / "results" / "videos" / str(random_seed),
        output_dir / "results" / "gifs" / str(random_seed),
    ]:
        subdir.mkdir(parents=True, exist_ok=True)


def make_config(output_dir: Path, random_seed: int, attacker: bool) -> QAgentConfig:
    results_dir = output_dir / "results"
    return QAgentConfig(
        gamma=0.999,
        alpha=0.0005,
        epsilon=1,
        render=False,
        eval_sleep=0.9,
        min_epsilon=0.01,
        eval_episodes=30,
        train_log_frequency=30,
        epsilon_decay=0.9999,
        video=False,
        eval_log_frequency=1,
        video_fps=5,
        video_dir=str(results_dir / "videos" / str(random_seed)),
        num_episodes=6001,
        eval_render=False,
        gifs=True,
        gif_dir=str(results_dir / "gifs" / str(random_seed)),
        eval_frequency=300,
        attacker=attacker,
        defender=not attacker,
        video_frequency=101,
        save_dir=str(results_dir / "data" / str(random_seed)),
    )


def main() -> None:
    random_seed = 0
    load_q_table = False
    attacker = False

    create_artefact_dirs(SCRIPT_DIR, random_seed)
    config = make_config(SCRIPT_DIR, random_seed, attacker=attacker)

    if attacker:
        env_name = "ransomgame-minimal_defense-v0"
        table_tag = "1784037638.6862357"
        q_table_filename = f"{table_tag}_attacker_q_table.npy"
    else:
        env_name = "ransomgame-minimal_attack-v0"
        q_table_filename = None

    env = gym.make(env_name, save_dir=config.save_dir)
    agent = RansomTabularQAgent(env.unwrapped, config)

    if load_q_table and q_table_filename:
        agent.Q_attacker = np.load(Path(agent.config.save_dir) / q_table_filename)
    else:
        agent.train()

    train_result = agent.train_result
    eval_result = agent.eval_result


if __name__ == "__main__":
    main()
