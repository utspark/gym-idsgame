import os
import gymnasium as gym
import gym_ransomgame.envs
import sys
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_ransomgame.agents.training_agents.q_learning.tabular_q_learning.ransom_tabular_q_agent import RansomTabularQAgent

def create_artefact_dirs(output_dir, random_seed):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_dir = output_dir + "/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    data_dir = results_dir + "/data/" + str(random_seed)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    video_dir = results_dir + "/videos/" + str(random_seed)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    gif_dir = results_dir + "/gifs/" + str(random_seed)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir

# Program entrypoint
if __name__ == '__main__':
    random_seed = 0
    create_artefact_dirs(default_output_dir(), random_seed)
    q_agent_config = QAgentConfig(gamma=0.999, alpha=0.0005, epsilon=1, render=False, eval_sleep=0.9,
                                  min_epsilon=0.01, eval_episodes=30, train_log_frequency=30,
                                  epsilon_decay=0.9999, video=False, eval_log_frequency=1,
                                  video_fps=5, video_dir=default_output_dir() + "/results/videos/" + str(random_seed), num_episodes=6001,
                                  eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs/" + str(random_seed),
                                  eval_frequency=300, attacker=True, defender=False, video_frequency=101,
                                  save_dir=default_output_dir() + "/results/data/" + str(random_seed))
    env_name = "ransomgame-minimal_defense-v0"
    env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + str(random_seed))
    attacker_agent = RansomTabularQAgent(env.unwrapped, q_agent_config)
    attacker_agent.train()
    train_result = attacker_agent.train_result
    eval_result = attacker_agent.eval_result