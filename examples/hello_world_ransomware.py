import gymnasium as gym
from gym_ransomgame.envs import RansomGameEnv


def attack_against_baseline_defense_env(render=False, seed=None, pause=False):
    versions = range(0, 20)
    version = versions[0]
    env_name = "ransomgame-minimal_defense-v" + str(version)
    env = gym.make(env_name)
    env.reset(seed=seed)
    done = False
    while not done:
        if render:
            env.render()
        if pause:
            input("Press Enter to continue...")
        attack_action = env.unwrapped.attacker_action_space.sample()
        defense_action = None
        a = (attack_action, defense_action)
        obs, reward, done, truncated, info = env.step(a)
        print(f"Action: {attack_action}, Reward: {reward}, Done: {done}, Info: {info}")
        print(f"Obs: {obs}")
        print("\n")


def main():
    render = False
    seed = 7
    pause = False
    attack_against_baseline_defense_env(render=render, seed=seed, pause=pause)

if __name__ == '__main__':
    main()