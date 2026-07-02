import gymnasium as gym
from typing import cast
from gym_idsgame.envs import IdsGameEnv

def attack_against_baseline_defense_env(render=False, seed=None, pause=False):
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-minimal_defense-v" + str(version)
    env = gym.make(env_name)
    unwrapped_env = cast(IdsGameEnv, env.unwrapped)
    env.reset(seed=seed)
    done = False
    while not done:
        if render:
            env.render()
        if pause:
            input("Press Enter to continue...")
        attack_action = unwrapped_env.attacker_action_space.sample()
        defense_action = None
        a = (attack_action, defense_action)
        obs, reward, done, truncated, info = env.step(a)


def attack_against_random_defense_env(render=False, seed=None, pause=False):
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-random_defense-v" + str(version)
    env = gym.make(env_name)
    unwrapped_env = cast(IdsGameEnv, env.unwrapped)
    env.reset(seed=seed)
    done = False
    while not done:
        if render:
            env.render()
        if pause:
            input("Press Enter to continue...")
        attack_action = unwrapped_env.attacker_action_space.sample()
        defense_action = None
        a = (attack_action, defense_action)
        obs, reward, done, truncated, info = env.step(a)

def defense_against_baseline_attack_env(render=False, seed=None, pause=False):
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-maximal_attack-v" + str(version)
    env = gym.make(env_name)
    unwrapped_env = cast(IdsGameEnv, env.unwrapped)
    env.reset(seed=seed)
    done = False
    while not done:
        if render:
            env.render()
        if pause:
            input("Press Enter to continue...")
        attack_action = None
        defense_action = unwrapped_env.defender_action_space.sample()
        a = (attack_action, defense_action)
        obs, reward, done, truncated, info = env.step(a)


def defense_against_random_attack_env(render=False, seed=None, pause=False):
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-random_attack-v" + str(version)
    env = gym.make(env_name)
    unwrapped_env = cast(IdsGameEnv, env.unwrapped)
    env.reset(seed=seed)
    done = False
    while not done:
        if render:
            env.render()
        if pause:
            input("Press Enter to continue...")
        attack_action = None
        defense_action = unwrapped_env.defender_action_space.sample()
        a = (attack_action, defense_action)
        obs, reward, done, truncated, info = env.step(a)

def two_agents_env(render=False, seed=None, pause=False):
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-v" + str(version)
    env = gym.make(env_name)
    unwrapped_env = cast(IdsGameEnv, env.unwrapped)
    env.reset(seed=seed)
    done = False
    while not done:
        if render:
            env.render()
        if pause:
            input("Press Enter to continue...")
        attack_action = unwrapped_env.attacker_action_space.sample()
        defense_action = unwrapped_env.defender_action_space.sample()
        a = (attack_action, defense_action)
        obs, reward, done, truncated, info = env.step(a)

def main():
    render = True
    seed = 0
    pause = True

    attack_against_baseline_defense_env(render=render, seed=seed, pause=pause)
    # attack_against_random_defense_env(render=render, seed=seed, pause=pause)
    # defense_against_baseline_attack_env(render=render, seed=seed, pause=pause)
    # defense_against_random_attack_env(render=render, seed=seed, pause=pause)
    # two_agents_env(render=render, seed=seed, pause=pause)

if __name__ == '__main__':
    main()