import gymnasium as gym

def attack_against_baseline_defense_env(render=False, seed=None, pause=False):
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-minimal_defense-v" + str(version)
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
        obs, reward, done, info = env.step(a)


def main():
    render = True
    seed = 0
    pause = True

    attack_against_baseline_defense_env(render=render, seed=seed, pause=pause)

if __name__ == '__main__':
    main()