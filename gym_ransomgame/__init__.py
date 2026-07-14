from gymnasium.envs.registration import register

# -------- Version 0 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id="ransomgame-minimal_defense-v0",
    entry_point="gym_ransomgame.envs:RansomGameMinimalDefenseV0Env",
    kwargs={"ransomgame_config": None, "save_dir": None, "initial_state_path": None},
)

register(
    id="ransomgame-minimal_attack-v0",
    entry_point="gym_ransomgame.envs:RansomGameMinimalAttackV0Env",
    kwargs={"ransomgame_config": None, "save_dir": None, "initial_state_path": None},
)
