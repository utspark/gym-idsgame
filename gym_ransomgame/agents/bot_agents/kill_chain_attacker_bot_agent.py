"""
A deterministic bot attack agent for the gym-ransomgame environment that walks the
ransomware kill chain in order.
"""

from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_ransomgame.envs.dao.game_config import GameConfig
from gym_ransomgame.envs.dao.game_state import AttackType, GameState


class KillChainAttackerBotAgent(BotAgent):
    """
    Class implementing a deterministic attack policy that advances to the lowest
    incomplete stage (reconnaissance -> compression -> exfiltration -> encryption).

    Stage completion is probabilistic, so the number of steps spent on each stage varies
    between episodes even though the policy itself is deterministic.

    This is a pure kill-chain policy and has no benign mode: the env never consults an
    attacker policy on a benign episode, it plays BenignBotAgent instead.
    """

    def __init__(self, game_config: GameConfig):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(KillChainAttackerBotAgent, self).__init__(game_config)

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy: the fixed representative action for the
        lowest incomplete stage (recon_mount, compress_gzip_1t, transfer_aws_1t,
        symm_AES_128b), or IDLE once every stage is done.

        :param game_state: the game state
        :return: action_id
        """
        stages = game_state.stages[0]
        stage = next((i for i, done_stage in enumerate(stages) if not done_stage), None)
        if stage is None:
            return AttackType.IDLE
        return GameState.STAGE_ACTIONS[stage][0]
