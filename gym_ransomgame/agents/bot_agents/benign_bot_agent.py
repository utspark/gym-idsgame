"""
A bot agent for the gym-ransomgame environment that stands in for the attacker on
benign episodes.
"""

from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_ransomgame.envs.dao.game_config import GameConfig
from gym_ransomgame.envs.dao.game_state import GameState


class BenignBotAgent(BotAgent):
    """
    Class implementing the behaviour of a system on which no ransomware is running.

    The environment plays this agent in place of the attacker whenever nature draws a
    benign episode, so it is what the defender has to learn to leave alone. Making it an
    agent rather than a special case inside the attacker policies means benign behaviour
    is defined in exactly one place, and it leaves room to grow: a benign workload that
    touches files and moves data would drive the detector scores that currently sit
    unused on GameState, which is what makes the defender's problem non-trivial.

    It samples a random benign background action (browsing, file I/O, media, spec
    benchmarks) each step. None of these carry a stage, so a benign episode still
    leaves every stage and progress bar at zero and runs until the step cap - only the
    detector-facing trace behind each step now varies instead of always being idle.
    """

    def __init__(self, game_config: GameConfig):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(BenignBotAgent, self).__init__(game_config)

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        return int(game_state.np_random.choice(GameState.BENIGN_ACTIONS))
