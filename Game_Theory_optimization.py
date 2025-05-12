from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
import numpy as np
import random

class GTOPokerAI1(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.aggressiveness = 0.5
        self.wins = 0
        self.losses = 0
        self.round_count = 0
        self.recent_results = []

    def adjust_aggressiveness(self):
        recent_win_rate = sum(self.recent_results) / len(self.recent_results) if self.recent_results else 0.5
        if recent_win_rate > 0.6:
            self.aggressiveness += 0.02
        elif recent_win_rate < 0.4:
            self.aggressiveness -= 0.02
        self.aggressiveness = max(0.1, min(2.0, self.aggressiveness))

    def monte_carlo_win_rate(self, hole_card, community_card, nb_simulation=1000, nb_player=2):
        hole = gen_cards(list(hole_card))  # Ensure lists
        community = gen_cards(list(community_card))
        return estimate_hole_card_win_rate(nb_simulation, hole, community, nb_player)

    def declare_action(self, valid_actions, hole_card, round_state):
        self.adjust_aggressiveness()
        win_rate = self.monte_carlo_win_rate(hole_card, round_state['community_card'])
        pot_size = round_state['pot']['main']['amount']
        amount_to_call = valid_actions[1]['amount']

        # Early-game aggressive play (Preflop: First 3 rounds)
        if self.round_count <= 3:
            if win_rate >= 0.45:
                return valid_actions[2]['action'], valid_actions[2]['amount']['max']
            elif 0.3 <= win_rate < 0.45:
                bet = random.randint(int(0.3 * valid_actions[2]['amount']['max']), int(0.5 * valid_actions[2]['amount']['max']))
                return valid_actions[2]['action'], bet
            elif 0.2 <= win_rate < 0.3:
                return valid_actions[1]['action'], amount_to_call
            else:
                return valid_actions[0]['action'], 0

        # Mid & Late-game (Flop, Turn, River)
        if len(round_state['community_card']) >= 3:
            if win_rate >= 0.65:
                return valid_actions[2]['action'], valid_actions[2]['amount']['max']
            elif 0.35 <= win_rate < 0.65:
                if random.random() < 0.1:
                    bet = random.randint(int(0.1 * valid_actions[2]['amount']['max']), int(0.3 * valid_actions[2]['amount']['max']))
                    return valid_actions[2]['action'], bet
                return valid_actions[1]['action'], amount_to_call
            elif 0.2 <= win_rate < 0.35:
                return valid_actions[1]['action'], amount_to_call
            else:
                return valid_actions[0]['action'], 0

        # Turn & River: More conservative, max raise only with very strong hands
        if win_rate >= 0.7:
            return valid_actions[2]['action'], valid_actions[2]['amount']['max']
        elif win_rate >= 0.5:
            return valid_actions[2]['action'], random.randint(int(0.2 * valid_actions[2]['amount']['max']), int(0.4 * valid_actions[2]['amount']['max']))
        elif win_rate >= 0.3:
            return valid_actions[1]['action'], amount_to_call
        return valid_actions[0]['action'], 0

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count += 1

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.uuid == winners[0]['uuid']:
            self.wins += 1
        else:
            self.losses += 1

