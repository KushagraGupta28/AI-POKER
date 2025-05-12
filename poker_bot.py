import numpy as np
import random
from math import floor
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

def fix_cards(cards):
    """
    Convert card representations from formats like "10s" (3 characters)
    to "Ts" (2 characters). If already formatted, leave unchanged.
    """
    fixed = []
    for card in cards:
        card = str(card).strip()
        if len(card) == 2:
            fixed.append(card)
        elif len(card) == 3 and card.startswith("10"):
            fixed.append("T" + card[2])
        else:
            raise ValueError("Invalid card format: " + card)
    return fixed

class GTOPokerAIm4(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        # CFR data structures: regret sums and cumulative strategy sums.
        self.regret_sum = {}
        self.strategy_sum = {}
        self.current_round_history = []  # Stores decision trajectory for regret updates

        # Heuristic parameters and performance counters.
        self.aggressiveness = 0.5
        self.wins = 0
        self.losses = 0
        self.round_count = 0
        self.iteration = 0

        # Monte Carlo simulation and CFR hyperparameters.
        self.mc_simulations = 1000
        self.initial_cfr_epsilon = 0.2  # Initial exploration parameter.
        self.cfr_epsilon = self.initial_cfr_epsilon
        self.initial_heuristic_weight = 0.5
        self.stack = 1000
        self.regret_decay = 0.999  # Decay factor to gradually reduce influence of old regrets.

    def monte_carlo_win_rate(self, hole_card, community_card, nb_simulation=1000, nb_player=2):
        if not isinstance(hole_card, list):
            hole_card = [hole_card]
        if not isinstance(community_card, list):
            community_card = []
        if not all(isinstance(card, str) and len(card) == 2 for card in hole_card):
            fixed_hole = fix_cards(hole_card)
        else:
            fixed_hole = hole_card.copy()
        if not all(isinstance(card, str) and len(card) == 2 for card in community_card):
            fixed_community = fix_cards(community_card)
        else:
            fixed_community = community_card.copy()
        hole = gen_cards(fixed_hole)
        community = gen_cards(fixed_community)
        return estimate_hole_card_win_rate(nb_simulation, hole, community, nb_player)

    def adjust_for_board_texture(self, community_card):
        if not community_card:
            return 0
        suits = [card[-1] for card in community_card]
        values = []
        for card in community_card:
            rank = card[:-1]
            if rank.isdigit():
                values.append(int(rank))
            else:
                rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11}
                if rank in rank_map:
                    values.append(rank_map[rank])
        if max(suits.count(suit) for suit in suits) >= 4:
            return -0.15
        unique_values = sorted(set(values))
        if len(unique_values) >= 4 and unique_values[-1] - unique_values[0] <= 4:
            return -0.10
        if len(set(values)) <= len(values) - 1:
            return -0.20
        return 0

    def adjust_aggressiveness(self, round_state):
        self.round_count += 1
        self.aggressiveness += 0.02
        if self.losses > self.wins:
            self.aggressiveness *= 0.9
        elif self.wins > self.losses:
            self.aggressiveness *= 1.05
        button_pos = round_state['dealer_btn']
        my_pos = next(i for i, p in enumerate(round_state['seats']) if p['uuid'] == self.uuid)
        if my_pos == (button_pos + 1) % len(round_state['seats']):
            self.aggressiveness *= 1.2
        elif my_pos == (button_pos + 2) % len(round_state['seats']):
            self.aggressiveness *= 1.1
        else:
            self.aggressiveness *= 0.9
        self.aggressiveness = max(0.1, min(2.0, self.aggressiveness))

    def evaluate_hand_strength(self, hole_card, community_card):
        win_rate = self.monte_carlo_win_rate(hole_card, community_card, nb_simulation=self.mc_simulations)
        adjustment = self.adjust_for_board_texture(community_card)
        return win_rate + adjustment

    def _create_info_set(self, hole_card, round_state):
        community = round_state['community_card']
        win_rate = self.evaluate_hand_strength(hole_card, community)
        pot_commitment = round_state['pot']['main']['amount'] / (self.stack + 1e-6)
        win_bucket = int(win_rate // 0.1)
        pot_bucket = int(pot_commitment // 0.2)
        sorted_hole = tuple(sorted(hole_card))
        street = round_state['street']
        return (sorted_hole, win_bucket, pot_bucket, street)

    def _get_strategy(self, info_set):
        if info_set not in self.regret_sum:
            return np.ones(3) / 3
        regrets = np.maximum(self.regret_sum[info_set], 0)
        total = regrets.sum()
        if total > 0:
            strategy = regrets / total
        else:
            strategy = np.ones(3) / 3
        # Dynamic exploration: decaying epsilon over iterations.
        self.cfr_epsilon = self.initial_cfr_epsilon * np.exp(-0.0001 * self.iteration)
        return (1 - self.cfr_epsilon) * strategy + self.cfr_epsilon * (np.ones(3) / 3)

    def _blend_strategies(self, cfr_strategy, win_rate, round_state, heuristic_weight):
        street_weights = {'preflop': 0.4, 'flop': 0.6, 'turn': 0.7, 'river': 0.8}
        base_weight = street_weights.get(round_state['street'], 0.5)
        cfr_weight = base_weight * (1 - heuristic_weight)
        hr_strategy = self._win_rate_strategy(win_rate, round_state)
        blended = cfr_weight * cfr_strategy + (1 - cfr_weight) * hr_strategy
        total = blended.sum()
        if total == 0:
            return np.ones(3) / 3
        return blended / total

    def _win_rate_strategy(self, win_rate, round_state):
        pot_odds = self._calculate_pot_odds(round_state)
        if round_state['street'] == 'preflop':
            if win_rate >= 0.5:
                return np.array([0.0, 0.2, 0.8])
            elif win_rate >= 0.3:
                return np.array([0.0, 0.7, 0.3])
            else:
                return np.array([1.0, 0.0, 0.0])
        else:
            if win_rate > pot_odds + 0.1:
                return np.array([0.0, 0.4, 0.6])
            elif win_rate > pot_odds:
                return np.array([0.0, 1.0, 0.0])
            else:
                return np.array([1.0, 0.0, 0.0])

    def _map_action(self, action_idx, valid_actions, hole_card, round_state):
        if action_idx == 2:
            win_rate = self.evaluate_hand_strength(hole_card, round_state['community_card'])
            min_r = valid_actions[2]['amount']['min']
            max_r = valid_actions[2]['amount']['max']
            raise_amount = min_r + (max_r - min_r) * (0.3 + win_rate * 0.7)
            return 'raise', int(raise_amount * self.aggressiveness)
        return [('fold', 0), ('call', valid_actions[1]['amount']), ('raise', 0)][action_idx]

    def _calculate_win_rate(self, hole_card, community):
        fixed_hole = hole_card if all(isinstance(card, str) and len(card) == 2 for card in hole_card) else fix_cards(hole_card)
        fixed_community = community if all(isinstance(card, str) and len(card) == 2 for card in community) else fix_cards(community)
        return estimate_hole_card_win_rate(self.mc_simulations, 2, gen_cards(fixed_hole), gen_cards(fixed_community))

    def _calculate_pot_odds(self, round_state):
        call_amount = round_state['pot']['main']['amount']
        return call_amount / (call_amount + self.stack)

    def declare_action(self, valid_actions, hole_card, round_state):
        self.adjust_aggressiveness(round_state)
        info_set = self._create_info_set(hole_card, round_state)
        cfr_strategy = self._get_strategy(info_set)
        win_rate = self.evaluate_hand_strength(hole_card, round_state['community_card'])
        heuristic_weight = self.initial_heuristic_weight * np.exp(-0.001 * self.iteration)
        blended_strategy = self._blend_strategies(cfr_strategy, win_rate, round_state, heuristic_weight)
        action_idx = np.random.choice(3, p=blended_strategy)
        action, amount = self._map_action(action_idx, valid_actions, hole_card, round_state)
        self.current_round_history.append({
            'info_set': info_set,
            'strategy': cfr_strategy,
            'action': action_idx,
            'reach_prob': 1.0
        })
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = np.zeros(3)
        self.strategy_sum[info_set] += cfr_strategy
        return action, amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        utility = 1 if self.uuid in [w['uuid'] for w in winners] else -1
        self._update_regrets(utility)
        self.current_round_history = []
        self.iteration += 1

    def _update_regrets(self, utility):
        cumulative_reach = 1.0
        for info_set in self.regret_sum:
            self.regret_sum[info_set] *= self.regret_decay
        for node in reversed(self.current_round_history):
            info_set = node['info_set']
            action = node['action']
            strategy = node['strategy']
            if info_set not in self.regret_sum:
                self.regret_sum[info_set] = np.zeros(3)
            for a in range(3):
                regret = utility * cumulative_reach * ((1 if a == action else 0) - strategy[a])
                self.regret_sum[info_set][a] += regret / max(strategy[a], 1e-6)
            cumulative_reach *= strategy[action]

    # ---- PyPokerEngine required methods ----
    def receive_game_start_message(self, game_info):
        self.player_num = game_info['player_num']
        self.opponent_stacks = [s['stack'] for s in game_info['seats'] if s['uuid'] != self.uuid]
        self.stack = next(s['stack'] for s in game_info['seats'] if s['uuid'] == self.uuid)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.current_round_history = []
        self.stack = next(s['stack'] for s in seats if s['uuid'] == self.uuid)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def set_uuid(self, uuid):
        self.uuid = uuid

    def respond_to_ask(self, message):
        valid_actions, hole_card, round_state = self.__parse_ask_message(message)
        return self.declare_action(valid_actions, hole_card, round_state)

    def receive_notification(self, message):
        msg_type = message["message_type"]
        if msg_type == "game_start_message":
            info = self.__parse_game_start_message(message)
            self.receive_game_start_message(info)
        elif msg_type == "round_start_message":
            round_count, hole, seats = self.__parse_round_start_message(message)
            self.receive_round_start_message(round_count, hole, seats)
        elif msg_type == "street_start_message":
            street, state = self.__parse_street_start_message(message)
            self.receive_street_start_message(street, state)
        elif msg_type == "game_update_message":
            new_action, round_state = self.__parse_game_update_message(message)
            self.receive_game_update_message(new_action, round_state)
        elif msg_type == "round_result_message":
            winners, hand_info, state = self.__parse_round_result_message(message)
            self.receive_round_result_message(winners, hand_info, state)

    def __parse_ask_message(self, message):
        return message["valid_actions"], message["hole_card"], message["round_state"]

    def __parse_game_start_message(self, message):
        return message["game_information"]

    def __parse_round_start_message(self, message):
        return message["round_count"], message["hole_card"], message["seats"]

    def __parse_street_start_message(self, message):
        return message["street"], message["round_state"]

    def __parse_game_update_message(self, message):
        return message["action"], message["round_state"]

    def __parse_round_result_message(self, message):
        return message["winners"], message["hand_info"], message["round_state"]