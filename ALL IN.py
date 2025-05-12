from pypokerengine.players import BasePokerPlayer # type: ignore
from pypokerengine.utils.card_utils import gen_cards# type: ignore

class AllInPokerBot(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        # No additional state is needed for an all-in bot

    def declare_action(self, valid_actions, hole_card, round_state):

        # Look for the raise action among valid_actions
        raise_actions = [action for action in valid_actions if action['action'] == 'raise']
        if raise_actions:
            # Always bet the maximum allowed (simulate an all-in)
            all_in_amount = raise_actions[0]['amount']['max']
            return raise_actions[0]['action'], all_in_amount
        else:
            # If raise is not available, fall back to call
            call_actions = [action for action in valid_actions if action['action'] == 'call']
            if call_actions:
                return call_actions[0]['action'], call_actions[0].get('amount', 0)
            else:
                # Otherwise, fold (should rarely occur in a normal game setting)
                return valid_actions[0]['action'], 0

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def set_uuid(self, uuid):
        self.uuid = uuid