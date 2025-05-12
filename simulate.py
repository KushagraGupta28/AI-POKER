from pypokerengine.api.game import setup_config, start_poker  # type: ignore
from gto1 import GTOPokerAI1  # type: ignore
from gto2 import GTOPokerAI2  # type: ignore

NUM_SIMULATIONS = 20

def simulate_single_game():
    player1 = GTOPokerAI1()
    player2 = GTOPokerAI2()

    config = setup_config(max_round=15, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="Player1", algorithm=player1)
    config.register_player(name="Player2", algorithm=player2)

    result = start_poker(config, verbose=1)
    return result

def calculate_win_percentage(results):
    win_counts = {}
    total_stacks = {}

    for result in results:
        players = result['players']
        sorted_players = sorted(players, key=lambda x: x['stack'], reverse=True)
        winner = sorted_players[0]['name']

        win_counts.setdefault(winner, 0)
        win_counts[winner] += 1

        for player in players:
            player_name = player['name']
            total_stacks.setdefault(player_name, 0)
            total_stacks[player_name] += player['stack']
    
    total_games = len(results)
    win_percentages = {name: (win_counts.get(name, 0) / total_games) * 100 for name in total_stacks}

    print("\n=== FINAL RESULTS ===")
    for name in total_stacks:
        print(f"{name} Win Percentage: {win_percentages.get(name, 0):.2f}%")
    print("\nTotal Stacks:")
    for name in total_stacks:
        print(f"{name}: {total_stacks[name]}")

def run_simulations(num_simulations):
    results = []
    for i in range(num_simulations):
        print(f"Starting Simulation {i + 1}/{num_simulations}...")
        result = simulate_single_game()
        results.append(result)

    calculate_win_percentage(results)

if __name__ == "__main__":
    run_simulations(NUM_SIMULATIONS)