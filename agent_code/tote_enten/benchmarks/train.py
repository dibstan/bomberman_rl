import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Coins
    self.coins = []
    self.coins_temp = 0

    # Kills 
    self.kills = []
    self.kills_temp = 0

    # Deaths
    self.deaths = []
    self.deaths_temp = 0

    # Bombs 
    self.bombs = []
    self.bombs_temp = 0

    # Destroyed crates
    self.crates_destroyed = []
    self.crates_destroyed_temp = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # Coins
    if 'COIN_COLLECTED' in events:
        self.coins_temp += 1
    
    if 'KILLED_OPPONENT' in events:
        self.kills_temp += 1

    if 'KILLED_SELF' in events or 'GOT_KILLED' in events:
        self.deaths_temp += 1

    if 'BOMB_DROPPED' in events:
        self.bombs_temp += 1

    if 'CATE_DESTROYED' in events:
        self.crates_destroyed_temp += 1   


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # update benchmarks with events
    if 'COIN_COLLECTED' in events:
        self.coins_temp += 1

    if 'KILLED_OPPONENT' in events:
        self.kills_temp += 1

    if 'KILLED_SELF' in events or 'GOT_KILLED' in events:
        self.deaths_temp += 1

    if 'BOMB_DROPPED' in events:
        self.bombs_temp += 1

    if 'CRATE_DESTROYED' in events:
        self.crates_destroyed_temp += 1   

    # save the benchmarks
    self.coins.append(self.coins_temp)
    with open("coins.pt", "wb") as file:
        pickle.dump(self.coins, file)
    self.coins_temp = 0

    self.kills.append(self.kills_temp)
    with open("kills.pt", "wb") as file:
        pickle.dump(self.kills, file)
    self.kills_temp = 0

    self.deaths.append(self.deaths_temp)
    with open("deaths.pt", "wb") as file:
        pickle.dump(self.deaths, file)
    self.deaths_temp = 0

    self.bombs.append(self.bombs_temp)
    with open("bombs.pt", "wb") as file:
        pickle.dump(self.bombs, file)
    self.bombs_temp = 0

    self.crates_destroyed.append(self.crates_destroyed_temp)
    with open("crates_destroyed.pt", "wb") as file:
        pickle.dump(self.crates_destroyed, file)
    self.crates_destroyed_temp = 0


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
