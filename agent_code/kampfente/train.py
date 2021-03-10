import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

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
    #self.history = {'UP' = [], 'RIGHT' = [], 'DOWN' = [], 'LEFT' = [], 'WAIT' = [], 'BOMB' = []}
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    except:
        self.model = None
    self.temp_model = self.model
    
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
    self.logger.info(self_action)
    # Idea: Add your own events to hand out rewards

    if ...:
        events.append(PLACEHOLDER_EVENT)
     
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    new_state_vector = state_to_features(new_game_state)
    
    #setting up model if necessary
    if self.model == None:
        self.temp_model = {'UP': [random.random() for i in range(len(new_state_vector))], 
        'RIGHT': [random.random() for i in range(len(new_state_vector))], 'DOWN': [random.random() for i in range(len(new_state_vector))],
        'LEFT': [random.random() for i in range(len(new_state_vector))], 'WAIT': [random.random() for i in range(len(new_state_vector))], 'BOMB': [random.random() for i in range(len(new_state_vector))]}
    
    #initializing with arbitrary alpha as hyperparameter and transition_history_size as batch-size:
    if old_game_state is not None:
        alpha = .1
        beta = .1
        old_state_vector = state_to_features(old_game_state)

    
        try:
            reward = reward_from_events(events)
        except:
            reward = 0

        gradient_vector = np.dot(np.transpose(old_state_vector) , reward + beta*q_func(self,new_state_vector) - np.dot(old_state_vector, self.temp_model[self_action]))
        self.temp_model[self_action] = self.temp_model[self_action] + alpha/ 2 * gradient_vector
    

    


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
    self.model = self.temp_model
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def q_func(self, X):
    q_array = []
    q_array.append(np.dot(X, self.model["UP"]))
    q_array.append(np.dot(X, self.model["RIGHT"]))
    q_array.append(np.dot(X, self.model["DOWN"]))
    q_array.append(np.dot(X, self.model["LEFT"]))
    q_array.append(np.dot(X, self.model["WAIT"]))
    q_array.append(np.dot(X, self.model["BOMB"]))
    return max(q_array)

