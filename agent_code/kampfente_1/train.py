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
TRANSITION_HISTORY_SIZE = 100  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
WAITING_EVENT = "WAIT"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    try:    # loading existing model
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    except:     # initializing a model
        self.model = {'UP': [0 for i in range(1740)], 'RIGHT': [0 for i in range(1740)], 'DOWN': [0 for i in range(1740)], 'LEFT': [0 for i in range(1740)], 'WAIT': [0 for i in range(1740)], 'BOMB': [0 for i in range(1740)]}
    
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

    if self_action == "WAIT":
        events.append(WAITING_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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

    # Updating the model
    experience_replay(self)
    #print(self.model)
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
        e.COIN_COLLECTED: 200,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -300,
        WAITING_EVENT: -50,
        e.INVALID_ACTION: -80,
        e.MOVED_LEFT: -50,
        e.MOVED_RIGHT: -50,
        e.MOVED_UP: -50,
        e.MOVED_DOWN: -50,
    }
    reward_sum = 0
    
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def experience_replay(self):
    #updating the model using batched gradient decent
    moves = {'UP':0, 'RIGHT':1, 'DOWN':2,'LEFT':3, 'WAIT':4, 'BOMB':5}
    moves_inv = {v: k for k, v in moves.items()}

    B = [[] for i in range(6)]  # creating the batches for each action from the transition data

    for transition in self.transitions:
        move = transition.action
        if move is not None:
            B[moves[move]].append(transition)

    alpha = 1     # training rate
    gamma = 1    # discount rate

    model = {'UP':[],'RIGHT':[],'DOWN':[], 'LEFT':[], 'WAIT':[], 'BOMB':[]}
    
    for i in range(6):

        if B[i] != []:

            X = []
            next_state = []
            Y_TD = []
            for data in B[i]:
                if data.next_state is not None:
                    X.append(data.state)
                    Y_TD.append(data.reward+gamma*q_func(self, data.next_state))
                    
            
                            
            Y = np.dot(X, self.model[moves_inv[i]])
            #print(Y_TD-Y)
            #print(np.dot(np.transpose(X), (Y_TD-Y)))
            self.model[moves_inv[i]] = self.model[moves_inv[i]]+alpha*np.clip(np.dot(np.transpose(X), (Y_TD-Y)),-10,10)
    print(self.model)
    
    


def q_func(self, X):
    q_array = []
    q_array.append(np.dot(X, self.model["UP"]))
    q_array.append(np.dot(X, self.model["RIGHT"]))
    q_array.append(np.dot(X, self.model["DOWN"]))
    q_array.append(np.dot(X, self.model["LEFT"]))
    q_array.append(np.dot(X, self.model["WAIT"]))
    q_array.append(np.dot(X, self.model["BOMB"]))
    return max(q_array)

