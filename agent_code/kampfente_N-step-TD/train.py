import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

# Transition cache
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.05    # learning rate
GAMMA = 0.2     # discount rate
N = 4   # N step temporal difference

# Events
WAITING_EVENT = "WAIT"
COIN_CHASER = "CLOSER_TO_COIN"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transitions = deque(maxlen=N)

    try:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    except:
        self.model = None
    self.temp_model = self.model
    
    #print(self.model["RIGHT"])

    
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
    
    # Auxillary Events
    if self_action == "WAIT":
        events.append(WAITING_EVENT)
    
    if old_game_state is not None:
        # Adding the last move to the transition cache
        self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
        # updating the model using n-step temporal difference
        n_step_TD(self, N)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    #self.states.append(last_state_vector)
    #with open("saved_states.pt", "wb") as file:
    #    pickle.dump(self.states, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1000,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -300,
        WAITING_EVENT: -100,
        e.INVALID_ACTION: -100,
        e.MOVED_DOWN: -40,
        e.MOVED_LEFT: -40,
        e.MOVED_RIGHT: -40,
        e.MOVED_UP: -40,
        COIN_CHASER: 30  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def n_step_TD(self, n):
    '''
        input: self, n: nummber of steps in n-step temporal differnece
        updates the model using n-step temporal differnce and gradient descent
    '''
    
    #setting up model if necessary
    D = len(self.transitions[0].state)      # feature dimension

    if self.model == None:
        init_beta = np.zeros(D)
        self.model = {'UP': init_beta, 
        'RIGHT': init_beta, 'DOWN': init_beta,
        'LEFT': init_beta, 'WAIT': init_beta, 'BOMB': init_beta}

    transitions_array = np.array(self.transitions, dtype=object)      # converting the deque to numpy array for conveniency
    
    # Updating the model
    if  np.shape(transitions_array)[0] == n:
        print('hi')

        action = transitions_array[0,1]     # relevant action executed

        first_state = transitions_array[0,0]    # first state saved in the cache

        last_state = transitions_array[-1,0]     # last state saved in the cache

        n_future_rewards = transitions_array[:-1,3]   # rewards of the n-1 next actions

        discount = np.ones(n-1)*GAMMA   # discount for the i th future reward: GAMMA^i 
        for i in range(1,n):
            discount[i-1] = discount[i-1]**i

        Q_TD = np.dot(discount, n_future_rewards) + GAMMA**n * Q_func(self, last_state)   # value estimate using n-step temporal difference
        
        Q = np.dot(first_state, self.model[action])     # value estimate of current model

        GRADIENT = Q_TD - Q     # gradient descent

        self.model[action] = self.model[action] + ALPHA * GRADIENT      # updating the model for the relevant action
        print(self.model)

def Q_func(self, state):
    '''
        input: self, state
        output: Q value of the best action
    '''

    vec_model = np.array(list(self.model.values()))     # vectorizing the model dict
    
    return np.max(np.dot(vec_model, state))      # return the max Q value
