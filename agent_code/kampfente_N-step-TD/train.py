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
ALPHA = 0.001    # learning rate
GAMMA = 0.2     # discount rate
N = 4   # N step temporal difference

# Events
WAITING_EVENT = "WAIT"
COIN_CHASER = "COIN_CHASER"
MOVED_AWAY_FROM_BOMB = "MOVED_AWAY_FROM_BOMB"
WAITED_IN_EXPLOSION_RANGE = "WAITED_IN_EXPLOSION_RANGE"
INVALID_ACTION_IN_EXPLOSION_RANGE = "INVALID_ACTION_IN_EXPLOSION_RANGE"

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

    #impot tile map
    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)
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

        #get positions of the player
        old_player_coor = old_game_state['self'][3]     
        new_player_coor = new_game_state['self'][3]
        
        #define event coin_chaser
        coin_coordinates = old_game_state['coins']
        old_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,old_player_coor),axis=0)
        new_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,new_player_coor),axis=0)

        if min(new_coin_distances) < min(old_coin_distances):   #if the distance to closest coin got smaller
            events.append(COIN_CHASER)

        #define events with bombs
        old_bomb_coors = old_game_state['bombs']

        dangerous_tiles = []            #this array will store all tuples with 'dangerous' tile coordinates
        for bomb in old_bomb_coors:
            for coor in self.exploding_tiles_map[bomb[0]]:
                dangerous_tiles.append(coor)

        if dangerous_tiles != []:       
            if old_player_coor in dangerous_tiles and new_player_coor not in dangerous_tiles:
                events.append(MOVED_AWAY_FROM_BOMB)
            if old_player_coor in dangerous_tiles and self_action == "WAIT":
                #print('waited')
                events.append(WAITED_IN_EXPLOSION_RANGE)
            if old_player_coor in dangerous_tiles and "INVALID_ACTION" in events:
                #print('invalid')
                events.append(INVALID_ACTION_IN_EXPLOSION_RANGE)
                
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
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -3,
        WAITING_EVENT: -1,
        e.INVALID_ACTION: -3,
        e.MOVED_DOWN: -.4,
        e.MOVED_LEFT: -.4,
        e.MOVED_RIGHT: -.4,
        e.MOVED_UP: -.4,
        COIN_CHASER: 3.5,
        MOVED_AWAY_FROM_BOMB: 3,
        WAITED_IN_EXPLOSION_RANGE: -3,
        INVALID_ACTION_IN_EXPLOSION_RANGE: -4 
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

        action = transitions_array[0,1]     # relevant action executed

        first_state = transitions_array[0,0]    # first state saved in the cache

        last_state = transitions_array[-1,0]     # last state saved in the cache

        n_future_rewards = transitions_array[:-1,3]   # rewards of the n-1 next actions

        discount = np.ones(n-1)*GAMMA   # discount for the i th future reward: GAMMA^i 
        for i in range(1,n):
            discount[i-1] = discount[i-1]**i

        Q_TD = np.dot(discount, n_future_rewards) + GAMMA**n * Q_func(self, last_state)   # value estimate using n-step temporal difference
        
        Q = np.dot(first_state, self.model[action])     # value estimate of current model

        GRADIENT = first_state * (Q_TD - Q)     # gradient descent
        
        self.model[action] = self.model[action] + ALPHA * np.clip(GRADIENT, -10,10)   # updating the model for the relevant action
        print(self.model)

def Q_func(self, state):
    '''
        input: self, state
        output: Q value of the best action
    '''

    vec_model = np.array(list(self.model.values()))     # vectorizing the model dict
    
    return np.max(np.dot(vec_model, state))      # return the max Q value
