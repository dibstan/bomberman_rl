import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import events as e
from .callbacks import state_to_features

# Transition cache
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability 
ALPHA = .01
PARAMS = {'random_state':0, 'warm_start':True, 'n_estimators':100, 'learning_rate':ALPHA, 'max_depth':3}  # parameters for the GradientBoostingRegressor
GAMMA = 0.01     # discount rate
N = 5   # N step temporal difference

# Auxillary events
WAITING_EVENT = "WAIT"
VALID_ACTION = "VALID_ACTION"
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

    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)

    try:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
   

    except:
        self.model = {'UP':GradientBoostingRegressor(**PARAMS),'RIGHT':GradientBoostingRegressor(**PARAMS),'DOWN':GradientBoostingRegressor(**PARAMS),
        'LEFT':GradientBoostingRegressor(**PARAMS),'WAIT':GradientBoostingRegressor(**PARAMS),'BOMB':GradientBoostingRegressor(**PARAMS)}
        
    self.isFitted = {'UP':False, 'RIGHT':False, 'DOWN':False, 'LEFT':False, 'WAIT':False, 'BOMB':False}

    self.fluctuations = []      # array for the fluctuations of each each round
    self.max_fluctuations = []      # array for the maximum fluctuations in all rounds

    
    
 
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state is not None:

        # Adding auxillary Events
        aux_events(self, old_game_state, self_action, new_game_state, events)
        #print(events, reward_from_events(self,events))


        # Adding the last move to the transition cache
        self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    

        # updating the model using stochastic gradient descent n-step temporal difference 
        n_step_TD(self, N)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    :param self: The same object that is passed to all of your callbacks.
    """      

    # Adding the last move to the transition cache
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, state_to_features(None), reward_from_events(self, events)))
    

    # updating the model using stochastic gradient descent n-step temporal difference 
    n_step_TD(self, N)


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # Store the fluctuations
    if len(self.fluctuations) != 0:
        self.max_fluctuations.append(np.max(self.fluctuations))     # saving the maximum fluctuation of round
        self.fluctuations = []      # resetting the fluctuation array
        with open('fluctuations.pt', 'wb') as file:
            pickle.dump(self.max_fluctuations, file)

    # delete history cache
    self.transitions = deque(maxlen=N)


def reward_from_events(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -20,
        WAITING_EVENT: -3,
        e.INVALID_ACTION: -7,
        COIN_CHASER: 7,
        VALID_ACTION: -1,
        MOVED_AWAY_FROM_BOMB: 0,
        WAITED_IN_EXPLOSION_RANGE: -1,
        INVALID_ACTION_IN_EXPLOSION_RANGE: -2 
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def aux_events(self, old_game_state, self_action, new_game_state, events):

    # Valid action
    if e.INVALID_ACTION not in events:
        events.append(VALID_ACTION)


    # Waiting
    if self_action == "WAIT":
        events.append(WAITING_EVENT)
    

    # Getting closer to coins
    # get positions of the player
    old_player_coor = old_game_state['self'][3]     
    new_player_coor = new_game_state['self'][3]
        
     
    #define event coin_chaser
    coin_coordinates = old_game_state['coins']
    if len(coin_coordinates) != 0:
        old_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,old_player_coor), axis=1)
        new_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,new_player_coor), axis=1)

    
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
    



def n_step_TD(self, n):
    '''
        input: self, n: nummber of steps in n-step temporal differnece
        updates the model using n-step temporal differnce and gradient descent
    '''
    
    D = len(self.transitions[0].state)      # feature dimension

    transitions_array = np.array(self.transitions, dtype=object)      # converting the deque to numpy array for conveniency
    
    # Updating the model
    if  np.shape(transitions_array)[0] == n:
        
        action = transitions_array[0,1]     # relevant action executed

        first_state = transitions_array[0,0]    # first state saved in the cache

        last_state = transitions_array[-1,2]     # last state saved in the cache

        n_future_rewards = transitions_array[:,3]   # rewards of the n next actions

        discount = np.ones(n)*GAMMA   # discount for the i th future reward: GAMMA^i 
        for i in range(0,n):
            discount[i] = discount[i]**i

        if action == 'BOMB':
                discount[-1] = 1
            
        if self.isFitted[action] == True and last_state is not None:
            Q_TD = np.dot(discount, n_future_rewards) + GAMMA**n * Q_func(self, last_state.reshape(1, -1))    # Array holding the n-step-TD Q-value for each instance in subbatch
            
            Q = self.model[action].predict(first_state)     # compute current prediction for state

            self.fluctuations.append(np.abs(Q_TD-Q))      # Store fluctuation

        else:
            Q_TD = np.dot(discount, n_future_rewards)
        
        #print(action)
        #print(n_future_rewards)
        #print(Q_TD)

        self.model[action].n_estimators += 1
        self.model[action].fit(first_state.reshape(1, -1), np.array([Q_TD]))



def Q_func(self, state):
    '''
        input: self, state
        output: Q value of the best action
    '''
    Q_values = []
    for action in self.model.keys():
        if self.isFitted[action] == True:
            Q_values.append(self.model[action].predict(state))
        else:
            Q_values.append(-np.inf)
    
    Q_max = np.max(Q_values, axis = 0)
    return Q_max      # return the max Q value
