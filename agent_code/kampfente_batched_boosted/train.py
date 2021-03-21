import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

from sklearn import exceptions
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

import events as e
from .callbacks import state_to_features

# History cache
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward','next_state'))

# Hyper parameters
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability 
ALPHA = .3      # learning rate
GAMMA = 0.01    # discount rate
KAPPA = 0.0   # adaption konstant for learning rate (if set to zero -> konstant learning rate)
PARAMS = {'random_state':0, 'warm_start':True, 'n_estimators':100, 'learning_rate':ALPHA, 'max_depth':3}  # parameters for the GradientBoostingRegressor
HIST_SIZE = 1000
N = 4   # N step temporal difference

# Auxillary events
WAITING_EVENT = "WAIT"
VALID_ACTION = "VALID_ACTION"
COIN_CHASER = "COIN_CHASER"
MOVED_OUT_OF_DANGER = "MOVED_AWAY_FROM_EXPLODING_TILE"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
STAYED_NEAR_BOMB = 'STAYED_ON_EXPLODING_TILE'
CRATE_CHASER = 'CRATE_CHASER'
BOMB_NEXT_TO_CRATE = 'BOMB_NEXT_TO_CRATE'
BOMB_DESTROYED_NOTHING = 'BOMB_DESTROYED_NOTHING'
BOMB_NOT_NEXT_TO_CRATE = 'BOMB_NOT_NEXT_TO_CRATE'
#LESS_DISTANCE_TO_BOMB = 'LESS_DISTANCE_TO_BOMB'

def setup_training(self):
    """
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # transition cache
    self.transitions = deque(maxlen=HIST_SIZE)
    
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.isFitted = {'UP': True, 'RIGHT':True, 'DOWN':True, 'LEFT':True, 'WAIT':True, 'BOMB':True}

    except:
        self.model = {'UP':GradientBoostingRegressor(**PARAMS),'RIGHT':GradientBoostingRegressor(**PARAMS),'DOWN':GradientBoostingRegressor(**PARAMS),
        'LEFT':GradientBoostingRegressor(**PARAMS),'WAIT':GradientBoostingRegressor(**PARAMS),'BOMB':GradientBoostingRegressor(**PARAMS)}
        self.isFitted = {'UP':False, 'RIGHT':False, 'DOWN':False, 'LEFT':False, 'WAIT':False, 'BOMB':False}

    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)
           
    self.fluctuations = []      # Fluctuations

    self.round = 0
    self.learning_rate = ALPHA
    
    
    
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



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    :param self: The same object that is passed to all of your callbacks.
    """      

    # Adding the last move to the transition cache
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, state_to_features(None), reward_from_events(self, events)))
    
    #updating game round counter
    self.round += 1

    #updating learning rate
    self.learning_rate = ALPHA/(1+KAPPA*self.round)

    # updating the model using batched gradient descent n-step temporal difference 
    experience_replay(self, N)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    #Store the fluctuations
    with open('fluctuations.pt', 'wb') as file:
        pickle.dump(self.fluctuations, file)

    # delete history cache
    self.transitions = deque(maxlen=HIST_SIZE)

   


def reward_from_events(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    game_rewards = {
        e.COIN_COLLECTED: 12,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -80,
        WAITING_EVENT: -3,
        e.INVALID_ACTION: -7,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        #VALID_ACTION: -2,
        COIN_CHASER: 1.5,
        MOVED_OUT_OF_DANGER: 5,
        STAYED_NEAR_BOMB: -5,
        MOVED_INTO_DANGER: -5,
        e.CRATE_DESTROYED: 5,   #2
        e.COIN_FOUND: 1,
        CRATE_CHASER: 0.5,
        BOMB_NEXT_TO_CRATE: 2,
        BOMB_NOT_NEXT_TO_CRATE: -3,
        BOMB_DESTROYED_NOTHING: -3
    }

    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum



def aux_events(self, old_game_state, self_action, new_game_state, events):

    # Valid action
    #if e.INVALID_ACTION not in events  :
        #events.append(VALID_ACTION)

    # Waiting
    if self_action == "WAIT":
        events.append(WAITING_EVENT)
    
    # get positions of the player in old state and new state (tuples (x,y) in this case)
    old_player_coor = old_game_state['self'][3]     
    new_player_coor = new_game_state['self'][3]
        
 
    #define event coin_chaser
    coin_coordinates = old_game_state['coins']      #get coin coordinates(also tuples in form (x,y))
    if len(coin_coordinates) != 0:                  #now calculate distance to all coins in respect to...
        old_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,old_player_coor), axis=1) #...old player position
        new_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,new_player_coor), axis=1) #...new player position

        if min(new_coin_distances) < min(old_coin_distances):   #if the distance to closest coin got smaller
            events.append(COIN_CHASER)                          # -> reward

    
    #define events with bombs
    old_bomb_coors = old_game_state['bombs']    #get bomb coordinates (careful: timer still included: ((x,y),t)) for each bomb)

    dangerous_tiles = []                        #this array will store all tuples with 'dangerous' tile coordinates
    for bomb in old_bomb_coors:
        for coor in self.exploding_tiles_map[bomb[0]]:      #for each bomb get all tiles that explode with that bomb...
            dangerous_tiles.append(coor)                    ##... and append them to dangerous_tiles


    if dangerous_tiles != []:

        #event in case the agent sucsessfully moved away from a dangerous tile -> reward     
        if old_player_coor in dangerous_tiles and new_player_coor not in dangerous_tiles:
            events.append(MOVED_OUT_OF_DANGER)

        #event in case agent stayed on a dangerous tile -> penalty
        if old_player_coor in dangerous_tiles and ("WAITED" in events or "INVALID_ACTION" in events):
            events.append(STAYED_NEAR_BOMB)
        
        #event in case agent moved onto a dangerous tile -> penalty
        if old_player_coor not in dangerous_tiles and new_player_coor in dangerous_tiles:
            events.append(MOVED_INTO_DANGER)

    #if bombs destroyed nothing
    if 'BOMB_EXPLODED' in events:                   #a bomb placed by our agent exploded
        if 'CRATE_DESTROYED' not in events:         #no crate got destroyed (in the future also include:no enemy got bombed)
            #print(BOMB_DESTROYED_NOTHING)
            events.append(BOMB_DESTROYED_NOTHING)   # -> penalty
    
    
    #define crate chaser: the agent gets rewarded if he moves closer to crates ONLY if he currently has a bomb
    field = old_game_state['field']
    rows,cols = np.where(field == 1)
    crates_position = np.array([rows,cols]).T       #all crate coordinates in form [x,y] in one array
    old_crate_distance = np.linalg.norm(crates_position-np.array([old_player_coor[0],old_player_coor[1]]),axis = 1)
    new_crate_distance = np.linalg.norm(crates_position-np.array([new_player_coor[0],new_player_coor[1]]),axis = 1)

    if old_crate_distance.size > 0:                 #if agent moved closer to the nearest crate and BOMB action is possible 
        if min(new_crate_distance) < min(old_crate_distance) and old_game_state['self'][2]: 
            #print(CRATE_CHASER)
            events.append(CRATE_CHASER)
        
        
    #define event for bomb next to crate
        if self_action == 'BOMB' and e.INVALID_ACTION not in events:                                       #if bomb is placed...
            #if min(old_crate_distance) == 1:    # ... give reward for each crate neighbouring bomb position                   
                #events.append(BOMB_NEXT_TO_CRATE)   
            for i in range(len(np.where(old_crate_distance==1)[0])):    # ... give reward for each crate neighbouring bomb position                   
                events.append(BOMB_NEXT_TO_CRATE)                   
            if len(np.where(old_crate_distance==1)[0]) == 0 :                                                       #bomb is not placed next to crate
                events.append(BOMB_NOT_NEXT_TO_CRATE)                   # -> penalty
                #print(BOMB_NOT_NEXT_TO_CRATE)
    


def experience_replay(self, n):
    '''
        input: self, n: nummber of steps in n-step temporal differnece
        updates the model using n-step temporal differnce and gradient descent
    '''
    
    D = len(self.transitions[0].state)      # feature dimension

    total_batch = np.array(self.transitions, dtype=object)      # converting the deque to numpy array for conveniency
    total_batch[-1,2] = np.zeros(D)
    
    total_batch_size = np.shape(total_batch)[0]
    effective_batch_size = total_batch_size - n 

    effective_batch = total_batch[:effective_batch_size]    # training instances that have n next instances


    # Creating the batches
    actions = list(self.model.keys())

    fluctuations = []   # Array for the fluctuations in each action

    for action in actions:
        # Finding indices for the wanted instances
        index = np.where(effective_batch[:,1] == action)[0]     # indices of instances with action in subbatch
        n_next_index = index[:, None] + np.arange(n+1)    # indices of the n next instances for every instance in this subbatch
        nth_next_index = index + n      # indices of the nth next instance following the instances in the subbatch

        # computing the arrays according to above indices
        try:
            states = np.stack(total_batch[:,0][index])
        except ValueError:
            states = np.array([])
        
        try:
            rewards = np.stack(total_batch[:,3][n_next_index])
        except ValueError:
            rewards = np.array([])

        try:
            nth_states = np.stack(total_batch[:,2][nth_next_index])
        except ValueError:
            nth_states = np.array([])


        #print(states, nth_states, rewards)
        if states != np.array([]):
            
            if np.shape(states)[0] == 1:
                states = states.reshape(1, -1)
                nth_states = nth_states.reshape(1, -1)

            discount = np.ones(n+1)*GAMMA   # discount for the i th future reward: GAMMA^i 
            for i in range(0,n+1):
                discount[i] = discount[i]**i
            
            if action == 'BOMB':
                discount[-1] = 1
            
            if self.isFitted[action] == True:
                Q_TD = np.dot(rewards,discount) + GAMMA**n * Q_func(self,nth_states)    # Array holding the n-step-TD Q-value for each instance in subbatch
                
                Q = self.model[action].predict(states)

                fluctuations.append(np.abs(np.mean(np.clip((Q_TD-Q),-10,10))))
            else:
                Q_TD = np.dot(rewards,discount)

            print(action, rewards, Q_TD)
            #print(action, Q_TD, Q)

            # updating the model
            self.model[action].learning_rate = self.learning_rate
            self.model[action].n_estimators += 3
            self.model[action].fit(states, Q_TD)

            #print(action, Q_TD, self.model[action].predict(states))
            
            self.isFitted[action] = True
            
           
    # saving the fluctuations
    if len(fluctuations) != 0:
        self.fluctuations.append(np.max(fluctuations))



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