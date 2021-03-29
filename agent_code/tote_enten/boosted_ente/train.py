import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

from sklearn import exceptions
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import events as e
from .callbacks import state_to_features

# History cache
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward','next_state'))

# Hyper parameters
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability 
PARAMS = {'random_state':0, 'warm_start':True, 'n_estimators':100, 'learning_rate':0.1, 'max_depth':3}  # parameters for the GradientBoostingRegressor
GAMMA = 0.01
N = 4 

# Auxillary events
WAITING_EVENT = "WAIT"
VALID_ACTION = "VALID_ACTION"
COIN_CHASER = "COIN_CHASER"
MOVED_AWAY_FROM_BOMB = "MOVED_AWAY_FROM_BOMB"
WAITED_IN_EXPLOSION_RANGE = "WAITED_IN_EXPLOSION_RANGE"
INVALID_ACTION_IN_EXPLOSION_RANGE = "INVALID_ACTION_IN_EXPLOSION_RANGE"

def setup_training(self):
    """
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # transition cache
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        
    except:
        self.model = {'UP':GradientBoostingRegressor(**PARAMS),'RIGHT':GradientBoostingRegressor(**PARAMS),'DOWN':GradientBoostingRegressor(**PARAMS),
        'LEFT':GradientBoostingRegressor(**PARAMS),'WAIT':GradientBoostingRegressor(**PARAMS),'BOMB':GradientBoostingRegressor(**PARAMS)}
    
    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)
        

    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.logger.info(self_action)


    if old_game_state is not None:
        #updating auxillary events
        aux_events(self, old_game_state, self_action, new_game_state, events)
        
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
    # Adding the last move to the transition cache
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, state_to_features(None), reward_from_events(self, events)))
    experience_replay(self, N)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)



def reward_from_events(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    game_rewards = {
        e.COIN_COLLECTED: 12,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -20,
        WAITING_EVENT: -3,
        e.INVALID_ACTION: -7,
        COIN_CHASER: 10,
        VALID_ACTION: -1,
        MOVED_AWAY_FROM_BOMB: 1,
        WAITED_IN_EXPLOSION_RANGE: -1,
        INVALID_ACTION_IN_EXPLOSION_RANGE: -2 
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def experience_replay(self, n):
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

    total_batch = np.array(self.transitions, dtype=object)      # converting the deque to numpy array for conveniency
    total_batch[-1,2] = np.zeros(D)
    
    
    total_batch_size = np.shape(total_batch)[0]
    effective_batch_size = total_batch_size - n 

    effective_batch = total_batch[:effective_batch_size]    # training instances that have n next instances

    # Creating the batches
    B = {'UP':{}, 'RIGHT':{}, 'DOWN':{},'LEFT':{}, 'WAIT':{}, 'BOMB':{}}
    actions = list(B.keys())

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

       
        # updating the model
        N = np.shape(rewards)[0]
            # size of subbatch
        
        discount = np.ones(n+1)*GAMMA   # discount for the i th future reward: GAMMA^i 
        for i in range(0,n+1):
            discount[i] = discount[i]**i
        

        #print(action, len(nth_states), len(states), np.sum(rewards, axis = 1))
        if len(rewards) != 0:
            X = states
            #print(np.shape(rewards))
            #if np.shape(rewards)[1] is None:
            #    Y = np.sum(rewards)
            #else:
                #shape
            Y = np.dot(rewards, discount)#np.sum(rewards, axis = 1)
            
            '''if self.model != None:
                Y = []  
            else:
                Y = rewards
            N = len(X)'''


            '''if self.model is not None:
                for i in range(N):

                    if B[action]['next_states'][i] is not None:     # not terminal state
                        
                        # computing the reward for the state according to temporal difference
                        q_value_future = []
                        try:
                            for move in self.model:
                                q_value_future.append(self.model[move].predict(np.reshape(B[action]['states'][i],(1,-1))))
                            future_reward = np.max(q_value_future)
                            Y.append(B[action]['rewards'][i] + GAMMA * future_reward)

                        except exceptions.NotFittedError:
                            self.model[action].fit(X,B[action]['rewards'])
                            Y.append(B[action]['rewards'][i])

                    else:
                        Y.append(B[action]['rewards'][i])'''
            #print(np.shape(X),np.shape(Y))
            if X != [] and Y != []:
                self.model[action].fit(X,Y)
                feature_augmentation(self,X, Y, action)

def feature_augmentation(self, X, Y, action):
    hshift_states, hshift_action = horizontal_shift(X, action)
    self.model[hshift_action].fit(hshift_states,Y)

    #update with vertically shifted state:
    vshift_states, vshift_action = vertical_shift(X, action)
    self.model[vshift_action].fit(vshift_states,Y)

    #update with turn right:
    rturn_states, rturn_action = turn_right(X, action)
    self.model[rturn_action].fit(rturn_states, Y)

    #update with turn left:
    lturn_states, lturn_action = turn_right(X, action)
    self.model[lturn_action].fit(lturn_states, Y)

    #update with turn left:
    fturn_states, fturn_action = turn_around(X, action)
    self.model[fturn_action].fit(fturn_states, Y) 

'''def experience_replay(self):
    self.logger.debug('Doing experience replay with the collected data.')
    # creating the training batches from the transition history
    B = {'UP':{'states':[], 'rewards':[], 'next_states':[]}, 'RIGHT':{'states':[], 'rewards':[], 'next_states':[]}, 'DOWN':{'states':[], 'rewards':[], 'next_states':[]}, 
    'LEFT':{'states':[], 'rewards':[], 'next_states':[]}, 'WAIT':{'states':[], 'rewards':[], 'next_states':[]}, 'BOMB':{'states':[], 'rewards':[], 'next_states':[]}}

    for transition in self.transitions:
        if transition.action is not None:
            B[transition.action]['states'].append(transition.state)
            B[transition.action]['rewards'].append(transition.reward)
            B[transition.action]['next_states'].append(transition.next_state)
    #print(B['LEFT']['rewards'])
    print(len(B['BOMB']['states']), B['BOMB']['rewards'])
    for action in B:
        X = B[action]['states']
        
        if self.model != None:
            Y = []  
        else:
            B[action]['rewards']
        N = len(X)


        if self.model is not None:
            for i in range(N):

                if B[action]['next_states'][i] is not None:     # not terminal state
                    
                    # computing the reward for the state according to temporal difference
                    q_value_future = []
                    try:
                        for move in self.model:
                            q_value_future.append(self.model[move].predict(np.reshape(B[action]['states'][i],(1,-1))))
                        future_reward = np.max(q_value_future)
                        Y.append(B[action]['rewards'][i] + GAMMA * future_reward)

                    except exceptions.NotFittedError:
                        self.model[action].fit(X,B[action]['rewards'])
                        Y.append(B[action]['rewards'][i])

                else:
                    Y.append(B[action]['rewards'][i])
        #print(np.shape(X),np.shape(Y))
        if X != [] and Y != []:
            self.model[action].fit(X,Y)
    '''       


def aux_events(self, old_game_state, self_action, new_game_state, events):

    # Valid action
    if e.INVALID_ACTION not in events:
        events.append(VALID_ACTION)


    # Waiting
    if self_action == "WAIT":
        events.append(WAITING_EVENT)
    
    if old_game_state is not None:
        # Getting closer to coins
        # get positions of the player
        old_player_coor = old_game_state['self'][3]     
        new_player_coor = new_game_state['self'][3]
            
        ############################################################################################################    
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
        ######################################################################################################



def horizontal_shift(state, action):
    #initializing the shifted state:
    shifted_state = np.copy(state)

    #shifting up to down:
    shifted_state[:,16:24] = state[:,24:32]
    shifted_state[:,24:32] = state[:,16:24]

    #shifting actions
    if action == "LEFT":
        new_action = "RIGHT"
    
    elif action == "RIGHT":
        new_action = "LEFT"

    else:
        new_action = action

    return shifted_state, new_action

def vertical_shift(state, action):
    #initializing the shifted state:
    shifted_state = np.copy(state)

    #shifting up to down:
    shifted_state[:,0:8] = state[:,8:16]
    shifted_state[:,8:16] = state[:,0:8]

    #shifting actions
    if action == "UP":
        new_action = "DOWN"
    
    elif action == "DOWN":
        new_action = "UP"

    else:
        new_action = action

    return shifted_state, new_action

def turn_right(state, action):
    #initializing the turned state:
    turned_state = np.copy(state)
    
    #up -> left 
    turned_state[:,0:8] = state[:,16:24]
    #down -> right
    turned_state[:,8:16] = state[:,24:32]
    #right -> up
    turned_state[:,16:24] = state[:,8:16]
    #left -> down
    turned_state[:,24:32] = state[:,0:8]

    #shifting actions
    if action == 'LEFT':
        new_action = 'UP'
    
    elif action == 'RIGHT':
        new_action = 'DOWN'

    elif action == 'DOWN':
        new_action = 'LEFT'
    
    elif action == 'UP':
        new_action = 'RIGHT'

    else:
        new_action = action
    
    return turned_state, new_action

def turn_left(state, action):
    #initializing the turned state:
    turned_state = np.copy(state)

    #up -> left 
    turned_state[:,0:8] = state[:,24:32]
    #down -> right
    turned_state[:,8:16] = state[:,16:24]
    #right -> up
    turned_state[:,16:24] = state[:,0:8]
    #left -> down
    turned_state[:,24:32] = state[:,8:16]

    #shifting actions
    if action == 'LEFT':
        new_action = 'DOWN'
    
    elif action == 'RIGHT':
        new_action = 'UP'

    elif action == 'DOWN':
        new_action = 'RIGHT'
    
    elif action == 'UP':
        new_action = 'LEFT'

    else:
        new_action = action

    return turned_state, new_action

def turn_around(state, action):
    #first turn:
    turn1, action1 = turn_left(state, action)

    #second turn:
    turn2, action2 = turn_left(turn1, action1)

    return turn2, action2

