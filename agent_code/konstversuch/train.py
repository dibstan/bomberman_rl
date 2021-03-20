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
GAMMA = 0.0001     # discount rate
N = 5   # N step temporal difference
CLIP = 20  # initial clip value
N_CLIPPER = 30      # number of fluctuations considering in auto clipping


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

    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)
    
    self.fluctuations = []      # array for the fluctuations of each each round
    self.max_fluctuations = []      # array for the maximum fluctuations in all rounds
    
    self.clip = CLIP    # clip value
    
    
 
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
    self.max_fluctuations.append(np.max(self.fluctuations))     # saving the maximum fluctuation of round
    self.fluctuations = []      # resetting the fluctuation array
    with open('fluctuations.pt', 'wb') as file:
        pickle.dump(self.max_fluctuations, file)

    # updating the clip value
    try:
        self.clip = np.mean(self.max_fluctuations[-N_CLIPPER:])
    except:
        self.clip = CLIP

    # delete history cache
    self.transitions = deque(maxlen=N)


def reward_from_events(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    game_rewards = {
        e.COIN_COLLECTED: 12,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -150,
        WAITING_EVENT: -3,
        e.INVALID_ACTION: -7,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        #VALID_ACTION: -2,
        COIN_CHASER: 1,
        MOVED_OUT_OF_DANGER: 8,
        STAYED_NEAR_BOMB: -5,
        MOVED_INTO_DANGER: -5,
        e.CRATE_DESTROYED: 8,   #2
        e.COIN_FOUND: 2,
        CRATE_CHASER: 0.5,
        BOMB_NEXT_TO_CRATE: 3,
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
    
    crates_position = np.array([rows,cols]).T      #all crate coordinates in form [x,y] in one array
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
            if len(np.where(old_crate_distance==1)[0]) == 0 :                                                  #bomb is not placed next to crate
                events.append(BOMB_NOT_NEXT_TO_CRATE)                   # -> penalty
                #print(BOMB_NOT_NEXT_TO_CRATE)




def n_step_TD(self, n):
    '''
        input: self, n: nummber of steps in n-step temporal differnece
        updates the model using n-step temporal differnce and gradient descent
    '''
    
    #setting up model if necessary
    D = len(self.transitions[0].state)      # feature dimension
    D = 225
    if self.model == None:
        init_beta = np.zeros(D)
        self.model = {'UP': init_beta, 
        'RIGHT': init_beta, 'DOWN': init_beta,
        'LEFT': init_beta, 'WAIT': init_beta, 'BOMB': init_beta}

    transitions_array = np.array(self.transitions, dtype=object)      # converting the deque to numpy array for conveniency
    if transitions_array[0,1] is not None:

        # Updating the model
        if  np.shape(transitions_array)[0] == n:
            
            action = transitions_array[0,1]     # relevant action executed

            first_state = transitions_array[0,0]    # first state saved in the cache

            last_state = transitions_array[-1,2]     # last state saved in the cache

            n_future_rewards = transitions_array[:,3]   # rewards of the n next actions

            discount = np.ones(n)*GAMMA   # discount for the i th future reward: GAMMA^i 
            for i in range(0,n):
                discount[i] = discount[i]**i

            if last_state is not None:  # value estimate using n-step temporal difference
                Q_TD = np.dot(discount, n_future_rewards) + GAMMA**(n+1) * Q_func(self, last_state) 
    
            else:
                Q_TD = np.dot(discount, n_future_rewards)
            
            '''print(action)
            print(n_future_rewards)
            print(Q_TD)'''
            #print(first_state)
            Q = np.dot(first_state, self.model[action])     # value estimate of current model
            
            self.fluctuations.append(abs(np.clip((Q_TD-Q),-self.clip, self.clip)))       # saving the fluctuation

            GRADIENT = first_state * np.clip((Q_TD - Q), -self.clip,self.clip)     # gradient descent
            
            self.model[action] = self.model[action] + ALPHA * GRADIENT   # updating the model for the relevant action
            #print(self.model)
            '''
            # Train with augmented data
            #update with horizontally shifted state:
            hshift_model_update, hshift_action = feature_augmentation(self, horizontal_shift, first_state, last_state, action, discount, n_future_rewards, n)
            self.model[hshift_action] = hshift_model_update

            #update with vertically shifted state:
            vshift_model_update, vshift_action = feature_augmentation(self, vertical_shift, first_state, last_state, action, discount, n_future_rewards, n)
            self.model[vshift_action] = vshift_model_update

            #update with turn left:
            left_model_update, left_turn_action = feature_augmentation(self, turn_left, first_state, last_state, action, discount, n_future_rewards, n)
            self.model[left_turn_action] = left_model_update

            #update with turn right:
            right_model_update, right_turn_action = feature_augmentation(self, turn_right, first_state, last_state, action, discount, n_future_rewards, n)
            self.model[right_turn_action] = right_model_update

            #update with turn around:
            fullturn_model_update, fullturn_action = feature_augmentation(self, turn_around, first_state, last_state, action, discount, n_future_rewards, n)
            self.model[fullturn_action] = fullturn_model_update
            '''


def feature_augmentation(self, aug_direction, first_state, last_state, action, disc, n_future_rew, n):
    shift_first_state, shift_action = aug_direction(first_state, action)
    
    if last_state is not None:  # value estimate using n-step temporal difference
        shift_last_state, shift_action = aug_direction(last_state, action)
        Q_TD_shift = np.dot(disc, n_future_rew) + GAMMA**n * Q_func(self, shift_last_state)   # value estimate using n-step temporal difference

    else:
        shift_last_state = None
        Q_TD_shift = np.dot(disc, n_future_rew)

    Q_shift = np.dot(shift_first_state, self.model[shift_action])     # value estimate of current model

    GRADIENT = shift_first_state * (Q_TD_shift - Q_shift)
    model_update = self.model[shift_action] + ALPHA * np.clip(GRADIENT, -self.clip,self.clip)   # updating the model for the relevant action

    return model_update, shift_action



def horizontal_shift(state, action):
    #initializing the shifted state:
    shifted_state = np.copy(state)

    #shifting up to down:
    shifted_state[18:27] = state[27:36]
    shifted_state[27:36] = state[18:27]

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
    shifted_state[0:9] = state[9:18]
    shifted_state[9:18] = state[0:9]

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
    turned_state[0:9] = state[18:27]
    #down -> right
    turned_state[9:18] = state[27:36]
    #right -> up
    turned_state[18:27] = state[9:18]
    #left -> down
    turned_state[27:36] = state[0:9]

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
    turned_state[0:9] = state[27:36]
    #down -> right
    turned_state[9:18] = state[18:27]
    #right -> up
    turned_state[18:27] = state[0:9]
    #left -> down
    turned_state[27:36] = state[9:18]

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



def Q_func(self, state):
    '''
        input: self, state
        output: Q value of the best action
    '''

    vec_model = np.array(list(self.model.values()))     # vectorizing the model dict
    
    return np.max(np.dot(vec_model, state))      # return the max Q value
