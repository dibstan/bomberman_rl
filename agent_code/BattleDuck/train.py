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
ALPHA = 0.01     # learning rate
GAMMA = 0.01     # discount rate
KAPPA = 0.     # adaption konstant for learning rate (if set to zero -> konstant learning rate)
N = 5   # N step temporal difference
CLIP = 10   # initial clip value
N_CLIPPER = np.inf      # number of fluctuations considering in auto clipping

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
DROPPED_BOMB_NEAR_ENEMY = 'DROPPED_BOMB_NEAR_ENEMY'
DROPPED_BOMB_NEXT_TO_ENEMY ='DROPPED_BOMB_NEXT_TO_ENEMY'
OPPONENT_CHASER = 'CHASED_OPPONENT'


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

    self.round = 0      # round counter

    self.learning_rate = ALPHA      # adaptive learning rate
    self.learning_rate_log = []
        
 
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
    
    self.round += 1

    self.learning_rate = ALPHA/(1+KAPPA*self.round)
    self.learning_rate_log.append(self.learning_rate)

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

    #Store the learning rate
    with open('learning_rate.pt', 'wb') as file:
        pickle.dump(self.learning_rate_log, file)

    # updating the clip value
    try:
        self.clip = np.mean(self.max_fluctuations[-N_CLIPPER:])
    except:
        self.clip = CLIP

    # delete history cache
    self.transitions = deque(maxlen=N)




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

        last_state = transitions_array[-1,2]     # last state saved in the cache

        n_future_rewards = transitions_array[:,3]   # rewards of the n next actions

        discount = np.ones(n)*GAMMA   # discount for the i th future reward: GAMMA^i 
        for i in range(0,n):
            discount[i] = discount[i]**i

        if last_state is not None:  # value estimate using n-step temporal difference
            Q_TD = np.dot(discount, n_future_rewards) + GAMMA**(n+1) * Q_func(self, last_state) 
   
        else:
            Q_TD = np.dot(discount, n_future_rewards)
     

        Q = np.dot(first_state, self.model[action])     # value estimate of current model
        
        self.fluctuations.append(np.clip(abs(Q_TD-Q), -self.clip, self.clip))       # saving the fluctuation

        GRADIENT = first_state * np.clip((Q_TD - Q), -self.clip,self.clip)     # gradient descent
        
        self.model[action] = self.model[action] + self.learning_rate * GRADIENT   # updating the model for the relevant action
        
        #print(self.model)

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
            


def Q_func(self, state):
    '''
        input: self, state
        output: Q value of the best action
    '''

    vec_model = np.array(list(self.model.values()))     # vectorizing the model dict
    
    return np.max(np.dot(vec_model, state))             # return the max Q value


def feature_augmentation(self, aug_direction, first_state, last_state, action, disc, n_future_rew, n):
    '''Updates the model using n_step_temporal difference with augmented states'''


    shift_first_state, shift_action = aug_direction(first_state, action)
    
    if last_state is not None:  # value estimate using n-step temporal difference
        shift_last_state, shift_action = aug_direction(last_state, action)
        Q_TD_shift = np.dot(disc, n_future_rew) + GAMMA**n * Q_func(self, shift_last_state)   # value estimate using n-step temporal difference

    else:
        shift_last_state = None
        Q_TD_shift = np.dot(disc, n_future_rew)

    Q_shift = np.dot(shift_first_state, self.model[shift_action])     # value estimate of current model

    GRADIENT = shift_first_state * (Q_TD_shift - Q_shift)
    model_update = self.model[shift_action] + ALPHA * np.clip(GRADIENT, -self.clip, self.clip)   # updating the model for the relevant action

    return model_update, shift_action




def horizontal_shift(state, action):
    '''This function mirrors along the vertical axis'''

    #initializing the shifted state:
    shifted_state = np.copy(state)

    #shifting left to right:
    shifted_state[14:21] = state[21:28]
    shifted_state[21:28] = state[14:21]

    #shifting actions
    if action == "LEFT":
        new_action = "RIGHT"
    
    elif action == "RIGHT":
        new_action = "LEFT"

    else:
        new_action = action

    return shifted_state, new_action

def vertical_shift(state, action):
    '''This function mirrors along the horizontal axis'''

    #initializing the shifted state:
    shifted_state = np.copy(state)

    #shifting up to down:
    shifted_state[0:7] = state[7:14]
    shifted_state[7:14] = state[0:7]

    #shifting actions
    if action == "UP":
        new_action = "DOWN"
    
    elif action == "DOWN":
        new_action = "UP"

    else:
        new_action = action

    return shifted_state, new_action

def turn_right(state, action):
    '''This function turns the board to the right by 90 degrees'''

    #initializing the turned state:
    turned_state = np.copy(state)
    
    #up -> left 
    turned_state[0:7] = state[14:21]
    #down -> right
    turned_state[7:14] = state[21:28]
    #right -> up
    turned_state[14:21] = state[7:14]
    #left -> down
    turned_state[21:28] = state[0:7]

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
    '''This function turns the board to the left by 90 degrees'''

    #initializing the turned state:
    turned_state = np.copy(state)

    #up -> left 
    turned_state[0:7] = state[21:28]
    #down -> right
    turned_state[7:14] = state[14:21]
    #right -> up
    turned_state[14:21] = state[0:7]
    #left -> down
    turned_state[21:28] = state[7:14]

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
    '''This function turns the board to the left by 180 degrees'''

    #first turn:
    turn1, action1 = turn_left(state, action)

    #second turn:
    turn2, action2 = turn_left(turn1, action1)

    return turn2, action2



def reward_from_events(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    game_rewards = {
        e.COIN_COLLECTED: 8,
        e.KILLED_OPPONENT: 20,
        e.KILLED_SELF: -80,
        e.WAITED: -1,
        e.INVALID_ACTION: -4,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        COIN_CHASER: 3,             
        MOVED_OUT_OF_DANGER: 5,
        STAYED_NEAR_BOMB: -5,
        MOVED_INTO_DANGER: -5,
        CRATE_CHASER: 1.5,
        BOMB_NEXT_TO_CRATE: 2,
        BOMB_NOT_NEXT_TO_CRATE: -2,
        DROPPED_BOMB_NEAR_ENEMY: 1,
        DROPPED_BOMB_NEXT_TO_ENEMY: 8, 
        OPPONENT_CHASER: 2
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def aux_events(self, old_game_state, self_action, new_game_state, events):
    '''Defining auxillary events for auxillary rewards to optimize training'''
    
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
    old_bomb_coors = old_game_state['bombs']                #get bomb coordinates (careful: timer still included: ((x,y),t)) for each bomb)

    dangerous_tiles = []                                    #this array will store all tuples with 'dangerous' tile coordinates
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
    
    #define crate chaser: the agent gets rewarded if he moves closer to crates ONLY if he currently has a bomb
    field = old_game_state['field']
    rows,cols = np.where(field == 1)
    crates_position = np.array([rows,cols]).T       #all crate coordinates in form [x,y] in one array
    old_crate_distance = np.linalg.norm(crates_position-np.array([old_player_coor[0],old_player_coor[1]]),axis = 1)
    new_crate_distance = np.linalg.norm(crates_position-np.array([new_player_coor[0],new_player_coor[1]]),axis = 1)

    if old_crate_distance.size > 0:                 #if agent moved closer to the nearest crate and BOMB action is possible 
        if min(new_crate_distance) < min(old_crate_distance) and old_game_state['self'][2]: 
            events.append(CRATE_CHASER)
        
    #get opponents
    enemys = []
    for others_coor in old_game_state['others']:
        enemys.append(others_coor[3])

    if self_action == 'BOMB' and e.INVALID_ACTION not in events:    #if bomb is placed...  

        #define event for bomb next to crate
        for i in range(len(np.where(old_crate_distance==1)[0])):    # ... give reward for each crate neighbouring bomb position                   
            events.append(BOMB_NEXT_TO_CRATE)        

        #define event for bomb not next to crate           
        if len(np.where(old_crate_distance==1)[0]) == 0 :           #bomb is not placed next to crate
            events.append(BOMB_NOT_NEXT_TO_CRATE)                   # -> penalty
            
        
        #define event for bomb near/next to opponent
        if len(old_game_state['others']) !=0:

            for others_coor in old_game_state['others']:

                if np.linalg.norm(np.subtract(old_player_coor,others_coor[3])) <=3:     #bomb placed near enemy -> reward
                    events.append(DROPPED_BOMB_NEAR_ENEMY)
                
                if np.linalg.norm(np.subtract(old_player_coor,others_coor[3])) == 1:    #bomb placed net to enemy -> reward
                    events.append(DROPPED_BOMB_NEXT_TO_ENEMY)
        

    #define opponent chaser
    if len(enemys) != 0:                                                            
        distances_old = np.linalg.norm(np.subtract(old_player_coor,enemys),axis=1)
        distances_new = np.linalg.norm(np.subtract(new_player_coor,enemys),axis=1)
        if min(distances_new) < min(distances_old):                                 #if agent moved closer to the closest enemy -> rewards
            events.append(OPPONENT_CHASER)