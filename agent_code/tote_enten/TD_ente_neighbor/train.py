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

# Auxillary events
WAITING_EVENT = "WAIT"
COIN_CHASER = "CLOSER_TO_COIN"
VALID_ACTION = "VALID_ACTION"



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
        #aux_events(self, old_game_state, self_action, new_game_state, events)
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
    #print(self.model['UP'][np.where(self.model['UP'] != 0)])

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)



def reward_from_events(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    game_rewards = {
        e.COIN_COLLECTED: 15,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -20,
        WAITING_EVENT: -3,
        e.INVALID_ACTION: -5,
        COIN_CHASER: 0,
        VALID_ACTION: 0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



'''def aux_events(self, old_game_state, self_action, new_game_state, events):

    # Valid action
    if e.INVALID_ACTION not in events:
        events.append(VALID_ACTION)


    # Waiting
    if self_action == "WAIT":
        events.append(WAITING_EVENT)
    

    # getting closer to coins
    # get positions of the player
    old_player_coor = old_game_state['self'][3]     
    new_player_coor = new_game_state['self'][3]
        
    #define event coin_chaser
    coin_coordinates = old_game_state['coins']
    old_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,old_player_coor), axis=1)
    new_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,new_player_coor), axis=1)
    
    if min(new_coin_distances) < min(old_coin_distances):   #if the distance to closest coin got smaller
        events.append(COIN_CHASER)'''



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
    if transitions_array[0,1] is not None:
        # Updating the model
        if  np.shape(transitions_array)[0] == n:
            
            action = transitions_array[0,1]     # relevant action executed

            first_state = transitions_array[0,0]    # first state saved in the cache

            last_state = transitions_array[-1,0]     # last state saved in the cache

            n_future_rewards = transitions_array[:-1,3]   # rewards of the n-1 next actions

            discount = np.ones(n-1)*GAMMA   # discount for the i th future reward: GAMMA^i 
            for i in range(1,n):
                discount[i-1] = discount[i-1]**i

            #update with original state
            Q_TD = np.dot(discount, n_future_rewards) + GAMMA**n * Q_func(self, last_state)   # value estimate using n-step temporal difference
            Q = np.dot(first_state, self.model[action])     # value estimate of current model

            GRADIENT = first_state * (Q_TD - Q)     # gradient descent
            
            self.model[action] = self.model[action] + ALPHA * np.clip(GRADIENT, -10,10)   # updating the model for the relevant action


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
    
    return np.max(np.dot(vec_model, state))      # return the max Q value


def feature_augmentation(self, aug_direction, first_state, last_state, action, disc, n_future_rew, n):
    shift_first_state, shift_action = aug_direction(first_state, action)
    shift_last_state, shift_action = aug_direction(last_state, action)

    Q_TD_shift = np.dot(disc, n_future_rew) + GAMMA**n * Q_func(self, shift_last_state)   # value estimate using n-step temporal difference
    Q_shift = np.dot(shift_first_state, self.model[shift_action])     # value estimate of current model

    GRADIENT = shift_first_state * (Q_TD_shift - Q_shift)
    model_update = self.model[shift_action] + ALPHA * np.clip(GRADIENT, -10,10)   # updating the model for the relevant action

    return model_update, shift_action

def vertical_shift(state, action):
    #initializing the shifted state:
    shifted_state = np.copy(state)

    #shifting up to down:
    shifted_state[10:15] = state[15:20]
    shifted_state[15:20] = state[10:15]

    #shifting actions
    if action == "UP":
        new_action = "DOWN"
    
    elif action == "DOWN":
        new_action = "UP"

    else:
        new_action = action

    return shifted_state, new_action

def horizontal_shift(state, action):
    #initializing the shifted state:
    shifted_state = np.copy(state)

    #shifting up to down:
    shifted_state[0:5] = state[5:10]
    shifted_state[5:10] = state[0:5]

    #shifting actions
    if action == "LEFT":
        new_action = "RIGHT"
    
    elif action == "RIGHT":
        new_action = "LEFT"

    else:
        new_action = action

    return shifted_state, new_action

def turn_left(state, action):
    #initializing the turned state:
    turned_state = np.copy(state)

    #up -> left 
    turned_state[0:5] = state[10:15]
    #down -> right
    turned_state[5:10] = state[15:20]
    #right -> up
    turned_state[10:15] = state[5:10]
    #left -> down
    turned_state[15:20] = state[0:5]

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

def turn_right(state, action):
    #initializing the turned state:
    turned_state = np.copy(state)

    #up -> left 
    turned_state[0:5] = state[15:20]
    #down -> right
    turned_state[5:10] = state[10:15]
    #right -> up
    turned_state[10:15] = state[0:5]
    #left -> down
    turned_state[15:20] = state[5:10]

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

def turn_around(state, action):
    #first turn:
    turn1, action1 = turn_left(state, action)

    #second turn:
    turn2, action2 = turn_left(turn1, action1)

    return turn2, action2