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
HIST_SIZE = 1000
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

    self.transitions = deque(maxlen=HIST_SIZE)

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
    

    # updating the model using batched gradient descent n-step temporal difference 
    n_step_TD(self, N)


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # resetting the transition cache
    self.transitions = deque(maxlen=HIST_SIZE)


def reward_from_events(self, events: List[str]) -> int:
    '''
        Input: self, list of events
        Output: sum of rewards resulting from the events
    '''
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -20,
        WAITING_EVENT: -7,
        e.INVALID_ACTION: -7,
        COIN_CHASER: 7,
        VALID_ACTION: -1
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
    

    # getting closer to coins
    # get positions of the player
    old_player_coor = old_game_state['self'][3]     
    new_player_coor = new_game_state['self'][3]
        
    #define event coin_chaser
    coin_coordinates = old_game_state['coins']
    old_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,old_player_coor), axis=1)
    new_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,new_player_coor), axis=1)
    
    if min(new_coin_distances) < min(old_coin_distances):   #if the distance to closest coin got smaller
        events.append(COIN_CHASER)



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

    total_batch = np.array(self.transitions, dtype=object)      # converting the deque to numpy array for conveniency
    total_batch[-1,2] = np.zeros(D)
    
    
    total_batch_size = np.shape(total_batch)[0]
    effective_batch_size = total_batch_size - n

    effective_batch = total_batch[:effective_batch_size]    # training instances that have n next instances

    # Creating the batches
    B = {'UP':{}, 'RIGHT':{}, 'DOWN':{},'LEFT':{}, 'WAIT':{}, 'BOMB':{}}
    actions = list(B.keys())
    
    for action in actions:
        # Finding indices for the wanted instances
        index = np.where(effective_batch[:,1] == action)[0]     # indices of instances with action in subbatch
        n_next_index = index[:, None] + np.arange(n)    # indices of the n next instances for every instance in this subbatch
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
        N = np.shape(rewards)[0]    # size of subbatch

        discount = np.ones(n)*GAMMA   # discount for the i th future reward: GAMMA^i 
        for i in range(0,n):
            discount[i] = discount[i]**i
        
        if N != 0: 
            Q_TD = np.dot(rewards,discount) + GAMMA**n * Q_func(self,nth_states)    # Array holding the n-step-TD Q-value for each instance in subbatch
            
            Q = np.dot(states,self.model[action])   # Array holding the prdicted Q-value for each instance in subbatch
            
            GRADIENT = np.dot(states.T, (Q_TD-Q))   # gradient descent
            
            self.model[action] = self.model[action] + (ALPHA / N) * GRADIENT    #updating the model
            



def Q_func(self, state):
    '''
        input: self, state
        output: Q value of the best action
    '''
    vec_model = np.array(list(self.model.values()))     # vectorizing the model dict
    Q_pred = np.dot(vec_model,state.T)
    Q_max = np.max(Q_pred, axis = 0)
    return Q_max      # return the max Q value