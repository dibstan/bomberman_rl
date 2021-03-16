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
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
WAITING_EVENT = "WAIT"
COIN_CHASER = "CLOSER_TO_COIN"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    try:
        with open("saved_states.pt", "rb") as file:
            self.states = pickle.load(file)
    except:
        self.states = []
        print("FUCK")
    #print(np.shape(self.states))
    
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    except:
        self.model = None
    self.temp_model = self.model
    
    #print(self.model["RIGHT"])
    
    # This Code needs to be executed when all states have been collected
    # and a new pca model has to be set up
    '''
    self.pca = PCA(self)
    with open("PCA.pt", "wb") as file:
        pickle.dump(self.pca, file)
    print(np.shape(self.pca))
    '''

    # This code needs to be executed when PCA has already done for feature reduction.
    with open("PCA.pt", "rb") as file:
        self.pca = pickle.load(file)

    
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
    
    # Idea: Add your own events to hand out rewards
    if self_action == "WAIT":
        events.append(WAITING_EVENT)
    
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    new_state_vector = state_to_features(new_game_state)
    
    #For feature reduction when pca has been done.
    #new_state_vector = np.dot(self.pca, new_state_vector)
    
    #saving new_game_state in self.states
    #For collecting new states for new PCA
    '''self.states.append(new_state_vector)'''
    
    #setting up model if necessary
    if self.model == None:
        init_beta = np.zeros(len(new_state_vector))
        self.temp_model = {'UP': init_beta, 
        'RIGHT': init_beta, 'DOWN': init_beta,
        'LEFT': init_beta, 'WAIT': init_beta, 'BOMB': init_beta}
        self.model = self.temp_model
    
    #initializing with arbitrary alpha as hyperparameter and transition_history_size as batch-size:
    if old_game_state is not None:
        alpha = 0.1
        beta = 0.5
        old_state_vector = state_to_features(old_game_state)
        #old_state_vector = np.dot(self.pca, old_state_vector)
        
        # Auxillary reward for getting closer to closest coin
        coins = np.arange(4, len(new_state_vector), 5)
        coin_dist_old = old_state_vector[coins]
        coin_dist_new = new_state_vector[coins]

        if max(coin_dist_new) > max(coin_dist_old):
            events.append(COIN_CHASER)

        try:
            reward = reward_from_events(self,events)
            
        except:
            reward = 0
        #print(reward + np.clip(beta*q_func(self,new_state_vector) - np.dot(old_state_vector, self.temp_model[self_action]), -500, 500))
        
        gradient_vector = np.dot(np.transpose(old_state_vector) , reward + np.clip(beta*q_func(self,new_state_vector) - np.dot(old_state_vector, self.temp_model[self_action]), -50000, 50000))
        #print(np.shape(gradient_vector))
        self.temp_model[self_action] = self.temp_model[self_action] + alpha/ 2 * gradient_vector
        self.model = self.temp_model
    


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
    
    last_state_vector = state_to_features(last_game_state)
    #last_state_vector = np.dot(self.pca, last_state_vector)
    
    reward = reward_from_events(self,events)
    alpha = .1
    beta = 0.5
    #print(reward + np.dot(last_state_vector, self.temp_model[last_action]))
    gradient_vector = np.dot(np.transpose(last_state_vector) , reward + beta*q_func(self, last_state_vector)- np.dot(last_state_vector, self.temp_model[last_action]))
    self.temp_model[last_action] = self.temp_model[last_action] + alpha/ 2 * gradient_vector
    # Store the model
    print(self.temp_model[last_action][np.where(self.temp_model[last_action] != 0)])
    self.model = self.temp_model

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



def q_func(self, X):
    q_array = []
    q_array.append(np.dot(X, self.model["UP"]))
    q_array.append(np.dot(X, self.model["RIGHT"]))
    q_array.append(np.dot(X, self.model["DOWN"]))
    q_array.append(np.dot(X, self.model["LEFT"]))
    q_array.append(np.dot(X, self.model["WAIT"]))
    q_array.append(np.dot(X, self.model["BOMB"]))
    return max(q_array)

def PCA(self):
    from sklearn.preprocessing import StandardScaler
    states = self.states
    print(np.shape(states))
    states_std = states - np.mean(states, axis = 0)
    

    states_cov = np.cov(np.transpose(states))

    eig_vals, eig_vecs = np.linalg.eig(states_cov)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]

    ind = np.ceil(1/3*len(eig_vecs)) 
    eig_vecs = eig_vecs[:500]
    
    return eig_vecs