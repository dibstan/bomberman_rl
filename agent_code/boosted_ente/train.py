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
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability 
PARAMS = {'random_state':0, 'warm_start':True, 'n_estimators':500, 'learning_rate':0.1, 'max_depth':500,'max_features': 500}  # parameters for the GradientBoostingRegressor
GAMMA = 0.8


# Events
WAITING_EVENT = "WAIT"
COIN_CHASER = "CLOSER_TO_COIN"
MOVED_AWAY_FROM_BOMB = "MOVED_AWAY_FROM_BOMB"
WAITED_IN_EXPLOSION_RANGE = "WAITED_IN_EXPLOSION_RANGE"

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

    # additional events
    if self_action == "WAIT":
        events.append(WAITING_EVENT)

    # Auxillary events
    if self_action == "WAIT":
        events.append(WAITING_EVENT)

    
    #define auxillary events depending on the transition
    if old_game_state is not None:

        #get positions of the player
        old_player_coor = old_game_state['self'][3]     
        new_player_coor = new_game_state['self'][3]
        
        #define event coin_chaser
        coin_coordinates = old_game_state['coins']
        old_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,old_player_coor),axis=0)
        new_coin_distances = np.linalg.norm(np.subtract(coin_coordinates,new_player_coor),axis=0)

        if max(new_coin_distances) > max(old_coin_distances):   #if the distance to closest coin got smaller
            events.append(COIN_CHASER)

        #define events with bombs
        old_bomb_coors = old_game_state['bombs']

        dangerous_tiles = []
        for bomb in old_bomb_coors:
            dangerous_tiles.append(self.exploding_tiles_map[bomb[0]])
        
        if dangerous_tiles != []:
            if old_player_coor in dangerous_tiles and new_player_coor not in dangerous_tiles:
                events.append(MOVED_AWAY_FROM_BOMB)
            if old_player_coor in dangerous_tiles and self_action == "WAIT":
                events.append(WAITED_IN_EXPLOSION_RANGE)
     
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, reward_from_events(self, events), state_to_features(new_game_state)))

    

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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, reward_from_events(self, events), None))
    
    experience_replay(self)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)



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
        e.INVALID_ACTION: -10
    }
    #print(events)
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #print(reward_sum)
    return reward_sum

def experience_replay(self):
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
    for action in B:
        X = B[action]['states']
        Y = [] #B[action]['rewards']
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
        #print(X,Y)
        if X != [] and Y != []:
            self.model[action].fit(X,Y)
           


