import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

# nemad tuple for History
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters 
TRANSITION_HISTORY_SIZE = 100  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.3 # discount rate
ALPHA = 0.5 # learning rate

# Events

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
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    try:    # try loading existing model
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    except:     
        self.model = None
    
    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)

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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.logger.info(self_action)

    
    #define auxillary events depending on the transition
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
            
    # appending gamestate to history
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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


    # Updating the model
    experience_replay(self)
    

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
   Auxillary rewards
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.WAITED: -3.5,
        e.INVALID_ACTION: -5,
        #e.MOVED_DOWN: 0,
        #e.MOVED_LEFT: 0,
        #e.MOVED_RIGHT: 0,
        #e.MOVED_UP: 0,
        COIN_CHASER: 3.5,
        MOVED_AWAY_FROM_BOMB: 5,
        WAITED_IN_EXPLOSION_RANGE: -5,
        INVALID_ACTION_IN_EXPLOSION_RANGE: -8  
    
    }
    reward_sum = 0
    
    for event in events:
        #if event == 'COIN_CHASER': print(event)
        #if event == 'INVALID_ACTION_IN_EXPLOSION_RANGE': print(event)
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def experience_replay(self):
    self.logger.debug('Doing experience replay with the collected data.')

    # creating the training batches from the transition history
    B = {'UP':{'states':[], 'rewards':[]}, 'RIGHT':{'states':[], 'rewards':[]}, 'DOWN':{'states':[], 'rewards':[]}, 
    'LEFT':{'states':[], 'rewards':[]}, 'WAIT':{'states':[], 'rewards':[]}, 'BOMB':{'states':[], 'rewards':[]}}

    D = 0   # feature dimension

    for transition in self.transitions:
        if transition.action is not None:
            B[transition.action]['states'].append(transition.state)
            B[transition.action]['rewards'].append(transition.reward)
            D = len(transition.state)

    
    #initializing the model
    if self.model is None:   
        init_beta = np.zeros(D)
        self.model = {'UP': init_beta, 
        'RIGHT': init_beta, 'DOWN': init_beta,
        'LEFT': init_beta, 'WAIT': init_beta, 'BOMB': init_beta}


    # updating the model
    for action in B:
        beta_update = []

        N = len(B[action]['states'])

        for i in range(N):
            X = B[action]['states'][i]
            total_reward = 0
            for j in range(i,N):
                total_reward += GAMMA**(j-i) * B[action]['rewards'][j]

            beta_update.append(np.dot(X, total_reward - np.clip(np.dot(X,self.model[action]),-3,3)))


        if beta_update != []:
            #diff = ALPHA/N * np.sum(beta_update,axis=0)
            self.model[action] = self.model[action] + ALPHA/N * np.sum(beta_update,axis=0)
            
            #print(ALPHA/N)
            #print(f'maximum diff {action}:',max(diff),min(diff))   

    #for action in B:
        #print(f'maximum beta {action}:',max(self.model[action]),min(self.model[action]))
    

