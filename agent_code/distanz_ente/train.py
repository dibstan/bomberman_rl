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
GAMMA = 0.03 # discount rate
ALPHA = 0.01 # learning rate

# Events
WAITING_EVENT = "WAIT"
COIN_CHASER = "CLOSER_TO_COIN"
MOVED_AWAY_FROM_BOMB = "MOVED_AWAY_FROM_BOMB"
WAITED_IN_EXPLOSION_RANGE = "WAITED_IN_EXPLOSION_RANGE"

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
        e.KILLED_SELF: -3,
        WAITING_EVENT: -1,
        e.INVALID_ACTION: -10,
        #e.MOVED_DOWN: -40,
        #e.MOVED_LEFT: -40,
        #e.MOVED_RIGHT: -40,
        #e.MOVED_UP: -40,
        #COIN_CHASER: 50,
        #MOVED_AWAY_FROM_BOMB: 50,
        #WAITED_IN_EXPLOSION_RANGE: -100  
    
    }
    reward_sum = 0
    
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def experience_replay(self):
    self.logger.debug('Doing experience replay with the collected data.')

    # creating the training batches from the transition history
    B = {'UP':{'states':[], 'rewards':[], 'next_states':[]}, 'RIGHT':{'states':[], 'rewards':[], 'next_states':[]}, 'DOWN':{'states':[], 'rewards':[], 'next_states':[]}, 
    'LEFT':{'states':[], 'rewards':[], 'next_states':[]}, 'WAIT':{'states':[], 'rewards':[], 'next_states':[]}, 'BOMB':{'states':[], 'rewards':[], 'next_states':[]}}

    D = 0   # feature dimension

    for transition in self.transitions:
        if transition.action is not None:
            B[transition.action]['states'].append(transition.state)
            B[transition.action]['rewards'].append(transition.reward)
            B[transition.action]['next_states'].append(transition.next_state)
            D = len(transition.state)

    
    #initializing the model
    if self.model is None:   
        init_beta = np.zeros(D)
        self.model = {'UP': init_beta, 
        'RIGHT': init_beta, 'DOWN': init_beta,
        'LEFT': init_beta, 'WAIT': init_beta, 'BOMB': init_beta}


    # updating the model
    for action in B:
        X = B[action]['states']
        Y_TD = []
        Y = []

        N = len(X)

        if self.model is not None:
            for i in range(N):

                if B[action]['next_states'][i] is not None:     # not terminal state
                    
                    # computing the reward for the state according to temporal difference
                    q_value_future = []

                    for move in self.model:
                        q_value_future.append(np.dot(self.model[move],B[action]['next_states'][i]))

                    future_reward = np.max(q_value_future)

                    
                    Y_TD.append(B[action]['rewards'][i] + GAMMA * future_reward)
                    
                    # computing the predicted reward
                    Y.append(np.dot(self.model[move], X[i]))

                else:   # terminal state
                    
                    Y_TD.append(B[action]['rewards'][i])

                    Y.append(np.dot(self.model[action], X[i]))

            if X != []:
                
                DESC  = np.dot(np.transpose(X), np.array(Y_TD)-np.array(Y))    # gradient descent
                
                self.model[action] = self.model[action] + ALPHA * np.clip(DESC, -10,10)
                

    #print(self.model['UP'][np.where(self.model['UP'] != 0)])

