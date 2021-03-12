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
    #self.history = {'UP' = [], 'RIGHT' = [], 'DOWN' = [], 'LEFT' = [], 'WAIT' = [], 'BOMB' = []}
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    except:
        self.model = None
    self.temp_model = self.model
    self.exploding_tiles_map = None

    
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
    # Idea: Add your own events to hand out rewards

    
     
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    new_state_vector = state_to_features(new_game_state)

    #first we need to get the exposion map of each bomb to determine 'bad' tiles
    #this is fixed thoughout every game, so we just calculate it once
    if self.exploding_tiles_map == None :  
        self.exploding_tiles_map = get_all_exploding_tiles(new_game_state['field'])  #dict where keys are tuples
    
    #get the coordinates of the bombs and the player
    old_player_coor = old_game_state['self'][3]
    new_player_coor = new_game_state['self'][3]
    old_bomb_coors = old_game_state['bombs'][:,0]

    dangerous_tiles = []
    for bomb in old_bomb_coors:
        dangerous_tiles.append(self.exploding_tiles_map(bomb))
    
    if dangerous_tiles != []:
        if old_player_coor in dangerous_tiles and new_player_coor not in dangerous_tiles:
            events.append(MOVED_AWAY_FROM_BOMB)
        if old_player_coor in dangerous_tiles and self_action == "WAIT":
            events.append(WAITED_IN_EXPLOSION_RANGE)
        
    



    if self_action == "WAIT":
        events.append(WAITING_EVENT)
    
    #setting up model if necessary
    if self.model == None:
        init_beta = np.zeros(len(new_state_vector))
        self.temp_model = {'UP': init_beta, 
        'RIGHT': init_beta, 'DOWN': init_beta,
        'LEFT': init_beta, 'WAIT': init_beta, 'BOMB': init_beta}
        self.model = self.temp_model
    #initializing with arbitrary alpha as hyperparameter and transition_history_size as batch-size:
    if old_game_state is not None:
        alpha = .1
        beta = .1

        #define coin event
        old_state_vector = state_to_features(old_game_state)
        coins = np.arange(4, len(new_state_vector), 5)
        coin_dist_old = old_state_vector[coins]
        coin_dist_new = new_state_vector[coins]

        if max(coin_dist_new) > max(coin_dist_old):
            events.append(COIN_CHASER)

        #define events with bombs
        old_player_coor = old_game_state['self'][3]
        new_player_coor = new_game_state['self'][3]
        old_bomb_coors = old_game_state['bombs'][:,0]

        dangerous_tiles = []
        for bomb in old_bomb_coors:
            dangerous_tiles.append(self.exploding_tiles_map(bomb))
        
        if dangerous_tiles != []:
            if old_player_coor in dangerous_tiles and new_player_coor not in dangerous_tiles:
                events.append(MOVED_AWAY_FROM_BOMB)
            if old_player_coor in dangerous_tiles and self_action == "WAIT":
                events.append(WAITED_IN_EXPLOSION_RANGE)
        
    #get the rewards
        try:
            reward = reward_from_events(self,events)
            
        except:
            reward = 0

        gradient_vector = np.dot(np.transpose(old_state_vector) , reward + beta*q_func(self,new_state_vector) - np.dot(old_state_vector, self.temp_model[self_action]))
        self.temp_model[self_action] = self.temp_model[self_action] + alpha/ 2 * gradient_vector
    

    


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
    
    last_state_vector = state_to_features(last_game_state)
    reward = reward_from_events(self,events)
    alpha = .1

    gradient_vector = np.dot(np.transpose(last_state_vector) , reward - np.dot(last_state_vector, self.temp_model[last_action]))
    self.temp_model[last_action] = self.temp_model[last_action] + alpha/ 2 * gradient_vector
    
    # Store the model
    self.model = self.temp_model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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
        COIN_CHASER: 30,
        MOVED_AWAY_FROM_BOMB: 40,
        WAITED_IN_EXPLOSION_RANGE: -100  
    
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



def get_all_exploding_tiles(field) -> dict:
    '''
    For each pixel where we can place a bomb, we search all the tiles blowing up with that bomb

    The function is rather complicated, so we just call it once in state_to features in the 
    beginning as a global varable

    :param field: input must be game_state_field (state itself is arbitrary)
    :return: dict where keys are coordinate tuples and values arrays of flattened coordinates
    '''
    
    np.where(field == -1,-1,0)     #set all walls to -1, rest to 0
    
    exploding_tiles = {}

    for i in range(17):
        for j in range(17):
            if field[i,j] == -1:
                continue
            coors_ij=[17*i + j]

            #first consider walking to the right, stop when encounter -1 or after 3 steps
            k,l = i,j
            while (field[k,l+1] !=-1) and np.abs(l-j)<3:
                l+=1
                coors_ij.append((k,l))
            #walking left:
            k,l = i,j
            while (field[k,l-1] !=-1) and np.abs(l-j)<3:
                l-=1
                coors_ij.append((k,l))
            #walking up:
            k,l = i,j
            while (field[k-1,l] !=-1) and np.abs(k-i)<3:
                k-=1
                coors_ij.append((k,l))
            #walking down
            k,l = i,j
            while (field[k+1,l] !=-1) and np.abs(k-i)<3:
                k+=1
                coors_ij.append((k,l))
            
            exploding_tiles[(i,j)] = np.array(coors_ij)

    return exploding_tiles

