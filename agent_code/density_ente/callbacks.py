import os
import pickle
import random
import sklearn as sk
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from itertools import product
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        ##weights = np.random.rand(len(ACTIONS))
        self.model = None ## weights / weights.sum()
    else:
        self.model = None
        #self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)

    ##############################################################################################       
    global exploding_tiles_map
    with open('explosion_map.pt', 'rb') as file:
        exploding_tiles_map = pickle.load(file)
    ##############################################################################################



def act(self, game_state: dict) -> str:
    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    self.logger.info(state_to_features(game_state))
    if self.model == None: random_prob = 0
    else: random_prob = 0

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action according to the epsilon greedy policy.")
        betas = np.array(list(self.model.values()))
        feature_vector = np.array(state_to_features(game_state))
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
        
        #print(move)
        return move #np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.1,0.1])
    
    #if we just want to play and not overwrite training data
    if not self.train: 
        self.logger.debug("Choosing action according to the epsilon greedy policy.")
        betas = np.array(list(self.model.values()))
        feature_vector = np.array(state_to_features(game_state))
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
        return move
        
    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=[.2,.2,.2,.2,.15,0.05])


def state_to_features(game_state: dict) -> np.array:
    
    player = np.array(game_state['self'][3])
    
    low_half = np.array(list(product(np.arange(0, player[0]), np.arange(0,17))), dtype = object)
   
    up_half = np.array(list(product(np.arange(player[0], 17), np.arange(0,17))), dtype = object)
    
    left_half = np.array(list(product(np.arange(0, 17), np.arange(0,player[1]))), dtype = object)
    
    right_half = np.array(list(product(np.arange(0, 17), np.arange(player[1], 17))), dtype = object)

    segments = np.array([low_half, up_half, left_half, right_half])
    
    field = game_state['field']

    coins = np.array(game_state['coins'])

    get_coin_density(segments, coins)

def get_coin_density(segments, coins):
    
    for segment in segments:
        print(np.shape(segment))
        print(np.shape(coins[:,None,:]))
        coins_num = np.where(np.sum(np.abs(segment-coins[:,None,:]), axis = 2)==0)
        print(np.sum(np.abs(segment-coins[:,None,:]), axis = 2))
        print(coins_num)

        #np.any(np.sum(np.abs(dangerous_tiles-neighbor_pos[i]), axis=1) == 0)WS