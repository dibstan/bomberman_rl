import os
import pickle
import random
import sklearn as sk
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from itertools import product


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
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # This code needs to be executed when PCA has already done for feature reduction.
    with open("PCA.pt", "rb") as file:
        self.pca = pickle.load(file)
    

def act(self, game_state: dict) -> str:
    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    self.logger.info(state_to_features(game_state))
    random_prob = 0.8

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action according to the epsilon greedy policy.")
        betas = np.array(list(self.model.values()))
        feature_vector = np.array(state_to_features(game_state))
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
        
        #print(move)
        return move #np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.1,0.1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=[.2,.2,.2,.2,.1,0.1])


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*


    Converts the game state to the input of your model, i.e.
    a feature vector.print(new_state_vector[0])

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None
    
    channels = []
    
    coordinates = np.array(list(product(np.arange(0,17),np.arange(0,17))))  # generating a list holding all possible coordinates of the field

    position_self = np.array(game_state['self'][3])     # position of player self


    # COINS
    position_coins = np.array(game_state['coins'])      # position of the coins
        
    d_coins = position_coins - position_self   # distance from coins to player

    for i in range(9):
        if i < len(d_coins):
            channels.append(d_coins[i])
        else:
            channels.append([0,0])

    # SELF    
    channels.append(position_self)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels).reshape(-1)
    # and return them as a vector
    
    return stacked_channels.reshape(-1)

'''def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*


    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends

    if game_state is None:
        return None
    # For example, you could construct several channels of equal shape, ...
    channels = []
    coordinates = np.array(list(product(np.arange(0,17),np.arange(0,17))))  # generating a list holding all possible coordinates of the field 
    channels.append([game_state['self'][1], int(game_state['self'][2] == True), game_state['self'][3][0], game_state['self'][3][1], game_state['round'], 0])#, game_state['step']])
    for xy in coordinates:
        field_state = game_state['field'][xy[0]][xy[1]]
        explosion_state = game_state['explosion_map'][xy[0]][xy[1]]
        bomb_countdown = -1
        for bomb in game_state['bombs']:
            if (xy[0],xy[1]) in bomb:
                bomb_countdown = bomb[1]
        coin_state = 0
        for coin in game_state['coins']:
            if (xy[0],xy[1]) == coin:
                coin = 1
        other_state = 0
        other_bomb = 0
        for other in game_state['others']:
            if (xy[0],xy[1]) == other[3]:
                other_state = 1
                other_score = other[1]
                other_bomb = int(other[2] == True)
        channels.append([field_state, explosion_state, bomb_countdown, coin_state, other_state, other_bomb])
    
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels).reshape(-1)
    # and return them as a vector
    
    return stacked_channels #stacked_channels.reshape(-1)'''