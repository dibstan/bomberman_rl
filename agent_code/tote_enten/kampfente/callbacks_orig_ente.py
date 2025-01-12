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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        ##weights = np.random.rand(len(ACTIONS))
        self.model = None ## weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    print(self.model)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    self.logger.info(state_to_features(game_state))
    #self.logger.info(game_state['bombs'])
    random_prob = 1

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action according to the epsilon greedy policy.")
        betas = list(self.model.values())
        feature_vector = state_to_features(game_state)
        
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
        print(move)
        return move #np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.1,0.1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=[.2,.2,.2,.2,.1,.1])


def state_to_features(game_state: dict) -> np.array:
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
    '''channels.append(np.array([game_state['round'], game_state['step'], None, None]))    # current game state holding the current round, step and score
    channels.append(np.array([game_state['self'][3][0], game_state['self'][3][1], int(game_state['self'][2] == True), game_state['self'][1]]))     # info about self: xpos, ypos, bomb available (0=False, 1=True), score
    for i in range(len(game_state['others'])):
        channels.append(np.array([game_state['others'][i][3][0], game_state['others'][i][3][1], int(game_state['others'][i][2] == True), game_state['others'][1]]))     # info about others: xpos, ypos, bomb available (0=False, 1=True), score
    for xy in coordinates:          
        channels.append(np.concatenate((xy, [game_state['field'][xy[0]][xy[1]], None])))   # all coordinates and the tile at that coordinate (1,-1,0)
        channels.append(np.concatenate((xy, [game_state['explosion_map'][xy[0]][xy[1]], None])))    # all coordinates and the current explosion state of that coordinate
    for i in range(len(game_state['bombs'])):
        channels.append([game_state['bombs'][i][0][0], game_state['bombs'][i][0][1], game_state['bombs'][i][1], None])    # info about the bombs: xpos, ypos, timer'''
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
    
    return stacked_channels #stacked_channels.reshape(-1)
