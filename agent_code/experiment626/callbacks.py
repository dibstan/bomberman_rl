import os
import pickle
import random
import sklearn as sk
from sklearn.feature_extraction import DictVectorizer
import numpy as np



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
        #weights = np.random.rand(len(ACTIONS))
        self.model = None ## weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


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
    random_prob = .5
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action according to the epsilon greedy policy.")
        betas = list(self.model.values())
        feature_vector = state_to_features(game_state)
        
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
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
    if game_state['step']==1 and game_state['round']==1 :  
        global exploding_tiles_map
        exploding_tiles_map = get_all_exploding_tiles(game_state['field'])  #dict where keys are tuples
  
    b = 5                           #number of features per pixel
    channels = np.zeros((17*17,b))  #here rows are flattned pixels and colums are certain features for each pixel
    
    #first learn field parameters(crate,wall,tile)
    tile_values = np.stack(game_state['field']).reshape(-1) #flatten field matrix
    channels[np.where(tile_values == 1),0] = 1              #crates               
    channels[np.where(tile_values == -1),0] = -1            #walls  

    #position of player
    player_coor = game_state['self'][3]
    player_coor_flat = 17 * player_coor[0] + player_coor[1]
    channels[player_coor_flat,1] = 1

    #postition of enemys
    for enemy in game_state['others']:      #maybe also create 'danger levels'
        enemy_coor = enemy[3]
        enemy_coor_flat = 17 * enemy_coor[0] + enemy_coor[1]
        channels[enemy_coor_flat,2] = 1

    #position of bombs and their timers as 'danger' values, existing explosion maps
    for bomb in game_state['bombs']:
        bomb_coor = bomb[0]                                         #now assign danger value to all exploding tiles
        channels[exploding_tiles_map[bomb_coor],3] = 4-bomb[1]/4    #danger level = time steps passed / time needed to explode

    explosion_map = game_state['explosion_map'].flatten()
    channels[np.where(explosion_map == 2),3] = 1
    channels[np.where(explosion_map == 1),3] = 1

    #position of coins
    for coin in game_state['coins']:
        A = 5                                      #hyperparameter indicating weight for nearest coins
        max_distance = np.linalg.norm([15,15])     #max distance player-coin 
        coin_distance = np.linalg.norm(np.subtract(game_state['self'][3], coin))   #get the distance to the player 
        coin_coor_flat = 17 * coin[0] + coin[1]
        channels[coin_coor_flat,4] = A * coin_distance / max_distance
    
    
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels).reshape(-1)
    # and return them as a vector
    
    return stacked_channels #stacked_channels.reshape(-1)



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
                coors_ij.append(17*k+l)
            #walking left:
            k,l = i,j
            while (field[k,l-1] !=-1) and np.abs(l-j)<3:
                l-=1
                coors_ij.append(17*k+l)
            #walking up:
            k,l = i,j
            while (field[k-1,l] !=-1) and np.abs(k-i)<3:
                k-=1
                coors_ij.append(17*k+l)
            #walking down
            k,l = i,j
            while (field[k+1,l] !=-1) and np.abs(k-i)<3:
                k+=1
                coors_ij.append(17*k+l)
            
            exploding_tiles[(i,j)] = np.array(coors_ij)

    return exploding_tiles



            


