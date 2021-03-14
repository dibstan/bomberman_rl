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
    else: random_prob = 1

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action according to the epsilon greedy policy.")
        betas = np.array(list(self.model.values()))
        feature_vector = np.array(state_to_features(game_state))
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
        
        #print(move)
        return move #np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.1,0.1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=[.2,.2,.2,.2,.15,0.05])


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
    
    #get player position:
    player = np.array(game_state['self'][3])
    
    #converting positions of coins and bombs
    position_coins = np.array(game_state['coins'])
    #print('coins:', position_coins)

    ################################################################################################################
    bomb_position = []
    for i in range(len(game_state['bombs'])):
        bomb_position.append([game_state['bombs'][i][0][0], game_state['bombs'][i][0][0]] )
    bomb_position = np.array(bomb_position)
    #print('bomb_position', bomb_position)
    #################################################################################################################


    #positions of neigboring tiles in the order (Left, Right, Up, Down)
    neighbor_pos = []
    neighbor_pos.append((player[0], player[1] - 1))
    neighbor_pos.append((player[0], player[1] + 1))
    neighbor_pos.append((player[0] - 1, player[1]))
    neighbor_pos.append((player[0] + 1, player[1]))
    neighbor_pos = np.array(neighbor_pos)
    
    # distance from coins to player
    if position_coins != []:
        d_coins = np.subtract(position_coins, player)   
        
        dist_norm = np.linalg.norm(d_coins, axis = 1)
        #print(dist_norm)
        closest_coin = position_coins[dist_norm.argmin()]
        #print(dist_norm.argmin())
        
        #find direction to go for closest coin:
        d_coins_neighbor = np.subtract(neighbor_pos, closest_coin)

        #finding the direction that brings us closer the closest coin
        closest_neighbor = np.linalg.norm(d_coins_neighbor, axis = 0).argmin()

    #creating channels for one-hot encoding
    channels = np.zeros((4,6))
    
    #describing field of agent:
    player_tile = np.zeros(2) #... adjust to 3 when Danger feature is added

###################################################################################################
    #searching for near bombs:
    
    if len(bomb_position)!=0:
        bomb_distances = np.linalg.norm( np.subtract(bomb_position, player) , axis = 1)
        close_bomb_indices = np.where(bomb_distances <= 4)[0]
   

    #get explosion map (remeber: explosions last for 2 steps)
    explosion_map = game_state['explosion_map']
###################################################################################################

    #each direction is encoded by [wall, crate, coin, bomb, priority] ...danger value to come

    for i in range(np.shape(neighbor_pos)[0]):
        
        #finding a wall:
        field_value = game_state['field'][neighbor_pos[i][0]][neighbor_pos[i][1]] 

        if field_value == -1:
            channels[i][0] = 1
        
        #finding crate
        if field_value == 1:
            channels[i][1] = 1

        #finding coin:
        if position_coins != []:
            if neighbor_pos[i] in position_coins:
                channels[i][2] = 1

            #describing priority:
            if i == closest_neighbor:
                channels[i][4] = 1

############################################################################################################

        #finding bomb:
        if len(bomb_position) != 0:
            
            #bomb on neighbor?
            if neighbor_pos[i] in bomb_position:
                channels[i][3] = 1

            #bomb on player position?
            if player in bomb_position:
                player_tile[1] = 1
            
            # are there dangerous tiles in the neighbors?
            bomb_tuples = [tuple(x) for x in bomb_position]
           
            for j in close_bomb_indices:                                                     #only look at close bombs
                if bomb_tuples[j] not in exploding_tiles_map.keys(): continue               
                dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])  #get all tiles exploding with close bombs
                if neighbor_pos in dangerous_tiles:                                     #if neighbor is on dangerous tile -> set danger value
                    channels[i,5] = 1                                                   #alternative danger value increasing with timer: (4-bomb_position[j,1])/4
        

            #are there already exploding tiles in the neighbors (remember:explosions last for 2 steps)

            if len(np.where(explosion_map != 0)[0]):                                    #check if there are current explosions
                if explosion_map[neighbor_pos[i,0],neighbor_pos[i,1]] != 0:
                    channels[i,5] = 1 

#############################################################################################################
            
        
        
    #combining current channels:
    stacked_channels = np.stack(channels).reshape(-1)

    #player on coin?
    if player in position_coins:
        player_tile[0] = 1

    #player on bomb?
    if len(bomb_position)!=0:
        if player in bomb_position:
            player_tile[1] = 1

    #combining neighbor describtion with current tile describtion:
    stacked_channels = np.concatenate((stacked_channels, player_tile))

    
    #does our player have a bomb?
    player_bomb = []
    if game_state['self'][2]:
        player_bomb.append(1)
    else:
        player_bomb.append(0)

    #combining and returning state_vector:
    stacked_channels = np.concatenate((stacked_channels, player_bomb))
    

    
    return stacked_channels