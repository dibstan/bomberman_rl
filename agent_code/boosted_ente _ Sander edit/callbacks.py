import os
import pickle
import random
import sklearn as sk
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from itertools import product
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


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
    
    with open('explosion_map.pt', 'rb') as file:
        self.exploding_tiles_map = pickle.load(file)

    ##############################################################################################       
    global exploding_tiles_map
    with open('explosion_map.pt', 'rb') as file:
        exploding_tiles_map = pickle.load(file)
    ##############################################################################################


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

        q_value={'UP':0,'RIGHT':0,'DOWN':0,'LEFT':0,'WAIT':0,'BOMB':0}
        for move in self.model:
            q_value[move] = self.model[move].predict(np.reshape(state_to_features(game_state),(1,-1)))
        move = list(q_value.keys())[np.argmax(list(q_value.values()))]
        print(q_value)
        #print(q_value)
        #print(move)

        return move

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=[.2,.2,.2,.2,.1,.1])



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

    #getting bomb position from state
    bomb_position = []
    for i in range(len(game_state['bombs'])):
        bomb_position.append([game_state['bombs'][i][0][0], game_state['bombs'][i][0][1]] )
    bomb_position = np.array(bomb_position)

    #getting position of other players from state:
    other_position = []
    for i in range(len(game_state['others'])):
        other_position.append([game_state['others'][i][3][0], game_state['others'][i][3][1]])
    other_position = np.array(other_position)
    

    #positions of neigboring tiles in the order (UP, DOWN, LEFT, RIGHT)
    neighbor_pos = []
    neighbor_pos.append((player[0], player[1] - 1))
    neighbor_pos.append((player[0], player[1] + 1))
    neighbor_pos.append((player[0] - 1, player[1]))
    neighbor_pos.append((player[0] + 1, player[1]))
    neighbor_pos = np.array(neighbor_pos)
    
    # distance from coins to player
    if position_coins.size > 0:
        d_coins = np.subtract(position_coins, player)   
        
        dist_norm = np.linalg.norm(d_coins, axis = 1)
        
        closest_coin = position_coins[dist_norm.argmin()]
        
        
        #find direction to go for closest coin:
        d_coins_neighbor = np.subtract(neighbor_pos, closest_coin)
        
        #finding the direction that brings us closer the closest coin
        closest_neighbor = np.linalg.norm(d_coins_neighbor, axis = 1)
        priority_index = np.argsort(closest_neighbor)


    # distance from others to player
    if other_position.size > 0:
        d_others = np.subtract(other_position, player)   
        
        dist_norm_others = np.linalg.norm(d_others, axis = 1)
        index_closest = dist_norm_others.argmin()
        closest_others = other_position[index_closest]
        
        
        #find direction to go for closest coin:
        d_others_neighbor = np.subtract(neighbor_pos, closest_others)
        
        #finding the direction that brings us closer the closest coin
        closest_neighbor_other = np.linalg.norm(d_others_neighbor, axis = 1)
        others_index = np.argsort(closest_neighbor_other)

    #creating channels for one-hot encoding
    channels = np.zeros((4,8))
    
    #describing field of agent:
    player_tile = np.zeros(2) #... adjust to 3 when Danger feature is added


    #searching for near bombs:
    
    if len(bomb_position) != 0:
        bomb_distances = np.linalg.norm(np.subtract(bomb_position, player) , axis = 1)
        close_bomb_indices = np.where(bomb_distances <= 4)[0]
   

    #get explosion map (remeber: explosions last for 2 steps)
    explosion_map = game_state['explosion_map']

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
        if position_coins.size > 0:
            if np.any(np.sum(np.abs(position_coins-neighbor_pos[i]), axis=1) == 0):
                channels[i][2] = 1


        #finding bomb:
        if len(bomb_position) != 0:
            
            #bomb on neighbor?
            if np.any(np.sum(np.abs(bomb_position-neighbor_pos[i]), axis=1) == 0):
                channels[i][3] = 1

            #bomb on player position?
            if player in bomb_position:
                player_tile[1] = 1
            
            # are there dangerous tiles in the neighbors?
            bomb_tuples = [tuple(x) for x in bomb_position]
            #print(bomb_tuples)
            
            for j in close_bomb_indices:                                                     #only look at close bombs
                #if bomb_tuples[j] not in exploding_tiles_map.keys(): continue               
                dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])         #get all tiles exploding with close bombs
                if np.any(np.sum(np.abs(dangerous_tiles-neighbor_pos[i]), axis=1) == 0):
                                                         #if neighbor is on dangerous tile -> set danger value
                    channels[i,5] = 1                                                   #alternative danger value increasing with timer: (4-bomb_position[j,1])/4

                #if player on dangerous tile, add 1 to player tile danger index
                if np.any(np.sum(np.abs(dangerous_tiles - player), axis=1) == 0):
                    player_tile[0] = 1

            #are there already exploding tiles in the neighbors (remember:explosions last for 2 steps)
            if len(np.where(explosion_map != 0)[0]):                                    #check if there are current explosions
                if explosion_map[neighbor_pos[i,0],neighbor_pos[i,1]] != 0:
                    channels[i,5] = 1 

            
    
    #describing priority: 
    if position_coins.size > 0:
        for i in range(len(priority_index)):
            if channels[priority_index[i]][0] != 1:
                channels[priority_index[i]][4] = 1
                break

    if other_position.size > 0:
        for i in range(len(others_index)):
            if channels[others_index[i]][0] != 1:
                channels[others_index[i]][6] = 1
                
                #does the player have a bomb?
                if game_state['others'][index_closest][2]:
                    channels[others_index[i]][7] = 1
                break 
        
    #combining current channels:
    stacked_channels = np.stack(channels).reshape(-1)


    #player on bomb?
    if len(bomb_position)!=0:
        if np.any(np.sum(np.abs(bomb_position- player), axis=1) == 0):
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
    

    #print(stacked_channels)
    return stacked_channels


 