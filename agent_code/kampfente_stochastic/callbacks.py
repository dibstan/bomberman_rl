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
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    
    global exploding_tiles_map
    with open('explosion_map.pt', 'rb') as file:
        exploding_tiles_map = pickle.load(file)
    



def act(self, game_state: dict) -> str:
    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    self.logger.info(state_to_features(game_state))
    random_prob = 0.5


    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action according to the epsilon greedy policy.")
        betas = np.array(list(self.model.values()))
        feature_vector = np.array(state_to_features(game_state))
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
        
        #print(move)
        return move 
    if not self.train:
        betas = np.array(list(self.model.values()))
        feature_vector = np.array(state_to_features(game_state))
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
        
        #print(betas)
        return move 
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

    #creating channels for one-hot encoding
    channels = np.zeros((4,9))
    
    #describing field of agent:
    player_tile = np.zeros(2)
    
    '''finding positions of coins, bombs, dangerous tiles, crates and walls'''

    #get player position:
    player = np.array(game_state['self'][3])

    #get field values
    field = game_state['field']

    #get explosion map (remeber: explosions last for 2 steps)
    explosion_map = game_state['explosion_map']

    #getting bomb position from state 
    '''try to vectorize'''
    bomb_position = get_bomb_position(game_state)
    
    #positions of neighboring tiles in the order (UP, DOWN, LEFT, RIGHT)
    neighbor_pos = get_neighbor_pos(player)
    
    #getting position of coins and priority of neighboring tiles
    position_coins, priority_index = get_coin_prio(game_state, neighbor_pos, player)

    #getting position of other players and direction
    other_position, others_index, index_closest = get_player_prio(game_state, neighbor_pos, player)


    #searching for near bombs:
    if len(bomb_position) != 0:
        bomb_distances = np.linalg.norm(np.subtract(bomb_position, player) , axis = 1)
        close_bomb_indices = np.where(bomb_distances <= 4)[0]


    '''filling neighboring tiles with values characterizing the direction'''

        

    #each direction is encoded by [wall, crate, coin, bomb, priority, danger, closest_other, other_bomb]
    for i in range(np.shape(neighbor_pos)[0]):
        
        #is the neighbor a wall, crate or coin?
        channels = get_neighbor_value(game_state, channels, neighbor_pos, position_coins, i)

        #finding bomb:
        if len(bomb_position) != 0:
            #setting bomb value or danger value to 1 if appropriate:
            channels, player_tile = get_neighbor_danger(game_state, channels, neighbor_pos, close_bomb_indices, exploding_tiles_map, bomb_position, player, player_tile, explosion_map, i)

    #print(player_tile)        
    #describing pritority for next free tile if a bomb has been placed
    if player_tile[0] == 1:
        free_tile = find_closest_free_tile(game_state, player, close_bomb_indices, bomb_position)
        if free_tile is not None:
            closest_free_index = get_tile_prio(free_tile, neighbor_pos)

            for i in range(len(closest_free_index)):
                if channels[closest_free_index[i]][0] != 1:
                    channels[closest_free_index[i]][8] = 1
                    break


    #describing priority: 
    if position_coins.size > 0:
        for i in range(len(priority_index)):
            if channels[priority_index[i]][0] != 1 and channels[priority_index[i]][1] != 1:
                channels[priority_index[i]][4] = 1
                break
    
    else:
        crates_position, crate_index = get_crate_prio(game_state, neighbor_pos, player, field)
        for i in range(len(crate_index)):
            if channels[crate_index[i]][0] != 1:
                channels[crate_index[i]][4] = 1
                break

    if other_position.size > 0:
        for i in range(len(others_index)):
            if channels[others_index[i]][0] != 1:
                channels[others_index[i]][6] = 1
                
                #does the player have a bomb?
                if game_state['others'][index_closest][2]:
                    channels[others_index[i]][7] = 1
                break 
        

    #player on bomb?
    if len(bomb_position)!=0:
        if np.any(np.sum(np.abs(bomb_position- player), axis=1) == 0):
            player_tile[1] = 1
    
    #combining current channels:
    stacked_channels = np.stack(channels).reshape(-1)
    
    #combining neighbor describtion with current tile describtion:
    stacked_channels = np.concatenate((stacked_channels, player_tile))
    
    #does our player have a bomb?
    own_bomb = []
    if game_state['self'][2]:
        own_bomb.append(1)
    else:
        own_bomb.append(0)
    
    stacked_channels = np.concatenate((stacked_channels, own_bomb))
    #print(stacked_channels)
    return stacked_channels

def get_coin_prio(game_state, neighbor_pos, player):
    
    #converting positions of coins
    position_coins = np.array(game_state['coins'])

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

        return position_coins, priority_index

    priority_index = []
    return position_coins, priority_index

def get_neighbor_pos(player):
    #positions of neigboring tiles in the order (UP, DOWN, LEFT, RIGHT)
    neighbor_pos = []
    neighbor_pos.append((player[0], player[1] - 1))
    neighbor_pos.append((player[0], player[1] + 1))
    neighbor_pos.append((player[0] - 1, player[1]))
    neighbor_pos.append((player[0] + 1, player[1]))
    
    return np.array(neighbor_pos)

def get_player_prio(game_state, neighbor_pos, player):
    
    #getting position of other players from state:
    other_position = []
    for i in range(len(game_state['others'])):
        other_position.append([game_state['others'][i][3][0], game_state['others'][i][3][1]])
    other_position = np.array(other_position)

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

        return other_position, others_index, index_closest

    index_closest = None
    others_index = []
    return other_position, others_index, index_closest

def get_bomb_position(game_state):
    #getting bomb position from state
    bomb_position = []
    for i in range(len(game_state['bombs'])):
        bomb_position.append([game_state['bombs'][i][0][0], game_state['bombs'][i][0][1]] )
    return np.array(bomb_position)

def get_neighbor_value(game_state, channels, neighbor_pos, position_coins, i):
    
        #importing field values in neighbor position
        field_value = game_state['field'][neighbor_pos[i][0]][neighbor_pos[i][1]] 
        
        #finding a wall:
        if field_value == -1:
            channels[i][0] = 1
        
        #finding crate
        if field_value == 1:
            channels[i][1] = 1

        #finding coin:
        if position_coins.size > 0:
            if np.any(np.sum(np.abs(position_coins-neighbor_pos[i]), axis=1) == 0):
                channels[i][2] = 1

        return channels

def get_neighbor_danger(game_state, channels, neighbor_pos, close_bomb_indices, exploding_tiles_map, bomb_position, player, player_tile, explosion_map, i):

    #bomb on neighbor?
    if np.any(np.sum(np.abs(bomb_position-neighbor_pos[i]), axis=1) == 0):
        channels[i][3] = 1

    #bomb on player position?
    if player in bomb_position:
        player_tile[1] = 1
    
    # are there dangerous tiles in the neighbors?
    bomb_tuples = [tuple(x) for x in bomb_position]
    
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

    return channels, player_tile
  
def find_closest_free_tile(game_state, player_pos, close_bomb_indices, bomb_position):

    field =  game_state['field']
    field = np.where(field == -1 , 1, field)            
    #print(field)                 #set crates and walls to 1 
    bomb_tuples = [tuple(x) for x in bomb_position]    

    for j in close_bomb_indices:                                                #only look at close bombs              
        dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])         #get all tiles exploding with close bombs
        for tile in dangerous_tiles:
            if field[tile[0],tile[1]] == 0: field[tile[0],tile[1]]=2 
    #print(field)
    neighbors = get_neighbor_pos(player_pos)
    #print(neighbors)
    q = deque()
    for neighbor in neighbors:
        
        if field[neighbor[0], neighbor[1]] != 1:
            #print(neighbor, field[neighbor[0], neighbor[1]])
            q.append(neighbor)
    
    #Number of searched tiles:
    it = 0
    #print(q)
    while it <= 50:
        it += 1

        pos = q.popleft()
        neighbors = get_neighbor_pos(pos)
        #print(neighbors)

        for neighbor in neighbors:
            #print(field[neighbor[0], neighbor[1]])
            if field[neighbor[0], neighbor[1]] == 0:
                closest_tile = neighbor
                return closest_tile
            if field[neighbor[0], neighbor[1]] == 2:
                q.append(neighbor)
            else:
                continue

    return None
    
def get_tile_prio(tile,neighbors):
    #finding the direction that brings us closer the closest coin
    closest_neighbor = np.linalg.norm(neighbors-tile, axis=1)
    priority_index = np.argsort(closest_neighbor)

    return priority_index

def get_crate_prio(game_state, neighbor_pos, player, field):

    crates_position = np.array([np.where(field == 1)[0], np.where(field == 1)[1]]).T
    
    distances = []
    
    # distance from crates to player
    if crates_position.size > 0:
                
        d_crates = np.subtract(crates_position, player)   
    
        dist_norm = np.linalg.norm(d_crates, axis = 1)
        
        closest_crate = crates_position[dist_norm.argmin()]
        
        #find direction to go for closest coin:
        d_crates_neighbor = np.linalg.norm(np.subtract(neighbor_pos, closest_crate), axis = 1)
        
        priority_index = np.argsort(d_crates_neighbor)
        
        return crates_position, priority_index
    
    priority_index = []
    return crates_position, priority_index
    
    
