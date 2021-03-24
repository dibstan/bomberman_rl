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
        #print(os.path.isfile("my-saved-model.pt"))
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
    if self.model == None: random_prob = 0
    else: random_prob = 0.7

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
        #print(betas)
        feature_vector = np.array(state_to_features(game_state))
        #print(feature_vector[8],feature_vector[18],feature_vector[28],feature_vector[38])
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
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

    #creating channels for one-hot encoding
    channels = np.zeros((4,6))


    #get player position:
    player = np.array(game_state['self'][3])
    
    #get field values
    field = np.array(game_state['field'])
    
    #get explosion map (remeber: explosions last for 2 steps)
    explosion_map = game_state['explosion_map']

    #getting bomb position from state 
   
    bomb_position = get_bomb_position(game_state)
    
    #positions of neighboring tiles in the order (UP, DOWN, LEFT, RIGHT)
    neighbor_pos = get_neighbor_pos(player)
    #print('neighbors : ',neighbor_pos)
    
    #getting segments
    segments = get_segments(player)

    #getting position of other players and distance
    dist_others, other_position = get_player_dist(game_state, segments, player)
   
    #getting position of crates and distance
    dist_crates, crates_position = get_crate_dist(field, segments, player)

    #searching for near bombs:
    player_on_bomb = False
    close_bomb_indices = []
    if len(bomb_position) != 0:
        bomb_distances = np.linalg.norm(np.subtract(bomb_position, player) , axis = 1)
        close_bomb_indices = np.where(bomb_distances <= 4)[0]

    #getting position of closest coin and assign each neighbor the number of steps to it 
    position_coins = np.array(game_state['coins'])
    if position_coins.size > 0:
        
        dist_coins = np.linalg.norm(position_coins-player,axis = 1)                     #find the closest coin
        closest_coin_index = np.argmin(dist_coins)
        closest_coin = position_coins[closest_coin_index]
        if np.linalg.norm(closest_coin - player) == 0:                                  #if player spawns on coin
            steps_to_coins = np.zeros(4)

        else:
            steps_to_coins = np.zeros(4)
            for j, neighbor in enumerate(neighbor_pos):                                     #get the number of steps from every neighbor to closest coin
                steps_to_coins[j] = search_for_obj(player, neighbor, closest_coin ,field)   #this function returns inverse stepnumber
        
            steps_to_coins = steps_to_coins * 1/max(steps_to_coins)                      

    #each direction is encoded by [wall, crate, coin, bomb, priority, danger, closest_other, other_bomb]
    for i in range(np.shape(neighbor_pos)[0]):
        
        #is the neighbor a wall, crate or coin?
        channels = get_neighbor_value(game_state, channels, neighbor_pos, position_coins, i)

        #finding bomb:
        if len(close_bomb_indices) != 0:
            #neighbor danger is described in get_neighbor_danger. give every neighbor its danger-value, player_on_bomb is a boolean (True if player on bomb)
            channels, player_on_bomb = get_neighbor_danger(channels, neighbor_pos, close_bomb_indices, bomb_position, player, explosion_map, i)
        
        #describing coin prio:
        if position_coins.size > 0:
            channels[i,2] = steps_to_coins[i]       #for each neighbor note number of steps to closest coin

    #describing pritority for next free tile if a bomb has been placed
    if player_on_bomb: 
        tile_count = find_closest_free_tile(game_state, player, close_bomb_indices, bomb_position,neighbor_pos)
        for j in range(4):
            channels[j,5] = tile_count[j]
    

    #describing distance to other players
    if other_position.size > 0:
        for i in range(len(dist_others)):       #same as for crates
            channels[i][3] = dist_others[i]
    

    #describing distance to crates
    if crates_position.size > 0:
        for i in range(len(dist_crates)):
            channels[i][4] = dist_crates[i]     #got to higher densitys of crates while ignoring walls

  
    
    #print('channels\n',channels)
    #combining current channels:
    stacked_channels = np.stack(channels).reshape(-1)
    

    own_bomb = []
    if game_state['self'][2]:
        own_bomb.append(1)
    else:
        own_bomb.append(0)
    
    stacked_channels = np.concatenate((stacked_channels, own_bomb))
    
    #print(stacked_channels)
    return stacked_channels


def get_neighbor_pos(player):
    #positions of neigboring tiles in the order (UP, DOWN, LEFT, RIGHT)
    neighbor_pos = []
    neighbor_pos.append((player[0], player[1] - 1))
    neighbor_pos.append((player[0], player[1] + 1))
    neighbor_pos.append((player[0] - 1, player[1]))
    neighbor_pos.append((player[0] + 1, player[1]))
    
    return np.array(neighbor_pos)

def get_player_dist(game_state, segments, player):
    
    #getting position of other players from state:
    other_position = []
    for i in range(len(game_state['others'])):
        other_position.append([game_state['others'][i][3][0], game_state['others'][i][3][1]])
    other_position = np.array(other_position)
    
    distances = []
    
    # distance from others to player
    if other_position.size > 0:
        for segment in segments:

            #maximum_dist = np.sqrt((segment[0,1] - segment[0,0])**2 + (segment[1,1] - segment[1,0])**2)
            
            others_in_segment = np.where((other_position[:,0] > segment[0,0]) & (other_position[:, 0] < segment[0,1]) & (other_position[:, 1] > segment[1,0]) & (other_position[:,0] < segment[1,1]))
            
            if len(others_in_segment[0]) == 0:
                distances.append(0)
                continue
            
            d_others = np.subtract(other_position[others_in_segment[0]], player)   
        
            dist_norm = np.linalg.norm(d_others, axis = 1)
        
            dist_closest = 2/ (1 + min(dist_norm))               #max dist?
            distances.append(dist_closest)

        return distances, other_position
    
    return distances, other_position

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

        return channels

def get_neighbor_danger(channels, neighbor_pos, close_bomb_indices, bomb_position, player, explosion_map, i):

    player_on_bomb = False
    # are there dangerous tiles in the neighbors?
    bomb_tuples = [tuple(x) for x in bomb_position]
    
    for j in close_bomb_indices:                                                    #only look at close bombs
        #if bomb_tuples[j] not in exploding_tiles_map.keys(): continue               
        dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])             #get all tiles exploding with close bombs
        if np.any(np.sum(np.abs(dangerous_tiles-neighbor_pos[i]), axis=1) == 0):
                                                                                    #if neighbor is on dangerous tile -> set danger value
            channels[i,5] = 2 / (1+np.linalg.norm(bomb_position[j]-neighbor_pos))   #danger value: 4 / distance to bomb+1  

        if i == 3 and np.any(np.sum(np.abs(dangerous_tiles-player), axis=1) == 0):
            player_on_bomb = True 

    #are there already exploding tiles in the neighbors (remember:explosions last for 2 steps)
    if len(np.where(explosion_map != 0)[0]):                                        #check if there are current explosions
        if explosion_map[neighbor_pos[i,0],neighbor_pos[i,1]] != 0:
            channels[i,5] = 1 

    return channels, player_on_bomb
  
def find_closest_free_tile(game_state, player_pos, close_bomb_indices, bomb_position,neighbor_pos):

    field =  game_state['field']
    field = np.where(field == -1 , 1, field)                                    #set crates and walls to 1                        
    bomb_tuples = [tuple(x) for x in bomb_position]    

    for j in close_bomb_indices:                                                #only look at close bombs              
        dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])         #get all tiles exploding with close bombs
        for tile in dangerous_tiles:
            if field[tile[0],tile[1]] == 0: field[tile[0],tile[1]]=2 
    
    #print(field.T)

    #for enemy in game_state['others']:                                          #since other players block moement, look at them as walls
        #field[enemy[3][0],enemy[3][1]] = 1

    tile_count = np.zeros(4)
    for j, neighbor in enumerate(neighbor_pos):
        tile_count[j] = width_search_danger(field,neighbor,player_pos)
        
    if np.sum(tile_count) == 0 : return tile_count

    tile_count_ratio = tile_count / np.sum(tile_count)
    
    return tile_count_ratio  

def width_search_danger(field,neighbor_pos,player_pos):
    if field[neighbor_pos[0],neighbor_pos[1]] == 1 : 
        return 0

    tiles = 0 
    history = [player_pos]
    q = deque()
    q.append(neighbor_pos)
    #print('neighbor start')
    while len(q) > 0:

        pos = q.popleft()
        history.append(pos)

        neighbors = get_neighbor_pos(pos)

        for neighbor in neighbors:
            #print('neighboor it', neighbor)

            if field[neighbor[0], neighbor[1]] == 0:                            #neighbor is on not exploding tile
                #print('safe', neighbor)
                tiles +=1 
                

            if field[neighbor[0], neighbor[1]] == 2:                            #check if neighbor is wall or crate, if not...
                #print('not crate')
                if not np.any(np.sum(np.abs(history - neighbor), axis=1) == 0): # if neighbor is already in the q, dont append
                    #print('not_back')
                    q.append(neighbor)                                          # neighbor is not yet in q
                    history.append(neighbor)

            else: continue

    return tiles

def get_crate_dist(field, segments, player):

    crates_position = np.array([np.where(field == 1)[0], np.where(field == 1)[1]]).T
    
    distances = []
    
    # distance from crates to player
    if crates_position.size > 0:
        for segment in segments:

            maximum_dist = np.sqrt((segment[0,1] - segment[0,0])**2 + (segment[1,1] - segment[1,0])**2)
            
            crates_in_segment = np.where((crates_position[:,0] > segment[0,0]) & (crates_position[:, 0] < segment[0,1]) & (crates_position[:, 1] > segment[1,0]) & (crates_position[:,0] < segment[1,1]))
            
            if len(crates_in_segment[0]) == 0:
                distances.append(0)
                continue
            
            d_crates = np.subtract(crates_position[crates_in_segment[0]], player)   
        
            dist_norm = np.linalg.norm(d_crates, axis = 1)
        

            dist_closest  =  len(dist_norm)/len(crates_position)
            #dist_closest = maximum_dist / (1 + min(dist_norm))
            #dist_closest = np.sum(maximum_dist / (1 + dist_norm))
            distances.append(dist_closest)
        
        return distances, crates_position
    
    return distances, crates_position

def get_segments(player):

    left_half =  np.array([[0, player[0]], [0, 17]]) #np.array(list(product(np.arange(0, player[0]), np.arange(0,17))), dtype = object)
   
    right_half = np.array([[player[0], 17], [0, 17]]) #np.array(list(product(np.arange(player[0], 17), np.arange(0,17))), dtype = object)
    
    up_half = np.array([[0, 17], [0, player[1]]])  #np.array(list(product(np.arange(0, 17), np.arange(0,player[1]))), dtype = object)
    
    low_half = np.array([[0, 17], [player[1], 17]])  #np.array(list(product(np.arange(0, 17), np.arange(player[1], 17))), dtype = object)

    segments = np.array([up_half, low_half, left_half, right_half])

    return segments

def search_for_obj(player_pos, neighbor_pos, obj ,field):
    #print('coin', obj)
    #print('player:',player_pos)


    if field[neighbor_pos[0],neighbor_pos[1]] != 0 : 
        return 0
    

    parents = [None]*17**2 

    flat_neighbor = 17 * neighbor_pos[0] + neighbor_pos[1]
    flat_player = 17 * player_pos[0] + player_pos[1]
    flat_obj = 17 * obj[0] + obj[1]
                    
    parents[flat_neighbor] = flat_neighbor
    parents[flat_player] = flat_player

    q = deque()                      
    q.append(neighbor_pos)              
 
    while len(q) > 0: 
          
        node = q.popleft()     
        if np.linalg.norm(node-obj) == 0:  
            break   

        for neighbor in get_neighbor_pos(node):

            if field[neighbor[0],neighbor[1]] == 0 :
               
                if parents[17 * neighbor[0] + neighbor[1]] is None:
                    
                    parents[17 * neighbor[0] + neighbor[1]] = 17 * node[0] + node[1]  
                    q.append(neighbor) 
         
        
    if parents[flat_obj] is None: 
        return 0                  

    path = [flat_obj]
    while path[-1] != flat_neighbor:
   
        path.append(parents[path[-1]])
    
    return 1/len(path)  


