from collections import deque
from random import shuffle
from itertools import product
import numpy as np
import random
import pickle


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0

    ##############################################################################################       
    global exploding_tiles_map
    with open('explosion_map.pt', 'rb') as file:
        exploding_tiles_map = pickle.load(file)
    ##############################################################################################


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info(state_to_features(game_state))
    random_prob = 0.4

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action according to the epsilon greedy policy.")
        betas = np.array(list(self.model.values()))
        feature_vector = np.array(state_to_features(game_state))
        move = list(self.model.keys())[np.argmax(np.dot(betas, feature_vector))]
        
        #print(move)
        return np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.1,0.1])

    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a




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
    field = np.array(game_state['field'])
    
    #get explosion map (remeber: explosions last for 2 steps)
    explosion_map = game_state['explosion_map']

    #getting bomb position from state 
    '''try to vectorize'''
    bomb_position = get_bomb_position(game_state)
    
    #positions of neighboring tiles in the order (UP, DOWN, LEFT, RIGHT)
    neighbor_pos = get_neighbor_pos(player)
    
    #getting segments
    segments = get_segments(player)
    
    #getting position of coins and distance of neighboring tiles
    dist_coins, position_coins = get_coin_dist(game_state, segments, player)

    #getting position of other players and distance
    dist_others, other_position = get_player_dist(game_state, segments, player)

    #getting position of crates and distance
    dist_crates, crates_position = get_crate_dist(field, segments, player)

    #searching for near bombs:
    if len(bomb_position) != 0:
        bomb_distances = np.linalg.norm(np.subtract(bomb_position, player) , axis = 1)
        close_bomb_indices = np.where(bomb_distances <= 4)[0]

        

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

    #describing coin distance: 
    if position_coins.size > 0:
        for i in range(len(dist_coins)):
            if channels[i][0] != 1 and channels[i][1] != 1:
                channels[i][4] = dist_coins[i]

    #describing distance to other players
    if other_position.size > 0:
        for i in range(len(dist_others)):
            if channels[i][0] != 1 and channels[i][1] != 1:
                channels[i][6] = dist_others[i]

    #describing distance to crates
    if crates_position.size > 0:
        for i in range(len(dist_crates)):
            if channels[i][0] != 1 and channels[i][1] != 1:
                channels[i][7] = dist_crates[i]

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

            maximum_dist = np.sqrt((segment[0,1] - segment[0,0])**2 + (segment[1,1] - segment[1,0])**2)
            
            others_in_segment = np.where((other_position[:,0] > segment[0,0]) & (other_position[:, 0] < segment[0,1]) & (other_position[:, 1] > segment[1,0]) & (other_position[:,0] < segment[1,1]))
            
            if len(others_in_segment[0]) == 0:
                distances.append(0)
                continue
            
            d_others = np.subtract(other_position[others_in_segment[0]], player)   
        
            dist_norm = np.linalg.norm(d_others, axis = 1)
        
            dist_closest = maximum_dist / (1 + min(dist_norm))
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
    field = np.where(field == -1 , 1, field)                                    #set crates and walls to 1                        
    bomb_tuples = [tuple(x) for x in bomb_position]    

    for j in close_bomb_indices:                                                #only look at close bombs              
        dangerous_tiles = np.array(exploding_tiles_map[bomb_tuples[j]])         #get all tiles exploding with close bombs
        for tile in dangerous_tiles:
            if field[tile[0],tile[1]] == 0: field[tile[0],tile[1]]=2 
    
    history = []
    q = deque()
    q.append(player_pos)

    while len(q) > 0:

        pos = q.popleft()
        history.append(pos)

        neighbors = get_neighbor_pos(pos)

        for neighbor in neighbors:

            if field[neighbor[0], neighbor[1]] == 0:                            #neighbor is on not exploding tile
                closest_tile = neighbor                                         # -> end search and return tile
                return closest_tile

            if field[neighbor[0], neighbor[1]] == 2:                            #check if neighbor is wall or crate, if not...
                if not np.any(np.sum(np.abs(history - neighbor), axis=1) == 0): # if neighbor is already in the q, dont append
                    q.append(neighbor)                                          # neighbor is not yet in q
                    history.append(neighbor)

            else:
                continue

    return None
    
def get_tile_prio(tile,neighbors):
    #finding the direction that brings us closer the closest coin
    closest_neighbor = np.linalg.norm(neighbors-tile, axis=1)
    priority_index = np.argsort(closest_neighbor)

    return priority_index

def get_coin_dist(game_state, segments, player):
    
    #converting positions of coins
    position_coins = np.array(game_state['coins'])
    #print("position Coins:",position_coins)
    # distance from coins to player
    if position_coins.size > 0:
        distances = []

        for segment in segments:

            maximum_dist = np.sqrt(2 * (15)**2)
            
            coins_in_segment = np.where((position_coins[:,0] > segment[0,0]) & (position_coins[:, 0] < segment[0,1]) & (position_coins[:, 1] > segment[1,0]) & (position_coins[:,0] < segment[1,1]))
            
            if len(coins_in_segment[0]) == 0:
                distances.append(0)
                continue
            
            d_coins = np.subtract(position_coins[coins_in_segment[0]], player)   
        
            dist_norm = np.linalg.norm(d_coins, axis = 1)
            #print('dist\n',dist_norm)
            dist_closest = np.sum(maximum_dist / (1 + dist_norm))
            #dist_closest = maximum_dist / (1 + min(dist_norm))
            #print('dist ratio\n',maximum_dist / (1 + dist_norm))
            distances.append(dist_closest)

        return distances, position_coins

    distances = []
    return distances, position_coins

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
        
            dist_closest = np.sum(maximum_dist / (1 + dist_norm))
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

    