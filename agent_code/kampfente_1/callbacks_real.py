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


'''def act(self, game_state: dict) -> str:
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
        print('move',move)

        return move #np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.1,0.1])

        #return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=[.2,.2,.2,.2,.1,.1])'''

def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
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
    channels.append([game_state['self'][1], int(game_state['self'][2] == True), game_state['self'][3][0], game_state['self'][3][1], game_state['round'], game_state['step']])
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
