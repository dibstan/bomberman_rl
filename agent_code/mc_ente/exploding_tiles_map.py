import numpy as np
import pickle

field=np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

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
            coors_ij=[(i,j)]

            #first consider walking to the right, stop when encounter -1 or after 3 steps
            k,l = i,j
            while (field[k,l+1] !=-1) and np.abs(l-j)<3:
                l+=1
                coors_ij.append((k,l))
            #walking left:
            k,l = i,j
            while (field[k,l-1] !=-1) and np.abs(l-j)<3:
                l-=1
                coors_ij.append((k,l))
            #walking up:
            k,l = i,j
            while (field[k-1,l] !=-1) and np.abs(k-i)<3:
                k-=1
                coors_ij.append((k,l))
            #walking down
            k,l = i,j
            while (field[k+1,l] !=-1) and np.abs(k-i)<3:
                k+=1
                coors_ij.append((k,l))
            
            exploding_tiles[(i,j)] = coors_ij
            #print(exploding_tiles[(i,j)])

    return exploding_tiles

map = get_all_exploding_tiles(field)

with open("explosion_map.pt", "wb") as file:
    pickle.dump(map, file)