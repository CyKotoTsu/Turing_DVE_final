import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def plot_v (file, window_size):
    '''
    Plot temporal velocity profile of a given file
    file: path to the file
    window_size: size of the window to average the velocity over. if None, no averaging is done
    '''
    with open(file, 'rb') as f:
        mask_lst, x_lst, p_lst, q_lst = pickle.load(f)
    

    # Select DVE cells
    dve_indices = np.where(mask_lst[-1] == 2)[0]

    # Calculate velocity
    v_lst = []
    for i in range(len(x_lst)-1):
        dve_x_i = x_lst[i][dve_indices]
        dve_x_i_1 = x_lst[i+1][dve_indices]

        # Mean Euclidean speed over DVE cells (scalar)
        v_i = np.mean(np.linalg.norm(dve_x_i_1 - dve_x_i, axis=1))
        v_lst.append(v_i)
    
    if window_size is not None:
        v_lst = np.convolve(v_lst, np.ones(window_size)/window_size, mode='valid')
    
    # Plot velocity profile
    plt.plot(v_lst)
    plt.show()

    return v_lst