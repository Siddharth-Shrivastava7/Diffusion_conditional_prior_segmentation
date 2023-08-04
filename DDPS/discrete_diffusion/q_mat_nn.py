import numpy as np
import torch 
import scipy

## Discrete Diffusion matrix adopted from D3PM-Pytorch github code 


## NearestNeighborDiffusion
## adjacency matrix of k=3 dervied from confusion matrix of oneformer model 
list_of_lists = [[0,        0.38,           0,           0,     0,     0,            0,              0,              0,       0.03,        0,      0,       0,   0.07,      0,       0,      0,        0,            0 ],
        [3.82,        0,           0.47,           0,     0,     0,            0,              0,              0,         0.65,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0.12,           0,           0,     0,     0.3,            0,              0,              1.32,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           10.84,           0,     3.71,     0,            0,              0,              1.99,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           9.97,           6.06,     0,     0,            0,              0,              2.18,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        1.38,           7.83,           0,     0,     0,            0,              0,              4.09,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           5.35,           0,     0,     2.23,            0,              0,              4.69,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           4.83,           0,     0,     1.22,            0,              0,              1.9,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           1.79,           0,     0,     0.31,            0,              0,              0,         0.53,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [1.6,       9.17,           0,           0,     0,     0,            0,              0,              9.71,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           11.51,           0,     0,     0.09,            0,              0,              1.01,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           3.65,           0,     0,     0,            0,              0,              0.53,         0,        0,      0,       0.97,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           1.47,           0,     0,     0,            0,              0,              0,         0,        0,      2.32,            0,      0,      0,       0,      0,        0,            5.16 ],
        [0.77,        0,           0.41,           0,     0,     0,            0,              0,              0.25,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
        [0,        0,           1.24,           0,     0,     0,            0,              0,              0.46,         0,        0,      0,       0,      3.96,         0,          0,         0,           0,      0 ],
        [0.61,        0,           0.83,           0,     0,     0,            0,              0,              0,         0,        0,      0,       0,       0.55,      0,           0,         0,            0,    0 ],
        [0,        0,           2.19,           0,     0,     0,            0,              0,              1.37,         0,        0,      0,       0,      0,      0,       0.38,      0,        0,            0 ],
        [0,        0,           2.61,           0,     0,     0,            0,              0,              0,         0,        0,      2.06,       0,      0,      0,       0,      0,        0,            2.16 ],
        [0,        1.58,           4.19,           0,     1.45,     0,            0,              0,              0,         0,        0,      0,       1.58,      0,      0,       0,      0,        0,            0 ]]

list_of_lists_arr = np.array(list_of_lists)  
## one-hot adjacency matrix 
adjacency_matrix_one_hot = list_of_lists_arr 
adjacency_matrix_one_hot[list_of_lists_arr > 0] = 1 

## from google_research/d3pm/text/diffusion
adjacency_matrix_one_hot = adjacency_matrix_one_hot + adjacency_matrix_one_hot.T

transition_rate = adjacency_matrix_one_hot - np.diagflat(np.sum(adjacency_matrix_one_hot, axis=1))
# define beta 
# beta = self.schedule(0) 

matrix = scipy.linalg.expm(
            np.array(beta * transition_rate, dtype=np.float64))

matrix / matrix.sum(0, keepdims=True) 