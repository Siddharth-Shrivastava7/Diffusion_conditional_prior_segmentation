# Python program to generate a heatmap
# which displays the value in each cell 
# corresponding to the given dataframe 
  
# import required libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
  
# defining index for the dataframe
idx = ['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic light', 'Traffic sign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle']
  
# defining columns for the dataframe
cols = ['R', 'S', 'B', 'W', 'F', 'P', 'Tl', 'Ts', 'V', 'T', 'Sk', 'Pe', 'Ri', 'C', 'Tr', 'Bu', 'Tra', 'M', 'Bi']

# [0,        0.38,           0,           0,     0,     0,            0,              0,              0,       0.03,        0,      0,       0,   0.07,      0,       0,      0,        0,            0 ]
# [3.82,        0,           0.47,           0,     0,     0,            0,              0,              0,         0.65,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0.12,           0,           0,     0,     0.3,            0,              0,              1.32,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           10.84,           0,     3.71,     0,            0,              0,              1.99,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           9.97,           6.06,     0,     0,            0,              0,              2.18,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        1.38,           7.83,           0,     0,     0,            0,              0,              4.09,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           5.35,           0,     0,     2.23,            0,              0,              4.69,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           4.83,           0,     0,     1.22,            0,              0,              1.9,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           1.79,           0,     0,     0.31,            0,              0,              0,         0.53,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [1.6,       9.17,           0,           0,     0,     0,            0,              0,              9.71,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           11.51,           0,     0,     0.09,            0,              0,              1.01,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           3.65,           0,     0,     0,            0,              0,              0.53,         0,        0,      0,       0.97,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           1.47,           0,     0,     0,            0,              0,              0,         0,        0,      2.32,            0,      0,      0,       0,      0,        0,            5.16 ]
# [0.77,        0,           0.41,           0,     0,     0,            0,              0,              0.25,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ]
# [0,        0,           1.24,           0,     0,     0,            0,              0,              0.46,         0,        0,      0,       0,      3.96,         0,          0,         0,           0,      0 ]
# [0.61,        0,           0.83,           0,     0,     0,            0,              0,              0,         0,        0,      0,       0,       0.55,      0,           0,         0,            0,    0 ]
# [0,        0,           2.19,           0,     0,     0,            0,              0,              1.37,         0,        0,      0,       0,      0,      0,       0.38,      0,        0,            0 ]
# [0,        0,           2.61,           0,     0,     0,            0,              0,              0,         0,        0,      2.06,       0,      0,      0,       0,      0,        0,            2.16 ]
# [0,        1.58,           4.19,           0,     1.45,     0,            0,              0,              0,         0,        0,      0,       1.58,      0,      0,       0,      0,        0,            0 ]
        
        
# entering values in the index and columns  
# and converting them into a panda dataframe



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
row_sums = list_of_lists_arr.sum(axis=1)
new_matrix = list_of_lists_arr / row_sums[:, np.newaxis]


## one-hot adjacency matrix 
adjacency_matrix_one_hot = list_of_lists_arr 
adjacency_matrix_one_hot[list_of_lists_arr > 0] = 1


df = pd.DataFrame(new_matrix.tolist(), columns=cols, index=idx)
df.to_string(col_space=50)

# df = pd.DataFrame([[0,        0.38,           0,           0,     0,     0,            0,              0,              0,       0.03,        0,      0,       0,   0.07,      0,       0,      0,        0,            0 ],
#         [3.82,        0,           0.47,           0,     0,     0,            0,              0,              0,         0.65,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0.12,           0,           0,     0,     0.3,            0,              0,              1.32,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           10.84,           0,     3.71,     0,            0,              0,              1.99,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           9.97,           6.06,     0,     0,            0,              0,              2.18,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        1.38,           7.83,           0,     0,     0,            0,              0,              4.09,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           5.35,           0,     0,     2.23,            0,              0,              4.69,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           4.83,           0,     0,     1.22,            0,              0,              1.9,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           1.79,           0,     0,     0.31,            0,              0,              0,         0.53,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [1.6,       9.17,           0,           0,     0,     0,            0,              0,              9.71,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           11.51,           0,     0,     0.09,            0,              0,              1.01,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           3.65,           0,     0,     0,            0,              0,              0.53,         0,        0,      0,       0.97,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           1.47,           0,     0,     0,            0,              0,              0,         0,        0,      2.32,            0,      0,      0,       0,      0,        0,            5.16 ],
#         [0.77,        0,           0.41,           0,     0,     0,            0,              0,              0.25,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0 ],
#         [0,        0,           1.24,           0,     0,     0,            0,              0,              0.46,         0,        0,      0,       0,      3.96,         0,          0,         0,           0,      0 ],
#         [0.61,        0,           0.83,           0,     0,     0,            0,              0,              0,         0,        0,      0,       0,       0.55,      0,           0,         0,            0,    0 ],
#         [0,        0,           2.19,           0,     0,     0,            0,              0,              1.37,         0,        0,      0,       0,      0,      0,       0.38,      0,        0,            0 ],
#         [0,        0,           2.61,           0,     0,     0,            0,              0,              0,         0,        0,      2.06,       0,      0,      0,       0,      0,        0,            2.16 ],
#         [0,        1.58,           4.19,           0,     1.45,     0,            0,              0,              0,         0,        0,      0,       1.58,      0,      0,       0,      0,        0,            0 ]],          
#         columns = cols, index = idx)
  
# displaying dataframe as an heatmap 
# with diverging colourmap as virdis
# df.style.background_gradient(cmap ='viridis')\
#         .set_properties(**{'font-size': '20px'})

# Displaying dataframe as an heatmap
# with diverging colourmap as RdYlBu
plt.imshow(df, cmap='Greys', interpolation='none')
  
# Displaying a color bar to understand
# which color represents which range of data
plt.colorbar()
  
# Assigning labels of x-axis 
# according to dataframe
plt.xticks(range(len(df)), df.columns)
  
# Assigning labels of y-axis 
# according to dataframe
plt.yticks(range(len(df)), df.index)

plt.savefig('adjacency_matrix_from_confusionmat_knn_3.png', format='png')

print('done')