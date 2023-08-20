import numpy as np
import torch
import math
import scipy

from torch.nn import functional as F

from .confusion_matrix import calculate_adjacency_matrix


ind_2_class_dict = {  
    0: "Road",
    1: "Sidewalk", 
    2: "Building",  
    3: "wall",
    4: "fence",  
    5: "Pole", 
    6: "Traffic light",
    7: "Traffic sign",
    8: "Vegetation", 
    9: "Terrain",
    10: "Sky",
    11: "Person",
    12: "Rider",
    13: "Car",
    14: "Truck",
    15: "Bus",
    16: "Train",
    17: "Motorcycle",
    18: "Bicycle"
}


def cos_fun_sch(step): 
    return math.cos((step + 0.008) / 1.008 * math.pi / 2) ** 2
    

## custom beta_schedule  (linear) / (expo)
def custom_schedule(beta_start = 0.0001, beta_end = 0.02, timesteps=20,dtype=torch.float64, type = 'expo'):
    # betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)
    # betas = -torch.log(torch.linspace(beta_start, beta_end, timesteps, dtype=dtype))
    # betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)**2 ## quadratic 
    if type == 'expo':
        betas = torch.logspace(beta_start, beta_end,steps=timesteps ,base = 10, dtype=dtype) ## expo space growth...increases slowly in the start and raipdy grows in the end! ## hyperparam tuned in such a way that classes in the dia confuses with the off dia classes 
    elif type == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype) 
    elif type == 'cosine': ## adapted from google research/d3pm
        betas = []
        for t in range(timesteps):
            t_start = t / timesteps
            t_end = (t + 1) / timesteps
            betas.append(np.minimum( 1 - cos_fun_sch(t_end) / cos_fun_sch(t_start) , 0.999)) ## max_beta is 0.999 
        betas = torch.from_numpy(np.array(betas), dtype = dtype)
    else:
        raise ValueError(
            f"Diffusion noise schedule of kind {type} is not supported.")
    return betas 


## Q-transition matrix based discerete diffusion
def similarity_transition_mat(betas, t, similarity_matrix, transition_mat_type, similarity_soft = True, k_nn = 3):
    """Computes transition matrix for q(x_t|x_{t-1}).
    Nearest neighbor transition matrix inspired from the text word embedding distance to introduce locality.
    Args:
        t: timestep. integer scalar.
    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = betas[t]
    beta_t = beta_t.cpu().numpy()
    similarity_matrix = np.array(similarity_matrix) ## as all the calculations here are done in numpy, later transforming into cuda tensor.
    # beta_t = beta_t * 100 ## increasing beta_t value 
    
    if transition_mat_type == 'band_diagonal':
        ## width paramater to be decided 
        dim = 20 ## number of different classes 
        width = 7 ## a hyper parameter 
        band = np.tri(
            dim, dim, width // 2, dtype=np.float64) - np.tri(
                dim, dim, -width // 2, dtype=np.float64)
        arr = band / band.sum(0, keepdims=True)
        matrix = beta_t * arr + (1 - beta_t) * np.eye(dim)
        matrix = torch.from_numpy(matrix).to(betas.device)  
        return matrix
        
    elif transition_mat_type == 'matrix_expo': # matrix_expo or sinkhorn method for base transition matrix
        if similarity_soft:
            '''
                have to  fill the way D3PM makes the base matrix using matrix expo method
            '''
        else: ## using adjacency matrix as mentioned in the paper 
            '''
                have to  fill the way D3PM makes the base matrix using matrix expo method
            '''
            adj, similar_classes = calculate_adjacency_matrix_knn(similarity_matrix, k=k_nn)
            adj_s = (adj + adj.T) / (2 * k_nn)
                
    elif transition_mat_type == 'sinkhorn_algorithm':
        matrix = similarity_matrix.copy()
        matrix = beta_t * matrix
        # ## sinkhorn algo for base matrix
        for _ in range(5): # number of iterations is a hyperparameter of sinkhorn's algo ## till in covergence 
            matrix = matrix / matrix.sum(1, keepdims=True)
            matrix = matrix / matrix.sum(0, keepdims=True)
        
        # matrix = (1 - beta_t)*np.eye(20) + beta_t * matrix  ## additional just for trying as given in d3pm original code
        
    else: 
        raise Exception("transition matrix type not implemented!")
       
    matrix = torch.from_numpy(matrix).to(betas.device) 
    
    ## NOTE below to include background in the transition matrix 
    matrix = F.pad(input=matrix, pad=(0, 1, 0, 1), mode='constant', value=0) ## we need to check this!
    matrix[-1,-1] = 1 # background remains the background ## we need to check this!
    
    # return torch.from_numpy(matrix).to(betas.device)
    return matrix


def logits_to_categorical(logits):
    uniform_noise = torch.rand_like(logits)
    ## # To avoid numerical issues clip the uniform noise to a minimum value
    uniform_noise = torch.clamp(uniform_noise, min=torch.finfo(uniform_noise.dtype).tiny, max=1.)
    gumbel_noise = - torch.log(-torch.log(uniform_noise))
    sample = (gumbel_noise + logits).argmax(dim=1)
    return sample

## using prototypes of 19 cityscapes semantic classes from "Rethinking Semantic Segmentation: A Prototype View" 
def similarity_among_classes(protos):
    cos = torch.nn.CosineSimilarity(dim=0)
    similarity_matrix = []
    for i in range(19):
        per_row_similarity = [] 
        for j in range(19):
            per_row_similarity.append(cos(protos[i][0], protos[j][0]))
        similarity_matrix.append(per_row_similarity)
    similarity_matrix_tensor = torch.FloatTensor(similarity_matrix)     

    # sim_test = similarity_matrix_tensor.clone() 
    # sim_test.fill_diagonal_(-1e17) ## filling extremely low numbers at diagonals for removing them from similarity consideration
    # probas_sim_test = F.softmax(sim_test, dim=1) ## applying softmax to convert into probability distribution 

    # return probas_sim_test, similarity_matrix_tensor
    return similarity_matrix_tensor

def calculate_adjacency_matrix_knn(similarity_matrix, k): 
    adj = similarity_matrix.copy()
    similar_classes = []
    for row in range(adj.shape[0]):
        row_sim_decrease_inds = np.argsort(adj[row])[::-1]
        knn_indexs = row_sim_decrease_inds[:k] 
        adj[row][knn_indexs] = 1
        adj[row][adj[row]!=1] = 0

        similar_classes.append([ (('Current Class:' + ind_2_class_dict[row]) ,(knn_indexs[i], ind_2_class_dict[knn_indexs[i]])) for i in range(knn_indexs.shape[0])])
    
    assert (sum(adj.sum(1)) / adj.shape[1]) == k
    return adj, similar_classes 