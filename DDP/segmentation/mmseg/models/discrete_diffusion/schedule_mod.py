import numpy as np
import torch
import math
import scipy

from torch.nn import functional as F

from .confusion_matrix import calculate_adjacency_matrix

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


## diffusion based on Q-transition matrix 
def similarity_transition_mat(betas, t, confusion_matrix, band_diagonal, matrix_expo = False, confusion = True, k_nn = 3):
    """Computes transition matrix for q(x_t|x_{t-1}).
    Nearest neighbor transition matrix inspired from the text word embedding distance to introduce locality.
    Args:
        t: timestep. integer scalar.
    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = betas[t]
    beta_t = beta_t.cpu().numpy()
    # beta_t = beta_t * 100 ## increasing beta_t value 
    
    if band_diagonal:
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
        
    else: # matrix_expo or sinkhorn method for base transition matrix
        if matrix_expo:
            if confusion: ## using confusion matrix of the model, directly
                # using confusion matrix not the conventitional one hot transition matrix 
                matrix_from_confusion = confusion_matrix.copy() ## it doesn't make sense to include background...so using confusion matrix as it is 
                np.fill_diagonal(matrix_from_confusion, 0) ## first zeroing out elements of confusion matrix for making transition rate matrix 
                matrix_from_confusion = matrix_from_confusion + matrix_from_confusion.T ## symmetricity required in transition rate 
                transition_rate = matrix_from_confusion - np.diagflat(np.sum(matrix_from_confusion, axis=1)) ## transition rate matrix from confusion matrix 
                betas_tt = torch.sum(betas[:t+1]).item() ## since t is starting from 0 ## cummulative steps matrix expo calc
                matrix = scipy.linalg.expm(
                            np.array(betas_tt * transition_rate, dtype=np.float64)) 
                # print('************', np.diag(matrix))
                # print('^^^^^^^^^^^^^^^^', np.max(matrix), np.min(matrix))
            else: ## using adjacency matrix as mentioned in the paper 
                ## building similarity matrix, rate and base matrix using adjacency matrix 
                ## adjacency matrix of k=3 dervied from confusion matrix of oneformer model 
                list_of_lists = [[0,        0.38,           0,           0,     0,     0,            0,              0,              0,       0.03,        0,      0,       0,   0.07,      0,       0,      0,        0,           0, 0],
                        [3.82,        0,           0.47,           0,     0,     0,            0,              0,              0,         0.65,        0,      0,       0,      0,      0,       0,      0,        0,            0, 00],
                        [0,        0.12,           0,           0,     0,     0.3,            0,              0,              1.32,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0, 0],
                        [0,        0,           10.84,           0,     3.71,     0,            0,              0,              1.99,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [0,        0,           9.97,           6.06,     0,     0,            0,              0,              2.18,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [0,        1.38,           7.83,           0,     0,     0,            0,              0,              4.09,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [0,        0,           5.35,           0,     0,     2.23,            0,              0,              4.69,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [0,        0,           4.83,           0,     0,     1.22,            0,              0,              1.9,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [0,        0,           1.79,           0,     0,     0.31,            0,              0,              0,         0.53,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [1.6,       9.17,           0,           0,     0,     0,            0,              0,              9.71,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [0,        0,           11.51,           0,     0,     0.09,            0,              0,              1.01,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [0,        0,           3.65,           0,     0,     0,            0,              0,              0.53,         0,        0,      0,       0.97,      0,      0,       0,      0,        0,            0,0],
                        [0,        0,           1.47,           0,     0,     0,            0,              0,              0,         0,        0,      2.32,            0,      0,      0,       0,      0,        0,            5.16,0],
                        [0.77,        0,           0.41,           0,     0,     0,            0,              0,              0.25,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
                        [0,        0,           1.24,           0,     0,     0,            0,              0,              0.46,         0,        0,      0,       0,      3.96,         0,          0,         0,           0,      0,0],
                        [0.61,        0,           0.83,           0,     0,     0,            0,              0,              0,         0,        0,      0,       0,       0.55,      0,           0,         0,            0,    0,0],
                        [0,        0,           2.19,           0,     0,     0,            0,              0,              1.37,         0,        0,      0,       0,      0,      0,       0.38,      0,        0,            0,0],
                        [0,        0,           2.61,           0,     0,     0,            0,              0,              0,         0,        0,      2.06,       0,      0,      0,       0,      0,        0,            2.16,0],
                        [0,        1.58,           4.19,           0,     1.45,     0,            0,              0,              0,         0,        0,      0,       1.58,      0,      0,       0,      0,        0,            0,0],
                        [0.7,       0.2,            0,              0,      0,      0,            0,              0,              0.1,       0,         0,     0,         0,       0,       0,      0,      0,        0,            0,0]
                        ] ## background class also added for now making its relative dependency with road, sidewalk and vegetation 
                # list_of_lists_arr = np.array(list_of_lists)  
                # # # ## one-hot adjacency matrix 
                # adjacency_matrix_one_hot = list_of_lists_arr 
                # adjacency_matrix_one_hot[list_of_lists_arr > 0] = 1 
                # # ## from google_research/d3pm/text/diffusion
                # adjacency_matrix_one_hot = (adjacency_matrix_one_hot + adjacency_matrix_one_hot.T) / (2 * 3) ## for building the symmetricity of adjacency matrix and k = 3
                adjacency_matrix_one_hot = calculate_adjacency_matrix(confusion_matrix=confusion_matrix, k=k_nn) ## for k nearest neighbours
                adjacency_matrix_soft = (adjacency_matrix_one_hot + adjacency_matrix_one_hot.T) / (2 * k_nn)
                transition_rate = adjacency_matrix_soft - np.diagflat(np.sum(adjacency_matrix_soft, axis=1))
                # ### building base matrix 
                matrix = scipy.linalg.expm(
                            np.array(beta_t * transition_rate, dtype=np.float64))  
            
            # matrix = (1 - beta_t)*np.eye(20) + beta_t * matrix
        else:
            ## building similarity matrix, rate and base matrix using confusion matrix 
            ### Dealing with confusion matrix for similarity matrix 
            # per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
            # confusion_matrix = \
            #     confusion_matrix.astype(np.float64) / per_label_sums
            # matrix = np.zeros((20,20)) ## not giving background any chance of coming in the prediction; whatsoever ## num_classes x num_classes  
            # np.fill_diagonal(confusion_matrix, 0) ## inplace function, as the proba of transferring to itself is quite high, wont ever transfer to any other class if this present so zeroing it out # commenting for now 
            # matrix = np.random.uniform(0,np.max(confusion_matrix), (20,20)) ## uniform distribution for background class 
            # matrix = (torch.ones((20,20), device = bt.device, dtype = torch.float64)*(1/20)) ## in torch 
            # matrix[:19, :19] = torch.tensor(confusion_matrix).to(bt.device) 
            # matrix = np.ones((20,20)) ## 20 is the number of classes; making a uniform transition matrix 
            # matrix[:19, :19] = confusion_matrix  ## this is similarity matrix...main thing as this says
            # np.fill_diagonal(matrix, 0) ## making dia zero so as to be in use in matrix expo method ## no changes to introduced in confusion matrix calc ## first making the matrix zeroing out the dia as there is severe dis balance, because of dia in confusion matrix 
            # np.fill_diagonal(matrix, (1-beta_t)*np.diag(confusion_matrix)) 
            # matrix = matrix + matrix.T ## no changes to introduced in confusion matrix calc ## as connectivity (similarity) should be symmetric among classes ## additional for symmetricity 
            # matrix = beta_t * matrix ## no changes to introduced in confusion matrix calc
            # matrix.fill_diagonal_(0)
            # print('********', np.unique(confusion_matrix))
            # print(np.max(confusion_matrix), np.min(confusion_matrix)) ## maximum is around 0.99 when dia is present else it is 0.15 
            # matrix = matrix / (2 * 3)  ## not required cause not using k nearest neighbours 
            # np.fill_diagonal(matrix, np.sum(matrix, axis=1)) ## adding each proba of transition equal to being staying there in the same class (for making it in same scale) >> thus high chance of being staying there
            # matrix_prev = matrix  ## initially what was the matrix before multiplying the beta_t scalar 
            # matrix = beta_t * (t+1) * matrix_prev
            # np.fill_diagonal(matrix, ((1 - 20*beta_t)*np.sum(matrix_prev, axis=1)))
            # print('>>>>>>>>>>', np.max(matrix))
            # matrix = (1 - beta_t)*np.eye(20) + beta_t * matrix  ## not working with sinkhorn ## 1. similar to uniform and absorbtion state transition matrix, 2. for transition into another state is like corrupting which confusion matrix stores info..for staying in the same, is like correct which gradually lowers down as time t increases...so this formulation make sense, of bringing it above sinkhorn algorithm
            # matrix = torch.from_numpy(matrix).to(bt.device) ## cuda out of memory here...alas!
            matrix = confusion_matrix.copy()
            matrix = beta_t * matrix
            # ## sinkhorn algo for base matrix
            for _ in range(5): # number of iterations is a hyperparameter of sinkhorn's algo ## till in covergence 
                matrix = matrix / matrix.sum(1, keepdims=True)
                matrix = matrix / matrix.sum(0, keepdims=True)
            # np.fill_diagonal(matrix, 0)
            # matrix = matrix / matrix.sum(1, keepdims=True) # rows should sum up to one exactly even if column a bit off from one 
            
            # print('>>>>>>>>>>', np.max(matrix))
            # print('*******************',matrix) ## sort of symmetric mostly 
            # print('RRRRRRRRRRRRRRRRRRR',matrix.sum(1, keepdims=True))  
            # print('CCCCCCCCCCCCCCCCCCC',matrix.sum(0, keepdims=True)) ## its also close to 1 but not exactly one .. its fine 
            # print('*************', np.diag(matrix)) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            # print('>>>>>>>>>>>>>>>>>>>', np.max(matrix))
            # matrix = matrix / matrix.sum(0, keepdims=True)      
            # # # matrix = (1 - bt[-1].cpu().numpy()) * np.eye(20) + bt[-1].cpu().numpy() * matrix ## additional just for trying as given in d3pm original code
            # matrix = (1 - beta_t)*np.eye(20) + beta_t * matrix  ## additional just for trying as given in d3pm original code
            # print('^^^^^^^^^^', matrix)
            # print('*************', np.diag(matrix))
            # matrix_prev = matrix  ## initially what was the matrix before multiplying the beta_t scalar 
            # matrix = beta_t * matrix_prev
            # np.fill_diagonal(matrix, (1 - beta_t*np.sum(matrix_prev, axis=1)))  ## this is the one i which thought, it work but it didn't work...
            # np.fill_diagonal(matrix, 0)
            # np.fill_diagonal(matrix, 1-beta_t)
            ## above 3 operations maintaining the doubly stochastic property of the matrix 
            # print('>>>>>>>>>>>>>>>>>>>>>', matrix, 'ttttttttimeeee', t)
            # print('RRRRRRRRRRRRRRRRRRR',matrix.sum(1, keepdims=True))  ## exactly 1 
            # print('CCCCCCCCCCCCCCCCCCC',matrix.sum(0, keepdims=True))  ## quite close to 1
        
        matrix = torch.from_numpy(matrix).to(betas.device) 
        # torch.save(matrix, 'q_matrix'+ str(t) + '.pt') ## saving the tensor for analysing it
        # print('time', t , 'matrix saved')
        # print('*******************',matrix.dtype) # double = float64 
        
        # return torch.from_numpy(matrix).to(betas.device)
        return matrix

def q_mats_from_onestepsdot(betas, num_timesteps, confusion_matrix, band_diagonal, matrix_expo = False, confusion = True, k_nn = 3): # return: Qt = Q_1.Q_2.Q_3...Q_t, input-arguments = set of betas values over diffusion timesteps and total number of diffusion timesteps
    if matrix_expo:
        q_cummulativesteps_mats = [similarity_transition_mat(betas, t, confusion_matrix, band_diagonal, matrix_expo, confusion, k_nn) 
                                for t in range(0, num_timesteps)]
        ## adding background class performace as well with a probability of 0 everywhere
        # q_mats = q_cummulativesteps_mats
        q_cummulativesteps_mats = torch.stack(q_cummulativesteps_mats, dim=0) 
        q_mats = F.pad(input=q_cummulativesteps_mats, pad=(0, 1, 0, 1), mode='constant', value=0) ## 20 x 20 matrix now  ## may be later need to change [20,20]th element to 1..check later
    else: 
        q_onestep_mats = [similarity_transition_mat(betas, t, confusion_matrix, band_diagonal, matrix_expo, confusion) 
                                for t in range(0, num_timesteps)]
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, num_timesteps):
            q_mat_t = torch.tensordot(q_mat_t, q_onestep_mats[t],
                                        dims=[[1], [0]])
            # q_mat_t.fill_diagonal_(0) ## forcefully making it zero ## not using it 
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0) 
        
    return q_mats

def q_pred(x_start, t, num_classes, q_mats, return_logits = False):  ## calculating q(x_t | x_0)
    B, H, W = x_start.shape # label map 
    q_mats_t = torch.index_select(q_mats, dim=0, index=t)
    x_start_onehot = F.one_hot(x_start.view(B, -1).to(torch.int64), num_classes).to(torch.float64)
    out = torch.matmul(x_start_onehot, q_mats_t)  
    out = out.view(B, num_classes, H, W) 
    if return_logits: 
        logits = torch.log(out + 1e-20)  ## eplison taken as 1e-20
        out_sample = logits_to_categorical(logits)
    else:
        out_sample = out.argmax(dim=1)  
    return out_sample 

def q_posterior(x_start, x_t, t, num_classes, q_mats, return_logits = False):


    
    pass 

def p_reverse(x_start, x_t, t, num_classes, q_mats, return_logits = False):
    
    ## two ways to arrive it; 
    '''
        1. The way D3PM arrived
        2. The way CCDM/Multinomial Diffusion arrived
        
        we will code both the ways!!
    '''
        
    
    pass 


def logits_to_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample = (gumbel_noise + logits).argmax(dim=1)
    return sample