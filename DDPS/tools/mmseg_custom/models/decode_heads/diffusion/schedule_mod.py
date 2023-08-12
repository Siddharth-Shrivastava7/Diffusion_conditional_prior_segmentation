import numpy as np
import torch
import math
import scipy

from torch.nn import functional as F
from .misc import extract, log_add_exp, log_1_min_a, index_to_log_onehot, sample_categorical, log_onehot_to_index

def cos_alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999, exp=3):
    att = np.arange(0, time_step)
    att = (np.cos((att + time_step) * math.pi * 0.5 / time_step) + 1)**exp
    att = att * (att_1 - att_T) + att_T
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]

    ctt = np.arange(0, time_step)
    ctt = (np.cos((ctt + time_step) * math.pi * 0.5 / time_step) + 1)**exp
    ctt = ctt * (ctt_1 - ctt_T) + ctt_T
    ctt = np.concatenate(([0], ctt))

    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


def alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


## this is being used : with time_step(#diffusion steps): 20 and N(#classes): 20
def alpha_schedule_torch(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = torch.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = torch.cat((torch.tensor([1]), att))
    at = att[1:] / att[:-1]
    bt = (1 - at) / N
    att = torch.cat((att[1:], torch.tensor([1])))
    btt = (1 - att) / N
    return at, bt, att, btt


## choices are between this and the above linear torch schedule one
def cos_alpha_schedule_torch(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999, exp=2):
    att = torch.arange(0, time_step)
    att = (torch.cos((att + time_step) * math.pi * 0.5 / time_step) + 1)**exp
    att = att * (att_1 - att_T) + att_T
    att = torch.cat((torch.tensor([1]), att))
    at = att[1:] / att[:-1]
    bt = (1 - at) / N
    att = torch.cat((att[1:], torch.tensor([1])))
    btt = (1 - att) / N
    return at, bt, att, btt


def q_pred(x_start, t, num_timesteps, num_classes, log_cumprod_at, log_cumprod_bt):           # q(xt|x0)
    # log_x_start can be onehot or not
    log_x_start = index_to_log_onehot(x_start, num_classes)
    t = (t + (num_timesteps + 1)) % (num_timesteps + 1)
    log_cumprod_at = extract(log_cumprod_at, t, log_x_start.shape)         # at~
    log_cumprod_bt = extract(log_cumprod_bt, t, log_x_start.shape)         # bt~
    log_probs = torch.cat([
        log_add_exp(log_x_start + log_cumprod_at, log_cumprod_bt)
    ], dim=1)  ## np.log(self.num_classes) is missing! (may its constant so they ignore, but it should be there as it may effect values here)

    sample_logits = sample_categorical(log_probs)
    return sample_logits


def q_pred_log(log_x_start, t, num_timesteps, log_cumprod_at, log_cumprod_bt):       # q(xt|x0)
    t = (t + (num_timesteps + 1)) % (num_timesteps + 1)
    log_cumprod_at = extract(log_cumprod_at, t, log_x_start.shape)         # at~
    log_cumprod_bt = extract(log_cumprod_bt, t, log_x_start.shape)         # bt~
    log_probs = torch.cat([
        log_add_exp(log_x_start + log_cumprod_at, log_cumprod_bt)
    ], dim=1)
    return log_probs


def q_pred_log_one_step(log_x_start, t, log_at, log_bt):
    log_at = extract(log_at, t, log_x_start.shape)         # at~
    log_bt = extract(log_bt, t, log_x_start.shape)         # bt~
    log_probs_one_step = torch.cat([
        log_add_exp(log_x_start + log_at, log_bt)
    ], dim=1)
    return log_probs_one_step


def q_posterior(x_start, x_t, t, num_timesteps, num_classes, log_cumprod_at, log_cumprod_bt, log_at, log_bt):       
    log_x_start = index_to_log_onehot(x_start, num_classes)
    log_x_t = index_to_log_onehot(x_t, num_classes)
     
    # compute q(x_t|x_0)
    log_xt_given_x_start = q_pred_log(log_x_start, t, num_timesteps, log_cumprod_at, log_cumprod_bt)     
    
    # compute q(x_t|x_t-1,x_0) = q(x_t|x_t-1)
    # [1] Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions     
    # the following actually compute q(x_t+1|x_t), but [1] says it is the same as q(x_t|x_t-1)
    # see the appendix of [1]               
    log_xt_given_x_t_minus_1 = q_pred_log_one_step(log_x_t, t, log_at, log_bt)     
    
    # compute q(x_t-1|x_0)
    log_xt_minus_1_given_x_start = q_pred_log(log_x_start, t-1, num_timesteps, log_cumprod_at, log_cumprod_bt)
    
    log_EV_xtmin_given_xt_given_xstart = log_xt_given_x_t_minus_1 + log_xt_minus_1_given_x_start - log_xt_given_x_start
    
    log_probs = torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    sample_logits = sample_categorical(log_probs)
    return sample_logits


def q_posterior_log(log_x_start, log_x_t, t, num_timesteps, num_classes, log_cumprod_at, log_cumprod_bt, log_at, log_bt):       
    # compute q(x_t|x_0)
    log_xt_given_x_start = q_pred_log(log_x_start, t, num_timesteps, log_cumprod_at, log_cumprod_bt)     
    
    # compute q(x_t|x_t-1,x_0) = q(x_t|x_t-1)
    # [1] Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions     
    # the following actually compute q(x_t+1|x_t), but [1] says it is the same as q(x_t|x_t-1)
    # see the appendix of [1]               
    log_xt_given_x_t_minus_1 = q_pred_log_one_step(log_x_t, t, log_at, log_bt)     
    
    # compute q(x_t-1|x_0)
    log_xt_minus_1_given_x_start = q_pred_log(log_x_start, t-1, num_timesteps, log_cumprod_at, log_cumprod_bt)
    
    log_EV_xtmin_given_xt_given_xstart = log_xt_given_x_t_minus_1 + log_xt_minus_1_given_x_start - log_xt_given_x_start
    
    log_probs = torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    return log_probs

## diffusion based on Q-transition matrix 
def _get_nearestneighbor_transition_mat(bt, t, confusion_matrix, band_diagonal):
    """Computes transition matrix for q(x_t|x_{t-1}).
    Nearest neighbor transition matrix inspired from the text word embedding distance to introduce locality.
    Args:
        t: timestep. integer scalar.
    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = bt[t]
    beta_t = beta_t.cpu().numpy()
    # beta_t = beta_t * 100 ## increasing beta_t value 
    
    if band_diagonal:
        ## width paramater to be decided 
        dim = 20 ## number of different classes 
        width = 10 ## a hyper parameter 
        band = np.tri(
            dim, dim, width // 2, dtype=np.float64) - np.tri(
                dim, dim, -width // 2, dtype=np.float64)
        arr = band / band.sum(0, keepdims=True)
        matrix = beta_t * arr + (1 - beta_t) * np.eye(dim)
        matrix = torch.from_numpy(matrix).to(bt.device)  
        return matrix
        
    else: # confusion matrix type transition
        
        ## building similarity matrix, rate and base matrix using adjacency matrix 
        ## adjacency matrix of k=3 dervied from confusion matrix of oneformer model 
        # list_of_lists = [[0,        0.38,           0,           0,     0,     0,            0,              0,              0,       0.03,        0,      0,       0,   0.07,      0,       0,      0,        0,           0, 0],
        #         [3.82,        0,           0.47,           0,     0,     0,            0,              0,              0,         0.65,        0,      0,       0,      0,      0,       0,      0,        0,            0, 00],
        #         [0,        0.12,           0,           0,     0,     0.3,            0,              0,              1.32,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0, 0],
        #         [0,        0,           10.84,           0,     3.71,     0,            0,              0,              1.99,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [0,        0,           9.97,           6.06,     0,     0,            0,              0,              2.18,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [0,        1.38,           7.83,           0,     0,     0,            0,              0,              4.09,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [0,        0,           5.35,           0,     0,     2.23,            0,              0,              4.69,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [0,        0,           4.83,           0,     0,     1.22,            0,              0,              1.9,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [0,        0,           1.79,           0,     0,     0.31,            0,              0,              0,         0.53,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [1.6,       9.17,           0,           0,     0,     0,            0,              0,              9.71,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [0,        0,           11.51,           0,     0,     0.09,            0,              0,              1.01,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [0,        0,           3.65,           0,     0,     0,            0,              0,              0.53,         0,        0,      0,       0.97,      0,      0,       0,      0,        0,            0,0],
        #         [0,        0,           1.47,           0,     0,     0,            0,              0,              0,         0,        0,      2.32,            0,      0,      0,       0,      0,        0,            5.16,0],
        #         [0.77,        0,           0.41,           0,     0,     0,            0,              0,              0.25,         0,        0,      0,       0,      0,      0,       0,      0,        0,            0,0],
        #         [0,        0,           1.24,           0,     0,     0,            0,              0,              0.46,         0,        0,      0,       0,      3.96,         0,          0,         0,           0,      0,0],
        #         [0.61,        0,           0.83,           0,     0,     0,            0,              0,              0,         0,        0,      0,       0,       0.55,      0,           0,         0,            0,    0,0],
        #         [0,        0,           2.19,           0,     0,     0,            0,              0,              1.37,         0,        0,      0,       0,      0,      0,       0.38,      0,        0,            0,0],
        #         [0,        0,           2.61,           0,     0,     0,            0,              0,              0,         0,        0,      2.06,       0,      0,      0,       0,      0,        0,            2.16,0],
        #         [0,        1.58,           4.19,           0,     1.45,     0,            0,              0,              0,         0,        0,      0,       1.58,      0,      0,       0,      0,        0,            0,0],
        #         [0.7,       0.2,            0,              0,      0,      0,            0,              0,              0.1,       0,         0,     0,         0,       0,       0,      0,      0,        0,            0,0]
        #         ] ## background class also added for now making its relative dependency with road, sidewalk and vegetation 
        # list_of_lists_arr = np.array(list_of_lists)  
        # # # ## one-hot adjacency matrix 
        # adjacency_matrix_one_hot = list_of_lists_arr 
        # adjacency_matrix_one_hot[list_of_lists_arr > 0] = 1 
        # ## from google_research/d3pm/text/diffusion
        # adjacency_matrix_one_hot = (adjacency_matrix_one_hot + adjacency_matrix_one_hot.T) / (2 * 3) ## for building the symmetricity of adjacency matrix and k = 3
        # transition_rate = adjacency_matrix_one_hot - np.diagflat(np.sum(adjacency_matrix_one_hot, axis=1))
        # matrix = scipy.linalg.expm(
        #             np.array(beta_t * transition_rate, dtype=np.float64)) 
        # matrix = scipy.linalg.expm(
        #             np.array(bt[0].cpu().numpy() * transition_rate, dtype=np.float64)) ## using the one as given in d3pm original code
        # print('************', np.diag(matrix))
        # print('^^^^^^^^^^^^^^^^', np.max(matrix), np.min(matrix))
        
        
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
        matrix = np.ones((20,20)) ## 20 is the number of classes; making a uniform transition matrix 
        matrix[:19, :19] = confusion_matrix  ## this is similarity matrix...main thing as this says
        # np.fill_diagonal(matrix, 0) ## making dia zero so as to be in use in matrix expo method ## no changes to introduced in confusion matrix calc ## first making the matrix zeroing out the dia as there is severe dis balance, because of dia in confusion matrix 
        np.fill_diagonal(matrix, beta_t*np.diag(confusion_matrix))
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

        # ### building rate matrix  
        # ## matrix exponential for rate matrix 
        # transition_rate = matrix - np.diagflat(np.sum(matrix, axis=1)) 
        
        # ### building base matrix 
        # matrix = scipy.linalg.expm(
        #             np.array(beta_t * transition_rate, dtype=np.float64))  ## base matrix 
        
        # matrix = (1 - beta_t)*np.eye(20) + beta_t * matrix  ## not working with sinkhorn ## 1. similar to uniform and absorbtion state transition matrix, 2. for transition into another state is like corrupting which confusion matrix stores info..for staying in the same, is like correct which gradually lowers down as time t increases...so this formulation make sense, of bringing it above sinkhorn algorithm
        
        # matrix = torch.from_numpy(matrix).to(bt.device) ## cuda out of memory here...alas!
        # ## sinkhorn algo for base matrix
        for _ in range(5): # number of iterations is a hyperparameter of sinkhorn's algo ## till in covergence 
            matrix = matrix / matrix.sum(1, keepdims=True)
            matrix = matrix / matrix.sum(0, keepdims=True)
        matrix = matrix / matrix.sum(1, keepdims=True) # rows should sum up to one exactly even if column a bit off from one 
        
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
        
        matrix = torch.from_numpy(matrix).to(bt.device) 
        # torch.save(matrix, 'q_matrix'+ str(t) + '.pt') ## saving the tensor for analysing it
        # print('time', t , 'matrix saved')
        # print('*******************',matrix.dtype) # double = float64 
        
        # return torch.from_numpy(matrix).to(bt.device)
        return matrix

def q_mats_from_onestepsdot(bt, num_timesteps, confusion_matrix, band_diagonal): # return: Qt = Q_1.Q_2.Q_3...Q_t, input-arguments = set of betas values over diffusion timesteps and total number of diffusion timesteps
    q_onestep_mats = [_get_nearestneighbor_transition_mat(bt, t, confusion_matrix, band_diagonal) 
                               for t in range(0, num_timesteps)]
    q_mat_t = q_onestep_mats[0]
    q_mats = [q_mat_t]
    for t in range(1, num_timesteps):
        q_mat_t = torch.tensordot(q_mat_t, q_onestep_mats[t],
                                      dims=[[1], [0]])
        # q_mat_t.fill_diagonal_(0) ## forcefully making it zero 
        q_mats.append(q_mat_t)
    q_mats = torch.stack(q_mats, dim=0) 
    return q_mats

def q_pred_from_mats(x_start, t, num_timesteps, num_classes, q_mats): 
    B, H, W = x_start.shape # label map
    # torch.save(q_mats, 'q_mats.pt') ## saving the tensor for analysing it
    # print('>>>>>>>>>>>>saving done')
    t = (t + (num_timesteps + 1)) % (num_timesteps + 1)  # having consistency with the original DDPS algo...so using this
    q_mats_t = torch.index_select(q_mats, dim=0, index=t)
    x_start_onehot = F.one_hot(x_start.view(B, -1).to(torch.int64), num_classes).to(torch.float64)
    out = torch.matmul(x_start_onehot, q_mats_t)
    # print('>>>>>>>>>>>>>>>', out.shape, out)
    # print('<<<<<<<<<<<<<<?>>>>>>>>>>>>>',out.unique(), t) ## not too much differnce 
    out = out.view(B, num_classes, H, W)
    # logits = out ## random testing 
    logits = torch.log(out.clamp(min=1e-30)) ## with relevant to original DDPS code 
    # logits = torch.log(out + 1e-6) ## with relevant to d3pm pytorch code
    sample_logits = sample_categorical(logits)
    return sample_logits