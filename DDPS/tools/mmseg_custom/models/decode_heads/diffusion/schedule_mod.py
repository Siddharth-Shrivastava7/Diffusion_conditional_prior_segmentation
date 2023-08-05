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


def alpha_schedule_torch(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = torch.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = torch.cat((torch.tensor([1]), att))
    at = att[1:] / att[:-1]
    bt = (1 - at) / N
    att = torch.cat((att[1:], torch.tensor([1])))
    btt = (1 - att) / N
    return at, bt, att, btt


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
def _get_nearestneighbor_transition_mat(bt, t):
    """Computes transition matrix for q(x_t|x_{t-1}).
    Nearest neighbor transition matrix inspired from the text word embedding distance to introduce locality.
    Args:
        t: timestep. integer scalar.
    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = extract(bt, t, bt.shape) ## here need to change the 't' dimension
    
    beta_t = beta_t.numpy()
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
    adjacency_matrix_one_hot = adjacency_matrix_one_hot + adjacency_matrix_one_hot.T ## for building the symmetricity of adjacency matrix
    transition_rate = adjacency_matrix_one_hot - np.diagflat(np.sum(adjacency_matrix_one_hot, axis=1))
    matrix = scipy.linalg.expm(
                np.array(beta_t * transition_rate, dtype=np.float64))
    matrix / matrix.sum(0, keepdims=True) 
    return torch.from_numpy(matrix)

def q_mats_from_onestepsdot(bt, num_timesteps): # return: Qt = Q_1.Q_2.Q_3...Q_t, input-arguments = set of betas values over diffusion timesteps and total number of diffusion timesteps
    q_onestep_mats = [_get_nearestneighbor_transition_mat(bt, t) 
                               for t in range(0, num_timesteps)]
    q_mats = [q_onestep_mats[0]]
    for t in range(1, num_timesteps):
        q_mat_t = torch.tensordot(q_mat_t, q_onestep_mats[t],
                                      dims=[[1], [0]])
        q_mats.append(q_mat_t)
    return q_mats

def q_pred_from_mats(x_start, t,num_timesteps, num_classes, bt): 
    q_mats = q_mats_from_onestepsdot(bt, num_timesteps)
    B, _, H, W = x_start.shape
    q_mats_t = torch.index_select(q_mats, dim=0, index=t)
    x_start_onehot = F.one_hot(x_start.view(B, -1).to(torch.int64), num_classes).to(torch.float32)
    out = torch.matmul(x_start_onehot, q_mats_t)
    out = out.view(B, num_classes, H, W)
    logits = torch.log(out.clamp(min=1e-30)) ## rather than "torch.log(out + 1e-6)"
    sample_logits = sample_categorical(logits)
    return sample_logits