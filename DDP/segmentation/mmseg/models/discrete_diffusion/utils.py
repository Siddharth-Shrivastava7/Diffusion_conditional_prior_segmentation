import numpy as np
import torch
import math
import scipy

from torch.nn import functional as F


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


def get_transition_rate(similarity_matrix): 
    per_label_sums = similarity_matrix.sum(axis=1)[:, np.newaxis]
    similarity_matrix_norm = similarity_matrix.astype(np.float64) / per_label_sums  
    transition_rate_matrix = similarity_matrix_norm - np.diag(np.sum(similarity_matrix_norm, axis=1)) ## transition_rate sum to zero (rowwise), for making a valid base transition matrix
    return transition_rate_matrix


def logits_to_categorical(logits):
    uniform_noise = torch.rand_like(logits)
    ## # To avoid numerical issues clip the uniform noise to a minimum value
    uniform_noise = torch.clamp(uniform_noise, min=torch.finfo(uniform_noise.dtype).tiny, max=1.)
    gumbel_noise = - torch.log(-torch.log(uniform_noise))
    sample = (gumbel_noise + logits).argmax(dim=1)
    return sample


def builder_fn(trans_rate, exponent):
    """
    Function that computes a matrix exponential. 

    Function that, given a float exponent parameter, returns a
    transition matrix T[i, j] = p(x_t = j | x_0 = i) representing a matrix
    exponetial with the given exponent.
    """
    trans_matrix = scipy.linalg.expm(np.array(trans_rate * exponent, dtype=np.float64))
    return trans_matrix

def compute_information_removal_samples_closed_form(trans_rate, init_distribution, min_exponent=1e-4,
                                                    max_exponent=1e5,
                                                    interpolation_steps=256):  
    """Compute mutual information by evaluating a closed form estimate.

    Chooses interpolation steps, then evaluates mutual information for each one.

    Args:
      builder_fn: Function that, given a float exponent parameter, returns a
        transition matrix T[i, j] = p(x_t = j | x_0 = i) representing a matrix
        exponetial with the given exponent.
      init_distribution: Initial distribution of semantic class labels.
      min_exponent: Smallest non-zero exponent to try.
      max_exponent: Largest exponent to try.
      interpolation_steps: How many interpolation steps to try.

    Returns:
      exponents: Array of exponents for which we computed relative mutual
        information removal.
      information_removals: Array of the information removal for each exponent.
    """
    
    query_exponents = np.geomspace(min_exponent, max_exponent,
                                  interpolation_steps)

    information_removals = [] 
    for exponent in query_exponents:  
        trans_matrix = builder_fn(trans_rate, exponent) 
        info_remove = compute_relative_information_removal(trans_matrix, init_distribution) 
        information_removals.append(info_remove) 
    information_removals = np.stack(information_removals)
    
    return query_exponents, information_removals

def compute_relative_information_removal(transition_matrix, init_distribution, use_perplexity=False):

    """Computes removal of (mutual) information after applying a transition matrix.

    I(x_t; x_0) = [ log p(x_0, x_t) - log p(x_0) - log p(x_t)]
                = H(x_0) + H(x_t) - H(x_0, x_t)
          result = 1 - I(x_t; x_0) / H(x_0)
                = 1 - (H(x_0) + H(x_t) - H(x_0, x_t)) / H(x_0)
                = (H(x_0, x_t) - H(x_t)) / H(x_0)

    Args:
      transition_matrix: float32 matrix such that transition_matrix[i, j] = p(x_t
        = j | x_0 = i)
      init_distribution: float32 matrix reprezenting p(x_0)
      use_perplexity: Use conditional perplexity(ish) instead of MI. Assumes
        convergence to uniform. < Uniform thing we don't have :) > 

    Returns:
      Normalized information removal, which should be zero for the identity
      matrix,
      and 1 for a transition matrix which does not depend on the initial state.
    """
    # Normalizations for stability ## with logarithms to work with 
    log_transition = np.log(transition_matrix)  ## log 
    log_transition = (log_transition - scipy.special.logsumexp(log_transition, axis=1, keepdims=True)) ## norm axis 1 differnt from the code (differ)
    log_initial = np.log(init_distribution)
    log_initial = (log_initial - scipy.special.logsumexp(log_initial, keepdims=True)) ## check as change  has been done here ## differ
    log_joint = log_initial[:, None] + log_transition ## differ
    log_marginal_after = scipy.special.logsumexp(log_joint, axis=0) ## differ 

    joint_entropy = -np.sum(np.where(log_joint == -np.inf, 0.0, np.exp(log_joint) * log_joint))
    initial_entropy = -np.sum(np.where(log_initial == -np.inf, 0.0, np.exp(log_initial) * log_initial)) 
    marginal_after_entropy = -np.sum(np.where(log_marginal_after == -np.inf, 0.0, np.exp(log_marginal_after) * log_marginal_after)) 

    if use_perplexity:
        dim = init_distribution.shape[0]
        conditional_perplexity = np.exp(joint_entropy - initial_entropy) 
        return (conditional_perplexity - 1) / (dim - 1) 
    else: 
        information_removal = (joint_entropy - marginal_after_entropy) / initial_entropy
    return information_removal 

def transition_rate_expm(matrix, target_diagonal=1e-3, renormalize_rows=True):  
    """Slightly improved expm for transition rate matrices.

    A transition rate matrix will always have columns that sum to zero, and will
    have nonnegative entries everywhere except the diagonal. We can ensure some
    stability by controlling the magnitude of the diagonal elements and
    renormalizing during each squaring to reduce error.
    
    Args:
    matrix: The matrix to compute a matrix exponential for.
    target_diagonal: Maximum magnitude of the diagonal elements for which it is
      "safe" to approximate e(tA) as I + tA. Will automatically perform more
      iterations until this is small enough to be a good approximation.
    renormalize_cols: Whether to renormalize the columns of the result, with the
      assumption that the rate matrix summed to zero across the columns. This
      property should always hold, so renormalizing can prevent errors from
      exploding.
    
    Returns:
    Approximation of expm(matrix).
    """ 
    max_diag = np.max(-np.diag(matrix)) 
    target_diagonal= ( 1 / 19) ## don't know, cause not going for uniform .. at stationary>>>
    iterations_for_diagonal = np.ceil(np.log2(max_diag) - np.log2(target_diagonal))
    iterations_for_mixing = np.ceil(np.log2(matrix.shape[0]))
    iterations = np.maximum(iterations_for_diagonal, iterations_for_mixing).astype(np.int32)
    tiny_approx = np.eye(matrix.shape[0]) + matrix / (2.0**iterations)    
    mat = tiny_approx.copy()
    for i in range(iterations): 
        mat = np.dot(mat, mat) 
        if renormalize_rows:
          mat = mat / np.sum(mat, axis=1, keepdims=True)
    return mat

def compute_information_removal_samples_by_squaring(rate_matrix,
                                                    init_distribution,
                                                    min_exponent=1e-4,
                                                    max_exponent=1e5,
                                                    interpolation_steps=256,
                                                    use_perplexity=False): 

    """Compute mutual information using repeated squaring.
    
      Reduces a bunch of repeated work by evaluating power-of-two exponents using
      repeated squaring, starting from a few different test offsets to fill the
      gaps between powers of two.
    
      Args:
        rate_matrix: Transition rate matrix of shape [vocab_size, vocab_size]
        init_distribution: Initial distribution of tokens.
        min_exponent: Smallest non-zero exponent to try.
        max_exponent: Largest exponent to try.
        interpolation_steps: Minimum number of interpolation steps to try.
        use_perplexity: Use conditional perplexity(ish) instead of MI
    
      Returns:
        exponents: Array of exponents for which we computed relative mutual
          information removal.
        information_removals: Array of the information removal for each exponent.
    """ 
    # How many powers of two do we need to fill the range?
    powers_of_two = 1 + np.ceil(np.log2(max_exponent) - np.log2(min_exponent)).astype(np.int32)  
    # How many shifts should we evaluate between each power of two? For instance,
    # in addition to evaluating at 1, 2, 4, 8, 16, 32 we might also evaluate at
    # 3/2, 3, 6, 12, 24, 48. Increasing interpolation steps will increase this.
    shifts = np.ceil(interpolation_steps / powers_of_two).astype(np.int32) 
    # Figure out the base exponents (1 and 3/2 in the above example, but there
    # may be more)
    base_exponents = np.exp2(np.log2(min_exponent) + np.linspace(0, 1, shifts, endpoint=False)) 
    
    for base_exponent in base_exponents:  
        base_matrix = transition_rate_expm(base_exponent * rate_matrix)   
        # base_matrix = scipy.linalg.expm(np.array(base_exponent * rate_matrix, dtype=np.float64)) ## using scipy instead of above base matrix calc
        mat = base_matrix
        ys = [] 
        for i in np.arange(powers_of_two): 
            exponent = base_exponent * (2.0**i) 
            info_removal = compute_relative_information_removal(mat, init_distribution, use_perplexity=use_perplexity) 
            mat = np.dot(mat, mat)
            mat = mat / np.sum(mat, axis=1, keepdims=True)
            ys.append((exponent, info_removal)) 
        ys = np.stack(ys)  
        exponents = ys[:,0]
        info_removals = ys[:, 1]
    return exponents.reshape([-1]), info_removals.reshape([-1]) 

def build_mutual_information_schedule(schedule_steps,
                                      exponents,
                                      information_removals,
                                      allow_out_of_bounds=False,
                                      kind="linear"): # "warn" ## logging
    """Compute a mutual-information-based schedule by interpolation.

      Args:
        schedule_steps: Desired number of steps in the schedule.
        exponents: Array of exponents for which we computed relative mutual
          information removal.
        information_removals: Array of the information removal for each exponent.
        allow_out_of_bounds: Whether to allow interpolation for mutual information
          values that are not encountered before `max_exponent`. If True, clips the
          schedule so that it ends at the mutual info for `max_exponent` instead of
          at the desired (near-one) amount of mutual information removal. If False,
          throws an error.
        kind: one of ['linear', 'cosine']. Used to determine the schedule used.
    
      Returns:
        schedule_info_removals: float32[schedule_steps] array giving the amount of
          relative information removal at each point in the schedule. Will linearly
          interpolate between 0 and 1, not including either endpoint, unless this
          goes out of bounds and `allow_out_of_bounds=True`, in which case it may
          linearly interpolate to some value smaller than 1. Note that this may
          not be exactly correct due to the interpolation, but it should be close.
        schedule_exponents: float32[schedule_steps] array with the exponents
          needed to obtain each level of information removal. Note that this array
          does NOT include zero or infinity at the beginning/end, which are needed
          to obtain zero or one information removal. The caller should take care of
          padding so that the schedule takes the appropriate number of steps, for
          instance by adding zero to the front and ensuring that the sequence is
          replaced by a mask at the last step.
    """
    exponents = np.array(exponents)
    information_removals = np.array(information_removals)
    # Sort by exponent.
    permutation = np.argsort(exponents) 
    exponents = exponents[permutation]
    information_removals = information_removals[permutation] 
    # Fix out-of-order information removals due to numerical error.
    cmax_info_removal = np.maximum.accumulate(information_removals)
    bad = information_removals <= np.concatenate([[0], cmax_info_removal[:-1]])
    exponents = exponents[~bad]
    information_removals = information_removals[~bad]  
    # Add zero at the start.
    exponents = np.concatenate([[0], exponents])
    information_removals = np.concatenate([[0], information_removals])    

    # Interpolate monotonically so that our exponents are non-decreasing
    interpolator = scipy.interpolate.PchipInterpolator(information_removals, exponents, extrapolate=False)  # monotonic cubic interpolation 

    if kind == "linear":
        schedule_info_removals = np.linspace(0, 1, schedule_steps + 2)[1:-1]   # skipping the first and the last step 
    
    elif kind == "cosine": 
        s = 0.008 
        def cosine_fn(step):
          return np.cos((step / schedule_steps + s) / (1 + s) * np.pi / 2) 
        
        schedule_info_removals = 1 - cosine_fn(np.arange(schedule_steps)) 
    else:
        raise ValueError(f"kind {kind} is not supported.") 

    if schedule_info_removals[-1] > information_removals[-1]: 
        if allow_out_of_bounds: 
            if allow_out_of_bounds == "warn": 
                ## logging 
                # "build_mutual_information_schedule: Requested mutual "
                # "information removal value schedule_info_removals[-1] for "
                # "schedule was larger than largest observed value "
                # information_removals[-1]. Clipping schedule to this largest "
                # "observed value; consider increasing extrapolation range.",
                pass 
            schedule_info_removals = (np.linspace(0, information_removals[-1], schedule_steps + 1)[1:]) 
        else:
            raise ValueError(
            "Requested mutual information removal value "
            f"{schedule_info_removals[-1]} for schedule was larger than largest "
            f"observed value {information_removals[-1]}") 
            
    schedule_exponents = interpolator(schedule_info_removals) 
    return schedule_info_removals, schedule_exponents 


def get_powers(schedule_steps, 
                transition_rate, 
                init_distribution,
                min_exponent=1e-4,
                max_exponent=1e5,
                interpolation_steps=256, 
                kind = 'linear', 
                allow_out_of_bounds=False): 
  
    # ## using this way < both the below methods provide similiar results > 
    query_exponents, query_info_removals = compute_information_removal_samples_closed_form(
                                                        transition_rate, 
                                                        init_distribution,
                                                        min_exponent,
                                                        max_exponent,
                                                        interpolation_steps) 
    # query_exponents, query_info_removals = compute_information_removal_samples_by_squaring(
    #                                                     transition_rate, 
    #                                                     init_distribution,
    #                                                     min_exponent,
    #                                                     max_exponent,
    #                                                     interpolation_steps) 
    
    _, middle_exponents = build_mutual_information_schedule(
          schedule_steps, 
          query_exponents, 
          query_info_removals, 
          kind, allow_out_of_bounds) 
    
    exponents = np.concatenate([np.zeros([1]), middle_exponents]) # shape -> (self.schedule_steps +1, )
    ## rounding it off 
    min_exponent = middle_exponents[0] 
    powers = np.round(exponents / min_exponent).astype(np.int32)  # shape -> (self.schedule_steps +1, )
    return powers