from random import sample
from numpy import unique, random
import pandas as pd
from pandas import cut

def balanced_sample_maker(X, y, random_seed=None):
    """ return a balanced data set by oversampling minority class
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # oversampling on observations of positive label
    sample_size = uniq_counts[0]
    over_sample_idx = random.choice(groupby_levels[1], size=sample_size, replace=True).tolist()
    balanced_copy_idx = groupby_levels[0] + over_sample_idx
    random.shuffle(balanced_copy_idx)

    return X[balanced_copy_idx, :], y[balanced_copy_idx]
    
    
def stratified_sample_maker(x, size, k=10, is_return_index=False):
    """ binning continuous variable and conduct stratified sampling
        
    Parameters:
    ==========
    * x: {vector-like}
    * size: integer, size of samples
    * k: integer, the number of bins to cut continous variable 
    * is_return_index: boolean, 
         True,  sampled instance's index in x
         False, returned sample instance index
    """
    
    # detect if x is categorical variables 
    inst_x = x[0]
    if isinstance(inst_x, (int, float)):
        if isinstance(inst_x, int):
            uniq_x = list(set(x))
            if len(uniq_x) < 0.2 * len(x):
                is_number = False
            else:
                is_number = True
        else:
            is_number = True
    else: 
        is_number = False
    
    # bininig continuous variable
    if is_number:
        probs = np.linspace(0, 1, k+1, endpoint=True)
        cut_points = pd.Series(x).quantile(probs).tolist()
        cut_labels = [i for i in range(k)]
        x_bins = cut(x, bins=cut_points, labels=cut_labels)
    else:
        cut_labels = list(set(x))
        x_bins = x
    
    # calcualte the observed probabilties
    vc_df = pd.Series(x_bins).value_counts()
    cut_labels = vc_df.index.tolist()
    obs_probs  = (vc_df / sum(vc_df)).tolist()
    cum_size = 0
    cum_prob = 0

    sample_bin_size = []
    for ii, obs_prob in enumerate(obs_probs):
        if ii < len(cut_labels)-1:
            sb_size = round(size * obs_prob, 0)
            cum_size += sb_size
            cum_prob += obs_prob
        else:
            sb_size = size - cum_size
            cum_size += sb_size
            cum_prob += obs_prob
        sample_bin_size.append(int(sb_size))
     
    # generate samples
    sample_index = []
    for ii, (cut_label, bin_samp_size) in enumerate(zip(cut_labels, sample_bin_size)):
        pool = [i for i, xb in enumerate(x_bins) if xb == cut_label]
        samples = sample(pool, k=bin_samp_size)
        sample_index = sample_index + samples
     
    if is_return_index:
        return sample_index 
    else:
        return [x[i] for i in sample_index]
    