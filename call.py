import pandas as pd
import numpy as np
# call segments
def find_runs(inarray):   # from Laurtis, get state with highest posterior(higher than 0.5 default) for each bin. 
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions

        for (a, b, c) in zip(ia[i], p, z):
            yield (a, b, c)

def Call(posterior, hmm_parameters, called):
    post_all = pd.read_csv(posterior)
    chr_all = post_all.Chr.unique()
    segments = []
    with open(called, 'w') as f:
        print("chr\tstart_window\tend_window\tstate\tmean_prob\tlength", file = f)
        for chr in chr_all:
            post_chr = post_all[post_all.Chr == chr].reset_index(drop=True)
            start = np.array(post_chr.Window)[0]
            length = len(post_chr)
            post_chr_T = np.zeros((2, length))
            post_chr_T[0,:] = np.array(post_chr.state1)[0:length]
            post_chr_T[1,:] = np.array(post_chr.state2)[0:length]
            state_with_highest_prob = np.argmax(post_chr_T[:,0:length], axis = 0)
            for (state, start_index, length_index) in find_runs(state_with_highest_prob):
                start_genome = start_index + start
                end_genome = start_genome + length_index - 1
                mean_prob = round(np.mean(post_chr_T[state, start_index:(start_index+length_index)]), 5)
                print(chr, start_genome, end_genome, hmm_parameters.state_names[state], mean_prob, length_index, sep = '\t', file=f) 
