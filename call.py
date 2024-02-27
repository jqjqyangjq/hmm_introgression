import pandas as pd
import numpy as np
from itertools import accumulate
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

def Call(posterior, hmm_parameters, called, window_size):
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
                s = start_genome * window_size
                end_genome = start_genome + length_index - 1
                e = end_genome * window_size + window_size
                mean_prob = round(np.mean(post_chr_T[state, start_index:(start_index+length_index)]), 5)
                print(chr, s, e, hmm_parameters.state_names[state], mean_prob, length_index, sep = '\t', file=f) 
def get_runs(posterior_file, penalty=0.2):
    posterior_ = pd.read_csv(posterior_file)
    frags = []
    chr = list(set(posterior_.Chr.to_list()))
    for i in chr:
        pp = posterior_[(posterior_['Chr'] == i)].reset_index(drop=True)
        s = pp.Window.to_list()[0]
        e = pp.Window.to_list()[-1]
        posterior = np.array(pp.state2)
        id_ = np.array(range(0,e-s+1))
        p0 = np.array(np.log(posterior + penalty))
        frag_score = 0
        while True:
            p = np.array([k for k in accumulate(p0, lambda x, y: max(x + y, 0))])
            pos_max, score_max = np.argmax(p), np.max(p)
            if score_max == 0.0:
                break
            else:
                pass
            zeros = np.where(p[:pos_max] == 0)[0]
            if len(zeros) == 0:
                pos_min = 0
            else:
                pos_min = np.max(zeros) + 1
            if pos_max != pos_min:
                frags.append((i, id_[pos_min]+s, id_[pos_max]+s, p[pos_max] - p[pos_min]))
            p0[pos_min : (pos_max + 1)] = 0
    out_ =  pd.DataFrame(frags, columns=["chrom","start", "end", "score"])
    out_.to_csv(posterior_file+f".called", index=False)
