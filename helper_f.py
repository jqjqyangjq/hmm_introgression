from collections import defaultdict
import pandas as pd
import numpy as np
from numba import njit
import json
from math import exp, ceil
import os 
import gzip
"""
def load_observations(gll_file, window_size):  # return g0 g1
    
    
    # free space for each chrom.
    # cap on depth
    gl = defaultdict(lambda:defaultdict(lambda: defaultdict(list)))
    with open(gll_file, "r") as data:
        for line in data:
            (
                chrom,
                pos,
                Anc,
                _,
                _,
                _,
                _,
                dep,
                G1,
                G2,
                G3,
                G4,
                G5,
                G6,
                G7,
                G8,
                G9,
                G10,
            ) = line.strip().split()
            '''
            A,  C,G,T,  aa,ac,ag,at,cc,cg,ct,gg,gt,tt
            For now, get max of G1
            '''
            '''
            get anc base from ref
            '''            
            if Anc == "A":
                g_0 = float(G1)
                g_1 = max(
                    np.array(
                        [
                            float(G2),
                            float(G3),
                            float(G4),
                            float(G5),
                            float(G6),
                            float(G7),
                            float(G8),
                            float(G9),
                            float(G10),
                        ]
                    )
                )
                if g_1 >= 1:
                    g_1 = 1
            elif Anc == "C":
                g_0 = float(G5)
                g_1 = max(
                    np.array(
                        [
                            float(G1),
                            float(G2),
                            float(G3),
                            float(G4),
                            float(G6),
                            float(G7),
                            float(G8),
                            float(G9),
                            float(G10),
                        ]
                    )
                )
                if g_1 >= 1:
                    g_1 = 1
            elif Anc == "G":
                g_0 = float(G8)
                g_1 = max(
                    np.array(
                        [
                            float(G1),
                            float(G2),
                            float(G3),
                            float(G4),
                            float(G5),
                            float(G6),
                            float(G7),
                            float(G9),
                            float(G10),
                        ]
                    )
                )
                if g_1 >= 1:
                    g_1 = 1
            elif Anc == "T":
                g_0 = float(G10)
                g_1 = max(
                    np.array(
                        [
                            float(G1),
                            float(G2),
                            float(G3),
                            float(G4),
                            float(G5),
                            float(G6),
                            float(G7),
                            float(G8),
                            float(G9),
                        ]
                    )
                )
                if g_1 >= 1:
                    g_1 = 1
            #window = int(pos) - int(pos) % 1000
            window = ceil(int(pos) / window_size) - 1
            gl["g_0"][chrom][window].append(g_0)
            gl["g_1"][chrom][window].append(g_1)
    obs_chrs = []
    obs_count = []
    for chrom in list(gl["g_0"].keys()):
        gl_0_ = []
        gl_1_ = []
        obs_count_ = []
        for window in list(gl["g_0"][chrom].keys()):
            gl_0_.append(np.array(gl["g_0"][chrom][window]))
            obs_count_.append(len(gl["g_0"][chrom][window]))
            gl_1_.append(np.array(gl["g_1"][chrom][window]))
        obs_chrs.append([gl_0_, gl_1_])
        obs_count.append(obs_count_)
    return obs_chrs, list(gl["g_0"].keys()), list(list(gl["g_0"][chrom].keys()) for chrom in list(gl["g_0"].keys()) ), obs_count
    '''
    return:
        full gll[chrom][window], chrom, window index for each chrom, and the actual count of snps in each window
    '''
"""
def load_observations(gll_file, window_size=1000, filter_depth = False, maximum_dep = None, minimum_dep = None):  # return g0 g1
    
    
    # free space for each chrom.
    # cap on depth
    gl = defaultdict(lambda:defaultdict(lambda: defaultdict(list)))

    with gzip.open(gll_file, "rt") as data:
        for line in data:
            if not line.startswith("#"):
                ne = line.strip().split()
                len_ne = len(ne)
                break
    if len_ne == 9:
        print(f"loading simulated data from {gll_file}")
        with gzip.open(gll_file, "rt") as data:
            for line in data:
                (
                    chrom,
                    pos,
                    _,
                    _,
                    _,
                    _,
                    _,
                    g0,
                    g1
                ) = line.strip().split()
                g_0 = float(g0)
                g_1 = float(g1)
                window = ceil(int(pos) / window_size) - 1
                gl["g_0"][chrom][window].append(g_0)
                gl["g_1"][chrom][window].append(g_1)
    if len_ne == 6:
        print(f"loading real data from {gll_file}")
        with gzip.open(gll_file, "rt") as data:
            if filter_depth is False:
                for line in data:
                    (
                        chrom,
                        pos,
                        _,
                        _,
                        g0,
                        g1
                    ) = line.strip().split()
                    g_0 = float(g0)
                    g_1 = float(g1)
                    window = ceil(int(pos) / window_size) - 1
                    gl["g_0"][chrom][window].append(g_0)
                    gl["g_1"][chrom][window].append(g_1)
            else:
                assert not (maximum_dep is None), "maximum_dep is not None"
                assert not (minimum_dep is None), "minimum_dep is not None"
                print(f"keep positions with depth <= {maximum_dep} or depth >= {minimum_dep}")
                for line in data:
                    (
                        chrom,
                        pos,
                        _,
                        dep,
                        g0,
                        g1
                    ) = line.strip().split()
                    if (int(dep) <= maximum_dep) and (int(dep) >= minimum_dep):
                        g_0 = float(g0)
                        g_1 = float(g1)
                        window = ceil(int(pos) / window_size) - 1
                        gl["g_0"][chrom][window].append(g_0)
                        gl["g_1"][chrom][window].append(g_1)
    obs_chrs = []
    obs_count = []
    for chrom in list(gl["g_0"].keys()):
        gl_0_ = []
        gl_1_ = []
        obs_count_ = []
        for window in list(gl["g_0"][chrom].keys()):
            gl_0_.append(np.array(gl["g_0"][chrom][window]))
            obs_count_.append(len(gl["g_0"][chrom][window]))
            gl_1_.append(np.array(gl["g_1"][chrom][window]))
        obs_chrs.append([gl_0_, gl_1_])
        obs_count.append(obs_count_)
    return obs_chrs, list(gl["g_0"].keys()), list(list(gl["g_0"][chrom].keys()) for chrom in list(gl["g_0"].keys()) ), obs_count
"""
def load_observations_rec(gll_file, window_size = 1000):  # return g0 g1
    
    
    # free space for each chrom.
    # cap on depth
    gl = defaultdict(lambda:defaultdict(lambda: defaultdict(list)))

    with gzip.open(gll_file, "rt") as data:
        for line in data:
            (
                chrom,
                pos,
                _,
                _,
                Anc,
                _,
                _,
                _,
                _,
                g0,
                g1
            ) = line.strip().split()
            g_0 = float(g0)
            g_1 = float(g1)
            window = ceil(int(pos) / window_size) - 1
            gl["g_0"][chrom][bin].append(g_0)
            gl["g_1"][chrom][bin].append(g_1)
    obs_chrs = []
    obs_count = []
    for chrom in list(gl["g_0"].keys()):
        gl_0_ = []
        gl_1_ = []
        obs_count_ = []
        for window in list(gl["g_0"][chrom].keys()):
            gl_0_.append(np.array(gl["g_0"][chrom][window]))
            obs_count_.append(len(gl["g_0"][chrom][window]))
            gl_1_.append(np.array(gl["g_1"][chrom][window]))
        obs_chrs.append([gl_0_, gl_1_])
        obs_count.append(obs_count_)
    return obs_chrs, list(gl["g_0"].keys()), list(list(gl["g_0"][chrom].keys()) for chrom in list(gl["g_0"].keys()) ), obs_count
"""

def get_mut_rates(mut_full, window_size, windows, obs_count, chr):
    '''
    window: window index
    a chromsome
    e.g. 0 1000000 1.5
         1000000 2000000 2.5

    '''
    mut = []
    mut_full = mut_full[mut_full['chrom'] == chr]
    assert len(mut_full)!=0, f"mutation rate missing for chromsome {chr}"
    current_window = 0
    inter = 0   #interval of mut rates
    while current_window <= len(windows):
        while inter < len(mut_full):  
            '''
            window 100
            100 * 1000 - 101 * 1000
            '''


            ''' go to current mut interval'''
            if ((mut_full.iloc[inter]['start'] <= (windows[current_window] * window_size)) and (mut_full.iloc[inter]['end'] >= (windows[current_window] + 1) * window_size)):
                if (mut_full.iloc[inter]['mut_rate'] == 0) and (obs_count[current_window] > 0):   
                    '''
                    might cause numerical problem. so far all 0 mut_rates are manully modifyed
                    '''
                    mut.extend([0.000125] * obs_count[current_window])
                else:   
                    mut.extend([mut_full.iloc[inter]['mut_rate']] * obs_count[current_window])
                current_window += 1
                break
            inter += 1
        if ((inter >= len(mut_full)) or (current_window >= len(windows))):
            break
    return mut
    '''
    seems to load correctly. checked.
    '''
"""
I am lazy, so I directly calculate mutation rate either from simulation or from empirical data.
No functions embedded so far.
Super-high mutation rates comes from mappability filter. Nominator is too small in some windows.



def get_mut_mis(afr_vcf, out_file, window_size, region_bed = None):
    # get mutatino rate from simulated 50 afrs.
    
    '''
    get weights for each window
    '''
    mut = defaultdict(lambda:defaultdict(list))
    snp = f"zcat {afr_vcf} | sed 1d | awk '{{sum=0; for(i=25; i<=NF; i++) sum+=$i; print $1, $2, sum}}' "
    for line in os.popen(snp):
        chrom, pos, c = line.strip().split()
        window = ceil(int(pos) / window_size) - 1
        '''
        count the number of mutations along each window
        '''
        if int(c) > 0:
            mut[chrom][window].append(1)



    if not region_bed is None:
        for chrom in mut.keys():
            for window in mut[chrom].keys():
                start = window * window_size
                end = (window + 1) * window_size 
                
"""
def find_runs(inarray):
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


def Call(obs, chroms, starts, variants, hmm_parameters):
    
    post_seq = (forward_probs * backward_probs).T

    segments = []
    for (chrom, chrom_start_index, chrom_length_index) in find_runs(chroms):
        state_with_highest_prob = np.argmax(post_seq[:,chrom_start_index:chrom_start_index + chrom_length_index-1], axis = 0)
        for (state, start_index, length_index) in find_runs(state_with_highest_prob):

            start_index = start_index + chrom_start_index
            end_index = start_index + length_index

            genome_start = starts[start_index]
            genome_end = starts[start_index + length_index - 1]
            genome_length =  length_index * window_size

            snp_counter = np.sum(obs[start_index:end_index])
            mean_prob = round(np.mean(post_seq[state, start_index:end_index]), 5)
            #variants_segment = flatten_list(variants[start_index:end_index])

            
            # Diploid or haploid
            if '_hap' in chrom:
                newchrom, ploidity = chrom.split('_')
            else:
                ploidity = 'diploid'
                newchrom = chrom

            segments.append([newchrom, genome_start,  genome_end, genome_length, hmm_parameters.state_names[state], mean_prob, snp_counter, ploidity, variants_segment]) 
        
    return segments