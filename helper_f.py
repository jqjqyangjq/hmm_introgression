from collections import defaultdict
import pandas as pd
import numpy as np
from numba import njit
import json
from math import exp, ceil

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
            """
            A,  C,G,T,  aa,ac,ag,at,cc,cg,ct,gg,gt,tt
            For now, get max of G1
            """
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
            100 * 100 - 101 * 1000
            '''
            ''' go to current mut interval'''
            if ((mut_full.iloc[inter]['start'] <= (windows[current_window] * window_size)) and (mut_full.iloc[inter]['end'] >= (windows[current_window] + 1) * window_size)):
                mut.extend([mut_full.iloc[inter]['mut_rate']] * obs_count[current_window])
                current_window += 1
                break
            inter += 1
        if ((inter >= len(mut_full)) or (current_window >= len(windows))):
            break
    return mut