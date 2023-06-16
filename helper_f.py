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
    for chrom in list(gl["g_0"].keys()):
        gl_0_ = []
        gl_1_ = []
        for window in list(gl["g_0"][chrom].keys()):
            gl_0_.append(np.array(gl["g_0"][chrom][window]))
            gl_1_.append(np.array(gl["g_1"][chrom][window]))
        obs_chrs.append([gl_0_, gl_1_])
    return obs_chrs, list(gl["g_0"].keys()), list(list(gl["g_0"][chrom].keys()) for chrom in list(gl["g_0"].keys()) )
    '''
    return:
        full gll, chrom, window index for each chrom
    '''
    
def mut_rates(mut_file, window, window_size):
    '''
    return:
        mutation rate for each indexed window given the input.
        
    '''
    pass
    