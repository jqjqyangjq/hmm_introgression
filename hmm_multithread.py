from collections import defaultdict
import pandas as pd
import numpy as np
from numba import njit
import json
from math import exp, ceil


from helper_f import load_observations, get_mut_rates

@njit
def snp2bin_scale(e_out, e_in, ix, scale):    
    scale[:] = 0
    for i, row in enumerate(ix):
        e_out[row] *= e_in[i]    # this is where numeric underflow happens
        while np.log(np.min(e_out[row])) < -1:   # if e_out[row] underflows.
            e_out[row] *= np.exp(1)
            scale[row] += 1        

def snp2bin_scale_log(e_out, e_in, ix, scale):
    
    return 0


class HMMParam:
    def __init__(self, state_names, starting_probabilities, transitions, emissions):
        self.state_names = np.array(state_names)
        self.starting_probabilities = np.array(starting_probabilities)
        self.transitions = np.array(transitions)
        self.emissions = np.array(emissions)

    def __str__(self):
        out = f"> state_names = {self.state_names.tolist()}\n"
        out += f"> starting_probabilities = {np.matrix.round(self.starting_probabilities, 4).tolist()}\n"
        out += f"> transitions = {np.matrix.round(self.transitions, 8).tolist()}\n"
        out += f"> emissions = {np.matrix.round(self.emissions, 8).tolist()}"
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.state_names}, {self.starting_probabilities}, {self.transitions}, {self.emissions})"


# Read HMM parameters from a json file
def read_HMM_parameters_from_file(filename):

    if filename is None:
        return get_default_HMM_parameters()

    with open(filename) as json_file:
        data = json.load(json_file)

    return HMMParam(
        state_names=data["state_names"],
        starting_probabilities=data["starting_probabilities"],
        transitions=data["transitions"],
        emissions=data["emissions"],
    )


# Set default parameters
def get_default_HMM_parameters():
    return HMMParam(
        state_names=["Human", "Archaic"],
        starting_probabilities=[0.98, 0.02],
        transitions=[[0.9999, 0.0001], [0.02, 0.98]],
        #emissions = [0.00006, 0.00006]
        emissions=[0.00001896, 0.0002],
    )


@njit
def fwd_step(alpha_prev, E, trans_mat):
    alpha_new = (alpha_prev @ trans_mat) * E
    n = np.sum(alpha_new)
    return alpha_new / n, n


@njit
def forward(probabilities, transitions, init_start):
    n = len(probabilities)
    forwards_in = np.zeros((n, len(init_start)))
    scale_param = np.ones(n)
    for t in range(n):    
        if t == 0:
            forwards_in[t, :] = init_start * probabilities[t, :]
            scale_param[t] = np.sum(forwards_in[t, :])
            forwards_in[t, :] = forwards_in[t, :] / scale_param[t]
        else:
            forwards_in[t, :], scale_param[t] = fwd_step(
                forwards_in[t - 1, :], probabilities[t, :], transitions
            )
    return forwards_in, scale_param


@njit
def bwd_step(beta_next, E, trans_mat, n):
    beta = (trans_mat * E) @ beta_next
    return beta / n


@njit
def backward(emissions, transitions, scales):
    n, n_states = emissions.shape
    beta = np.ones((n, n_states))
    for i in range(n - 1, 0, -1):
        beta[i - 1, :] = bwd_step(beta[i, :], emissions[i, :], transitions, scales[i])
    return beta


def logoutput(pars, loglikelihood, iteration, log_file):
    n_states = len(pars.emissions)
    with open(log_file, "a") as f:
    # Make header
        if iteration == 0:
            print_emissions = "\t".join(["emis{0}".format(x + 1) for x in range(n_states)])
            print_starting_probabilities = "\t".join(
                ["start{0}".format(x + 1) for x in range(n_states)]
            )
            print_transitions = "\t".join(
                ["trans{0}_{0}".format(x + 1) for x in range(n_states)]
            )
            print(
                "it",
                "ll",
                print_starting_probabilities,
                print_emissions,
                print_transitions,
                sep="\t",
                file = f
            )
            print(
                "it",
                "ll",
                print_starting_probabilities,
                print_emissions,
                print_transitions,
                sep="\t"            )

        # Print parameters
        print_emissions = "\t".join([str(x) for x in np.matrix.round(pars.emissions, 8)])
        print_starting_probabilities = "\t".join(
            [str(x) for x in np.matrix.round(pars.starting_probabilities, 5)]
        )
        print_transitions = "\t".join(
            [str(x) for x in np.matrix.round(pars.transitions, 6).diagonal()]
        )
        print(
            iteration,
            round(loglikelihood, 6),
            print_starting_probabilities,
            print_emissions,
            print_transitions,
            sep="\t",
            file = f
        )
        print(
            iteration,
            round(loglikelihood, 6),
            print_starting_probabilities,
            print_emissions,
            print_transitions,
            sep="\t"
        )
def write_post_to_file(Z, chr_index, w_index, outfile):
    Z_df = pd.DataFrame()
    for i in range(len(Z)):
    # Create temporary dataframes for each element
        temp_df = pd.DataFrame()
        temp_df['Chr'] = [chr_index[i]] * len(w_index[i])  # Assign Chr value
        temp_df['Window'] = [item for item in w_index[i]]  # Extract the values from W[i]
        temp_df['state1'] = [item for item in Z[i][:,0]]  # Flatten and assign values from Z[i]
        temp_df['state2'] = [item for item in Z[i][:,1]]
        # Concatenate temporary dataframe to the main dataframe
        Z_df = pd.concat([Z_df, temp_df], ignore_index=True)
    Z_df.to_csv(outfile, index=False)
    
    
def linearize_obs(raw_obs, w):   # index all the snps [0,0,0..,1,1,...]
    g = (w_ind for i, (s,w_ind) in enumerate(zip((raw_obs[0]), w)) for j in s)
    SNP2BIN = np.fromiter(g, int)
    GLL = np.vstack((np.concatenate(raw_obs[0]), np.concatenate(raw_obs[1]))).T
    return GLL, SNP2BIN

def update_snp_prob(SNP, GLL, p, m_rates):
    """
    """
    update_geno_emissions(SNP, p, m_rates)      # per site P(G_lr=g | Z_l)
    SNP *= GLL[:, np.newaxis, :]       # per site P(O_lr, G_lr=g | Z_l) = P(O_lr | G_lr=g) P(G_lr=g | Z_l)


def update_geno_emissions(SNP, p, m_rates):
    """calculate P(G | Z)
    at least one derived allele at the pisition
    p_0derived = e^(-lambda) * lambda^0 / 0! = e^(-lambda)
    
    snp1 = 1 - p_0derived = 1 - e^(-lambda) with lambda = T / tu (* mu), approxiamte with snp1 = T / tu (* mu) 
    
    
    
    broadcast for all sites
    if m_mates is too small, SNP tends to be 1
    """
    ### approxiamation does make a little difference, but both still show the same pattern
    
    SNP[:, 0, 1] = np.array(m_rates) * p[0]
    SNP[:, 1, 1] = np.array(m_rates) * p[1]

    SNP[:, 0, 0] = (1 - SNP[:, 0, 1])
    SNP[:, 1, 0] = (1 - SNP[:, 1, 1])
    '''
    SNP[:, 0, 0] = np.exp(-np.array(m_rates) * p[0])
    SNP[:, 1, 0] = np.exp(-np.array(m_rates) * p[1])
    SNP[:, 0, 1] = 1 - SNP[:, 0, 0]
    SNP[:, 1, 1] = 1 - SNP[:, 1, 0]
    '''
    
def update_emissions_scale(E, SNP, GLL, p, SNP2BIN, scale, m_rates):
    update_snp_prob(SNP, GLL, p, m_rates)    # get SNP = P(O_lr, G_lr=g | Z_l) = P(O_lr | G_lr=g) P(G_lr=g | Z_l) = GLL * p_g.  numeric underflow safe?
    '''
    SNP per se seems to be calculated correctly.
    '''
    E[:] = 1
    snp2bin_scale(E, np.sum(SNP, 2), SNP2BIN, scale)  #also return the array of scaling factors
    '''
    get P(O_l | Z_l) = prod r ( P(O_lr | Z_l) ) 
    for each Z, P(O_l | Z_l) = prod r ( P(O_lr = g1 | Z_l) + P(O_lr = g2 | Z_l) )
    '''
    

def update_post_geno(PG, SNP, Z, SNP2BIN):
    """
    from admixfrog
    calculate P(G, Z | O)
    
    P(G_lr=g, Z_l| O) = P(Z|O) P(G | Z, O)
               = P(Z|O) P(O|G) P(G|Z) / P(O | Z)
               = P(Z|O) P(O|G) P(G|Z) / sum_g P(O, G=g | Z)
               = P(Z_l|O) P(O_lr, G_lr | Z_l) / sum_g P(O_lr, G_lr=g | Z)           See derivation of lambda in my letax note
    PG[n_snp x n_geno]: P( G| O'), prob of genotype given observations,
        parameters from previous iteration
    
    SNP[n_snp x n_states x n_geno]: P(O_lr, G_L=lr | Z_l)
    Z[n_bin x n_states]: P(Z | O')
    SNP2BIN [n_snp] : index assigning snps to windows
    """
    PG[:] = Z[SNP2BIN, :, np.newaxis] * SNP   # P(Z_l|O ) * P(O_lr, G_lr=g | Z_l)  
    
    '''
    as np.sum(SNP, 2) = P(O_lr | Z_l) 
    '''
    
    PG /= np.sum(SNP, 2)[:, :, np.newaxis]    # P(G_lr=g, Z_l | O), see above
    PG[np.isnan(PG)] = 0.0
    np.clip(PG, 0, 1, out=PG)

    assert np.all(PG >= 0)
    assert np.all(PG <= 1)
    assert np.allclose(np.sum(PG, (1, 2)), 1, atol=1e-6)

    return PG

def TrainModel(raw_obs, chr_index, w, obs_count, pars, post_file, window_size = 1000, 
               epsilon=1e-4, maxiterations=1000, log_file = None, m_rates_file = None):
    '''
    raw_obs, [[obs_chr1], [obs_chr2], ...]
    chr [chr1, chr2, ...]
    w [[chr1_w1, chr1_w2...], [chr2_w1,...], ...]
    w_index: full index from w1 to w-1
    '''
    
    
    
    #pars = read_HMM_parameters_from_file(input_pars)
    n_chr = len(chr_index)
    n_states = len(pars.starting_probabilities)
    n_windows = np.ones(n_chr)
    GLL = []
    SNP2BIN = []
    w_index = []
    Z = []
    E = []
    n_gt = 2
    previous_ll = -np.Inf
    SNP = []
    PG = []
    S = []
    fwd = []
    bwd = []
    scales = []
    m_rates = []
    if not m_rates_file is None:
        m_rates_full = pd.read_csv(m_rates_file, sep = '\t', 
                               header = None, names = ["chrom", "start", "end", "mut_rate"],
                               dtype = {"chrom": str, "start": int, "end": int, "mut_rate": float})
    for chr in range(n_chr):
        n_windows[chr] = w[chr][-1] - w[chr][0] + 1
        GLL_, SNP2BIN_ = linearize_obs(raw_obs[chr], w[chr])
        w_start = w[chr][0]
        w_index_ = np.arange(w_start, w[chr][-1] + 1)    # including missing windows in between
        SNP2BIN_ -= w_start
        n_snp = len(GLL_)
        n_windows_ = round(n_windows[chr])
        SNP_ = np.zeros((n_snp, n_states, n_gt))
        Z_ = np.zeros((n_windows_, n_states))  # P(Z | O)
        E_ = np.ones((n_windows_, n_states))  # P(O | Z)
        GLL.append(GLL_)
        SNP2BIN.append(SNP2BIN_)
        w_index.append(w_index_)
        Z.append(Z_)
        E.append(E_)
        SNP.append(SNP_)
        PG_ = np.zeros((n_snp, n_states, n_gt))
        PG.append(PG_)
        if m_rates_file is None:
            m_rates_ = np.ones(n_snp)
        else:
            m_rates_ = get_mut_rates(m_rates_full, window_size, w[chr], obs_count[chr], chr_index[chr])
        m_rates.append(m_rates_)
        assert len(m_rates_) == n_snp, f"missing part of mutation rates for obs in {chr}-th chromsome"
    # the data and an indexing array
    # adapted from admixfrog
    # create arrays for posterior, emissions
    for i in range(maxiterations):
        p = pars.emissions
        top = 0
        #bot = 0
        bot = np.zeros(2)
        new_trans_ = np.zeros((n_chr, n_states, n_states))
        new_ll = 0
        normalize = []
        for chr in range(n_chr):
            n_windows_ = round(n_windows[chr])
            S = np.zeros(n_windows_)
            update_emissions_scale(E[chr], SNP[chr], GLL[chr], p, SNP2BIN[chr], S, m_rates[chr])
            '''
            update E for per window(product of all sites)
            update S, scaling factor for each window
            Get P(O_l|Z_l),   by go through all r sites and g genotypes
            also update SNP  SNP = P(O_lr, G_lr=g | Z_l)
            '''
            fwd, scales = forward(E[chr], pars.transitions, pars.starting_probabilities)
            bwd = backward(E[chr], pars.transitions, scales)
            Z[chr][:] = fwd * bwd
            update_post_geno(PG[chr], SNP[chr], Z[chr], SNP2BIN[chr])
            '''
            P(G_lr = g, Z_l | O', theta)  see my(Jiaqi) letax note.
            '''
            assert np.allclose(np.sum(PG[chr], (1, 2)), 1)
            # PG = P(Z|O) P(O, G | Z) / sum_g P(O, G=g | Z) for all sites   .  affected by missing data?
            top += np.sum(PG[chr][:,:,1], axis = 0)
            bot[0] += np.sum(PG[chr][:,0,0]) + np.sum(PG[chr][:,0,1] * m_rates[chr])
            bot[1] += np.sum(PG[chr][:,1,0]) + np.sum(PG[chr][:,1,1] * m_rates[chr])
            #bot += np.sum(PG[chr],(0,2))
            for s1 in range(n_states):
                for s2 in range(n_states):
                    new_trans_[chr][s1, s2] = np.sum(
                        fwd[:-1, s1] * pars.transitions[s1, s2] * E[chr][1:, s2] * bwd[1:, s2] / scales[1:]
                    )
            new_ll += np.sum(np.log(scales)) -  np.sum(S)
            normalize.append(np.sum(Z[chr], axis = 0))      # sum over posterior across chrs
        
        new_p = top / bot
        new_trans = np.sum(new_trans_, 0)
        new_trans /= new_trans.sum(axis=1)[:, np.newaxis] #Ben/Laurits 
        



        normalize = np.sum(normalize, axis = 0)
        new_starting_probabilities = normalize/np.sum(normalize, axis=0) 
        pars = HMMParam(
            pars.state_names,
            new_starting_probabilities,
            new_trans,
            new_p
        )  
        logoutput(pars, new_ll, i, log_file = log_file)
        
        #if (new_ll - previous_ll < epsilon) & (new_ll - previous_ll > 0):  # epsilon ...
        if (new_ll - previous_ll < epsilon):
            break
        
        previous_ll = new_ll

    # Iterate over the elements in Z, W, and Chr
    write_post_to_file(Z, chr_index, w_index, post_file)

    #return pars, Z, E, PG, SNP, GLL, SNP2BIN    # all arrays must be same length
    return pars

def decode_from_params(raw_obs, chr_index, w, obs_count, pars, post_file, window_size = 1000,
             m_rates_file = None):
    n_chr = len(chr_index)
    n_states = len(pars.starting_probabilities)
    n_windows = np.ones(n_chr)
    GLL = []
    SNP2BIN = []
    w_index = []
    Z = []
    E = []
    n_gt = 2
    previous_ll = -np.Inf
    SNP = []
    PG = []
    S = []
    fwd = []
    bwd = []
    scales = []
    m_rates = []
    if not m_rates_file is None:
        m_rates_full = pd.read_csv(m_rates_file, sep = '\t', 
                               header = None, names = ["chrom", "start", "end", "mut_rate"],
                               dtype = {"chrom": str, "start": int, "end": int, "mut_rate": float})
    for chr in range(n_chr):
        n_windows[chr] = w[chr][-1] - w[chr][0] + 1
        GLL_, SNP2BIN_ = linearize_obs(raw_obs[chr], w[chr])
        w_start = w[chr][0]
        w_index_ = np.arange(w_start, w[chr][-1] + 1)    # including missing windows in between
        SNP2BIN_ -= w_start
        n_snp = len(GLL_)
        n_windows_ = round(n_windows[chr])
        SNP_ = np.zeros((n_snp, n_states, n_gt))
        Z_ = np.zeros((n_windows_, n_states))  # P(Z | O)
        E_ = np.ones((n_windows_, n_states))  # P(O | Z)
        GLL.append(GLL_)
        SNP2BIN.append(SNP2BIN_)
        w_index.append(w_index_)
        Z.append(Z_)
        E.append(E_)
        SNP.append(SNP_)
        PG_ = np.zeros((n_snp, n_states, n_gt))
        PG.append(PG_)
        if m_rates_file is None:
            m_rates_ = np.ones(n_snp)
        else:
            m_rates_ = get_mut_rates(m_rates_full, window_size, w[chr], obs_count[chr], chr_index[chr])
        m_rates.append(m_rates_)
        assert len(m_rates_) == n_snp, f"missing part of mutation rates for obs in {chr}-th chromsome"
    # the data and an indexing array
    # adapted from admixfrog
    # create arrays for posterior, emissions
    p = pars.emissions
    for chr in range(n_chr):
        n_windows_ = round(n_windows[chr])
        S = np.zeros(n_windows_)
        update_emissions_scale(E[chr], SNP[chr], GLL[chr], p, SNP2BIN[chr], S, m_rates[chr])
        fwd, scales = forward(E[chr], pars.transitions, pars.starting_probabilities)
        bwd = backward(E[chr], pars.transitions, scales)
        Z[chr][:] = fwd * bwd
    write_post_to_file(Z, chr_index, w_index, post_file)

def write_HMM_to_file(hmmparam, outfile):
    data = {key: value.tolist() for key, value in vars(hmmparam).items()}
    json_string = json.dumps(data, indent = 2)
    with open(outfile, 'w') as out:
        out.write(json_string)
