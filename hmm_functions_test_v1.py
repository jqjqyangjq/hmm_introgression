from collections import defaultdict
import pandas as pd
import numpy as np
from numba import njit
import json
from math import exp


@njit
def snp2bin(e_out, e_in, ix):   # numeric underflow   E = 1
    for i, row in enumerate(ix):
        e_out[row] *= e_in[i]
        
        if np.min(e_out[row]) < 1e-250:
            e_out *= 10
            #print("normalize")


'''
e_out is always reset to 1
e_in is np.sum(SNP,2), hence no numeric underflow itself
track current scaling parameters

'''


@njit
def snp2bin_scale(e_out, e_in, ix, scale):    
    scale[:] = 0
    for i, row in enumerate(ix):
        e_out[row] *= e_in[i]    # this is where numeric underflow happens
        while np.log(np.min(e_out[row])) < -1:   # if e_out[row] underflows.
            e_out[row] *= np.exp(1)
            scale[row] += 1
        '''
        while (np.log(np.min(e_out[row])) - np.exp(scale[row])) < -10:
            e_out[row] *= np.exp(1)
            scale[row] += 1
            #print("normalize")
        '''

def load_observations(gll_file):  # return g0 g1
    gl = defaultdict(lambda: defaultdict(list))
    # with open("/mnt/diversity/jiaqi/hmm/hmm_extend_sim/sim/gll_cov_10_cont_0_deam_0/chrom_1.obs_gll_sorted.txt_n20k", 'r') as data:
    with open(gll_file, "r") as data:
        for line in data:
            (
                pos,
                Anc,
                _,
                _,
                _,
                _,
                _,
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
            window = int(pos) - int(pos) % 1000
            gl["g_0"][window].append(g_0)
            gl["g_1"][window].append(g_1)
    gl_0 = []
    gl_1 = []
    for window in list(gl["g_0"].keys()):
        gl_0.append(np.array(gl["g_0"][window]))
        gl_1.append(np.array(gl["g_1"][window]))
    return gl_0, gl_1
    # g_0     likelihood being no derived allele.
    # g_1     sum of the other 9 likelihoods being >= 1 derived allele.   (g_1 - g_0)


class HMMParam:
    def __init__(self, state_names, starting_probabilities, transitions, emissions):
        self.state_names = np.array(state_names)
        self.starting_probabilities = np.array(starting_probabilities)
        self.transitions = np.array(transitions)
        self.emissions = np.array(emissions)

    def __str__(self):
        out = f"> state_names = {self.state_names.tolist()}\n"
        out += f"> starting_probabilities = {np.matrix.round(self.starting_probabilities, 3).tolist()}\n"
        out += f"> transitions = {np.matrix.round(self.transitions, 3).tolist()}\n"
        out += f"> emissions = {np.matrix.round(self.emissions, 3).tolist()}"
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
        emissions=[0.0002, 0.00001896],
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
            sep="\t", file = log_file
        )

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
        round(loglikelihood, 4),
        print_starting_probabilities,
        print_emissions,
        print_transitions,
        sep="\t",
        file = log_file
    )


def approx_obs(obs):
    """approximation"""
    P0 = np.array([np.prod(i) for i in obs[0]])
    P1_sum = np.array([np.sum(o1 / o0) for o0, o1 in zip(obs[0], obs[1])])

    P1 = P0 * P1_sum / 1000  # fix multiple genotype stuff

    return np.stack((P0, P1), 1)


def linearize_obs(raw_obs):   # index all the snps [0,0,0..,1,1,...]
    g = (i for i, s in enumerate(raw_obs[0]) for j in s)
    SNP2BIN = np.fromiter(g, int)
    GLL = np.vstack((np.concatenate(raw_obs[0]), np.concatenate(raw_obs[1]))).T
    return GLL, SNP2BIN

def update_snp_prob(SNP, GLL, p):
    """
    calculate P(O_lr | G_lr, Z_l) = P(O_lr | G_lr) P(G_lr | Z_l)
    """
    update_geno_emissions(SNP, p)      # per site P(G_lr=g | Z_l)
    SNP *= GLL[:, np.newaxis, :]       # per site P(O_lr, G_lr=g | Z_l) = P(O_lr | G_lr=g) P(G_lr=g | Z_l)


def update_geno_emissions(SNP, p):
    """calculate P(G | Z)
    we have k sites, 2 GT, 2 states
    as k is large, we do things in memory

    this is total overkill, but can be used later to add complexity
    """
    SNP[:, 0, 0] = 1 - p[0]
    SNP[:, 1, 0] = 1 - p[1]
    SNP[:, 0, 1] = p[0]
    SNP[:, 1, 1] = p[1]

def update_emissions(E, SNP, GLL, p, SNP2BIN):   # sum_g P(O, G | Z) = P(O | Z)
    update_snp_prob(SNP, GLL, p)    # get SNP = P(O_lr, G_lr=g | Z_l) = P(O_lr | G_lr=g) P(G_lr=g | Z_l) = GLL * p_g.  numeric underflow safe
    E[:] = 1  # reset
    snp2bin(E, np.sum(SNP, 2), SNP2BIN)  # possible numeric underflow
    # np.sum(SNP,2): P(O_lr | Z_l)     (assume we go through all possible G_lr)
    # sum up all sites in window (product), so E_L = P(O_l | Z_l)

    
def update_emissions_scale(E, SNP, GLL, p, SNP2BIN, scale):
    update_snp_prob(SNP, GLL, p)    # get SNP = P(O_lr, G_lr=g | Z_l) = P(O_lr | G_lr=g) P(G_lr=g | Z_l) = GLL * p_g.  numeric underflow safe
    E[:] = 1
    snp2bin_scale(E, np.sum(SNP, 2), SNP2BIN, scale)  #also return the array of scaling factors


def update_emissions2(E, GLL, p, SNP2BIN):
    """simplified version that tries to not calculate all the latent states"""
    E[:] = 1  # reset
    snp2bin(E[:, 1], (GLL[:,0] * (1-p[1]) + GLL[:, 1] * p[1]), SNP2BIN)
    snp2bin(E[:, 0], (GLL[:,0] * (1-p[0]) + GLL[:, 1] * p[0]), SNP2BIN)

def update_post_geno(PG, SNP, Z, SNP2BIN):
    """from admixfrog
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
    
    PG /= np.sum(SNP, 2)[:, :, np.newaxis]    
    PG[np.isnan(PG)] = 0.0
    np.clip(PG, 0, 1, out=PG)

    assert np.all(PG >= 0)
    assert np.all(PG <= 1)
    assert np.allclose(np.sum(PG, (1, 2)), 1, atol=1e-6)

    return PG

def TrainModel(raw_obs, mutrates, input_pars, epsilon=1e-4, maxiterations=1000, log_file = None):
    pars = read_HMM_parameters_from_file(input_pars)
    n_states = len(pars.starting_probabilities)
    n_windows = len(raw_obs[0])
    n_gt = 2

    previous_ll = -np.Inf

    # the data and an indexing array
    GLL, SNP2BIN = linearize_obs(raw_obs)
    n_snps = len(GLL)

    # adapted from admixfrog
    # create arrays for posterior, emissions
    Z = np.zeros((n_windows, n_states))  # P(Z | O)
    E = np.ones((n_windows, n_states))  # P(O | Z)

    SNP = np.zeros(
        (n_snps, n_states, n_gt)
    )  # P(O, G| Z), scaled such that  the max is 1
    PG = np.zeros((n_snps, n_states, n_gt))  # P(G Z | O)
    S = np.zeros(n_windows)  # set scaling factor
    # Train parameters using Baum Welch algorithm
    for i in range(maxiterations):
        p = pars.emissions
        #updates both E and SNP
        update_emissions_scale(E, SNP, GLL, p, SNP2BIN, S)  
        # update E for per window(product of all sites)
        # update S, scaling factor for each window
        # get P(O_l|Z_l),   by go through all r sites and g genotypes
                
        # also update SNP  SNP = P(O_lr, G_lr=g | Z_l)

        fwd, scales = forward(E, pars.transitions, pars.starting_probabilities)
        bwd = backward(E, pars.transitions, scales)
        Z[:] = fwd * bwd

        update_post_geno(PG, SNP, Z, SNP2BIN) 
        '''
        get 
        P(G_lr = g, Z_l | O', theta)  see my(Jiaqi) letax note.
        '''
        assert np.allclose(np.sum(PG, (1, 2)), 1)
        # PG = P(Z|O) P(O, G | Z) / sum_g P(O, G=g | Z) for all sites
        
        #PG refers to P(O_lr, G_lr=g | Z_l)
        # Update emission
        new_p = np.sum(PG[:, :, 1] / np.sum(PG, (0,2)),0)

        # update transitions, from old hmmix
        new_trans = np.zeros((n_states, n_states))
        for s1 in range(n_states):
            for s2 in range(n_states):
                new_trans[s1, s2] = np.sum(
                    fwd[:-1, s1] * pars.transitions[s1, s2] * E[1:, s2] * bwd[1:, s2] / scales[1:]
                )

        new_trans /= new_trans.sum(axis=1)[:, np.newaxis]

        pars = HMMParam(
            pars.state_names,
            pars.starting_probabilities,
            new_trans,
            new_p,
        )
        new_ll = np.sum(np.log(scales)) -  np.sum(S)
        logoutput(pars, new_ll, i, log_file = log_file)

        if (new_ll - previous_ll < epsilon) & (new_ll - previous_ll > 0):  # epsilon ...
            break

        previous_ll = new_ll

    return pars, Z, E, PG, SNP, GLL, SNP2BIN

def TrainModel2(raw_obs, mutrates, input_pars, epsilon=1e-4, maxiterations=1000):
    pars = read_HMM_parameters_from_file(input_pars)
    n_states = len(pars.starting_probabilities)
    n_windows = len(raw_obs[0])
    n_gt = 2

    previous_ll = -np.Inf

    # the data and an indexing array
    GLL, SNP2BIN = linearize_obs(raw_obs)
    n_snps = len(GLL)

    # adapted from admixfrog
    # create arrays for posterior, emissions
    Z = np.zeros((n_windows, n_states))  # P(Z | O)
    E = np.ones((n_windows, n_states))  # P(O | Z)

    PG = np.zeros((n_snps, n_states, n_gt))  # P(G Z | O)

    # Train parameters using Baum Welch algorithm
    for i in range(maxiterations):
        p = pars.emissions

        update_emissions2(E, GLL, p, SNP2BIN)

        fwd, scales = forward(E, pars.transitions, pars.starting_probabilities)
        bwd = backward(E, pars.transitions, scales)
        Z[:] = fwd * bwd

        update_post_geno(PG, Z, SNP2BIN)
        # Update emission
        new_p = np.sum(PG[:, :, 1] / np.sum(PG[:, 0]), 0)

        # update transitions
        new_trans = np.zeros((n_states, n_states))
        for s1 in range(n_states):
            for s2 in range(n_states):
                new_trans[s1, s2] = np.sum(
                    fwd[:-1, s1] * pars.transitions[s1, s2] * E[1:, s2] * bwd[1:, s2] / scales[1:]
                )
        #np.clip(new_trans, )
        new_trans /= new_trans.sum(axis=1)[:, np.newaxis]

        pars = HMMParam(
            pars.state_names,
            pars.starting_probabilities,
            new_trans,
            new_p,
        )
        new_ll = np.sum(np.log(scales))
        logoutput(pars, new_ll, i)

        if (new_ll - previous_ll < epsilon) & (new_ll - previous_ll > 0):  # epsilon ...
            break

        previous_ll = new_ll

    return pars, Z, E, PG, GLL, SNP2BIN

#gl_0_all, gl_1_all = load_observations("/mnt/diversity/jiaqi/hmm/hmm_extend_sim/sim/ind_4_gll_cov_10_cont_0.02_deam_0.02/chrom_1.obs_gll_sorted_3m.txt")
#gl_0_all, gl_1_all = load_observations("/mnt/diversity/jiaqi/hmm/hmm_extend_sim/sim/ind_4_gll_cov_10_cont_0.01_deam_0.02/chrom_1.obs_gll_sorted_20n_tail_recal_map35_100.txt")
'''
observations = [gl_0_all, gl_1_all]
print("load done")

pars, Z, E, PG, SNP, GLL, SNP2BIN = TrainModel(
   raw_obs=observations,
   mutrates=None,
   input_pars=None,
   epsilon=5 * 1e-5,
   maxiterations=1500,
)
print("train model done")
np.savetxt('/home/jiaqi_yang/hmm/test_cont_0.01_deam_0.02_tail.csv', Z , delimiter=',')
'''
def run_test(input_file, post_file):
    gl_0_all, gl_1_all = load_observations(input_file)
    observations = [gl_0_all, gl_1_all]
    pars, Z, E, PG, SNP, GLL, SNP2BIN = TrainModel(
        raw_obs=observations,
        mutrates=None,
        input_pars=None,
        epsilon=0.5 * 1e-5,
        maxiterations=1500
    )
    print("train model done")
    np.savetxt(post_file, Z , delimiter=',')
