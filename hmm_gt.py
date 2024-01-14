from helper_f import load_observations
from collections import defaultdict
import pandas as pd
import numpy as np
from numba import njit
import json
from math import exp, ceil,factorial
from hmm import write_post_to_file
class HMMParam:
    def __init__(self, state_names, starting_probabilities, transitions, emissions): 
        self.state_names = np.array(state_names)
        self.starting_probabilities = np.array(starting_probabilities)
        self.transitions = np.array(transitions)
        self.emissions = np.array(emissions)


    def __str__(self):
        out = f'> state_names = {self.state_names.tolist()}\n'
        out += f'> starting_probabilities = {np.matrix.round(self.starting_probabilities, 3).tolist()}\n'
        out += f'> transitions = {np.matrix.round(self.transitions, 3).tolist()}\n'
        out += f'> emissions = {np.matrix.round(self.emissions, 3).tolist()}'
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state_names}, {self.starting_probabilities}, {self.transitions}, {self.emissions})'
        
# Read HMM parameters from a json file
def read_HMM_parameters_from_file(filename):

    if filename is None:
        return get_default_HMM_parameters()

    with open(filename) as json_file:
        data = json.load(json_file)

    return HMMParam(state_names = data['state_names'], 
                    starting_probabilities = data['starting_probabilities'], 
                    transitions = data['transitions'], 
                    emissions = data['emissions'])

# Set default parameters
def get_default_HMM_parameters():
    return HMMParam(state_names = ['Human', 'Archaic'], 
                    starting_probabilities = [0.98, 0.02], 
                    transitions = [[0.9999,0.0001],[0.02,0.98]], 
                    emissions = [0.04, 0.4])

# Save HMMParam to a json file
def write_HMM_to_file(hmmparam, outfile):
    data = {key: value.tolist() for key, value in vars(hmmparam).items()}
    json_string = json.dumps(data, indent = 2) 
    with open(outfile, 'w') as out:
        out.write(json_string)


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


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# HMM functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


def Emission_probs_poisson(emissions, observations, weights):
    n = len(observations)
    n_states = len(emissions)
    
    # observations values
    fractorials = np.zeros(n)
    for i, obs in enumerate(observations):
        fractorials[i] = factorial(int(obs))

    probabilities = np.zeros( (n, n_states) ) 
    for state in range(n_states): 
        probabilities[:,state] = (np.exp( - emissions[state] * weights) *  ((emissions[state] * weights )**observations )) / fractorials

    return probabilities

@njit
def fwd_step(alpha_prev, E, trans_mat):
    alpha_new = (alpha_prev @ trans_mat) * E
    n = np.sum(alpha_new)
    return alpha_new / n, n

@njit
def forward(probabilities, transitions, init_start):

    n = len(probabilities)
    forwards_in = np.zeros( (n, len(init_start)) ) 
    scale_param = np.ones(n)

    for t in range(n):
        if t == 0:
            forwards_in[t,:] = init_start  * probabilities[t,:]
            scale_param[t] = np.sum( forwards_in[t,:])
            forwards_in[t,:] = forwards_in[t,:] / scale_param[t]
        else:
            forwards_in[t,:], scale_param[t] =  fwd_step(forwards_in[t-1,:], probabilities[t,:], transitions) 

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
        beta[i - 1,:] = bwd_step(beta[i,:], emissions[i,:], transitions, scales[i])
    return beta


def TrainModel_GT(hmm_parameters, weights, obs, chr_index, call_index, post_file, log_file, rec, w, maxiterations = 1000, epsilon = 0.0001, window_size = 1000,):  # w window index
    """
    Trains the model once, using the forward-backward algorithm. 
    chr_index, weights, obs, call_index, w
    """
    n_chr = len(chr_index)
    n_windows = np.ones(n_chr)
    n_states = len(hmm_parameters.starting_probabilities)
    Z = []
    E = []
    for chr in range(n_chr):
        n_windows[chr] = w[chr][-1] - w[chr][0] + 1
        n_windows_ = round(n_windows[chr])
        Z_ = np.zeros((n_windows_, n_states))  # P(Z | O)
        E_ = np.ones((n_windows_, n_states))  # P(O | Z)
        Z.append(Z_)
    previous_ll = -np.inf
    for iteration in range(maxiterations):
        new_emissions_matrix = np.zeros((n_states))
        top = np.zeros(2)
        bot = np.zeros(2)
        new_trans_ = np.zeros((n_chr, n_states, n_states))
        new_ll = 0
        normalize = []
        for chr in range(n_chr):
            emissions = Emission_probs_poisson(hmm_parameters.emissions, obs[chr], weights[chr])
            forward_probs, scales = forward(emissions, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
            backward_probs = backward(emissions, hmm_parameters.transitions, scales)
            new_ll += np.sum(np.log(scales))

            # Update emission, use only called_snpns
            top[0] += np.sum(forward_probs[[call_index[chr]], 0] * backward_probs[[call_index[chr]], 0] * obs[chr][call_index[chr]])
            bot[0] += np.sum(forward_probs[[call_index[chr]], 0] * backward_probs[[call_index[chr]], 0] * (weights[chr][call_index[chr]]) )
            
            top[1] += np.sum(forward_probs[[call_index[chr]], 1] * backward_probs[[call_index[chr]], 1] * obs[chr][call_index[chr]])
            bot[1] += np.sum(forward_probs[[call_index[chr]], 1] * backward_probs[[call_index[chr]], 1] * (weights[chr][call_index[chr]]) )
            # Update starting probs
            Z[chr][:] = forward_probs * backward_probs 
            normalize.append(np.sum(Z[chr], axis = 0))      # sum over posterior across chrs
            # Update Transition probs 
            for state1 in range(n_states):
                for state2 in range(n_states):
                    new_trans_[chr][state1, state2] = np.sum( forward_probs[:-1,state1]  * hmm_parameters.transitions[state1, state2] * emissions[1:,state2] * backward_probs[1:,state2] / scales[1:] )
        new_emissions_matrix = top/bot
        new_trans = np.sum(new_trans_, 0)
        new_trans /= new_trans.sum(axis=1)[:, np.newaxis] #Ben/Laurits
        normalize = np.sum(normalize, axis = 0)
        new_starting_probabilities = normalize/np.sum(normalize, axis=0) 
        hmm_parameters = HMMParam(
            hmm_parameters.state_names,
            new_starting_probabilities,
            new_trans,
            new_emissions_matrix
        )
        logoutput(hmm_parameters, new_ll, iteration, log_file)
        if (new_ll - previous_ll < epsilon):
            if not rec:
                #not taking into account recombination rates    
                break
            else:
                """
                new_trans[0] = 
                """
        if rec:
            #scale transition paramters to account for local recombination rates
            break
        previous_ll = new_ll
    
    write_post_to_file(Z, chr_index, w, post_file)
    return HMMParam(hmm_parameters.state_names,new_starting_probabilities, new_trans, new_emissions_matrix)


