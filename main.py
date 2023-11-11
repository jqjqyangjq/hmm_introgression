#!/usr/bin/env python3
import argparse
import numpy as np
from helper_f import load_observations
from hmm import TrainModel, write_HMM_to_file, read_HMM_parameters_from_file, decode_from_params
import pandas as pd
from call import Call


def main():
    parser = argparse.ArgumentParser(add_help = True)

    subparser = parser.add_subparsers(dest = 'mode')

    # Train model
    train_subparser = subparser.add_parser('train', help='Train HMM')
    train_subparser.add_argument("-gll_file", metavar='', 
                                 help="[required] gll file taken deam into account", type=str, required = True)
    train_subparser.add_argument("-mut_file", metavar='',
                                 help="file with mutation rates (default is mutation rate is uniform)", type=str, default=None)
    train_subparser.add_argument("-param", metavar='',
                                 help="markov parameters file (default is human/neanderthal like parameters)", type=str, default= None)
    train_subparser.add_argument("-out", metavar='',
                                 help="outputfile (default is a file named trained.json)", default = 'trained.json')
    train_subparser.add_argument("-window_size", metavar='',
                                 help="size of bins (default is 1000 bp)", type=int, default = 1000)
    train_subparser.add_argument("-posterior", metavar='',
                                 help="output of posterior from fwd-bwd", default = "Posterior.txt")
    train_subparser.add_argument("-log_file", metavar='',
                                 help="output log info of baum-welch", default = "EM_iterration.log")
    train_subparser.add_argument("-iteration", metavar='', help = "max iteration for EM", type = int, default = 1000)
    train_subparser.add_argument("-filter_depth", action='store_true', help = "whetther set a uniform filter on coverage",  default = False)
    train_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = None)
    train_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = None)
    train_subparser.add_argument("-not_est_transition",  action='store_true', help = "estimate transition parameter or not", default= False)
    train_subparser.add_argument("-transition_1", metavar='',  help = "transition_param_1", default= None, type=float)
    train_subparser.add_argument("-transition_2", metavar='',  help = "transition_param_2", default= None, type=float)

    decode_subparser = subparser.add_parser('decode', help='Decode posterior from params trained')
    decode_subparser.add_argument("-filter_depth", action='store_true', help = "whetther set a uniform filter on coverage", default = False)
    decode_subparser.add_argument("-gll_file", metavar='', 
                                 help="[required] gll file taken deam into account", type=str, required = True)
    decode_subparser.add_argument("-mut_file", metavar='',
                                 help="file with mutation rates (default is mutation rate is uniform)", type=str, default=None)
    decode_subparser.add_argument("-param", metavar='',
                                 help="markov parameters file (default is human/neanderthal like parameters)", type=str, default= None)
    decode_subparser.add_argument("-window_size", metavar='',
                                 help="size of bins (default is 1000 bp)", type=int, default = 1000)
    decode_subparser.add_argument("-posterior", metavar='',
                                 help="output of posterior from fwd-bwd", default = "Posterior.txt")
    decode_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = None)
    decode_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = None)
                                
    call_subparser = subparser.add_parser('call', help='call fragments based on posterior')
    call_subparser.add_argument("-posterior", metavar='',
                                 help="output of posterior from fwd-bwd")
    call_subparser.add_argument("-param", metavar='',
                                 help="markov parameters file (default is human/neanderthal like parameters)", type=str, default= None)
    call_subparser.add_argument("-called", metavar='',
                                    help="output of called fragments", default = "called.txt")
    args = parser.parse_args()
    if args.mode == 'train':
        if not hasattr(args, 'gll_file'):
            parser.print_help()
            return
        hmm_parameters = read_HMM_parameters_from_file(args.param)
        if args.transition_1 is not None:
            hmm_parameters[0] = [1- args.transition_1, args.transition_1]
        if args.transition_2 is not None:
            hmm_parameters[1] = [args.transition_2, 1 - args.transition_2]
        print(hmm_parameters)
        print(f"filter depth: {args.filter_depth}")
        print(f"fixed transition parameters: {args.not_est_transition}")
        print(args.param)
        #obs, _, _, _, mutrates, weights = Load_observations_weights_mutrates(args.obs, args.weights, args.mutrates, args.window_size, args.haploid)
        observation, chrs, windows, obs_count = load_observations(args.gll_file, args.window_size, args.filter_depth, args.maximum_dep, args.minimum_dep)
        print('-' * 40)
        print('> Output is',args.out) 
        print('> Window size is',args.window_size, 'bp') 
        print('-' * 40)
        hmm_parameters = TrainModel(raw_obs = observation,
                                 chr_index = chrs,
                                 w= windows,
                                 obs_count = obs_count,
                                 window_size = args.window_size,
                                 post_file= args.posterior,
                                 log_file = args.log_file,
                                 pars = hmm_parameters,
                                 m_rates_file = args.mut_file,
                                 maxiterations=args.iteration,
                                 not_est_trans = args.not_est_transition)
        write_HMM_to_file(hmm_parameters, args.out)
    if args.mode == "decode":
        print("decoding")
        if not hasattr(args, 'gll_file'):
            parser.print_help()
            return
        hmm_parameters = read_HMM_parameters_from_file(args.param)
        observation, chrs, windows, obs_count = load_observations(args.gll_file, args.window_size, args.filter_depth, args.maximum_dep, args.minimum_dep)
        print('-' * 40)
        print(hmm_parameters)
        print('> Window size is',args.window_size, 'bp') 
        print('-' * 40)
        decode_from_params(raw_obs = observation,
                        chr_index = chrs,
                        w= windows,
                        obs_count = obs_count,
                        window_size = args.window_size,
                        post_file= args.posterior,
                        pars = hmm_parameters,
                        m_rates_file = args.mut_file)
    if args.mode == "call":
        print("calling fragments using posterior")
        if not hasattr(args, 'posterior'):
            print("provide decoded posterior file for calling fragments")
            return
        if not hasattr(args, 'param'):
            print("use the default paprameters")
            hmm_parameters = read_HMM_parameters_from_file(args.param)
        hmm_parameters = read_HMM_parameters_from_file(args.param)
        Call(args.posterior, hmm_parameters, args.called)
        print("call finished")
if __name__ == "__main__":
    main()
