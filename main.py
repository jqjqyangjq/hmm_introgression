import argparse
import numpy as np
from helper_f import load_observations
from hmm import TrainModel, write_HMM_to_file, read_HMM_parameters_from_file
import pandas as pd



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
    args = parser.parse_args()
    
    
    if args.mode == 'train':
        if not hasattr(args, 'gll_file'):
            parser.print_help()
            return
        hmm_parameters = read_HMM_parameters_from_file(args.param)
        print(args.param)
        #obs, _, _, _, mutrates, weights = Load_observations_weights_mutrates(args.obs, args.weights, args.mutrates, args.window_size, args.haploid)
        observation, chrs, windows, obs_count = load_observations(args.gll_file, args.window_size)
        print('-' * 40)
        print(hmm_parameters)
        print('> Output is',args.out) 
        print('> Window size is',args.window_size, 'bp') 
        print('-' * 40)
        write_HMM_to_file(hmm_parameters, args.out)
        hmm_parameters = TrainModel(raw_obs = observation,
                                 chr_index = chrs,
                                 w= windows,
                                 obs_count = obs_count,
                                 window_size = args.window_size,
                                 post_file= args.posterior,
                                 log_file = args.log_file,
                                 pars = hmm_parameters,
                                 m_rates_file = args.mut_file)
        write_HMM_to_file(hmm_parameters, args.out)

if __name__ == "__main__":
    main()
    
'''
observation, chrs, windows, obs_count = load_observations("/mnt/diversity/jiaqi/hmm/hmm_extend_sim/sim/ind_4_gll_cov_10_cont_0.01_deam_0.01/m_chrs.txt",
                                                          1000)
hmm_parameters = read_HMM_parameters_from_file(None)
hmm_parameters = TrainModel(raw_obs = observation,
                                 chr_index = chrs,
                                 w= windows,
                                 obs_count = obs_count,
                                 window_size = 1000,
                                 post_file= "/home/jiaqi_yang/hmm/wrap_Posterior.txt",
                                 log_file = "/home/jiaqi_yang/hmm/wrap_EM_iterration.log",
                                 pars = hmm_parameters,
                                 m_rates_file = "/mnt/diversity/jiaqi/hmm/hmm_extend_sim/sim/ind_4_gll_cov_10_cont_0.01_deam_0.01/m_chrs_mut.txt")
'''