#!/usr/bin/env python3
import argparse
import numpy as np
from helper_f import load_observations, load_observations_gt, get_obs_gll, get_obs_gt
from hmm import TrainModel, write_HMM_to_file, read_HMM_parameters_from_file, decode_from_params
import pandas as pd
from call import Call
from hmm_gt import TrainModel_GT

def main():
    parser = argparse.ArgumentParser(add_help = True)

    subparser = parser.add_subparsers(dest = 'mode')
    getobs_subparser = subparser.add_parser('get_obs', help='convert either genotype or gll file to observation file')  # converting snpAD results to observation, per chr. might need to concatenate later.
    getobs_subparser.add_argument("-type", metavar='', 
                                 help="[required] data type, either 'gll' by snpAD, or per individual 'gt' by snpad/genotypecaller", type=str, required = True)
    getobs_subparser.add_argument("-vcf", metavar='', help="[required] vcf file", type=str, default= None, required = True)
    getobs_subparser.add_argument("-ancestral", metavar='',
                                 help="[required] directory pointing to ancestral fasta file", type=str, default=None, required = True)
    getobs_subparser.add_argument("-out", metavar='',
                                 help="output file for observations, used later for this HMM", type=str, default= None, required= True)
    getobs_subparser.add_argument("-mask", metavar='', help = "mask file for vcf when generating observations from vcf (vcf might not be filtered)", type = str, default = None)
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
    train_subparser.add_argument("-rec", action='store_true', help = "binning genome by genetic length, default False. rec map in bed format required",  default = False)
    train_subparser.add_argument("-rec_map", metavar='',
                                 help="recombination map in bed format (interpolated)", type=str, default=None)
    train_subparser.add_argument("-rec_window_trim", action='store_true', help = "trim the last the end first windows by genetic length",  default = False)
    train_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = None)
    train_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = None)
    train_subparser.add_argument("-not_est_transition",  action='store_true', help = "estimate transition parameter or not", default= False)
    train_subparser.add_argument("-transition_1", metavar='',  help = "transition_param_1", default= None, type=float)
    train_subparser.add_argument("-transition_2", metavar='',  help = "transition_param_2", default= None, type=float)
    
    decode_subparser = subparser.add_parser('decode', help='Decode posterior from params trained')
    decode_subparser.add_argument("-filter_depth", action='store_true', help = "whetther set a uniform filter on coverage", default = False)
    decode_subparser.add_argument("-rec", action='store_true', help = "binning genome by genetic length, default False. rec map in bed format required",  default = False)
    decode_subparser.add_argument("-rec_window_trim", action='store_true', help = "trim the last the end first windows by genetic length",  default = False)

    decode_subparser.add_argument("-rec_map", metavar='',
                                 help="recombination map in bed format (interpolated)", type=str, default=None)
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
    decode_subparser.add_argument("-log_file", metavar='',
                                 help="output log info of forward-backward decoding", default = None)
    decode_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = None)
    decode_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = None)

    train_gt_subparser = subparser.add_parser('gt_mode', help='Train HMM')
    train_gt_subparser.add_argument("-data_type", metavar='',help="[required] data type: modern genotypes or ancient genotyes", type=str, default = "ancient")

    train_gt_subparser.add_argument("-count_file", metavar='', 
                                 help="[required] count of derived alleles", type=str, required = True)
    train_gt_subparser.add_argument("-max_variants_per_window", metavar='',
                                 help="maximum number of variants per window allowed", type=int, default = 20)
    train_gt_subparser.add_argument("-filter_depth", action='store_true', help = "whetther set a uniform filter on coverage",  default = False)
    train_gt_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = 0)
    train_gt_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = 200)
    train_gt_subparser.add_argument("-mask_file", metavar='',
                                 help="file with integrated mask", type=str, default=None)
    train_gt_subparser.add_argument("-mut_file", metavar='',
                                 help="file with mutation rates (default is mutation rate is uniform)", type=str, default=None)
    train_gt_subparser.add_argument("-param", metavar='',
                                 help="markov parameters file (default is human/neanderthal like parameters)", type=str, default= None)
    train_gt_subparser.add_argument("-out", metavar='',
                                 help="outputfile (default is a file named trained.json)", default = 'trained.json')
    train_gt_subparser.add_argument("-window_size", metavar='',
                                 help="size of bins (default is 1000 bp)", type=int, default = 1000)
    train_gt_subparser.add_argument("-posterior", metavar='',
                                 help="output of posterior from fwd-bwd", default = "Posterior.txt")
    train_gt_subparser.add_argument("-log_file", metavar='',
                                 help="output log info of baum-welch", default = "EM_iterration.log")
    train_gt_subparser.add_argument("-iteration", metavar='', help = "max iteration for EM", type = int, default = 1000)
    call_subparser = subparser.add_parser('call', help='call fragments based on posterior')
    call_subparser.add_argument("-posterior", metavar='',
                                 help="output of posterior from fwd-bwd")
    call_subparser.add_argument("-param", metavar='',
                                 help="markov parameters file (default is human/neanderthal like parameters)", type=str, default= None)
    call_subparser.add_argument("-called", metavar='',
                                    help="output of called fragments", default = "called.txt")
    args = parser.parse_args()
    if args.mode == 'get_obs':
        print(f"producing observation file for HMM from {args.vcf}, to observation outfile {args.out}, with mask file: {args.mask}")
        if not hasattr(args, 'ancestral'):
            parser.print_help()
            return
        print(f"ancestral fasta file is {args.ancestral}")
        if not hasattr(args, 'out'):
            parser.print_help()
            return
        if args.type == "gll":
            print("getting observations from gll file")
            get_obs_gll(args.vcf, args.ancestral, args.out, args.mask)
        elif args.type == "gt":
            print("getting observations from genotype file")
            get_obs_gt(args.vcf, args.ancestral, args.out, args.mask)
        else:
            print("please specify the data type, either 'gll' or 'gt'")
            return
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
        with open(args.log_file, 'w') as log_file:
            log_file.write(f"fixed transition parameters: {args.not_est_transition}\n")
            if args.rec is True:
                log_file.write(f"using rec map: {args.rec_map}\n")
            else:
                log_file.write(f"window size: {args.window_size}\n")
            log_file.write(f"filter depth: {args.filter_depth}\n")
            log_file.write(f"max depth: {args.maximum_dep}\n")
            log_file.write(f"min depth: {args.minimum_dep}\n")
            log_file.write(f"mut rate file: {args.mut_file}\n")
        observation, chrs, windows, m_rates = load_observations(args.gll_file, args.window_size, args.filter_depth, args.maximum_dep, args.minimum_dep,
        args.rec, args.rec_map, args.mut_file)
        print('-' * 40)
        print('> Output is',args.out) 
        print('> Window size is',args.window_size, 'bp') 
        print('-' * 40)
        hmm_parameters = TrainModel(raw_obs = observation,
                                 chr_index = chrs,
                                 w= windows,
                                 post_file= args.posterior,
                                 log_file = args.log_file,
                                 pars = hmm_parameters,
                                 m_rates = m_rates,
                                 maxiterations=args.iteration,
                                 not_est_trans = args.not_est_transition)
        write_HMM_to_file(hmm_parameters, args.out)
    if args.mode == "decode":
        print("decoding")
        if not hasattr(args, 'gll_file'):
            parser.print_help()
            return
        if not args.log_file is None:
            with open(args.log_file, 'a') as log_file:
                print("decoding", file = log_file)
                print(f"mut rate file: {args.mut_file}", file = log_file)
                print(f"posterior file: {args.posterior}", file = log_file)
                log_file.write(f"window size: {args.window_size}\n")
                log_file.write(f"filter depth: {args.filter_depth}\n")
                if args.filter_depth:
                    log_file.write(f"max depth: {args.maximum_dep}\n")
                    log_file.write(f"min depth: {args.minimum_dep}\n")
                if args.rec:
                    log_file.write(f"using rec map: {args.rec_map}\n")
        hmm_parameters = read_HMM_parameters_from_file(args.param)
        print("decoding")
        print(f"mut rate file: {args.mut_file}")
        print(f"posterior file: {args.posterior}")
        print(f"window size: {args.window_size}\n")
        print(f"filter depth: {args.filter_depth}\n")
        if args.filter_depth:
            print(f"max depth: {args.maximum_dep}\n")
            print(f"min depth: {args.minimum_dep}\n")
        if args.rec:
            print(f"using rec map: {args.rec_map}")
        observation, chrs, windows, m_rates = load_observations(args.gll_file, args.window_size, args.filter_depth, args.maximum_dep, args.minimum_dep,
        args.rec, args.rec_map, args.mut_file)
        print('-' * 40)
        print(hmm_parameters)
        print('> Window size is',args.window_size, 'bp') 
        print('-' * 40)
        decode_from_params(raw_obs = observation,
                        chr_index = chrs,
                        w= windows,
                        post_file= args.posterior,
                        pars = hmm_parameters,
                        m_rates = m_rates)
    if args.mode == 'gt_mode':   # same as Larits. But starts with the first callable position.
        if not hasattr(args, 'count_file'):
            parser.print_help()
            return

        print("gt mode")
        hmm_parameters = read_HMM_parameters_from_file(args.param)
        print(args.param)
        #print(f"maximum number of variants per window allowed is {args.max_variants_per_window}")
        if args.data_type == "modern":
            print(f"loading snp-only input data {args.count_file} with modern mask {args.mask_file}, with window size {args.window_size}")
            print(f"maximum number of variants per window allowed is {args.max_variants_per_window}")
            chr_index, weights, obs, call_index, w = load_observations_gt(args.count_file, args.mask_file, args.window_size, args.max_variants_per_window, "modern", args.filter_depth, args.minimum_dep, args.maximum_dep)
        if args.data_type == "ancient":
            print(f"loading all-pos input data {args.count_file} with ancient per-individual mask {args.mask_file}, with window size {args.window_size}")
            print(f"maximum number of variants per window allowed is {args.max_variants_per_window}")
            chr_index, weights, obs, call_index, w = load_observations_gt(args.count_file, args.mask_file, args.window_size, args.max_variants_per_window, "ancient", args.filter_depth, args.minimum_dep, args.maximum_dep)
        #check for zero:

        print(f"finished loading {args.count_file}")
        print(f"finished loading {args.mask_file}")
        print('-' * 40)
        print(hmm_parameters)
        print('> Output is',args.out) 
        print('> Window size is',args.window_size, 'bp') 
        print('-' * 40)
        with open(args.log_file, 'w') as log_file:
            log_file.write(f"{hmm_parameters}\n")
            log_file.write(f"maximum number of variants per window allowed is {args.max_variants_per_window}\n")
            log_file.write(f"output file: {args.out}\n")
            log_file.write(f"window size: {args.window_size}\n")
            log_file.write(f"mut rate file: {args.mut_file}\n")
            log_file.write(f"posteiors file: {args.posterior}\n")
            if args.filter_depth:
                log_file.write(f"filter depth: {args.filter_depth}\n")
                log_file.write(f"max depth: {args.maximum_dep}\n")
                log_file.write(f"min depth: {args.minimum_dep}\n")
            else:
                log_file.write("no filter on depth\n")
        hmm_parameters = TrainModel_GT(obs = obs,
                                weights = weights,
                                chr_index = chr_index,
                                w= w,
                                window_size = args.window_size,
                                call_index= call_index,
                                post_file= args.posterior,
                                log_file = args.log_file,
                                hmm_parameters = hmm_parameters,
                                m_rates_file = args.mut_file,
                                maxiterations=args.iteration)
        write_HMM_to_file(hmm_parameters, args.out)     

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
