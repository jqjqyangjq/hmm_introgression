#!/usr/bin/env python3
import argparse
import numpy as np
from helper_f import load_observations, load_observations_gt, get_obs_gll, get_obs_gt
from anno import anno
from hmm import TrainModel, write_HMM_to_file, read_HMM_parameters_from_file, decode_from_params
import pandas as pd
from call import Call, get_runs
from hmm_gt import TrainModel_GT
import time
def main():
    parser = argparse.ArgumentParser(add_help = True)

    subparser = parser.add_subparsers(dest = 'mode')
    getobs_subparser = subparser.add_parser('get_obs', help='convert either genotype or gll file to observation file')  # converting snpAD results to observation, per chr. might need to concatenate later.
    getobs_subparser.add_argument("-type", metavar='', 
                                 help="[required] data type, either 'gll' by snpAD, or per individual 'gt' by snpad/genotypecaller", type=str, required = True)
    getobs_subparser.add_argument("-vcf", metavar='', help="[required] vcf file", type=str, default= None, required = True)
    getobs_subparser.add_argument("-vcf_type", metavar='', help="vcf type. if ancient(default), record depth. otherwise assume a filtered vcf, an so all depth is 0 (will not be used)", type=str, default= "ancient")
    getobs_subparser.add_argument("-ancestral", metavar='',
                                 help="[required] directory pointing to ancestral fasta file", type=str, default=None, required = True)
    getobs_subparser.add_argument("-out", metavar='',
                                 help="output file for observations, used later for this HMM", type=str, default= None, required= True)
    getobs_subparser.add_argument("-mask", metavar='', help = "mask file for vcf when generating observations from vcf (vcf might not be filtered)", type = str, default = None)
    getobs_subparser.add_argument("-ind", metavar='', help = "ind to be used for vcf, when vcf has multiple individuals", type = str, default = None)
    # Train model
    train_subparser = subparser.add_parser('train', help='Train HMM')
    train_subparser.add_argument("-gll_file", metavar='', 
                                 help="[required] gll file taken deam into account", type=str, required = True)
    train_subparser.add_argument("-mut_file", metavar='',
                                 help="file with mutation rates (default is mutation rate is uniform)", type=str, default=None)
    train_subparser.add_argument("-param", metavar='',
                                 help="markov parameters file (default is human/neanderthal like parameters)", type=str, default= None)
    train_subparser.add_argument("-out", metavar='',
                                 help="prefix of output files", default = 'train')
    train_subparser.add_argument("-window_size", metavar='',
                                 help="size of bins (default is 1000 bp)", type=int, default = 1000)
    train_subparser.add_argument("-iteration", metavar='', help = "max iteration for EM", type = int, default = 2000)
    train_subparser.add_argument("-filter_depth", action='store_true', help = "whetther set a uniform filter on coverage",  default = False)
    train_subparser.add_argument("-rec", action='store_true', help = "binning genome by genetic length, default False. rec map in bed format required",  default = False)
    train_subparser.add_argument("-rec_map", metavar='',
                                 help="recombination map in bed format (interpolated)", type=str, default=None)
    train_subparser.add_argument("-not_est_starting_prob",  action='store_true', help = "estimate starting probabilities or not", default= False)
    train_subparser.add_argument("-starting_prob",  metavar='', help = "starting probabilities for archaic state", default= None, type = float)
    train_subparser.add_argument("-rec_window_trim", action='store_true', help = "trim the last the end first windows by genetic length",  default = False)
    train_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = 150)
    train_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = 0)
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
    decode_subparser.add_argument("-out", metavar='',
                                 help="prefix of output files", default = "decode")
    decode_subparser.add_argument("-penalty", metavar='',
                                 help="penalty score used for calling fragments", default = 0.2, type = float)
    decode_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = None)
    decode_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = None)

    train_gt_subparser = subparser.add_parser('gt_mode', help='Train HMM')
    train_gt_subparser.add_argument("-data_type", metavar='',help="[required] data type: modern genotypes or ancient genotyes", type=str, default = "ancient")

    train_gt_subparser.add_argument("-not_est_transition",  action='store_true', help = "estimate transition parameter or not", default= False)
    train_gt_subparser.add_argument("-transition_1", metavar='',  help = "transition_param_1", default= None, type=float)
    train_gt_subparser.add_argument("-transition_2", metavar='',  help = "transition_param_2", default= None, type=float)
    train_gt_subparser.add_argument("-not_est_starting_prob",  action='store_true', help = "estimate starting probabilities or not", default= False)
    train_gt_subparser.add_argument("-starting_prob",  metavar='', help = "starting probabilities for archaic state", default= None, type = float)
    train_gt_subparser.add_argument("-count_file", metavar='', 
                                 help="[required] count of derived alleles", type=str, required = True)
    train_gt_subparser.add_argument("-max_variants_per_window", metavar='',
                                 help="maximum number of variants per window allowed", type=int, default = 50)
    train_gt_subparser.add_argument("-filter_depth", action='store_true', help = "whetther set a uniform filter on coverage",  default = False)
    train_gt_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = 0)
    train_gt_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = 200)
    train_gt_subparser.add_argument("-mask_file", metavar='',
                                 help="file with integrated mask", type=str, default=None)
    train_gt_subparser.add_argument("-rec_map", metavar='',
                                 help="rec_map, indexed by windows", type=str, default=None)
    train_gt_subparser.add_argument("-mut_file", metavar='',
                                 help="file with mutation rates (default is mutation rate is uniform)", type=str, default=None)
    train_gt_subparser.add_argument("-param", metavar='',
                                 help="markov parameters file (default is human/neanderthal like parameters)", type=str, default= None)
    train_gt_subparser.add_argument("-out", metavar='',
                                 help="outputfile (default is a file named trained.json)", default = 'trained')
    train_gt_subparser.add_argument("-window_size", metavar='',
                                 help="size of bins (default is 1000 bp)", type=int, default = 1000)
    train_gt_subparser.add_argument("-iteration", metavar='', help = "max iteration for EM", type = int, default = 1000)
    train_gt_subparser.add_argument("-penalty", metavar='', help = "penalty score used for calling fragments", type = float, default = 0.2)
    call_subparser = subparser.add_parser('call', help='call fragments based on posterior')
    call_subparser.add_argument("-posterior", metavar='',
                                 help="output of posterior from fwd-bwd")
    #call_subparser.add_argument("-param", metavar='',
    #                             help="markov parameters file (default is human/neanderthal like parameters)", type=str, default= None)
    #call_subparser.add_argument("-called", metavar='',
    #                                help="output of called fragments", default = "called.txt")
    #call_subparser.add_argument("-window_size", metavar='', help="window_size", default= 1000, type = int)
    call_subparser.add_argument("-penalty", metavar='', help='penalty score used', default=0.2)
    anno_subparser = subparser.add_parser('anno', help='call fragments based on posterior')
    anno_subparser.add_argument("-called", metavar='',
                                 help="called file", default=None, type = str)
    anno_subparser.add_argument("-anno", metavar='',
                                 help="annotated file", type=str, default= None)
    anno_subparser.add_argument("-sample", metavar='',
                                    help="archaic individuals used for annotation", default = "")
    anno_subparser.add_argument("-vcf", metavar='',
                                    help="vcf used for comparison", default = "")
    anno_subparser.add_argument("-map_file", metavar='', help = "map file used for annotation. one map with all chrs", default = None)
    anno_subparser.add_argument("-map", metavar='', help = "map used for annotation", default = "AA_Map")
    anno_subparser.add_argument("-window_size", metavar='', help = "window size", type = int, default = 1000)
    anno_subparser.add_argument("-count_file", metavar='', help = "observation", type = str, default = None)
    anno_subparser.add_argument("-filter_depth", action='store_true', help = "whetther set a uniform filter on coverage",  default = False)
    anno_subparser.add_argument("-maximum_dep", metavar='', help = "max depth for per position", type = int, default = 0)
    anno_subparser.add_argument("-minimum_dep", metavar='', help = "min depth for per position", type = int, default = 200)
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
            print(f"getting observations from genotype file, vcf type is {args.vcf_type}")
            get_obs_gt(args.vcf, args.ancestral, args.out, args.mask, args.vcf_type, args.ind)
        else:
            print("please specify the data type, either 'gll' or 'gt'")
            return

    if args.mode == 'train':
        if not hasattr(args, 'gll_file'):
            parser.print_help()
            return
        hmm_parameters = read_HMM_parameters_from_file(args.param)
        if args.transition_1 is not None:
            hmm_parameters.transitions[0] = [1-args.transition_1, args.transition_1]
        if args.transition_2 is not None:
            hmm_parameters.transitions[1] = [args.transition_2, 1-args.transition_2]
        if args.not_est_starting_prob:
            print("not estimating starting probabilities")
            hmm_parameters.starting_probabilities = [1-args.starting_prob, args.starting_prob]
        print(hmm_parameters)
        print(f"filter depth: {args.filter_depth}")
        print(f"fixed transition parameters: {args.not_est_transition}")
        print(args.param)
        with open(args.out+".log", 'w') as log_file:
            log_file.write(f"gll mode\n")
            log_file.write(f"input_file {args.gll_file}\n")
            log_file.write(f"fixed transition parameters: {args.not_est_transition}\n")
            if args.rec is True:
                log_file.write(f"using rec map: {args.rec_map}\n")
            else:
                log_file.write(f"using physical length for binning\n")
                log_file.write(f"window size: {args.window_size}\n")
            log_file.write(f"filter depth: {args.filter_depth}\n")
            log_file.write(f"max depth: {args.maximum_dep}\n")
            log_file.write(f"min depth: {args.minimum_dep}\n")
            log_file.write(f"mut rate file: {args.mut_file}\n")
        t1 = time.time()
        observation, chrs, windows, m_rates = load_observations(args.gll_file, args.window_size, args.filter_depth, args.maximum_dep, args.minimum_dep, args.rec_map, args.mut_file)
        t2 = time.time()
        print(f"loading time: {t2 - t1}")
        print('-' * 40)
        print('> Output is',args.out)
        print('-' * 40)
        hmm_parameters = TrainModel(raw_obs = observation,
                                 chr_index = chrs,
                                 w= windows,
                                 post_file= args.out+".posterior",
                                 log_file = args.out+".log",
                                 pars = hmm_parameters,
                                 m_rates = m_rates,
                                 maxiterations=args.iteration,
                                 not_est_trans = args.not_est_transition,
                                 not_est_starting_prob= args.not_est_starting_prob)
        write_HMM_to_file(hmm_parameters, args.out+".param")
    if args.mode == "decode":
        print("decoding")
        if not hasattr(args, 'gll_file'):
            parser.print_help()
            return
        with open(args.out+".log", 'a') as log_file:
            print("decoding, gll", file = log_file)
            print(f"mut rate file: {args.mut_file}", file = log_file)
            print(f"posterior file: {args.out}.posteripr", file = log_file)
            print(f"input file: {args.gll_file}", file = log_file)
            print(f"parameter file: {args.param}\n", file = log_file)
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
        print(f"posterior file: {args.out}.posterior")
        print(f"window size: {args.window_size}\n")
        print(f"filter depth: {args.filter_depth}\n")
        if args.filter_depth:
            print(f"max depth: {args.maximum_dep}\n")
            print(f"min depth: {args.minimum_dep}\n")
        if args.rec:
            print(f"using rec map: {args.rec_map}")
        t1 = time.time()
        observation, chrs, windows, m_rates = load_observations(args.gll_file, args.window_size, args.filter_depth, args.maximum_dep, args.minimum_dep,
        args.rec, args.rec_map, args.mut_file)
        t2 = time.time()
        print(f"loading time: {t2 - t1}")
        print('-' * 40)
        print(hmm_parameters)
        print('> Window size is',args.window_size, 'bp') 
        print('-' * 40)
        decode_from_params(raw_obs = observation,
                        chr_index = chrs,
                        w= windows,
                        post_file= args.out+".redecode.posterior",
                        pars = hmm_parameters,
                        m_rates = m_rates)
        get_runs(args.out+".redecode.posterior", args.penalty) 
        #Call(args.out+".posterior", hmm_parameters, args.out+".called", args.window_size)
        print("call finished")
    if args.mode == 'gt_mode':   # same as Larits. But starts with the first callable position.
        if not hasattr(args, 'count_file'):
            parser.print_help()
            return

        print("gt mode")
        hmm_parameters = read_HMM_parameters_from_file(args.param)
        print(args.param)
        #print(f"maximum number of variants per window allowed is {args.max_variants_per_window}")
        t1 = time.time()

        print(f"loading input data {args.count_file} with {args.data_type} mask {args.mask_file}, with window size {args.window_size}")
        print(f"maximum number of variants per window allowed is {args.max_variants_per_window}")
        chr_index, weights, obs, call_index, w, rec = load_observations_gt(args.count_file, 
        args.mask_file, args.window_size, args.max_variants_per_window, args.data_type, args.rec_map, args.filter_depth, args.minimum_dep, args.maximum_dep, args.mut_file)
        #check for zero:
        t2 = time.time()
        print(f"loading time: {t2 - t1}")
        print(f"finished loading {args.count_file}")
        print(f"finished loading {args.mask_file}")
        print('-' * 40)
        if args.transition_1 is not None:
            hmm_parameters.transitions[0] = [1-args.transition_1, args.transition_1]
        if args.transition_2 is not None:
            hmm_parameters.transitions[1] = [args.transition_2, 1-args.transition_2]
        if args.not_est_starting_prob:
            print("not estimating starting probabilities")
            hmm_parameters.starting_probabilities = [1-args.starting_prob, args.starting_prob]
        print(hmm_parameters)
        print('> Output is',args.out) 
        print('> Window size is',args.window_size, 'bp') 
        print('-' * 40)
        with open(args.out+".log", 'w') as log_file:
            log_file.write(f"gt mode\n")
            log_file.write(f"input type {args.data_type}\n")
            log_file.write(f"input_file {args.count_file}\n")
            log_file.write(f"{hmm_parameters}\n")
            log_file.write(f"maximum number of variants per window allowed is {args.max_variants_per_window}\n")
            log_file.write(f"output file prefix: {args.out}\n")
            log_file.write(f"window size: {args.window_size}\n")
            log_file.write(f"mut rate file: {args.mut_file}\n")
            log_file.write(f"posteiors file: {args.out}.posterior\n")
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
                                post_file= args.out+".posterior",
                                log_file = args.out+".log",
                                rec = rec,
                                hmm_parameters = hmm_parameters,
                                maxiterations=args.iteration,
                                not_est_trans = args.not_est_transition,
                                not_est_starting_prob= args.not_est_starting_prob)
        write_HMM_to_file(hmm_parameters, args.out+".param")
        get_runs(args.out+".posterior", args.penalty)     
        #Call(args.out+".posterior", hmm_parameters, args.out+".called", args.window_size)
        print("call finished")
    if args.mode == "call":
        print("calling fragments using posterior")
        get_runs(args.posterior, args.penalty)
        print("call finished")
    if args.mode == "anno":
        print(f"annotating fragments using observation file {args.count_file} and called file {args.called}")
        print(f"using archaic vcf {args.vcf}")
        print(f"using archaic individuals {args.sample}")
        print(f"using filter on depth {args.filter_depth}")
        print(f"sample names {args.sample}")
        print(f"out file {args.anno}.anno")
        if args.filter_depth:
            print(f"max depth: {args.maximum_dep}")
            print(f"min depth: {args.minimum_dep}")
        if not args.map_file is None:
            print(f"using map file {args.map_file}, map {args.map} to calculate genetic length")
        anno(gt_file = args.count_file, 
             called = args.called,
             vcf = args.vcf,
             out = args.anno,
             samples = args.sample,
             map_file = args.map_file,
             map = args.map,
             window_size = args.window_size,
             filter_depth = args.filter_depth,
             minimum_dep = args.minimum_dep,
             maximum_dep = args.maximum_dep)
if __name__ == "__main__":
    main()
"""
hmm_parameters = read_HMM_parameters_from_file(None)
chr_index, weights, obs, call_index, w, rec = load_observations_gt("/mnt/diversity/jiaqi/Den_Anc/Han/NA18552_file/NA18552.chrAuto.obs.gz", 
"/mnt/diversity/jiaqi/hmm/mask_new/1kg_accessibility_map35_100_simpleRepeat.rm_1kg_sgdp_outgroup.bed", 
1000, 50, "Modern", None, False, 0, 100, 
"/mnt/diversity/jiaqi/Den_Anc/1kg_hgdp_outgroup/hmmix_mut_rate_1kg_1M_chrAll_safe")
print("done")
"""