from collections import defaultdict
import pandas as pd
import numpy as np
from numba import njit
import json
from math import exp, ceil
import os 
import gzip

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
def get_mut_rates_gt(mut_full, window_size, w, chr):
    '''
    window: window index
    a chromsome
    e.g. 0 1000000 1.5
         1000000 2000000 2.5

    '''
    mut = []
    mut_full = mut_full[mut_full['chrom'] == chr]
    assert len(mut_full)!=0, f"mutation rate missing for chromsome {chr}"
    current_window  = 0  #start from the first window with data, might be window_index 600.
    inter = 0   #interval of mut rates
    while current_window <= len(w):
        while inter < len(mut_full):  
            '''
            window 100
            100 * 1000 - 101 * 1000
            '''
            ''' go to current mut interval'''
            if ((mut_full.iloc[inter]['start'] <= (w[current_window] * window_size)) and (mut_full.iloc[inter]['end'] >= (w[current_window] + 1) * window_size)):     
                if (mut_full.iloc[inter]['mut_rate'] == 0):   
                    '''
                    might cause numerical problem. so far all 0 mut_rates are manully modifyed
                    '''
                    mut.extend([0.1])
                else:
                    mut.extend([mut_full.iloc[inter]['mut_rate']])
                current_window += 1   # move to next window
                break
            inter += 1
            #print(current_window)
        if ((inter >= len(mut_full)) or (current_window >= len(w))):
            break
    print("mut rates load done") 
    return np.array(mut)

def load_fasta(fasta_file):
    '''
    Read a fasta file with a single chromosome in and return the sequence as a string
    '''
    fasta_sequence = ''
    with open(fasta_file) as data:
        for line in data:
            if not line.startswith('>'):
                fasta_sequence += line.strip().upper()

    return fasta_sequence

def get_obs_gll(vcf, fa, out, mask = None):
    s = f"zcat {vcf}"
    if not mask is None:
        s  = f"bcftools view -R {mask} {vcf}"
    with open(out, 'w') as f:
        fa = load_fasta(fa)
        for line in os.popen(s):
            if not line.startswith('#'):
                GLs = defaultdict(lambda:0.0)
                chr, pos, _, A, BCD, _, _, _, _, info = line.strip().split()
                pos = int(pos)
                anc = fa[pos-1]
                if A in "ACGT":
                    B, C, D = BCD.strip().split(',')
                    _, DP, _, _, _, _, _, _, GL = info.strip().split(':')
                    GL = np.array([float(x) for x in GL.strip().split(',')])
                    GL = np.power(10, GL)
                    GLs[f'{A}{A}'] = GL[0]
                    GLs[f'{A}{B}'] = GL[1]
                    GLs[f'{B}{B}'] = GL[2]
                    GLs[f'{A}{C}'] = GL[3]
                    GLs[f'{B}{C}'] = GL[4]
                    GLs[f'{C}{C}'] = GL[5]
                    GLs[f'{A}{D}'] = GL[6]
                    GLs[f'{B}{D}'] = GL[7]
                    GLs[f'{C}{D}'] = GL[8]
                    GLs[f'{D}{D}'] = GL[9]
                    if anc == "A":
                        print(chr, pos, anc, DP, GLs['AA'], 
                        np.max(
                            [GLs['AC'], GLs['CA'], GLs['AG'], GLs['GA'], GLs['AT'], GLs['TA'], 
                            GLs['CC'], 
                            GLs['CG'], GLs['GC'], GLs['CT'], GLs['TC'], 
                            GLs['GG'], 
                            GLs['GT'], GLs['TG'], 
                            GLs['TT']]), 
                            file = f, sep = '\t')
                    elif anc == "C":
                        print(chr, pos, anc, DP, GLs['CC'], 
                        np.max(
                            [GLs['AA'], 
                            GLs['AC'], GLs['CA'], GLs['AG'], GLs['GA'], GLs['AT'], GLs['TA'], 
                            GLs['CG'], GLs['GC'], GLs['CT'], GLs['TC'], 
                            GLs['GG'], 
                            GLs['GT'], GLs['TG'], 
                            GLs['TT']]), 
                            file = f, sep = '\t')
                    elif anc == "G":
                        print(chr, pos, anc, DP, GLs['GG'], 
                        np.max(
                            [GLs['AA'], 
                            GLs['AC'], GLs['CA'], GLs['AG'], GLs['GA'], GLs['AT'], GLs['TA'], 
                            GLs['CC'], 
                            GLs['CG'], GLs['GC'], GLs['CT'], GLs['TC'], 
                            GLs['GT'], GLs['TG'], 
                            GLs['TT']]), 
                            file = f, sep = '\t')
                    elif anc == "T":
                        print(chr, pos, anc, DP, GLs['TT'], 
                        np.max(
                            [GLs['AA'], 
                            GLs['AC'], GLs['CA'], GLs['AG'], GLs['GA'], GLs['AT'], GLs['TA'], 
                            GLs['CC'], 
                            GLs['CG'], GLs['GC'], GLs['CT'], GLs['TC'], 
                            GLs['GG'], 
                            GLs['GT'], GLs['TG']]), 
                            file = f, sep = '\t')

def get_obs_gt(vcf, fa, out, mask = None):
    s = f"zcat {vcf}"
    if not mask is None:
        s = f"bcftools view -R {mask} {vcf}"
    with open(out, 'w') as f:
        fa = load_fasta(fa)
        for line in os.popen(s):
            if not line.startswith("#"):
                chr, pos, _, A, BCD, _, _, _, _, info = line.strip().split()    # A, ref
                if A != "ACGT" :
                    pos = int(pos)
                    anc = fa[pos-1]
                    if anc in "ACGT":
                        gt, dep = info.split(':')[0:2]
                        if gt == "0/0":    #   homozygous reference, BCD = '.'
                            if anc == A:
                                print(chr, pos, anc, dep, 0, A+A, sep = '\t', file = f)
                            else:
                                print(chr, pos, anc, dep, 2, A+A, sep = '\t', file = f)
                        elif gt == "1/1":   # homozygous alternative.    anc must be alt
                            B = BCD.split(',')[0]
                            if anc == B:
                                print(chr, pos, anc, dep, 0, BCD[0]+BCD[0], sep = '\t', file = f)
                            else:
                                print(chr, pos, anc, dep, 2, BCD[0]+BCD[0], sep = '\t', file = f)
                        elif gt == "0/1":
                            print(chr, pos, anc, dep, 1, A+BCD[0], sep = '\t',  file = f)

def get_weights(bedfile, window_size): # return weights from the first window to the last window
    first_window = 0
    last_window = 0
    weights = defaultdict(lambda: defaultdict(float))
    with open(f"{bedfile}") as data:
        print(f"loading mask file {bedfile}")
        for line in data:
            chr, start, end = line.strip().split('\t') 
            start = int(start)
            end = int(end)
            start_window = ceil((start+0.5) / window_size) - 1
            end_window = ceil((end - 0.5) / window_size) - 1
            if start_window == end_window: # same window   start window 1, end window 1
                weights[chr][start_window] += (end - start) / window_size
            else:
                weights[chr][start_window] += (window_size*(start_window+1) - start) / window_size
            
                if end_window > start_window + 1:       # fill in windows in the middle   start_window 1, end_window 4, so window 2 window3 will be filled as 1
                    for window_tofill in range(start_window + 1, end_window):
                        weights[chr][window_tofill] += float(1)
                    weights[chr][end_window] += (end - end_window * window_size) / window_size
                    
                else:    # e.g. start window 1, end window 2
                    weights[chr][start_window + 1] += (end - end_window * window_size) / window_size
        return weights

def load_observations(gll_file, window_size=1000, filter_depth = False, maximum_dep = None, minimum_dep = None, 
rec = False, rec_bed = None, mut_bed = None):  # return g0 g1
    
    
    # free space for each chrom.
    # cap on depth
    gl = defaultdict(lambda:defaultdict(lambda: defaultdict(list)))

    with gzip.open(gll_file, "rt") as data:
        for line in data:
            if not line.startswith("#"):
                ne = line.strip().split()
                len_ne = len(ne)
                break
    if rec == True:
        print(f"binning data based on {rec_bed} ")
        print(f"loading mut rates based on {mut_bed} ")
        m = defaultdict(list)
        rec_bed = pd.read_csv(rec_bed, sep = '\t', dtype = {'chr':str, 'start':int, 'end':int, 'window':int})
        loop_rec = rec_bed.iterrows()
        row_rec = next(loop_rec)[1]
        row_rec_ = next(loop_rec)[1]
        if len_ne == 6:
            print(f"loading real data from {gll_file}")
            with gzip.open(gll_file, "rt") as data:
                if not mut_bed is None:
                    mut_bed = pd.read_csv(mut_bed, sep = '\t', names = ['chr', 'start', 'end', 'rate'], dtype = {'chr':str, 'start':int, 'end':int, 'rate':float})
                    loop_mut= mut_bed.iterrows()
                    row_mut = next(loop_mut)[1]
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
                            #window = ceil(int(pos) / window_size) - 1
                            while row_rec.chr != chrom:  # go the the current chromomsome. Chromosomes must be in order.
                                row_rec, row_rec_ = row_rec_, next(loop_rec)[1]
                            if row_rec.start > int(pos): # recombination has not started yet. do not record
                                continue
                            try :
                                while row_rec.end < int(pos):  # pos goes to next recombination interval
                                    row_rec, row_rec_ = row_rec_, next(loop_rec)[1]
                            except row_rec_.chr != chrom:   # recombination at current chromosome ends. do not record anymore
                                continue   # do not record
                            except StopIteration:   # whole recombinatino ends. end
                                break
                            while row_mut.chr != chrom:
                                row_mut = next(loop_rec)[1]
                            while row_mut.end < int(pos):
                                row_mut = next(loop_rec)[1]

                            window = row_rec.window
                            m[chrom].append(row_mut.rate)
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
                                #window = ceil(int(pos) / window_size) - 1
                                while row_rec.chr != chrom:  # go the the current chromomsome. Chromosomes must be in order.
                                    row_rec, row_rec_ = row_rec_, next(loop_rec)[1]
                                if row_rec.start > int(pos): # recombination has not started yet. do not record
                                    continue
                                try :
                                    while row_rec.end < int(pos):  # pos goes to next recombination interval
                                        row_rec, row_rec_ = row_rec_, next(loop_rec)[1]
                                except row_rec_.chr != chrom:   # recombination at current chromosome ends. do not record anymore
                                    continue   # do not record
                                except StopIteration:   # whole recombinatino ends. end
                                    break
                                while row_mut.chr != chrom:
                                    row_mut = next(loop_mut)[1]
                                while row_mut.end < int(pos):
                                    row_mut = next(loop_mut)[1]
                                window = row_rec.window
                                m[chrom].append(row_mut.rate)
                                gl["g_0"][chrom][window].append(g_0)
                                gl["g_1"][chrom][window].append(g_1)
                else: # No mut bed provided. For real data, not mut bed should be provided when tranning parameters
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
                            #window = ceil(int(pos) / window_size) - 1
                            while row_rec.chr != chrom:  # go the the current chromomsome. Chromosomes must be in order.
                                row_rec, row_rec_ = row_rec_, next(loop_rec)[1]
                            if row_rec.start > int(pos): # recombination has not started yet. do not record
                                continue
                            try :
                                while row_rec.end < int(pos):  # pos goes to next recombination interval
                                    row_rec, row_rec_ = row_rec_, next(loop_rec)[1]
                            except row_rec_.chr != chrom:   # recombination at current chromosome ends. do not record anymore
                                continue   # do not record
                            except StopIteration:   # whole recombinatino ends. end
                                break
                            window = row_rec.window
                            m[chrom].append(1)
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
                                while row_rec.chr != chrom:  # go the the current chromomsome. Chromosomes must be in order.
                                    row_rec, row_rec_ = row_rec_, next(loop_rec)[1]
                                if row_rec.start > int(pos): # recombination has not started yet. do not record
                                    continue
                                try :
                                    while row_rec.end < int(pos):  # pos goes to next recombination interval
                                        row_rec, row_rec_ = row_rec_, next(loop_rec)[1]
                                except row_rec_.chr != chrom:   # recombination at current chromosome ends. do not record anymore
                                    continue   # do not record
                                except StopIteration:   # whole recombinatino ends. end
                                    break
        
                                window = row_rec.window
                                m[chrom].append(1)
                                gl["g_0"][chrom][window].append(g_0)
                                gl["g_1"][chrom][window].append(g_1)
        else:
            print("data format not supported (cuurently not desiged not simulated dataset)")
        obs_chrs = []
        obs_count = []
        m_rates = []
        for chrom in list(gl["g_0"].keys()):
            gl_0_ = []
            gl_1_ = []
            obs_count_ = []
            for window in list(gl["g_0"][chrom].keys()):
                gl_0_.append(np.array(gl["g_0"][chrom][window]))
                gl_1_.append(np.array(gl["g_1"][chrom][window]))
            obs_chrs.append([gl_0_, gl_1_])

        for chrom in list(m.keys()):
            m_rates.append(np.array(m[chrom]))
        return obs_chrs, list(gl["g_0"].keys()), list(list(gl["g_0"][chrom].keys()) for chrom in list(gl["g_0"].keys()) ), m_rates


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
    elif len_ne == 6:
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
    else:
        print("data format not supported (cuurently not desiged not simulated dataset)")
    obs_chrs = []
    obs_count = []
    m_rates = []
    if not mut_bed is None:
        m_rates_full = pd.read_csv(mut_bed, sep = '\t', 
                               header = None, names = ["chrom", "start", "end", "mut_rate"],
                               dtype = {"chrom": str, "start": int, "end": int, "mut_rate": float})
    for index, chrom in enumerate(list(gl["g_0"].keys())):
        gl_0_ = []
        gl_1_ = []
        obs_count_ = []
        for window in list(gl["g_0"][chrom].keys()):
            gl_0_.append(np.array(gl["g_0"][chrom][window]))
            obs_count_.append(len(gl["g_0"][chrom][window]))
            gl_1_.append(np.array(gl["g_1"][chrom][window]))
        obs_chrs.append([gl_0_, gl_1_])
        obs_count.append(obs_count_)
        if mut_bed is None:
            m_rates_ = np.ones(sum(obs_count_))
        else:
            m_rates_ = get_mut_rates(m_rates_full, window_size, list(list(gl["g_0"][chrom].keys())), obs_count[index], list(gl["g_0"].keys())[index])
        m_rates.append(m_rates_)
        l = len(m_rates_)
        l1 = sum(obs_count_)
        assert l==l1, f"missing part of mut rates for obs in {chrom} chromsome, {l} mut records, {l1} obs records"
    return obs_chrs, list(gl["g_0"].keys()), list(list(gl["g_0"][chrom].keys()) for chrom in list(gl["g_0"].keys()) ), m_rates


def load_observations_gt(gt_file, mask_file, window_size, max_variants, data_type, 
filter_depth, minimum_dep, maximum_dep):  # return number of variants
    chr = []
    chr_index = []  #chrs
    windows = []
    weights = []  # weights    [1,0,1,0,2,...] from the first bin with data to last bin with data
    obs = []    # [0,0,0,1,0,0...][0,1,1,0,...] from the first bin with data to last bin with data
    call_index = []    #[0,1,3,5,...]   bins with data
    # assuming full length 100M
    window_all = []

    call = get_weights(mask_file, window_size)

    for chr in list(call.keys()):
        first_w = list(call[chr].keys())[0]  # the first window callable
        last_w = list(call[chr].keys())[-1]  # the last window callable
        
        weights_ = np.zeros(last_w - first_w + 1)
        call_index_ = []
        
        for i in list(call[chr].keys()):   # loop through all windows with data
            i = int(i)
            call_index_.append(i - first_w)
        for w in range(first_w, last_w+1):
            weights_[w-first_w] = call[chr][w]
        weights.append(weights_)              #
        
        call_index.append(call_index_)    # [0,1,3,5,...]   bins with data, count from the first bin with data.
    snp = defaultdict(lambda:defaultdict(int))
    if gt_file.endswith(".gz"):
        if data_type == "ancient":
            sss = f"zcat {gt_file} | awk '($5 > 0)'"
            if filter_depth is True:
                for line in os.popen(sss):
                    #chr, pos = line.strip().split()
                    chr, pos, _, dep = line.strip().split()[0:4]
                    if (int(dep) <= maximum_dep) and (int(dep) >= minimum_dep):
                        window = ceil(int(pos) / window_size) - 1
                        snp[chr][window]+=1
            else:
                for line in os.popen(sss):
                    #chr, pos = line.strip().split()
                    chr, pos, = line.strip().split()[0:2]
                    window = ceil(int(pos) / window_size) - 1
                    snp[chr][window]+=1
        else:
            with gzip.open(gt_file, 'rt') as data:
                for line in data:
                    #chr, pos = line.strip().split()
                    chr, pos = line.strip().split()[0:2]
                    window = ceil(int(pos) / window_size) - 1
                    snp[chr][window]+=1
    else:  # not gz, must be from simulations
        if data_type == "ancient":
            sss = f"cat {gt_file} | awk '($5 > 0)'"
            for line in os.popen(sss):
                #chr, pos = line.strip().split()
                chr, pos = line.strip().split()[0:2]
                window = ceil(int(pos) / window_size) - 1
                snp[chr][window]+=1
        else:
            with open(gt_file, 'r') as data:
                for line in data:
                    #chr, pos = line.strip().split()
                    chr, pos = line.strip().split()[0:2]
                    window = ceil(int(pos) / window_size) - 1
                    snp[chr][window]+=1            
    for chr in list(snp.keys()):   # has to start from (chr)1
        
        print(chr)
        chr_index.append(int(chr))
        first_w = sorted(list(call[chr].keys()))[0]  # assuming all are chr1
        last_w = sorted(list(call[chr].keys()))[-1]
        window_all.append(list(range(first_w, last_w+1)))
        snps = np.zeros(last_w - first_w + 1)
        for window in list(snp[chr].keys()):
            snps[window-first_w] = snp[chr][window]   # snp from the first bin with data.
            if snps[window-first_w] > max_variants:
                snps[window-first_w] = max_variants
                print(f"window {window}, chr{chr} has more than {max_variants} variants, set to {max_variants}, file {gt_file}")
            #assert weights[int(chr)-1][window-first_w] >0 , f"weights for window {window} is 0 but there are derived in this window."

        obs.append(snps)
    return chr_index, weights, obs, call_index, window_all


"""
I am lazy, so I directly calculate mutation rate either from simulation or from empirical data.
No functions embedded so far.
Super-high mutation rates comes from mappability filter. Nominator is too small in some windows.
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