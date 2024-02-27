from collections import defaultdict
import numpy as np
import pandas as pd
import os
from math import ceil
def anno(gt_file, called, vcf, out, samples, map_file = None, map = "AA_Map", window_size = 1000,
filter_depth = "False", minimum_dep = 0, maximum_dep = 100):  # return number of variants
    called = pd.read_csv(called, dtype={'chrom':str, 'start':int, 'end':int})
    called = called.sort_values(by = ['chrom', 'start']).reset_index(drop=True)
    chrs = called['chrom'].unique()
    print(chrs)
    print(f"out file is {out}")
    if not map_file is None:
        map_anno = pd.read_csv(map_file, sep = '\s+',dtype = {'Chrom':str, 'Physical_Pos':int })
        map_anno = map_anno[['Chrom','Physical_Pos',f"{map}"]].rename(columns = {'Chrom':'chr', 'Physical_Pos':'pos', f"{map}":'map_len'})
    chr = []
    chr_index = []  #chrs
    windows = []
    weights = []  # weights    [1,0,1,0,2,...] from the first bin with data to last bin with data
    obs = []    # [0,0,0,1,0,0...][0,1,1,0,...] from the first bin with data to last bin with data
    call_index = []    #[0,1,3,5,...]   bins with data
    # assuming full length 100M
    window_all = []
    snp = defaultdict(lambda:defaultdict(lambda: defaultdict(lambda:defaultdict(str))))
    #still loading all the snps, and create a temp .pos file for vcf
    with open(out+".err", 'w') as err:
        with open(out+"temp", 'w') as f:
            if gt_file.endswith(".gz"):
                sss = f"zcat {gt_file} | awk '($5 > 0)'"
                if filter_depth is True:
                    for line in os.popen(sss):
                        #chr, pos = line.strip().split()
                        chr, pos, anc, dep, _, gt = line.strip().split()[0:6]
                        if (int(dep) <= maximum_dep) and (int(dep) >= minimum_dep):
                            window = ceil(int(pos) / window_size) - 1
                            gt_ = gt.replace(anc, "")
                            if len(gt_) == 0:
                                print(f"error calls: {chr}\t{pos}\t{anc}\t{gt}", file = err)
                            else:
                                gt = gt_[0]
                                snp[chr][window][pos] = gt
                                print(f"{chr}\t{pos}\t{gt}", file = f)
                else:
                    for line in os.popen(sss):
                        #chr, pos = line.strip().split()
                        chr, pos, anc, dep, _, gt = line.strip().split()[0:6]
                        window = ceil(int(pos) / window_size) - 1
                        snp[chr][window][pos] = gt
                        gt = gt.replace(anc, "")
                        gt = gt[0]
                        snp[chr][window][pos] = gt
                        print(f"{chr}\t{pos}\t{gt}", file = f)
            else:
                print("not supporting simulations at the moment")
                return None
    snp_anno = defaultdict(lambda:defaultdict(lambda: defaultdict(int)))
    samples_list = ",".join(samples.split(","))
    samples = samples.split(",")
    vcfs = vcf.split(",")
    print("loading vcfs for comparision")
    for vcf_ in vcfs:
        print(f"Using samples {samples_list}")
        for line in os.popen(f"bcftools view -R {out+'temp'} -s {samples_list} {vcf_}"):
            if not line.startswith("#"):
                chr, pos = line.strip().split()[0:2]
                window = ceil(int(pos) / window_size) - 1
                ref = line.strip().split()[3]
                alt = line.strip().split()[4]
                for ind, gt in enumerate(line.strip().split()[9:]):  #loop through all archaic individuals
                    gt = gt.split(':')[0]
                    if gt == "./.":
                        continue
                    if gt == "0/0":
                        gt = ref+ref
                    elif gt == "1/1":
                        gt = alt+alt
                    elif gt == "0/1":
                        gt = ref+alt
                    elif gt == "1/2":
                        gt = alt.split(",")[0]+alt.split(",")[1]
                    if (snp[chr][window][pos] == gt[0]) or (snp[chr][window][pos] == gt[1]):
                        snp_anno[chr][window][samples[ind]+"_match"] +=1
                    else:
                        snp_anno[chr][window][samples[ind]+"_mismatch"] +=1
    
    S_info = []
    for sample in samples:
        S_info.append(sample+"_match")
        S_info.append(sample+"_total")
    print("loading vcfs for comparision done")
    with open(out, 'w') as f:
        if not map_file is None:
            print("chrom","start", "end", "length", "map_len", "\t".join(S_info), "source1", "source2", sep = '\t', file = f)
            for chr in chrs:
                chr = str(chr)
                call_chr = called[called['chrom'] == chr].reset_index(drop=True)
                map_chr = map_anno[map_anno['chr'] == chr].reset_index(drop=True)
                call_rows = call_chr.iterrows()
                loop_map = map_chr.iterrows()
                row = next(loop_map)[1]
                row_1 = next(loop_map)[1]
                for i in range(len(call_chr)):
                    print(i)
                    if row.pos > call_chr.start[i]*window_size:  # start of the called frag is before the first rec pos. so is 0
                        start_gen = row.map_len
                        if (call_chr.end[i]*window_size+window_size) < row.pos:  #end of the called frag is before the first rec pos
                            end_gen = row.map_len
                        else:
                            while (row_1.pos <= (call_chr.end[i]*window_size+window_size)):
                                try:
                                    row, row_1 = row_1, next(loop_map)[1]
                                except StopIteration:
                                    break
                            if row_1.pos <= (call_chr.end[i]*window_size+window_size):
                                end_gen = row_1.map_len
                            else:
                                end_gen = row.map_len + (row_1.map_len - row.map_len) / (row_1.pos - row.pos) * (call_chr.end[i]*window_size+window_size - row.pos)
                            gen_len = end_gen - start_gen
                    else:  
                        while (row_1.pos <= call_chr.start[i]*window_size):
                            try:
                                row, row_1 = row_1, next(loop_map)[1]
                            except StopIteration:
                                break
                        if row_1.pos < call_chr.start[i]*window_size:
                            start_gen = row_1.map_len
                        else:
                            start_gen = row.map_len + (row_1.map_len - row.map_len) / (row_1.pos - row.pos) * (call_chr.start[i]*window_size - row.pos)                        
                        while (row_1.pos < (call_chr.end[i]*window_size+window_size)):
                            try:
                                row, row_1 = row_1, next(loop_map)[1]
                            except StopIteration:
                                break
                        if row_1.map_len <= (call_chr.end[i]*window_size+window_size):
                            end_gen = row_1.map_len
                        else:
                            end_gen = row.map_len + (row_1.map_len - row.map_len) / (row_1.pos - row.pos) * (call_chr.end[i]*window_size + window_size - row.pos)
                    gen_len = round(end_gen - start_gen, 4)
                    M = []
                    M_ = []
                    #print(gen_len)
                    for sample in samples:
                        m = 0
                        mism = 0               
                        for window in list(snp_anno[chr].keys()):
                            if (window >= call_chr.start[i]) and (window <= call_chr.end[i]):
                                m += snp_anno[chr][window][sample+"_match"]
                                mism += snp_anno[chr][window][sample+"_mismatch"]
                        t = m+mism
                        if (t) == 0:
                            M.append(0)
                        else:
                            r = round(m/t,4)
                            M.append(r)  #M float value
                        M_.append(f"{m}/{t}")  # M_ string info, to record
                    samples_ = samples.copy()
                    L1 = np.argmax(np.array(M))
                    if M[L1] > 0.05:
                        sample1 = samples_[L1]
                    else:
                        sample1 = "NA"
                    M.pop(L1)
                    samples_.pop(L1)
                    L2 = np.argmax(np.array(M))
                    if M[L2] > 0.05:
                        sample2 = samples_[L2]
                    else:
                        sample2 = "NA"
                    print(call_chr.chrom[i],call_chr.start[i]*window_size,call_chr.end[i]*window_size+window_size, (call_chr.end[i]-call_chr.start[i]+1)*window_size, 
                          gen_len, "\t".join(M_), sample1, sample2, sep = '\t', file = f)
        else:
            for chr in chrs:
                call_chr = called[called['chrom'] == chr].reset_index(drop=True)
                for i in range(len(call_chr)):
                    M = []
                    M_ = []
                    for sample in samples:
                        m = 0
                        mism = 0               
                        for window in list(snp_anno[chr].keys()):
                            if (window >= call_chr.start[i]) and (window <= call_chr.end[i]):
                                m += snp_anno[chr][window][sample+"_match"]
                                mism += snp_anno[chr][window][sample+"_mismatch"]
                        t = m+mism
                        if (t) == 0:
                            M.append(0)
                        else:
                            r = round(m/t,4)
                            M.append(r)
                        M_.append(f"{m}/{t}")
                    samples_ = samples.copy()
                    L1 = np.argmax(np.array(M))
                    if M[L1] > 0.05:
                        sample1 = samples_[L1]
                    else:
                        sample1 = "NA"
                    M.pop(L1)
                    samples_.pop(L1)
                    L2 = np.argmax(np.array(M))
                    if M_[L2] > 0.05:
                        sample2 = samples_[L2]
                    else:
                        sample2 = "NA"
                    print(call_chr.chrom[i],call_chr.start[i]*window_size,call_chr.end[i]*window_size+window_size, (call_chr.end[i]-call_chr.start[i]+1)*window_size, 
                          gen_len, "\t".join(M_), sample1, sample2, sep = '\t', file = f)