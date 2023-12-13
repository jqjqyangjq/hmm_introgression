# hmm_introgression
A beta version for simulation.

This is to extend the original himmix model to low-coverage and damaged genomic data.
The emission process has been totally re-written, similar to admixfrog.  
Currently can bin the genome by physical length or genetic length(testing).  
Mutrates are not used when training for parameters, assuming that emission parameters are trained as the average across the whole genone, but are used when decoding via forawrd-backward algorith. Mutrates are calculated using 1000g(imputed) sub-Saharan africans falling within strict accessibility filters, by counting the positions showing any derived alleles.  
Assume pre-calculated gll input file, taking damamge pattern into account.  
Previous simulations show that this works for covearge >5x, but for real data seems we can go lower  
Also include pretty much similar gt mode to LS hmmix but different in : Optimizing emission parameters using windows with data only. (LS used all windows). In addition, support binning the genome by genetic length(hg19 AA map, as DECODE map is missing huge chunk of each chromosoem)
