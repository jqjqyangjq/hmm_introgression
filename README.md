# hmm_introgression
See Skov et al., 2018

This is to extend the original himmix model to low-coverage and damaged genomic data. I rewrote scripts so it's covenient to adapt to my project for low-coverage data. 

In case using glls, The emission process has been totally re-written, similar to admixfrog. Otherwise similar to LS's, but different in: 1, Using bins with data for traning parameters. 2, train each chromosome as an independent Markov chain, but EM uses averaged expectations. 

Currently the genome is still binned by physical lengths, but can be annotated by a recombination map. Have shown not easy to bin by genetic length, unless using a time-continous markov chain and train the rate parameter.

When using glls, Mutrates are not used when training for parameters, assuming that emission parameters are trained as the average across the whole genone, but are used when decoding via forawrd-backward algorith. Mutrates are calculated using 1000g(imputed) sub-Saharan africans falling within strict accessibility filters, by counting the positions showing any derived alleles. Both getting outgroup list and calculating mut rates are using LS's functions, but modified the output of outgroup list from LS as that included the fixed derived alleles (not in the vcf but would be present when comparing human reference genome with ancestral fasta) outside the mask, which would inflate mutation rates in some windows with high-level of missingness. 

Assume pre-calculated gll input file, taking damamge pattern into account.  
Previous simulations show that this works for covearge >5x, but for real data seems we can go lower 

update: Indel-realignment seems to be crutial. which prevents HMM to go to lower coverage. But as long as genome is deep enough for realignment, there is no huge difference between using gll and gts. : )    
