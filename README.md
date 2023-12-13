# hmm_introgression
A beta version for simulation.

This is to extend the original himmix model to low-coverage and damaged genomic data.
The emission process has been totally re-written.

Assume a pre-calculated mutrate file across the genome, otherwise assume constant mutation rates
(testing by simulation)

Assume pre-calculated gll input file, taking damamge pattern into account.
(testing for uncertainty in gll calculating)
test  
Also include pretty much similar gt mode to LS hmmix but different in : Optimizing emission parameters using windows with data only. (LS used all windows)
