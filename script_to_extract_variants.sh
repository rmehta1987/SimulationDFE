#!/bin/bash

bcftools query -f '%CHROM\t%POS\t%INFO/non_cancer_AC_asj\n'  gnomad.exomes.r2.1.1.sites.1.vcf.bgz > aji_ac.vcf
temp = np.loadtxt('/home/rahul/PopGen/SimulationSFS/gnomAD_data/aji_ac.vcf')
temp2 = temp[:,-1].astype(int) # get last column
#sample_size = 4786*2 - 1 Get sample size (2*N) for AJI it is 4786*2 - 1
#get only non zeros
temp2 = temp2[np.nonzero(temp2)]
thebins = np.arange(1,sample_size+1)
temphist,bins = np.histogram(temp2, bins=thebins)  
plt.stairs(np.log(temphist[1:]),bins[1:]) # skipped the singleton bins fo now

# Finish non-neuro field non_neuro_AC_fin
# finish non_neuro_ sample size = 16734
bcftools query -f '%CHROM\t%POS\t%INFO/non_cancer_AC_fin\n'  gnomad.exomes.r2.1.1.sites.1.vcf.bgz > aji_ac.vcf
