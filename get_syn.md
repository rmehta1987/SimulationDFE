# Get SFS of specific type of variants variants

## First we get filtered allele counts of a specific population
## So for the Jewish population it would be:
# This name is put in the FILTER field for variants that get filtered. Note that there must be a 1-to-1 mapping between filter expressions and filter names.

`./gatk VariantFiltration -V ../SimulationSFS/gnomAD_data/gnomad.exomes.r2.1.1.sites.1.vcf.bgz -filter "AC_asj > 0.1" --filter-name "AC" -O ../SimulationSFS/gnomAD_data/ac_asj.vcf.gz`


## Similarly for the non_cancer subset for the Finnish Population it is:
`./gatk VariantFiltration -V ../SimulationSFS/gnomAD_data/gnomad.exomes.r2.1.1.sites.1.vcf.bgz -filter "non_cancer_AC_fin > 0.1" --filter-name "non_cancer_AC_fin" -O ../SimulationSFS/gnomAD_data/fin_output.vcf.gz`

## The reason for > 0.1 is to get integer values otherwise it just returns 0 :/ 

## Second step is to then to filter chromosome - position - allele count, VEP column stores information about synonynous and type of variant 

`./gatk VariantsToTable --variant ../SimulationSFS/gnomAD_data/synonymous_output.vcf.gz -F CHROM -F POS -F AC -F vep -O ../SimulationSFS/gnomAD_data/synonymous_output3.table`

## Get the sample size by finding the largest allele count
`awk -v max=0 'NR>1 {if($3>max) {max=$3}} END {print "max =",max+0}' synonymous_output3.table `


## Example of the VEP column and it's fields in the output table

## Examples of VEP fields and VEP columns of specific random SNPS

`##INFO=<ID=vep,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP. Format: Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|Existing_variation|ALLELE_NUM|DISTANCE|STRAND|FLAGS|VARIANT_CLASS|MINIMISED|SYMBOL_SOURCE|HGNC_ID|CANONICAL|TSL|APPRIS|CCDS|ENSP|SWISSPROT|TREMBL|UNIPARC|GENE_PHENO|SIFT|PolyPhen|DOMAINS|HGVS_OFFSET|GMAF|AFR_MAF|AMR_MAF|EAS_MAF|EUR_MAF|SAS_MAF|AA_MAF|EA_MAF|ExAC_MAF|ExAC_Adj_MAF|ExAC_AFR_MAF|ExAC_AMR_MAF|ExAC_EAS_MAF|ExAC_FIN_MAF|ExAC_NFE_MAF|ExAC_OTH_MAF|ExAC_SAS_MAF|CLIN_SIG|SOMATIC|PHENO|PUBMED|MOTIF_NAME|MOTIF_POS|HIGH_INF_POS|MOTIF_SCORE_CHANGE|LoF|LoF_filter|LoF_flags|LoF_info">`

`# 1	861360	2	T|upstream_gene_variant|MODIFIER|SAMD11|ENSG00000187634|Transcript|ENST00000341065|protein_coding|||||||||||1|4332|1|cds_start_NF|SNV|1|HGNC|28706|||||ENSP00000349216|||UPI000155D47A||||||||||||||||||||||||||||||||||,T|synonymous_variant|LOW|SAMD11|ENSG00000187634|Transcript|ENST00000342066|protein_coding|2/14||ENST00000342066.3:c.39C>T|ENST00000342066.3:c.39C>T(p.%3D)|122|39|13|C|tgC/tgT||1||1||SNV|1|HGNC|28706|YES|||CCDS2.2|ENSP00000342313|Q96NU1|Q5SV95&I7FV93&A6PWC8|UPI0000D61E04||||||||||||||||||||||||||||||||||DE_NOVO_DONOR_POS:-39&INTRON_START:861394&MUTANT_DONOR_MES:5.46466523333833&DE_NOVO_DONOR_PROB:0.00797631277913458&EXON_END:861393&DE_NOVO_DONOR_MES:-1.08956501872978&INTRON_END:865534&EXON_START:861302&DE_NOVO_DONOR_MES_POS:-35,T|synonymous_variant|LOW|SAMD11|ENSG00000187634|Transcript|ENST00000420190|protein_coding|2/7||ENST00000420190.1:c.39C>T|ENST00000420190.1:c.39C>T(p.%3D)|128|39|13|C|tgC/tgT||1||1|cds_end_NF|SNV|1|HGNC|28706|||||ENSP00000411579||Q5SV95&I7FV93&A6PWC8|UPI000155D47C||||hmmpanther:PTHR10417&hmmpanther:PTHR10417:SF5||||||||||||||||||||||||||||||DE_NOVO_DONOR_MES_POS:-35&EXON_START:861302&INTRON_END:865534&DE_NOVO_DONOR_MES:-1.08956501872978&EXON_END:861393&DE_NOVO_DONOR_PROB:0.00797631277913458&MUTANT_DONOR_MES:5.46466523333833&INTRON_START:861394&DE_NOVO_DONOR_POS:-39,T|synonymous_variant|LOW|SAMD11|ENSG00000187634|Transcript|ENST00000437963|protein_coding|2/5||ENST00000437963.1:c.39C>T|ENST00000437963.1:c.39C>T(p.%3D)|99|39|13|C|tgC/tgT||1||1|cds_end_NF|SNV|1|HGNC|28706|||||ENSP00000393181||Q5SV95&I7FV93|UPI000155D47B||||hmmpanther:PTHR10417&hmmpanther:PTHR10417:SF5||||||||||||||||||||||||||||||DE_NOVO_DONOR_POS:-39&INTRON_START:861394&MUTANT_DONOR_MES:5.46466523333833&EXON_END:861393&DE_NOVO_DONOR_PROB:0.00797631277913458&DE_NOVO_DONOR_MES:-1.08956501872978&INTRON_END:865534&EXON_START:861302&DE_NOVO_DONOR_MES_POS:-35,T|synonymous_variant|LOW|AL645608.1|ENSG00000268179|Transcript|ENST00000598827|protein_coding|6/6||ENST00000598827.1:c.240G>A|ENST00000598827.1:c.240G>A(p.%3D)|240|240|80|S|tcG/tcA||1||-1||SNV|1|Clone_based_ensembl_gene||YES||||ENSP00000471152||M0R0C9|UPI0000D61E05||||||||||||||||||||||||||||||||||,T|upstream_gene_variant|MODIFIER|RP11-54O7.3|ENSG00000223764|Transcript|ENST00000609207|retained_intron|||||||||||1|4964|-1||SNV|1|Clone_based_vega_gene||YES|||||||||||||||||||||||||||||||||||||||||,T|regulatory_region_variant|MODIFIER|||RegulatoryFeature|ENSR00001576148|promoter|||||||||||1||||SNV|1||||||||||||||||||||||||||||||||||||||||||||`

`# 1	865543	15	A|upstream_gene_variant|MODIFIER|SAMD11|ENSG00000187634|Transcript|ENST00000341065|protein_coding||||||||||rs370992396|1|149|1|cds_start_NF|SNV|1|HGNC|28706|||||ENSP00000349216|||UPI000155D47A|||||||A:0.0005414||||||||A:0|A:5.221e-05|A:0|A:0.0002887|A:0.001144|A:0|A:0|A:0||||||||||||,A|synonymous_variant|LOW|SAMD11|ENSG00000187634|Transcript|ENST00000342066|protein_coding|3/14||ENST00000342066.3:c.81G>A|ENST00000342066.3:c.81G>A(p.%3D)|164|81|27|G|ggG/ggA|rs370992396|1||1||SNV|1|HGNC|28706|YES|||CCDS2.2|ENSP00000342313|Q96NU1|Q5SV95&I7FV93&A6PWC8|UPI0000D61E04||||hmmpanther:PTHR12247&hmmpanther:PTHR12247:SF67|||A:0.0005414||||||||A:0|A:5.221e-05|A:0|A:0.0002887|A:0.001144|A:0|A:0|A:0||||||||||||EXON_START:865535&DE_NOVO_DONOR_MES_POS:-176&DE_NOVO_DONOR_MES:1.27122694829028&INTRON_END:866418&MUTANT_DONOR_MES:8.39701401588071&DE_NOVO_DONOR_PROB:0.0610947897378682&EXON_END:865716&DE_NOVO_DONOR_POS:-171&INTRON_START:865717,A|synonymous_variant|LOW|SAMD11|ENSG00000187634|Transcript|ENST00000420190|protein_coding|3/7||ENST00000420190.1:c.81G>A|ENST00000420190.1:c.81G>A(p.%3D)|170|81|27|G|ggG/ggA|rs370992396|1||1|cds_end_NF|SNV|1|HGNC|28706|||||ENSP00000411579||Q5SV95&I7FV93&A6PWC8|UPI000155D47C||||hmmpanther:PTHR10417&hmmpanther:PTHR10417:SF5|||A:0.0005414||||||||A:0|A:5.221e-05|A:0|A:0.0002887|A:0.001144|A:0|A:0|A:0||||||||||||DE_NOVO_DONOR_MES:1.27122694829028&INTRON_END:866418&EXON_START:865535&DE_NOVO_DONOR_MES_POS:-176&DE_NOVO_DONOR_POS:-171&INTRON_START:865717&MUTANT_DONOR_MES:8.39701401588071&DE_NOVO_DONOR_PROB:0.0610947897378682&EXON_END:865716,A|synonymous_variant|LOW|SAMD11|ENSG00000187634|Transcript|ENST00000437963|protein_coding|3/5||ENST00000437963.1:c.81G>A|ENST00000437963.1:c.81G>A(p.%3D)|141|81|27|G|ggG/ggA|rs370992396|1||1|cds_end_NF|SNV|1|HGNC|28706|||||ENSP00000393181||Q5SV95&I7FV93|UPI000155D47B||||hmmpanther:PTHR10417&hmmpanther:PTHR10417:SF5|||A:0.0005414||||||||A:0|A:5.221e-05|A:0|A:0.0002887|A:0.001144|A:0|A:0|A:0||||||||||||INTRON_START:865717&DE_NOVO_DONOR_POS:-171&EXON_END:865716&DE_NOVO_DONOR_PROB:0.0610947897378682&MUTANT_DONOR_MES:8.39701401588071&INTRON_END:866418&DE_NOVO_DONOR_MES:1.27122694829028&DE_NOVO_DONOR_MES_POS:-176&EXON_START:865535,A|intron_variant|MODIFIER|AL645608.1|ENSG00000268179|Transcript|ENST00000598827|protein_coding||4/5|ENST00000598827.1:c.186+13C>T|||||||rs370992396|1||-1||SNV|1|Clone_based_ensembl_gene||YES||||ENSP00000471152||M0R0C9|UPI0000D61E05|||||||A:0.0005414||||||||A:0|A:5.221e-05|A:0|A:0.0002887|A:0.001144|A:0|A:0|A:0||||||||||||`

## Third step is to filter out variants by type (loss-of-function, synonymous) to generate their specific SFS
`grep '|synonymous_variant|' synonymous_output3.table > out2.txt`


## Fourth step is to generate the sfs spectrum of the synonmyous variants

### Get the 3rd column of output table
`awk '{print $3}' out2.txt > checkawk.txt`

## Generate SFS using python script
```
temp = np.loadtxt('/home/rahul/PopGen/SimulationSFS/gnomAD_data/checkawk.txt')
temp2 = temp[:].astype(int) # get last column
#sample_size = 4786*2 - 1 Get sample size (2*N) for AJI it is 4786*2 - 1
#get non zeros
temp2 = temp2[np.nonzero(temp2)]
thebins = np.arange(1,sample_size+1)
temphist,bins = np.histogram(temp2, bins=thebins)
```

