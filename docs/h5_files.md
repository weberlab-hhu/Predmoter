# H5 files
Predmoter uses h5 files as input and main output data. H5 files, 
Hierarchical Data Format 5 (HDF5), are designed to store and organize large
amounts of data. They consist of two major components:   
    
- Datasets (multidimensional arrays)
- Groups (container structures; can hold datasets and other groups)
     
## 1. H5 file architecture
### 1. Input files
The architecture of the h5 files used as input data for Predmoter looks like this:
```raw
<file>.h5  
| 
|
├── data  
|     ├── X (encoded DNA sequence)
|     ├── seqids (chromosome/scaffold names)
|     ├── species
|     └── start_ends (start and end base per chunk)
|
└── evaluation
      ├── <dataset>_meta
      |     └── bam_files
      ├── <dataset>_coverage (in reads per base pair)
      └── <dataset>_spliced_coverage
```
    
#### 1.1. The data group
The base h5 file and all arrays of the data group is created using ``fasta2h5.py``.
     
##### 1.1.1. X
The encoded DNA sequence. The original fasta file is cut into chunks of the same
size (default: 21384 base pairs). The shape of this array is (N, C, 4). The C
represents the chunk/sequence length and N the number of chunks. The four represents
the nucleotides/bases (example shape: (11202, 21384, 4)). The nucleotide encoding
is as follows:
- [1, 0, 0, 0] = C
- [0, 1, 0, 0] = A
- [0, 0, 1, 0] = T
- [0, 0, 0, 1] = G
- [0.25, 0.25, 0.25, 0.25] = N
- [0, 0, 0, 0] = padding

Padding is used to pad out too short sequences or chromosome ends, since chromosomes
are rarely exactly divisible by the sequence length. Arrays always need to have
compatible shapes to be concatenated, so the sequence length needs to be identical
between all chunks. Shorter sequences will therefore be padded. Predmoter considers
Ns padding as well (so [0, 0, 0, 0]), as most Ns are gaps in the genome assembly
and as such not informative.

##### 1.1.2 seqids
The ``seqids`` represent the fasta file headers/identifiers of the sequence
(chromosome, scaffold, contig, etc.) that a chunk comes from. The shape is (N,), i.e.
the number of chunks.

##### 1.1.3. species
The species ID given by the user. The shape is (N,), i.e. the number of chunks.
Technically useful if multiple species are in one file. Predmoter assumes that
there is just one species per file, which is especially important for testing and
predicting. It should be possible to use multiple species in one file during
training, but it wasn't tested.

##### 1.1.4. start_ends
The start and end coordinates of a given chunk (in that order). When the end
coordinate is smaller than the start coordinate (e.g. [21384, 0]), then the
chunk is from the negative strand. So [0, 21384] is the first 21384 bp of a given
sequence of the positive strand. On the positive strand start is inclusive and
end exclusive (it's the other way around on the negative strand). If
``|start - end|`` is less than the chunk length then there is padding. The shape is
(N, 2), i.e. two values (start and end) per chunk.
      
#### Examples
##### 1. Chunk order
The DNA sequence order inside the h5 files is: chromosome 1-positive strand,
chromosome 1-negative strand, chromosome 2-positive strand,
chromosome 2-negative strand, etc.

Example of one chromosome:
- sequence length: 21384 bp
- length of chromosome 1: 51321 bp
     
|           |            |                |                |                |                |            |
|:----------|:-----------|:---------------|:---------------|:---------------|:---------------|:-----------|
| chunk     | 0          | 1              | 2              | 3              | 4              | 5          |
| strand    | +          | +              | +              | -              | -              | -          |
| start/end | [0, 21384] | [21384, 42768] | [42768, 51321] | [51321, 42768] | [42768, 21384] | [21384, 0] |

All strands are represented from 5' to 3', so the negative strand is in descending
order. Chunks 2 and 3 contain padding and chunk 3 is the reverse complement of chunk 2.     
      
##### 2. Padding
bases =  A, T, C, G, N    
padding = P  
(both encoded inside the file)     
      
positive strand:

|            |                 |                 |                 |
|:-----------|:----------------|:----------------|:----------------|
| chunk      | 0               | 1               | 2               |
| mini_array | [A, A, T, G, A] | [C, A, T, N, C] | [G, T, T, P, P] |
| start/end  | [0, 5]          | [5, 10]         | [10, 13]        |
     
negative strand:

|            |                 |                 |                 |
|:-----------|:----------------|:----------------|:----------------|
| chunk      | 3               | 4               | 5               |
| mini_array | [A, A, C, P, P] | [G, N, A, T, G] | [T, C, A, T, T] |
| start/end  | [13, 10]        | [10, 5]         | [5, 0]          |

The padding is on the positive and negative strand always at the end of the array,
so chunk 2 and 3 are **not** exact mirrors/reverse complements of each other, while
for example chunk 0 and 5 are.
     
#### 1.2 The evaluation group
All arrays of the evaluation group are added to the base h5 file using
``add_ngs_coverage.py``. Multiple datasets can be added to the h5 file.
      
##### 1.2.1 dataset_meta
The meta of a dataset contains the array ``bam_files``. This is a list of the bam file
names (including the path) that were used to add the dataset. The shape is (B,), i.e.
the number of bam files. Multiple bam files can be added per dataset, but it is not
possible to add additional bam files to an existing dataset. In that case,
``add_ngs_coverage`` needs to be used on the desired h5 file not containing the dataset
and all bam files (old and new) can be added together.
     
##### 1.2.2. dataset_coverage
Coverage is the number of reads mapping at a given position (cigar M or X). The
shape is (N, C, B). The C represents the chunk/sequence length, N the number of
chunks, and B the number of bam files (example shape: (11202, 21384, 5)). Predmoter
uses the average of all bam files per dataset to train.
      
##### 1.2.3 dataset_spliced_coverage
Spliced coverage is the number of reads mapping with a gap, that is a read deletion
or spliced out section at a given position (cigar N or D). The shape is the same as
the shape of dataset_coverage.
     
### 2. Prediction file
The prediction h5 file contains:   
```raw
<input_file_basename>_prediction.h5
|
├── data  
|     ├── seqids
|     ├── start_ends
|     └── species
|
└── prediction
      ├── predictions
      ├── model_name
      └── datasets
```
   
The ``seqids``, ``start_ends`` and ``species`` arrays are taken from the input file.
The predictions array contains the models' predictions. The shape is (N, C, D);
the C represents the chunk/sequence length, N the number of chunks and D the number
of datasets predicted. The metadata contains the models' filename (including the path),
and the datasets the model predicted in the correct order (e.g., "atacseq", "h3k4me3").
The order of datasets is very important, as it is used by the Converter to convert each
dataset into its own bigwig or bedgraph file. The Converter also needs ``seqids`` and
``start_ends`` to calculate chromosome order and chromosome lengths.
    
## 2. H5 file creation
**Tools:** Helixer, GeenuFF (optional), Predmoter
(Stiehler et al., 2021; Weberlab: Institute of Plant Biochemistry, HHU: GitHub, n.d.)     
     
ATAC- and ChIP-seq data is PCR amplified. Therefore, you can not determine from which
strand which read originated unless you used barcodes to get strand specific data.
Hence, the coverage information of the read is always added to both strands. The
advantages are:
- Open chromatin and closed chromatin regions always apply to both strands anyway.
- The addition to both strands allows for built-in data augmentation.
     
### 2.1. Create h5 file from fasta file
```bash
fasta2h5.py --species <full_species_name> --h5-output-path <species/filename>.h5 \
--fasta-path <path_to_genome>/<species>.fa
# full species name example: Arabidopsis_thaliana
```

> **H5 base file:** To add annotations or datasets, the fasta h5 file needs to be
> created **first**. The additional data will then be added to the fasta h5 file.
    
### 2.2. Add ATAC- and/or ChIP-seq data
Multiple datasets should be added separately. The shift argument is a special case
for ATAC-seq data to shift the reads +4 (+ strand) or -5 (- strand) base pairs as it is
typically done.
```bash
python3 <path_to_helixer>/helixer/evaluation/add_ngs_coverage.py \
-s <full_species_name> -d <species>.h5 -b <path_to_bam_files>/*.bam \
--dataset-prefix <ngs_prefix> --unstranded --threads <number_of_bam_files_to_add> \
(--shift)
# multiple threads are faster, but --threads 0 also works
# ngs prefix example: atacseq
# --unstranded only for unstranded data
```
    
### 2.3. Add annotation(s) (optional)
To add the annotation to the h5 file, they need to be converted from gff/ggf3 to a
sqlite file using GeenuFF. If helixer and reference are desired one of them needs to
be added using ``--add-additional <annotation_name>`` (helixer is the better option;
suggested annotation name: helixer_post).
[More details about the annotation data](https://github.com/weberlab-hhu/Helixer/blob/main/docs/h5_data.md)
```bash
# to sqlite
import2geenuff.py --gff3 <helixer_annotation>.gff3 --fasta <species>.fa \
--species <full_species_name> --db-path <helixer_geenuff>.sqlite3 \
--log-file output.log
# --gff3 <reference>.gff to convert the reference annotation

# add annotation
geenuff2h5.py --h5-output-path <species>.h5 --input-db-path <helixer_geenuff>.sqlite3 \
--no-multiprocess --modes y,anno_meta,transitions
# modes: don't add X again when using add_additional
```
    
Resulting new file structure (when adding the reference annotation not in alternative
and adding the helixer annotation to alternative):
```raw
<file>.h5  
| 
|
├── data  
|     ├── X (encoded DNA sequence)
|     ├── seqids (chromosome/scaffold names)
|     ├── species
|     ├── start_ends (start and end base per chunk)
|     ├── err_samples
|     ├── fully_intergenic_samples
|     ├── gene_lengths
|     ├── is_annotated
|     ├── phases
|     ├── sample_weights
|     ├── transitions
|     └── y
|
├── evaluation
|     ├── <dataset>_meta
|     |     └── bam_files
|     ├── <dataset>_coverage (in reads per base pair)
|     └── <dataset>_spliced_coverage
|     
└── alternative
      └── helixer_post
           ├── err_samples
           ├── fully_intergenic_samples
           ├── gene_lengths
           ├── is_annotated
           ├── phases
           ├── sample_weights
           ├── transitions
           └── y
```
     
**Common errors:**  
- GeenuFF is very strict, so frequently converting the annotation to sqlite fails and
the reference gff needs to be edited slightly for it to be converted.
- The full species names don't match up between script calls.
     
## References
- Stiehler, F., Steinborn, M., Scholz, S., Dey, D., Weber, A. P. M., & Denton,
A. K. (2021). Helixer: cross-species gene annotation of large eukaryotic genomes
using deep learning. Bioinformatics, 36(22–23), 5291–5298.
https://doi.org/10.1093/BIOINFORMATICS/BTAA1044
- SRA Toolkit - GitHub. (n.d.). Retrieved September 19, 2022,
from https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit
- Weberlab: Institute of Plant Biochemistry, HHU: GitHub. (n.d.). Retrieved
May 23, 2022, from https://github.com/weberlab-hhu