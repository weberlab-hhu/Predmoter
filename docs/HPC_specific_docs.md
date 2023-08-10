# HPC specific docs
These instructions are specific to working with the high performance cluster (HPC)
of the Heinrich Heine University in Düsseldorf (HHU). The documentation is intended
for members of the Weberlab institute to help work with this and similar
projects on the HPC. The HPC lacks internet access since 2020, resulting in the need
for a few workarounds.
      
**Tool:**
- RNAsleek 0.2.0 (Weberlab: Institute of Plant Biochemistry, HHU: GitHub)    
     
The other tools required are installed on the HPC and will be loaded automatically
by the scripts. A [list of the other tools](data_preprocessing.md) won't be presented
here.

## 1. Mount directory
Mounting a directory is useful and required when moving large amounts of data, like
sequencing data, to and from the HPC. The mounted directory gets treated like another
folder on your local machine and copying data this way you won't slow down the HPC
log in nodes.
```bash
# mount directory
mkdir <mount_directory>
sshfs <username>@<ip>:<directory_to_mount> <mount_directory>

# mount HPC directory
sshfs <username>@storage.hpc.rz.uni-duesseldorf.de:<directory_to_mount> <mount_directory>

# unmount, if mount is broken
fusermount -u <mount_directory>
```
     
## 2. Virtual environment
- add ``PIP_CONFIG_FILE=/software/python/pip.conf`` to your .bashrc
- execute ``module load Python/3.8.3``; you can also use another available Python
version, check for them with ``module avail Python``
- execute ``module load intel`` first before installing python packages needing a C
interpreter like pysam
- if you want to install a repository, download it first to your local machine
and then copy the folder to your HPC home directory

> **Note:** When using the virtual environment **always** load Python first
> (``module load Python/3.8.3``) before activating the environment.
    
```bash
# create an environment
python3 -m venv <name_of_your_vitual_environment>

# activate virtual environment
source <name_of_your_vitual_environment>/bin/activate

# install package
pip install wheel  # wheel always needs to be installed first!

# install github repository
cd <repository>
pip install -r requirements.txt
pip install .
```
     
> **WARNING:** If your requirements.txt contains lines like:
``git+https://github.com/janinamass/dustdas@master``, it will not work on
> the HPC, so those requirements should be commented out, downloaded on a local
> machine and then transferred to the HPC and installed like any other repository.
> Also delete those requirements from the ``install_requires`` and/or
> ``dependency_links`` list in ``setup.py``.

## 3. Data preprocessing using RNAsleek
RNAsleek is originally a semi-automated pipeline for processing public RNA-seq
samples on the HPC. It can also be used for other NGS-data now. Additional scripts
and script changes specific to ATAC- and ChIP-seq are generated using the low level
script ``setup_qsubs.py`` in ``side_scripts``.    
    
> **WARNING:** Expected virtual environment (env) names, locations and/or branches by
> the qsub files:
>- **Helixer:** env: ``<hpc_home_directory/helix_venv>``
>- **deepTools:** env: ``<hpc_home_directory>/deeptools_env``
>- **RNAsleek:** repository branch: flexi
>- **HPC-Project:** HelixerOpt
     
### 3.0. Setup job files
**Tools:** RNAsleek (flexi branch), **helper_scripts:** setup_qsubs.py  
**Info:** It's best to do this step on a local machine and not on the HPC.
     
### 3.1. RNAsleek
RNAsleek installation: see https://github.com/weberlab-hhu/RNAsleek/tree/flexi.   
   
```bash
# Directory setup and creation of bash scripts for each fastq file in SraRunInfo.csv
python3 <path_to_RNAsleek>/rnasleek.py -d <project_directory> \
-s <project_directory>/SraRunInfo.csv \
-c <project_directory>/config.ini setup

# Example directory name: Athaliana for Arabidopsis thaliana
````
The SraRunInfo.csv can be downloaded from SRA by selecting the desired accessions
and sending them to the SraRunInfo.csv file. It is necessary to just select accessions
of one species to be in the same file. The config.ini is a text file specifying which
job files should be written. An example file for *A. thaliana* and ATAC-seq looks
like this:   
```bash
# ordered sections for each job in the pipeline
# section names must match job names with the ending 'Job' removed
[Filters]
scientific_name = Arabidopsis thaliana
taxid = 3702
library_strategy = ATAC-seq

[Fetch]

[Trimming]
# todo install_directory = $HOME/extra_programs/Trimmomatic-0.36/
# all adapters must be in $install_directory/adapters/
# todo ILLUMINACLIP = TruSeq3-SE.fa:2:30:10
# todo MAXINFO = 36:0.7
[Fastqc]

[BWA]
sp = Athaliana

[Flagstat]

[MarkDuplicates]
```
   
The library strategy needs to be replaced with ChIP-Seq for the ChIP-seq experiments.
Common errors include:
- The library specified in the SraRunInfo.csv does not match the filter library as
it was named Tn-Seq instead of ATAC-seq or ChIP-seq instead of ChIP-Seq. Either edit
the SraRunInfo.csv or the config.ini file.
- The SRS sample names are identical for multiple replicates. If they are technical
replicates, this can be ignored. If they are biological replicates or different
experiments, change the sample names for example to their SRR accession in the
SraRunInfo.csv.  
- Generally something in your config.ini isn't lining up with the name/accession in
the SraRunInfo.csv (e.g. taxid, library strategy, etc.)
                     
Required programs/directories in your home directory on the HPC:
- ``extra_programs/Trimmomatic-0.36`` &rarr; inside: trimmomatic-0.36.jar
- ``extra_programs/Trimmomatic-0.36/adapters`` &rarr; inside: Illumina adapter
fasta files
- ``extra_programs/FastQC``
- ``bin/fastqc`` &rarr; symlink to ``extra_programs/FastQC``    
       
Required paths in your .bashrc (in your home directory on the HPC):    
- <pre>export PATH=$PATH:$HOME/bin</pre>
- <pre>export PATH=$PATH:$HOME/.local/bin</pre>
- <pre>export PATH=$PATH:$HOME/Helixer/HelixerPost/target/release (for later)</pre>
     
    
#### 3.2. Setup genome
If you haven't setup indexes for the species/program (e.g. BWA) yet, run:   
```bash
rnasleek -d <project_directory> -s <project_directory>/<SraRunInfo.csv> \
-c <project_directory>/<config.ini> genome \
-f <genome.fa> -g <genome.gff> -s <species_name>
```
If your genome doesn't have an annotation, just create an empty gff file. The
species name Athaliana would result in the setup of a genome_prep.qsub and
genome_prep.sh as well as copying the fasta and gff files into a genome directory
(will be created, if it doesn't exist already).  
```raw
atacseq (super_folder)
|
├── genomes
|     └── Athaliana
|            ├── Athaliana.fa
|            └── Athaliana.gff
|
└── Athaliana (project_directory)
```
    
#### 3.3. Setup extra qsubs
Afterwards ``setup_qsubs.py`` is needed (works with the same virtual
environment/requirements as RNAsleek). These extra files are specifically to quality
control ATAC- and ChIP-seq data, as well as adjusting some RNAsleek files. One script,
``helixer_pred.qsub``, assumes the helixer model path is
``$HOME/Helixer/helixer_models/land_plant_v0.3_a_0080.h5``.   
    
```bash
python3 setup_qsubs.py -d <project_directory> -c <project_directory>/<config.ini> \
--dset <dataset_prefix> --ref --usr-name <hpc_user_name>
# example dataset prefix: atacseq
# --ref if you want to add the reference annotation to the final h5 file
# (if there is none, don't choose it)
```
   
The project directory and config.ini are the same as before. Only one dataset prefix
can be given. It is recommended to have the different datasets in different super
folders.
    
#### 3.4. Check jobs
After each job run:   
```bash
python3 <path_to_RNAsleek>/rnasleek.py -d <project_directory> -s <project_directory>/SraRunInfo.csv \
-c <project_directory>/config.ini check
```
This checks if the jobs created by RNAsleek ran successfully. If the program hasn't
setup job files to help rerun them and the expected job output files are not reported
missing in the ``output_report.txt``, the jobs were run successfully. If the output
files are not there check the error files in logs. This doesn't check the additional
jobs to infer quality control and creation of the h5 files.     
     
Common errors are:    
- ran out of walltime, solution: increase walltime
- ran out of memory, solution: increase memory
       
#### 3.5. Final directory structure
Here an example of the final directory structure (just one example species is depicted):   
    
>**Note:** Please execute all qsub scripts from your species folder.
> So change your directory to  for example ``atacseq/Athaliana`` and execute a
> script like so: ``qsub qsubs/trimmomatic.qsub``.
     
```raw
atacseq (super_folder)
|
├── genomes
|     └── Athaliana
|            ├── Athaliana.fa
|            ├── assembly_data_report.jsonl (genome information; should be included manually)
|            └── Athaliana.gff
|
└── Athaliana (project_directory, execute qsubs from here)
      ├── deduplicated
      ├── deepqc
      ├── fastqc
      ├── fastqs
      ├── flagstat
      ├── h5
      ├── heixer_pred
      ├── logs 
      ├── mapped
      ├── qsubs
      ├── scripts
      ├── trimmed
      ├── SraRunInfo.csv
      ├── config.ini
      └── sample_ids.txt
``` 
     
### 3.6. Pipeline
Following the exact order of these qsubs (left to right and up to down) results in
h5 files created from public NGS data (fastq files).

| Category | Download data |      Trimming       | Quality control |           Mapping            | Quality control | Deduplication/Data cleaning |                                                      Quality control                                                       |              H5 file creation              |
|:--------:|:-------------:|:-------------------:|:---------------:|:----------------------------:|:---------------:|:---------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------:|
|  Tools   |  SRA Toolkit  | Trimmomatic<br>Java |     FastQC      |       BWA<br>SamTools        |    SamTools     | Picard<br>Java<br>SamTools  |                                              deepTools<br>Helixer<br>gffread                                               |                  Helixer                   |
|  qsubs   |       /       |  trimmomatic.qsub   |   fastqc.qsub   | genome_prep.qsub<br>bwa.qsub |  flagstat.qsub  |    mark_duplicates.qsub     | plotCoverage.qsub<br> plotFingerprint.qsub<br>tobigwig.qsub<br>helixer_pred.qsub<br>computeMatrix.qsub<br>plotHeatmap.qsub | create_fasta_h5.qsub<br>add_"dataset".qsub |
     
Additionally, the helixer or reference annotations can be added to the final h5
file using Geenuff and the scripts: ``sqlite.qsub`` and ``add_helixer.qsub``, and/or
``ref_sqlite.qsub`` and ``add_reference.qsub``.   
    
### 3.7. Download data     
Downloading the data can only be done on a local machine and not on the HPC. 
When first configuring the SRA toolkit, be sure to set the local
file caching to a folder of sufficient size, so **not** your home
directory (see https://github.com/ncbi/sra-tools/wiki/05.-Toolkit-Configuration).
An example bash script (multi_fetch.sh) to fetch data automatically is:
```bash
#!/bin/bash

source $HOME/.bashrc
mydir="<path_to_superfolder>/atacseq"
organisms="Athaliana Mtruncatula Osativa Zmays"

for org in $organisms; do
        cd $mydir/$org
        ls scripts/fetchS* |xargs -n1 -P2 -I% bash % 2> fetch.err 1> fetch.out
done

# Activate sleek_venv (RNAsleek) before and use ./multi_fetch.sh &
# to do before starting: make sure the path to sra-toolkit
# (<path_to>/sratoolkit.3.0.0-ubuntu64/bin) is in your .bashrc 
```
This script can be run in the background, so that, as long as your PC is on and has
a stable internet connection, the fetch jobs keep running.
```bash
bash multi_fetch.sh &

disown <job_id>
```
The job_id will be printed in the terminal when you send the script into the
background. Example: ``[1] 55423``. So the command would be ``disown 55423``.
     
**Special case:** single cell ATAC-seq
There is no way (yet) to choose/detect if the data is single cell ATAC-seq, so you
need to adjust this manually in the fetch scripts:
```bash
prefetch <SRR_accsession>
fasterq-dump <SRR_accsession> -S --include-technical

gzip <SRR_accsession>_3.fastq <SRR_accsession>_4.fastq
```
Unzipped fastq files also work, as trimmomatic accepts them as well.
It is however necessary to rename the forward read from <SRR_accsession>_3.fastq to
<SRR_accsession>_1.fastq and the reverse reads from <SRR_accsession>_4.fastq to
<SRR_accsession>_2.fastq.
    
## Hints
- ATAC-seq data mostly uses Nextera adapters, so replace the standard
TruSeq3-PE-2.fa with NexteraPE-PE.fa (side_scripts/replace_adapters.sh).
- Check the html output files/multiqc output files to determine adapter content,
base quality, etc. If you didn't trimm the correct adapters repeat trimming and
fastqc.
- Multiqc (from RNAsleek docs):
```bash
rnasleek -d <project_directory> -s <RunInfo_file> -c <config.ini> multiqc
cd <project_directory>/multiqc/
multiqc .  # html output is more detailed than fastqc
cd ../multiqc_untrimmed
multiqc .
cd ../..

# pdf output just for RNA-seq data
python <path_to>/RNAsleek/viz/summarizer.py <project_directory> <RunInfo_file> \
-o <output.pdf>
```
- When mapping to a large genome, adding more threads/cpus to bwa helps:
``bwa-mem2 mem -t4`` and ``#PBS -l select=1:ncpus=4:mem=52gb``
- Deduplication common errors:
  - the logged error file just ends without signaling that picard finished,
  there is no output bam file and walltime was not the issue
  - java.lang.OutOfMemoryError: Java heap space
  - java.lang.OutOfMemoryError: GC overhead limit exceeded
    
  The **solution** is extending the java heap space, so replacing
-Xmx1G with -Xmx2G or even -Xmx3G.
