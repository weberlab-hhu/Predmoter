import os
import argparse
from configobj import ConfigObj
# consider logging results instead of should be empty


def main(directory, config, dset, usr_name, ref):
    print("Note: This is just an additional helper script to use RNAsleek for ATAC- and ChIP-seq data, "
          "as it was originally designed for RNA-seq data. Please run RNAsleek BEFORE this script, as it "
          "depends on some outputs of RNAsleek.")

    config = ConfigObj(config)
    sci_spec = config["Filters"]["scientific_name"].replace(" ", "_")

    # e.g Arabidopsis thaliana -> Athaliana
    # Pyrus x bretschneideri -> Pbretschneideri
    spec = sci_spec[0] + sci_spec.split("_")[-1]

    # number of samples/threads
    with open(f"{directory}/sample_ids.txt", "r") as f:
        lines = f.readlines()
        threads = len(lines)

    log_start = """## Log-File setup
export LOGFILE=$PBS_O_WORKDIR/logs/$PBS_JOBNAME"."$PBS_JOBID".log"
echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" START" > $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE"""

    log_end = """## and more logging
qstat -f $PBS_JOBID >> $LOGFILE

echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" END" >> $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE"""

    redirect_err_out = "2> logs/$PBS_JOBNAME.e${PBS_JOBID%\[*} 1> logs/$PBS_JOBNAME.o${PBS_JOBID%\[*}"

    print("Setting up additional qsubs ...")

    # part 1: plotCoverage
    with open(f"{directory}/qsubs/plotCoverage.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=3:mem=500mb
#PBS -l walltime=5:59:00
#PBS -A HelixerOpt
#PBS -N plotCoverage
#PBS -e should_be_empty.err
#PBS -o deduplicated/coverage.txt

module load Python/3.8.3
source $HOME/deeptools_env/bin/activate

cd $PBS_O_WORKDIR

{log_start}

files=`ls deduplicated/*.bam`
$HOME/deepTools/bin/plotCoverage --numberOfProcessors 3 -b $files \\
--plotFile deduplicated/{spec}Coverage.png --smartLabels \\
--outRawCounts deduplicated/raw.txt {redirect_err_out}

{log_end}"""
        f.write(text)

    # part 2: plotFingerprint
    with open(f"{directory}/qsubs/plotFingerprint.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=3:mem=450mb
#PBS -l walltime=5:59:00
#PBS -A HelixerOpt
#PBS -N plotFingerprint
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

module load Python/3.8.3
source $HOME/deeptools_env/bin/activate

cd $PBS_O_WORKDIR

{log_start}

files=`ls deduplicated/*.bam`
$HOME/deepTools/bin/plotFingerprint --numberOfProcessors 3 -b $files --plotFile deduplicated/{spec}Fingerprint.png \\
--smartLabels --binSize 500 --numberOfSamples 500_000 {redirect_err_out}

# if you have a genome equal or smaller than 250 Mbp,
# please adjust the defaults of binsize (500) and numberOfSamples (500_000),
# so that binsize x numberOfSamples < genome size

{log_end}"""
        f.write(text)

    # part 3: to bigwig
    with open(f"{directory}/qsubs/tobigwig.qsub", "w") as f:
        # f-string, would mess up arrays --> piece text together
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:mem=750mb
#PBS -l walltime=4:59:00
#PBS -A HelixerOpt
#PBS -N to_bigwig
#PBS -J1-{threads}
#PBS -r y
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

module load Python/3.8.3
source $HOME/deeptools_env/bin/activate

cd $PBS_O_WORKDIR

{log_start}
\n""" + """array1=(deduplicated/*.bam)
array2=( "${array[@]##*/}" )  # strip off directory names
array2=( "${array2[@]%.bam}" ) # strip off extensions

# write to files
printf "%s\\n" "${array[@]}" > deepqc/bam_files.txt
printf "%s\\n" "${array2[@]}" > deepqc/basenames.txt

# PBS arrays
file=`sed $PBS_ARRAY_INDEX"q;d" deepqc/bam_files.txt`
basename=`sed $PBS_ARRAY_INDEX"q;d" deepqc/basenames.txt`

$HOME/deepTools/bin/bamCoverage -b $file -o deepqc/$basename.bw -of bigwig \\
2> logs/$PBS_JOBNAME.e${PBS_JOBID%\[*}.$PBS_ARRAY_INDEX 1> logs/$PBS_JOBNAME.o${PBS_JOBID%\[*}.$PBS_ARRAY_INDEX
\n""" + f"{log_end}"
        f.write(text)

    # part 4: helixer prediction
    with open(f"{directory}/qsubs/helixer_pred.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=15gb
#PBS -l walltime=23:55:00
#PBS -A "HelixerOpt"
#PBS -N helixer_pred
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out


module load Python/3.8.3
module load CUDA/11.2.2
module load cuDNN/8.1.1
module load HDF/5-1.8.12
source $HOME/.bashrc
source $HOME/helix_venv/bin/activate


cd $PBS_O_WORKDIR

{log_start}

Helixer.py --fasta-path ../genomes/{spec}/{spec}.fa \\
--species {sci_spec} --gff-output-path helixer_pred/{spec}_land_plant_v0.3_a_0080_helixer.gff3 \\
--model-filepath $HOME/Helixer/helixer_models/land_plant_v0.3_a_0080.h5 --batch-size 150 --subsequence-length 21384 \\
--temporary-dir /gpfs/scratch/{usr_name} {redirect_err_out}

{log_end}"""
        f.write(text)

    # part 5: computeMatrix
    with open(f"{directory}/qsubs/computeMatrix.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=5:59:00
#PBS -A HelixerOpt
#PBS -N computeMatrix
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

module load Python/3.8.3
source $HOME/deeptools_env/bin/activate
module load Cufflinks/2.2.1

cd $PBS_O_WORKDIR

{log_start}

# convert to bed
gffread helixer_pred/{spec}_land_plant_v0.3_a_0080_helixer.gff3 \\
--bed -o deepqc/{spec}_helixer.bed

# compute matrix
files=`ls deepqc/*.bw`
$HOME/deepTools/bin/computeMatrix reference-point -S $files -R deepqc/{spec}_helixer.bed --smartLabels \\
-o deepqc/{spec}_matrix.mat.gz --upstream 1500 --downstream 1500 {redirect_err_out}

{log_end}"""
        f.write(text)

    # part 6: plotHeatmap
    with open(f"{directory}/qsubs/plotHeatmap.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:mem=5gb
#PBS -l walltime=1:55:00
#PBS -A HelixerOpt
#PBS -N plotHeatmap
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

module load Python/3.8.3
source $HOME/deeptools_env/bin/activate

cd $PBS_O_WORKDIR

{log_start}

$HOME/deepTools/bin/plotHeatmap -m deepqc/{spec}_matrix.mat.gz -o deepqc/{spec}_tss_heatmap.png \\
{redirect_err_out}

{log_end}"""
        f.write(text)

    # part 7: create h5 fasta
    with open(f"{directory}/qsubs/create_fasta_h5.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=23:55:00
#PBS -A "HelixerOpt"
#PBS -N h5_fasta
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

cd $PBS_O_WORKDIR
module load Python/3.8.3
source $HOME/helix_venv/bin/activate

{log_start}

fasta2h5.py --species {sci_spec} --h5-output-path h5/fasta/{spec}.h5 \
--fasta-path ../genomes/{spec}/{spec}.fa {redirect_err_out}

{log_end}"""
        f.write(text)

    # part 8: add ngs dataset
    extra = f"--shift {redirect_err_out}" if dset == "atacseq" else f"{redirect_err_out}"
    prefix = "chipseq" if dset == "h3k4me3" else dset
    # if more datasets are added, replace prefix with dset

    with open(f"{directory}/qsubs/add_{prefix}.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus={threads}:mem=6gb
#PBS -l walltime=47:55:00
#PBS -A "HelixerOpt"
#PBS -N h5_{prefix}
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

cd $PBS_O_WORKDIR
module load Python/3.8.3
source $HOME/helix_venv/bin/activate

{log_start}

cp h5/fasta/{spec}.h5 h5
python3 $HOME/Helixer/Helixer/helixer/evaluation/add_ngs_coverage.py -s {sci_spec} -d h5/{spec}.h5 \\
-b deduplicated/*.bam --dataset-prefix {dset} --unstranded --threads {threads} {extra}

{log_end}"""
        f.write(text)

    # part 9: sqlite
    with open(f"{directory}/qsubs/sqlite.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:mem=3500mb
#PBS -l walltime=1:55:00
#PBS -A "HelixerOpt"
#PBS -N helixer_geenuffDB
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

module load Python/3.8.3
source $HOME/helix_venv/bin/activate

cd $PBS_O_WORKDIR

{log_start}

import2geenuff.py --gff3 helixer_pred/{spec}_land_plant_v0.3_a_0080_helixer.gff3 \\
--fasta ../genomes/{spec}/{spec}.fa --species {sci_spec} \\
--db-path helixer_pred/{spec}_helixer_GEENUFF.sqlite3 --log-file helixer_pred/output.log \\
{redirect_err_out}

{log_end}"""
        f.write(text)

    # part 10: add helixer annotation
    with open(f"{directory}/qsubs/add_helixer.qsub", "w") as f:
        text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=120:55:00
#PBS -A "HelixerOpt"
#PBS -N add_helixer_annotation
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

module load Python/3.8.3
source $HOME/helix_venv/bin/activate

cd $PBS_O_WORKDIR

{log_start}

geenuff2h5.py --h5-output-path h5/{spec}.h5 --input-db-path helixer_pred/{spec}_helixer_GEENUFF.sqlite3 \\
--add-additional helixer_post --no-multiprocess --modes y,anno_meta,transitions {redirect_err_out}

{log_end}"""
        f.write(text)

    if ref:
        # part 11: reference sqlite
        with open(f"{directory}/qsubs/ref_sqlite.qsub", "w") as f:
            text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:mem=3500mb
#PBS -l walltime=1:55:00
#PBS -A "HelixerOpt"
#PBS -N ref_geenuffDB
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

module load Python/3.8.3
source $HOME/helix_venv/bin/activate

cd $PBS_O_WORKDIR

{log_start}

import2geenuff.py --gff3 ../genomes/{spec}/{spec}.gff \\
--fasta ../genomes/{spec}/{spec}.fa --species {sci_spec} \\
--db-path helixer_pred/{spec}_ref_GEENUFF.sqlite3 --log-file helixer_pred/ref_output.log {redirect_err_out}

{log_end}"""
            f.write(text)

        # part 12: add reference
        with open(f"{directory}/qsubs/add_reference.qsub", "w") as f:
            text = f"""#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=48:55:00
#PBS -A "HelixerOpt"
#PBS -N add_ref_annotation
#PBS -e should_be_empty.err
#PBS -o should_be_empty.out

module load Python/3.8.3
source $HOME/helix_venv/bin/activate

cd $PBS_O_WORKDIR

{log_start}

geenuff2h5.py --h5-output-path h5/{spec}.h5 --input-db-path helixer_pred/{spec}_ref_GEENUFF.sqlite3 \\
--add-additional reference --no-multiprocess --modes y,anno_meta,transitions {redirect_err_out}

{log_end}"""
            f.write(text)

    print("Changing bwa and mark_duplicates qsub and script files ...")

    # part 1: replace old sam flags in every mark_duplicates file
    for file in os.listdir(f"{directory}/scripts"):
        if "mark_duplicates" in file:
            with open(f"{directory}/scripts/{file}", "r") as f:
                text = f.read()

            text = text.replace(" O=deduplicated/S", " O=deduplicated/marked/S")
            text = text.replace("-F 1024 deduplicated/", "-F 1796 deduplicated/marked/")

            with open(f"{directory}/scripts/{file}", "w") as f:
                f.write(text)

        # part 1.2 bwa threads
        if "bwa" in file:
            with open(f"{directory}/scripts/{file}", "r") as f:
                text = f.read()

            text = text.replace("bwa-mem2 mem ", "bwa-mem2 mem -t 4 ")

            with open(f"{directory}/scripts/{file}", "w") as f:
                f.write(text)

    # part 2: change qsubs
    for file in os.listdir(f"{directory}/qsubs"):
        if "bwa" in file:
            with open(f"{directory}/qsubs/{file}", "r") as f:
                text = f.read()

            text = text.replace(":ncpus=1:mem=2500mb", ":ncpus=4:mem=6gb")

            with open(f"{directory}/qsubs/{file}", "w") as f:
                f.write(text)

        if "mark_duplicates" in file:
            with open(f"{directory}/qsubs/{file}", "r") as f:
                text = f.read()

            text = text.replace(":mem=1200mb", ":mem=3200mb")

            with open(f"{directory}/qsubs/{file}", "w") as f:
                f.write(text)

    print("Adding subdirectories ...")

    os.makedirs(f"{directory}/deduplicated/marked", exist_ok=True)
    os.makedirs(f"{directory}/deepqc", exist_ok=True)
    os.makedirs(f"{directory}/helixer_pred", exist_ok=True)
    os.makedirs(f"{directory}/h5", exist_ok=True)
    os.makedirs(f"{directory}/h5/fasta", exist_ok=True)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Setup qsubs",
                                     description="Setup additional qsub files needed for ATAC/ChIP-seq.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--directory", type=str, default=None, required=True,
                        help="qsub files are written to <directory>/qsubs; "
                             "some scripts in <directory>/scripts are changed")
    parser.add_argument("-c", "--config", type=str, default=None, required=True,
                        help="config.ini, the same type used by RNAsleek; CAUTION: library types need to "
                             "be written like this to match the conventions on NCBI's SRA: ATAC-seq, ChIP-Seq")
    parser.add_argument("--dset", type=str, default=None, required=True,
                        help="dataset name/prefix for later h5 file creation, e.g. atacseq, h3k4me3, ...")
    parser.add_argument("--usr-name", type=str, required=True, help="HPC user name")
    parser.add_argument("--ref", action="store_true",
                        help="should the qsubs for adding the reference annotation be added")
    args = parser.parse_args()

    args.directory = args.directory.rstrip("/")

    dict_args = vars(args)

    main(**dict_args)
