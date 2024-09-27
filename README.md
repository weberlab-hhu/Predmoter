# Predmoter 
Predict promoter and enhancer associated next generation sequencing (NGS) data,
Assay for Transposase Accessible Chromatin using sequencing (ATAC-seq) and
histone (H3K4me3) Chromatin immunoprecipitation DNA-sequencing (ChIP-seq),
base-wise for plant species.    
     
Pretrained models can be found at: https://github.com/weberlab-hhu/predmoter_models
(see also [4.5 Inference](#45-inference) if interested in predicting with Predmoter).
[Instructions for code tests](predmoter/test/README.md) are provided.
    
## Table of contents
1. [Disclaimer](#1-disclaimer)
2. [Aim](#2-aim)
3. [Install](#3-install)
   1. [GPU requirements](#31-gpu-requirements)
   2. [Software requirements](#32-software-requirements)
   3. [Installation guide](#3-install)
      1. [Manual installation](#331-manual-installation)
      2. [Docker/Singularity](#332-dockersingularity)
4. [Usage](#4-usage)
    1. [Directories](#41-directories)
    2. [Input files](#42-input-files)
    3. [Training](#43-training)
       1. [Start training](#431-start-training)
       2. [Resume training](#432-resume-training)
       3. [Reproducibility](#433-reproducibility)
    4. [Testing](#44-testing)
    5. [Inference](#45-inference)
       1. [Predicting](#451-predicting)
       2. [Smoothing predictions](#452-smoothing-predictions)
       3. [Prediction results: useful tools](#453-prediction-results-useful-tools)
5. [References](#references)
6. [Citation](#citation)

## 1. Disclaimer
This software is undergoing active testing and development. Build on it at your
own risk.     
    
## 2. Aim
Cross-species prediction of plant promoter/enhancer regions in the DNA with
Deep Neural Networks. ATAC-seq is a technique to assess genome-wide chromatin
accessibility (open chromatin regions) (Buenrostro et al., 2013). Regulatory
elements, like promoters and enhancers, commonly reside in these open chromatin
regions. The histone ChIP-seq data, specifically H3K4me3, which is primarily
present at active genes (Santos-Rosa et al., 2002), is used to give the network
more context, partly improving the ATAC-seq predictions.        
    
## 3. Install
### 3.1 GPU requirements
For realistically sized datasets, a GPU will be necessary for acceptable performance.
(Predictions can be generated on a CPU with the
[available models](https://github.com/weberlab-hhu/predmoter_models), but it will
take around three times longer. The CPU predictions will be the same as the GPU
predictions would be. See [here](docs/performance.md) for more information about
performance.)
   
The example below and all provided models should run on an Nvidia GPU with 11GB
Memory (here a GTX 1080 Ti). The CPU used was E5-2640v4 (Broadwell).   
    
### 3.2 Software requirements
The code was always run on Linux (Ubuntu).   
    
A known conflict leading to the program failing is using the listed Pytorch
Lightning version (1.6.4) and PyTorch 1.12, when loading a pretrained model.
The saved Adam optimizer state conflicts with a new implementation in PyTorch 1.12
compared to 1.11 (https://github.com/pytorch/pytorch/issues/80809).
This is fixed in later versions.     
   
Software versions:   
- **CUDA**: 11.7.1
- **cuDNN**: 8.7.0
- **Python**: 3.8.3
- **PyTorch**: 2.0.0
- **Pytorch Lightning**: 2.0.2
- **Numpy**: 1.23.0
- **H5py**: 3.7.0
- **Numcodecs**: 0.10.0
- **pyBigWig**: 0.3.22 (not usable on Windows)
- **zlib**: 1.2.11 (for pyBigWig)
- **libcurl**: 7.52.1 (for pyBigWig)
   
### 3.3 Installation
#### 3.3.1 Manual installation
For the manual installation see
the [manual installation instructions](docs/manual_install.md).
    
#### 3.3.2 Docker/Singularity
TBA
    
## 4. Usage
>**NOTE**: Please keep the output files, especially the ``predmoter.log`` file,
> as it contains valuable information on setups, models and data you used!
    
For a list of all Predmoter options see the
[detailed description](docs/Predmoter_options.md).     
For detailed information about performance see the
[performance documentation](docs/performance.md).
    
### 4.1 Directories
Predmoter chooses the input files according to the directory name. You need to
provide an input directory (``-i <input_directory>``) that contains these folders:
- **train**: the h5 files used as training set
- **val**: the h5 files used as validation set 
- **test**: the h5 files used as test set, in case you want to validate
using a trained model
    
Predicting is only possible on single files. The parameter ``-f/--filepath`` needs
to be used instead. (see [Predicting](#45-inference))
    
### 4.2 Input files
For more details see the [h5 file documentation](docs/h5_files.md).     
The data used in this project is stored in h5 files. H5 files (also called HDF5,
Hierarchical Data Format 5) are designed to store and organize large amounts of
data. They consist of two major components:   
    
- Datasets (multidimensional arrays)
- Groups (container structures; can hold datasets and other groups)
     
The input files are created using Helixer (https://github.com/weberlab-hhu/Helixer):   
```bash
# create genome h5 file
fasta2h5.py --species <generic_name>_<specific_name> \
--h5-output-path <species>.h5 --fasta-path <path_to_genome>/<species>.fa

# add the NGS (ATAC-/ChIP-seq) coverage
python3 <path_to_Helixer>/helixer/evaluation/add_ngs_coverage.py \
-s <generic_name>_<specific_name> -d <species>.h5 -b *.bam --dataset-prefix <ngs_prefix> \
--unstranded --threads 0
```
    
Input file architecture example (when using Helixer dev branch):
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
    
An example dataset would be "atacseq". Multiple datasets can be added. Poisson
distribution is assumed by Predmoter for all datasets, as it is designed for
ATAC- and ChIP-seq data.    
   
> **NOTE**: The input data is cut into chunks. The standard sequence length
> of these chunks is 21384 base pairs. Predmoter accepts other sequence lengths
> as long as they are divisible by the chosen step (stride) to the power of the
> chosen number of CNN layers. 
    
### 4.3 Training
#### 4.3.1 Start training
Training of Predmoter is of right now deterministic and reproducible. This is
achieved by using a seed. If no seed is provided Predmoter will choose a random
seed. The seed state is saved each epoch and included in the saved model, so
training can be resumed with the last seed state. Of course the same input data
needs to be provided as well, the input files are always alphabetically sorted
by Predmoter. (helpful: [performance documentation](docs/performance.md))

```bash
# start training  with default parameters
Predmoter.py -i <input_directory> -o <output_directory> -m train
# optional: --prefix <prefix> and all model configurations (cnn-layers, step, etc.)

# start training with custom setup
Predmoter.py -i <input_directory> -o <output_directory> -m train --seed 1827 \
--model-type bi-hybrid --lstm-layers 2 --cnn-layers 3 -b 120 -e 60 --patience 10 \
--datasets atacseq h3k4me3
```
   
Three model types are possible: a CNN, a CNN + LSTM (hybrid) and a CNN +
bidirectional LSTM (bi-hybrid). The model can be trained using different
dataset combinations. All model hyperparameters (including the datasets and
their order), callbacks (that are supposed to get saved), optimizer state,
and the weights and biases get saved to a model file named:
``predmoter_epoch=<epoch>_<checkpoint_metric>=<value>.ckpt``. The last
checkpoint saved is also copied to ``last.ckpt``. If `` --save-top-k`` is
-1 (default), models get saved after each epoch. If it is 3 the top three
models according to the checkpoint metric tracked are saved.   
     
> Hint: The prefix test can be confusing because test_metrics.log is the default
> log file name for the test metrics.    
      
**Outputs:**
```raw
<output_directory>  
| 
├── <prefix>_predmoter.log
|
├── <prefix>_metrics.log
|
|
└── <prefix>_checkpoints
     └── model checkpoint files
```
   
The prefix (``--prefix``) is useful, to separate multiple different setups
in one output folder. The checkpoint directory will always be in the output
directory under <prefix>_checkpoints. This directory should also be empty at
the start of the training unless the training is resumed. The ``predmoter.log``
file saves information about the model setup (model summary), input files,
datasets used and duration of the training. The ``metrics.log`` file keeps track
of the models' metrics per epoch; current metrics are: avg_val_loss,
avg_val_accuracy, avg_train_loss and avg_train_accuracy. The loss is the
Poisson negative log likelihood loss and the "accuracy" the Pearson correlation
coefficient.   
    
#### 4.3.2 Resume training
>**NOTE**: Do not move your log files, change the prefix if one was chosen
> or select a different input or output directory when you resume the training.
```bash
Predmoter.py -i <input_directory> -o <output_directory> -m train \
--model <model_checkpoint> -resume-training -e <epochs> -b <batch_size>
```
    
If no model checkpoint is given Predmoter will search for the model
``output_directory/<prefix>_checkpoints/last.ckpt``.
>**NOTE:** The epochs are max_epochs. So, if your model already trained for
> 12 epochs, you define ``-e 15`` and you give it the last checkpoint as input,
> it will only train for 3 more epochs and **not** for an additional 15!!
    
## 4.3.3 Reproducibility
Predmoter is reproducible to a point. When you choose the exact same setup (including
input data) and train for 3 epochs or for 2 epochs and then resume for 1 epoch, the
results (metrics) will be identical. Setups known to screw with the reproducibility
are switching between devices (CPU, GPU), changing the number of workers/devices or
using different hardware than before.
     
>**Warning**: From the [LSTM Pytorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html):   
>There are known non-determinism issues for RNN functions on some versions
> of cuDNN and CUDA. You can enforce deterministic behavior by setting the
> following environment variables:
>   
>On CUDA 10.1, set environment variable ``CUDA_LAUNCH_BLOCKING=1``.
> This may affect performance.   
>   
>On CUDA 10.2 or later, set environment variable (note the leading colon symbol)
> ``CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:2.``  
>    
> See the [cuDNN 8 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/release-notes/rel_8.html)
> for more information.
>   
>Issues with non-deterministic behavior of LSTMs weren't encountered so far
> during training of Predmoter.
    
### 4.4 Testing
Testing will be applied to all h5 files individually in ``<input_directory>/test``.    
When testing on the CPU, the results very slightly differ from the GPU results
(differences occurred after the third/fourth decimal place).    
**Outputs:**
```raw
<output_directory>  
| 
├── <prefix>_predmoter.log
|
└── <prefix>_test_metrics.log
```
The logging information will be appended to ``<prefix>_predmoter.log`` if it already
exists.   
If the model just used one dataset the metrics file will contain: the h5 file name,
the ``<dataset>_avg_val_loss`` and the ``<dataset>_avg_val_accuracy``. Otherwise,
it contains: the h5 file name, the total_avg_val_loss and accuracy and the
avg_val_loss and accuracy for each dataset. If the given file does not contain
target data of a given dataset the results will be NaN.   
    
**Command:**   
```bash
Predmoter.py -i <input_directory> -o <output_directory> -m test \
--model <model_checkpoint> -b <batch_size>
# optional: --prefix <prefix>
```
     
### 4.5 Inference
> **IMPORTANT**: Models trained on a GPU can be used to generate predictions on the
> CPU. The CPU predictions will be the same as the GPU predictions would be.
> Predicting on the CPU will take longer.    
> (*Side Note*: Since the results slightly differ when testing on the CPU and the
> network's predictions are rounded to integers, it's likely that there are slight
> differences in CPU and GPU predictions that, due to the rounding, do **not**
> affect the final results.)
    
#### 4.5.1 Predicting
Predictions will be applied to an individual fasta or h5 file only.
> **NOTE**: The ATAC-seq input data was shifted (+4 bp on "+" strand and -5 bp on
> "-" strand per read), so predictions are as well. If a fasta file is used the chosen
> ``--subsequence-length``, default 21384 base pairs, needs to be divisible by the 
> model's step to the power of the number of CNN layers. This condition is fulfilled
> by the provided models (https://github.com/weberlab-hhu/predmoter_models) and the
> default subsequence length. The batch size depends on the capacity of the CPU/GPU.
> The default batch size is 120. If the error ``RuntimeError:CUDA out of memory.``
> or other memory errors occur, try setting a lower batch size.
    
**Command:**   
```bash
# predict and convert h5 output file to bigWig file directly
Predmoter.py -f <input_file> -o <output_directory> -m predict \
--model <model_checkpoint> -b <batch_size> -of bigwig (--species <species>)
# optional: --prefix <prefix>, species is required for a fasta input file

# convert h5 output file after prediction
convert2coverage.py -i <predictions.h5> -o <output_directory> -of bigwig \
--basename <basename>
# optional: --prefix <prefix>
```
    
**Outputs:**
```raw
<output_directory>  
| 
├── <prefix>_predmoter.log
|
├── <input_file_basename>_<dataset>_avg_strand.bw/bg.gz
|
└── <input_file_basename>_prediction.h5
```
    
The h5 predictions file can be converted to bigWig or bedGraph files.
The result is one file per dataset predicted. Training and predictions are
done on both + and - strand (essentially providing built-in data augmentation),
as ATAC- and ChIP-seq data is usually 'unstranded' (open chromatin applies to
both strands). BigWig/bedGraph files are also non-strand-specific, so the average
of the predictions for + and - strand are calculated. If just + or - strand is
preferred, convert the h5 file later on using convert2coverage.py.    
The logging information will be appended to ``<prefix>_predmoter.log`` if it
already exists.   
      
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
   
The seqids, start_ends and species arrays are taken from the input file.
The predictions array contains the models' predictions. The metadata contains
the models' name (including the path), as a reminder if you ever loose the log file,
and the datasets the model predicted in the correct order (e.g., "atacseq", "h3k4me3").
    
> **Warning:** Predmoter can be used on Windows by excluding the pyBigWig in the
> requirements and commenting out the ``import pyBigWig`` in 
> ``predmoter.utilities.converter``, as pyBigWig is only available on Linux.
> The only coverage file output possible would then be bedGraph files.
    

#### 4.5.2 Smoothing predictions
Sometimes it can be useful to smooth Predmoter's predictions (see
https://github.com/weberlab-hhu/predmoter_models Section 2. Model performance).
The smoothing of the raw predictions from the h5 file is applied via a
'rolling mean' with a given window size. The smoothing is only applied per
chromosome. Since reading all predictions for a large chromosome into RAM is,
depending on the hardware, not feasible, the 'rolling mean' is always applied
to a subsequence of a given chromosome. The length of this subsequence is defined
as the 'maximum values allowed to be loaded into RAM', here 21,384,000 base
pairs. The last subsequence is usually shorter. Smoothing can only be used with
convert2coverage.py when converting from h5 to bigWig or bedGraph files.
    
**Command:**   
```bash
# convert h5 output file after prediction with smoothing
convert2coverage.py -i <predictions.h5> -o <output_directory> -of bigwig \
--basename <basename> --window-size 150  # the window size is an example
# optional: --prefix <prefix>
```
    
#### 4.5.3 Prediction results: useful tools
Another helpful option to convert bigWig to bedGraph files and vice versa
on Linux is using the binaries from [UCSC Genome Browser](http://hgdownload.soe.ucsc.edu/admin/exe/).    
    
```bash
# make the binary executable
chmod 764 bedGraphToBigWig  # converts bedGraph to bigwig

# execute the binaries
./bedGraphToBigWig in.bedGraph chrom.sizes out.bw
./bigWigToBedGraph in.bigWig out.bedGraph
```

It is also possible to call peaks from the bedGraph track with MACS (Zhang et al. 2008):
```bash
# ATAC-seq peak calling example
macs3 bdgpeakcall -i <atac_seq.bg> -o <output.bed> --min-length 100 --max-gap 20 --cutoff 25
# since the Predmoter bedgraph files are coverage tracks the cutoff needs to be higher
# than the MACS default which assumes MACS scores

# ChIP-seq peak calling example
macs3 bdgbroadcall -i <chip_seq.bg> -o <output.bed> -c 35 -C 15 -G 500
```
     
## References
Buenrostro, J. D., Giresi, P. G., Zaba, L. C., Chang, H. Y., & Greenleaf,
W. J. (2013). Transposition of native chromatin for fast and sensitive
epigenomic profiling of open chromatin, DNA-binding proteins and nucleosome
position. Nature Methods, 10(12), 1213–1218. https://doi.org/10.1038/nmeth.2688    
     
Santos-Rosa, H., Schneider, R., Bannister, A. J., Sherriff, J., Bernstein, B. E.,
Emre, N. C. T., Schreiber, S. L., Mellor, J., & Kouzarides, T. (2002).
Active genes are tri-methylated at K4 of histone H3. Nature, 419(6905), 407–411.
https://doi.org/10.1038/nature01080   

Zhang, Y., Liu, T., Meyer, C. A., Eeckhoute, J., Johnson, D. S.,
Bernstein, B. E., Nussbaum, C., Myers, R. M., Brown, M., Li, W., Shirley, X. S. (2008).
Model-Based Analysis of ChIP-Seq (MACS). Genome Biology, 9(9) , 1–9.
https://doi.org/10.1186/GB-2008-9-9-R137
    
## Citation
Kindel, F., Triesch, S., Schlüter, U., Randarevitch, L.A., Reichel-Deland, V.,
Weber, A.P.M., Denton, A.K. (2024) Predmoter—cross-species prediction of plant
promoter and enhancer regions. Bioinformatics Advances, 4(1), vbae074.
https://doi.org/10.1093/bioadv/vbae074