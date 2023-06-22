# Predmoter 
Predict ATAC- and ChIP-seq (H3K4me3) read coverage per base pair for plant species.   
    
## Disclaimer
This software is undergoing active testing and development. Build on it at your
own risk.   
    
## Aim
Cross-species prediction of plant promoter/enhancer regions in the DNA with
Deep Neural Networks. ATAC-seq is a technique to assess genome-wide chromatin
accessibility (open chromatin regions) (Buenrostro et al., 2013). Regulatory
elements, like promoters and enhancers, commonly reside in these open chromatin
regions. The histone ChIP-seq data, specifically H3K4me3, which is primarily
present at active genes (Santos-Rosa et al., 2002), is used to give the network
more context, improving the ATAC-seq predictions.   
    
## Install
### GPU requirements
For realistically sized datasets, a GPU will be necessary for acceptable performance.   
   
The example below and all provided models should run on an Nvidia GPU with 11GB
Memory (here a GTX 1080 Ti). The CPU used was E5-2640v4 (Broadwell).   
    
### Software requirements
The code was always run on Linux (Ubuntu).   
    
A known conflict leading to the program failing is using the listed Pytorch
Lightning version (1.6.4) and PyTorch 1.12, when loading a pretrained model.
The saved Adam optimizer state conflicts with a new implementation in PyTorch 1.12
compared to 1.11 (https://github.com/pytorch/pytorch/issues/80809).
This is fixed in the later versions.     
   
Sofware versions:   
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
   
### Installation guide
#### Manual installation
For the manual installation see
the [manual installation instructions](docs/manual_install.md).
    
#### Docker/Singularity
TBA
    
## Usage
>**NOTE**: Please keep the output files, especially the ``predmoter.log`` file,
> as it contains valuable information on setups, models and data you used!
    
### Directories
Predmoter chooses the input files according to the directory name. You need to
provide an input directory (``-i <input_directory>``) that contains these folders:
- **train**: the h5 files used as training set
- **val**: the h5 files used as validation set 
- **test**: the h5 files used as test set, in case you want to validate
using a trained model
    
Predicting is only possible on single files. The parameter ``-f/--filepath`` needs
to be used instead.

    
### Input files
For more details see the [h5 file documentation](docs/h5_files.md).     
The data used in this project is stored in h5 files. H5 files (also called HDF5,
Hierarchical Data Format 5) are designed to store and organize large amounts of
data. They consist of two major components:   
    
- Datasets (multidimensional arrays)
- Groups (container structures; can hold datasets and other groups)
     
The input files are created using Helixer (https://github.com/weberlab-hhu/Helixer):   
```bash
# create genome h5 file
python3 fasta2h5.py --species <generic_name>_<specific_name> \
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
    
### Training
#### Start training
Training of Predmoter is of right now deterministic and reproducible. This is
achieved by using a seed. If no seed is provided Predmoter will choose a random
seed. The seed state is saved each epoch and included in the saved model, so
training can be resumed with the last seed state. Of course the same input data
needs to be provided as well, the input files are always alphabetically sorted
by Predmoter.    
    
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
>Issues with non-deterministic behaiviour of LSTMs weren't encountered so far
> during training of Predmoter.

```bash
# start training  with default parameters
python3 Predmoter.py -i <input_directory> -o <output_directory> -m train
# optional: --prefix <prefix> and all model configurations (cnn-layers, step, etc.)

# start training with custom setup
python3 Predmoter.py -i <input_directory> -o <output_directory> -m train --seed 1827 \
--model-type bi-hybrid --lstm-layers 2 --cnn-layers 3 -b 120 -e 60 --patience 10 --datasets atacseq h3k4me3
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
    
#### Resume training 
>**NOTE**: Do not move your log files, change the prefix if one was chosen
> or select a different input or output directory when you resume the training.
```bash
python3 Predmoter.py -i <input_directory> -o <output_directory> -m train \
--model <model_checkpoint> -resume-training -e <epochs> -b <batch_size>
```
    
If the model checkpoint given is just the file name without the file path,
Predmoter will search for the model ``output_directory/<prefix>_checkpoints/last.ckpt``.
>**NOTE:** The epochs are max_epochs. So, if your model already trained for
> 12 epochs, you define ``-e 15`` and you give it the last checkpoint as input,
> it will only train for 3 more epochs and **not** for an additional 15!!
    
### Testing
Validation will be applied to all h5 files individually in input_directory/test.    
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
python3 Predmoter.py -i <input_directory> -o <output_directory> -m test \
--model <model_checkpoint> --test-batch-size <test_batch_size>
# optional: --prefix <prefix>
```
     
### Predicting
Predictions will be applied to an individual files only.   
> **NOTE**: The ATAC-seq input data was shifted (+4 bp on "+" strand and -5 bp on
> "-" strand per read), so predictions are as well.
    
**Outputs:**
```raw
<output_directory>  
| 
├── <prefix>_predmoter.log
|
└── <input_file_basename>_prediction.h5
```
    
Optionally the h5 predictions file can be converted to bigwig or bedgraph files.
The result is one file per dataset predicted. Training and predictions are
done on both + and - strand (essentially providing built-in data augmentation),
as ATAC- and ChIP-seq data is usually 'unstranded' (open chromatin applies to
both strands). Bigwig/bedgraph files are also non-strand-specific, so the average
of the predictions for + and - strand are calculated. The file naming convention is:
<basename_of_input_file>_dataset_avg_strand.bw/bg.gz. If just + or - strand is
preferred, convert the h5 file later on using convert2coverage.py.
     
**Command:**   
```bash
python3 Predmoter.py -f <input_file> -o <output_directory> -m predict \
--model <model_checkpoint> -pb <predict_batch_size> 
# optional: --prefix <prefix>, -of <output_format>
```
   
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
   
    
Predmoter can be used on Windows by excluding the pyBigWig in the requirements and
commenting out the ``import pyBigWig`` in ``predmoter.utilities.converter``.
The only coverage file output possible would then be bedgraph files. A conversion to
bigwig is only possible on Linux either directly via Predmoter from the predictions
h5 file or via the binaries from UCSC Genome Browser
(http://hgdownload.soe.ucsc.edu/admin/exe/).    
    
```bash
# make the binary executable
chmod 764 bedGraphToBigWig

# execute the binary
./bedGraphToBigWig in.bedGraph chrom.sizes out.bw
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
