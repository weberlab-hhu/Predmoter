# Predmoter 
Predict ATAC- and ChIP-seq (H3K4me3) read coverage per base pair for plant species.   
    
## Disclaimer
This software is undergoing active testing and development. Build on it at your own risk.   
    
## Aim
Cross-species prediction of plant promoter/enhancer regions in the DNA with Deep Neural Networks. ATAC-seq is a technique to assess genome-wide chromatin accessibility (open chromatin regions). Regulatory elements, like promoters and enhancers, commonly reside in these open chromatin regions. The histone ChIP-seq data is used to give the network more context, improving the ATAC-seq predictions.   
    
## Install
### GPU requirements
For realistically sized datasets, a GPU will be necessary for acceptable performance.   
   
The example below and all provided models should run on an Nvidia GPU with 11GB Memory (here a GTX 1080 Ti). The CPU used was E5-2640v4 (Broadwell).   
    
### Software requirements
The code was always run on Linux (Ubuntu).   
   
Parts of the code were also tested with Pytorch Lightning version 1.8.6.   
A known conflict leading to the program failing is using the listed Pytorch Lightning version (1.6.4) and PyTorch 1.12, when loading a pretrained model. The saved Adam optimizer state conflicts with a new implementation in PyTorch 1.12 compared to 1.11 (https://github.com/pytorch/pytorch/issues/80809). If this is fixed in the later versions of these two packages is unknown.     
   
Sofware versions:   
- **CUDA**: 11.2.2
- **cuDNN**: 8.1.1
- **Python**: 3.8.3
- **PyTorch**: 1.11.0
- **Pytorch Lightning**: 1.6.4
- **Numpy**: 1.23.0
- **H5py**: 3.7.0
- **Numcodecs**: 0.10.0
   
### Installation guide
#### Code
First download the code from GitHub.
```bash
# from a directory of your choice
git clone https://github.com/weberlab-hhu/Predmoter.git

```
   
#### Virtual environment (optional)
We recommend installing all the python packages in a virtual environment: https://docs.python-guide.org/dev/virtualenvs/

For example, create and activate an environment called "env":
```bash
python3 -m venv env
source env/bin/activate

# deactivation
deactivate
```
    
#### Dependencies of Predmoter
``` bash
# from the Helixer directory
pip install -r requirements.txt
```
    
## Usage
>**NOTE**: Please keep the output files, especially the ``predmoter.log`` file, as it contains valuable information on setups, models and data you used!!
    
### Directories
Predmoter chooses the input files according to the directory name. You need to provide an input directory (``-i <input_directory>``) that contains these folders:
- **train**: the h5 files used as training set
- **val**: the h5 files used as validation set
- **test**: the h5 files used as test set, in case you want to validate using a trained model
- **predict**: the h5 files used as prediction set, in case you want to predict using a trained model
    
### Input files
The data used in this project is stored in h5 files. H5 files (also called HDF5, Hierarchical Data Format 5) are designed to store and organize large amounts of data. They constist of two major components:   
    
- Datasets (multidimensional arrays)
- Groups (container structures; can hold datasets and other groups)
     
The input files are created using Helixer (https://github.com/weberlab-hhu/Helixer):   
```bash
# create genome h5 file
python3 <path_to_helixer>/HelixerPrep/fasta2h5.py --species <generic_name>_<specific_name> \
--h5-output-path <species>.h5 --fasta-path <path_to_genome>/<species>.fa

# add the NGS (ATAC-/ChIP-seq) coverage
python3 <path_to_helixer>/HelixerPrep/helixer/evaluation/add_ngs_coverage.py \
-s <generic_name>_<specific_name> -d <species>.h5 -b *.bam --dataset-prefix <ngs_prefix> \
--unstranded --threads 0

# the bam files are: mapped, sorted, indexed, cleaned, deduplicated, and quality controlled
```
    
Input file architecture example:
```raw
<file>.h5  
| 
├── <dataset>_meta
|     └── bam_files
|
|
├── data  
|     ├── X (encoded DNA sequence)
|     ├── seqids (chromosome/scaffold names)
|     ├── species
|     └── starts_ends
|
└── evaluation
      ├── <dataset>_coverage (in reads per base pair)
      └── <dataset>_spliced_coverage
```
    
An example dataset would be "atacseq". Multiple datasets can be added.   
   
> **NOTE**: The input data is cut into chunks. The standard sequence length of these chunks is 21384 base pairs. Predmoter accepts other sequence lengths as long as they are divisible by the chosen step (stride) to the power of the chosen number of CNN layers. 
    
### Training
#### Start training
Training of Predmoter is of right now deterministic and reproducible. This is achieved by using a seed. If no seed is provided Predmoter will choose a random seed between 0 and 2000. The seed state is saved each epoch and included in the saved model, so training can be s´resumed with the last seed state.

```bash
# start training  with default parameters
python3 <path_to_predmoter>/main.py -i <input_directory> -o <output_directory> -m train
# optional: --prefix <prefix>

# start training with custom setup
python3 <path_to_predmoter>/main.py -i <input_directory> -o <output_directory> -m train --seed 1827 \
--model-type bi-hybrid --lstm-layers 2 --cnn-layers 3 -b 120 -e 60 --patience 10 --datasets atacseq h3k4me3
```
   
Three model types are possible: a CNN, a CNN + LSTM (hybrid) and a CNN + bidirectional LSTM (bi-hybrid). The model can be trained using different dataset combinations. All model hyperparameters (including the datasets and their order), callbacks (that are supposed to get saved), optimizer state, and the weights and biases get saved to a model file named: ``predmoter_epoch=<epoch>_<checkpoint_metric>=<value>.ckpt``. The last checkpoint saved is also copied to ``last.ckpt``. If `` --save-top-k`` is -1 (default), models get saved after each epoch. If it is 3 the top three models according to the checkpoint metric tracked are saved.   
   
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
   
The prefix (``--prefix``) is useful, to seperate multiple different setups in one output folder. The checkpoint directory will always be in the output directory under checkpoints unless otherwise specified (which is not recommended). This directory should also be empty at the start of the training. The ``predmoter.log`` file saves information about the model setup (model summray), used files, used datasets and duration of each epoch. The ``metrics.log`` file keeps track o the models metrics per epoch; current metrics are: avg_val_loss, avg_val_accuracy, avg_train_loss and avg_train_accuracy. The loss is the Poisson negative log likelihood loss and the "accuracy" the Pearson correlation coefficient.   
    
#### Resume training 
>**NOTE**: Do not move your log files, change the prefix if one was chosen or select a different input or output directory when you resume the training.
```bash
python3 <path_to_predmoter>/main.py -i <input_directory> -o <output_directory> -m train \
--model <model_checkpoint> -resume-training -e <epochs> -b <batch_size>
```
    
If the model checkpoint given is just the file name without the file path, Predmoter will search for the model in ``output_directory/<prefix>_checkpoints``. If that is desired, remember to use the same prefix as before.  
>**NOTE:** The epochs are max_epochs. So, if your model already trained for 12 epochs, you define ``-e 15`` and you give it the last checkpoint as input, it will only train for 3 more epochs and **not** for an additional 15!
    
### Validation/Testing
Validation will be applied to all h5 files individually in input_directory/test.    
**Outputs:**
```raw
<output_directory>  
| 
├── <prefix>_predmoter.log
|
└── <prefix>_val_metrics.log
```
The logging information will be appended to ``<prefix>_predmoter.log`` if it already exists.   
If the model just used one dataset the metrics file will contain: the h5 file name, the ``<dataset>_avg_val_loss`` and the ``<dataset>_avg_val_accuracy``. Otherwise it contains: the h5 file name, the total_avg_val_loss and accuracy and the avg_val_loss and accuracy for each dataset. If the given file does not contain target data of a given dataset the results will be NaN.   
    
**Command:**   
```bash
python3 <path_to_predmoter>/main.py -i <input_directory> -o <output_directory> -m validate \
--model <model_checkpoint> --test-batch-size <test_batch_size>
# optional: --prefix <prefix>
```
     
### Predicting
Predictions will be done to all h5 files individually in input_directory/predict.   
**Outputs:**
```raw
<output_directory>  
| 
├── <prefix>_predmoter.log
|
└── <prefix>_<input_file_name>_prediction.h5
```
   
The logging information will be appended to ``<prefix>_predmoter.log`` if it already exists.   
The prediction h5 file contains:   
```raw
<prefix>_<input_file_name>_prediction.h5
|
├── data  
|     ├── seqids
|     └── species
|
└── prediction
      ├── predictions
      └── prediction_meta
            ├── model_name
            └── datasets
```
   
The seqids and species arrays are taken from the input file. The predictions array contains the models' predictions. The meta data contains the models' name (including the path), as a reminder if you ever loose the log file, and the datasets the model predicted in the correct order (e.g., "atacseq", "h3k4me3").   
      