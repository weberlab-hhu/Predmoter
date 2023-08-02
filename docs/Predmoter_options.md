# Predmoter options
## Main Program
### 1. General/Main options
| Argument  | Explanation                                       |
|:----------|:--------------------------------------------------|
| --help/-h | show help messages and exit                       |
| --version | show Predmoter version and exit                   |
| --mode/-m | **required**, valid modes: train, test or predict |
### 2. Data input/output parameters
| Argument            | Explanation                                                                                                                                                                                                                                                                                                                | Default           |
|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|
| --input-dir/-i      | containing one train and one val directory for training and/or a test directory for testing (directories must contain h5 files)                                                                                                                                                                                            | current directory |
| --output-dir/-o     | output: log file(s), checkpoint directory with model checkpoints (if training), predictions (if predicting)                                                                                                                                                                                                                | current directory |
| --filepath/-f       | input file to predict on, either h5 (or fasta file, in development)                                                                                                                                                                                                                                                        | /                 |
| --output-format/-of | output format for predictions, if unspecified will output no additional files besides the h5 file (valid: bigwig (bw), bedgraph (bg)); the file naming convention is: ``<basename_of_input_file>_dataset_avg_strand.bw/bg.gz``, if just + or - strand is preferred, convert the h5 file later on using convert2coverage.py | /                 |
| --prefix            | prefix for log files and the checkpoint directory                                                                                                                                                                                                                                                                          | /                 |
### 3. Configuration parameters
| Argument          | Explanation                                                                                                                                                           | Default                                                                          |
|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
| --resume-training | add argument to resume training                                                                                                                                       | /                                                                                |
| --model           | model checkpoint file for predicting, testing or resuming training (provide full path)                                                                                | ``<outdir>/<prefix>_checkpoints/last.ckpt`` (only if resume-training, else None) |
| --datasets        | the dataset prefix(es) to use; are overwritten by the model checkpoint's dataset prefix(es) if one is chosen (in case of resuming training, testing, predicting)      | atacseq, h3k4me3 (list)                                                          |
| --ram-efficient   | if true will use RAM efficient data class (see [documentation about performance](performance.md)), **Warning:** Don't move the input data while Predmoter is running. | True                                                                             |
### 4. Model parameters
| Argument            | Explanation                                                                                                                                                                                                                  | Default   |
|:--------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|
| --model-type        | the type of model to train, valid types are: cnn, hybrid (CNN + LSTM), bi-hybrid (CNN + bidirectional LSTM)                                                                                                                  | bi-hybrid |
| --cnn-layers        | number of convolutional layers (e.g. 3 layers: 3 layer convolution and 3 layer deconvolution                                                                                                                                 | 1         |
| --filter-size       | filter size for convolution, is scaled up per layer after the first by ``up``                                                                                                                                                | 64        |
| --kernel-size       | kernel size for convolution                                                                                                                                                                                                  | 9         |
| --step              | stride for convolution                                                                                                                                                                                                       | 2         |
| --up                | multiplier used for up-scaling the convolutional filter per layer after the first                                                                                                                                            | 2         |
| --dilation          | dilation should be kept at the default, as it isn't useful for sequence data                                                                                                                                                 | 1         |
| --lstm-layers       | LSTM layers                                                                                                                                                                                                                  | 1         |
| --hidden-size       | LSTM units per layer                                                                                                                                                                                                         | 128       |
| --bnorm             | add a batch normalization layer after each convolutional layer                                                                                                                                                               | True      |
| --dropout           | adds a dropout layer with the specified dropout value (between 0. and 1.) after each LSTM layer except the last; if it is 0. no dropout layers are added; if there is just one LSTM layer specifying dropout will do nothing | 0.        |
| --learning-rate/-lr | learning rate for training (default recommended)                                                                                                                                                                             | 0.001     |
### 5.Trainer/callback parameters
| Argument        | Explanation                                                                                                                                                                      | Default          |
|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| --seed          | for reproducibility, if not provided will be chosen randomly                                                                                                                     | /                |
| --ckpt-quantity | quantity to monitor for checkpoints; loss: poisson negative log likelihood, accuracy: Pearson's r (valid: avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy)    | avg_val_accuracy |
| --save-top-k    | saves the top k (e.g. 3) models; -1 means every model gets saved; the last model checkpoint saved is named ``last.ckpt``                                                         | -1               |
| --stop-quantity | quantity to monitor for early stopping; loss: poisson negative log likelihood, accuracy: Pearson's r (valid: avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy) | avg_train_loss   |
| --patience      | allowed epochs without the stop quantity improving before stopping training                                                                                                      | 5                |
| --batch-size/-b | batch size for training, validation, test or prediction sets                                                                                                                     | 120              |
| --device        | device to train on                                                                                                                                                               | gpu              |
| --num-devices   | number of devices to train on (see [documentation about performance](performance.md)) , devices have to be on the same machine (leave at default for test/predict)               | 1                |
| --num-workers   | how many subprocesses to use for data loading                                                                                                                                    | 0                |
| --epochs/-e     | number of training runs; **Attention:** max_epochs, so when training for 2 epochs and then resuming training (``--resume-training``) for 4 additional epochs, you need ``-e 6``  | 5                |

## Convert to coverage
| Argument            | Explanation                                                                                         | Default           |
|:--------------------|:----------------------------------------------------------------------------------------------------|:------------------|
| --input-file/-i     | **required**, input h5 predictions file (Predmoter output)                                          | /                 |
| --output-dir/-o     | output directory for all converted files                                                            | current directory |
| --output-format/-of | output format for predictions (valid: bigwig (bw), bedgraph (bg))                                   | bigwig            |
| --basename          | **required**, basename of the output files, naming convention: ``basename_dataset_strand.bw/bg.gz`` | /                 |
| --strand            | if not specified the average of both strands is used, else + or - can be selected                   | avg               |
