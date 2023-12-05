# Testdata <a id="testdata"></a>
The artificial test data was created using
[a jupyter notebook](testdata_creation.ipynb).
The fake DNA sequence was cut into subsequences of 21384 bp.  Both strands, plus and
minus, were created. The artificial test data contain artificial but coherent
``start_ends``. They are also necessary to create coherent predictions files. The
last file ``pred_data_2.h5`` has the subsequence length 42768 bp. Some files have
artificial special cases to test:
- Padding: As few chromosomes, scaffolds or contigs were divisible by this number,
  sequence ends as well as short sequences were padded with the vector
  ``[0., 0., 0., 0.]``. Padded base pairs are masked during training. Non-divisible
  chromosomes were artificially created for the test cases.
- Gaps: If a subsequence only contained N bases, here referred to as “gap
  subsequence”, it is filtered out during training.
- Ns: N bases not stretching over an entire subsequence. These aren't filtered out.
- Flagged sequences (blacklisting): Unplaced scaffolds and non-nuclear sequences
  can be flagged and filtered out. Adding this option, for every h5 file containing
  the array ``data/blacklist``, the "blacklisted" sequences are filtered out during
  training and/or testing.

The files ``train_data_2.h5``, ``train_data_4.h5``and ``train_data_5.h5`` come with
a corresponding sequence report file ``<file>_sequence_report.jsonl``, which can be
used to create the ``data/blacklist`` dataframe for ``train_data_4.h5`` and
``train_data_5.h5`` is used to test adding this dataframe to ``train_data_2.h5`` via
[add_blacklist.py](../../side_scripts/add_blacklist.py).

     
| File           | Artificial dataset prefixes | Subsequences            | Artificial special cases                                                             |
|:---------------|:----------------------------|:------------------------|:-------------------------------------------------------------------------------------|
| train_data_1   | atacseq, h3k4me3            | 18                      | padded bases, Ns                                                                     |
| train_data_2   | atacseq, h3k4me3, moaseq    | 16                      | padded bases, Ns, 1 gap, (2 flagged sequence IDs not used to add ``data/blacklist``) |
| train_data_3   | moaseq                      | 6                       | /                                                                                    |
| train_data_4   | atacseq, h3k4me3            | 14 (8 after filtering)  | padded bases, 3 flagged sequence IDs                                                 |
| train_data_5   | atacseq                     | 20 (14 after filtering) | padded bases, Ns, 1 gap, 3 flagged sequence IDs (1 in between "valid" chromosomes)   |
| val_data_1     | atacseq, h3k4me3            | 8                       | /                                                                                    |
| val_data_2     | atacseq, h3k4me3, moaseq    | 4                       | padded bases                                                                         |
| val_data_3     | atacseq                     | 8                       | padded bases, 1 gap                                                                  |
| predict_data_1 | /                           | 16                      | padded bases, 1 gap                                                                  |
| predict_data_2 | /                           | 10                      | padded bases, 1 gap                                                                  |
    
Scripts for testing Predmoter will follow soon. In the meantime there are 5 major
tests that can be performed. The most important test is the [third](#3-predicting).
    
# Tests <a id="tests"></a>
## 1. Training <a id="1-training"></a>
```bash
# create necessary directory structure
mkdir <input_directory>/train
mkdir <input_directory>/val

# copy the test data
cp <path_to_predmoter>/Predmoter/predmoter/testdata/train_data* <input_directory>/train
cp <path_to_predmoter>/Predmoter/predmoter/testdata/val_data* <input_directory>/val

# train with default parameters
Predmoter.py -m train -i <input_directory> -o <output_directory> -e 2 -b 10
# optional:
# --seed <seed> --prefix <prefix> --device [gpu or cpu] [model configurations]

# test resume training
Predmoter.py -m train -i <input_directory> -o <output_directory> -e 4 \
-b 10 --resume-training
```
The output should be two files: ``<prefix_>predmoter.log``, containing important
information about the training procedure, and ``<prefix_>metrics.log``, containing the
metrics (avg_val_loss, avg_val_accuracy, avg_train_loss, avg_train_accuracy) per
epoch.
> The metrics are just test cases and don't reflect the actual results when
> using non-artificial data. If any memory issues/errors occur, consider downsizing
> the network, e.g., ``--hidden-size 32`` and ``--filter-size 32``. The same prefix
> (when a prefix was chosen before) and output directory need to be selected when
> resuming training
> (see [here](https://github.com/weberlab-hhu/Predmoter#432-resume-training) for
> more information).
    
## 2. Testing <a id="2-testing"></a>
```bash
# create necessary directory structure
mkdir <input_directory>/test

# copy the test data
cp <path_to_predmoter>/Predmoter/predmoter/testdata/train_data* <input_directory>/test
cp <path_to_predmoter>/Predmoter/predmoter/testdata/val_data* <input_directory>/test

# test
Predmoter.py -m test -i <input_directory> -o <output_directory> -b 10 --model <model>
```
The model used for testing the test mode of Predmoter can either be a model generated
when testing the train mode of Predmoter or use one of the available
[pretrained models](https://github.com/weberlab-hhu/predmoter_models). The output
should be two files: ``<prefix_>predmoter.log``, containing important
information about the testing procedure, and ``<prefix_>test_metrics.log``, containing
the avg_val_loss and avg_val_accuracy per dataset the chosen model can predict,
if the model can predict multiple datasets the total loss and accuracy are also
included.
    
### 3. Predicting <a id="3-predicting"></a>
```bash
Predmoter.py -m predict \
-f <path_to_predmoter>/Predmoter/predmoter/testdata/pred_data.fa \
--species Species_artificialis -o <output_directory> -b 10 --model <model>

# test prediction on a h5 file
Predmoter.py -m predict \
-f <path_to_predmoter>/Predmoter/predmoter/testdata/pred_data_1.h5 \
-o <output_directory> -b 10 --model <model>
# using pred_data_2.h5 is also possible
```
Again, the model used for testing the test mode of Predmoter can either be a model
generated when testing the train mode of Predmoter or use one of the available
[pretrained models](https://github.com/weberlab-hhu/predmoter_models). The output
should be two files: ``<prefix_>predmoter.log``, containing important
information about the prediction procedure, and ``<input_file_basename>_predictions.h5``
(see [here](https://github.com/weberlab-hhu/Predmoter#45-inference) for more
information). Optionally, a bigWig or bedGraph file will be created per dataset
when using the option ``-of [bg or bw]``.
    
## 4. Conversion <a id="4-conversion"></a>
```bash
convert2coverage.py -i <output_directory>/pred_data_predictions.h5 \
-o <output_directory> -of [bw or bg] --basename pred_data
```
The output should be one bigWig or bedGraph file per dataset predicted and
the ``<prefix_>predmoter.log`` file, containing important information about the
conversion procedure.

## 5. Flagging subsequences <a id="5-flagging-subsequences"></a>
```bash
# copy files
cp <path_to_predmoter>/Predmoter/predmoter/testdata/train_data_2.h5 <output_directory>
cp <path_to_predmoter>/Predmoter/predmoter/testdata/train_data_2_sequence_report.jsonl \
<output_directory>

python3 <path_to_predmoter>/Predmoter/side_scripts/add_blacklist.py \
-i <output_directory>/train_data_2_sequence_report.jsonl \
-h5 <output_directory>/train_data_2.h5
```
The information about flagged sequences will be added to the give h5 file under the
dataframe name ``data/blacklist``.
