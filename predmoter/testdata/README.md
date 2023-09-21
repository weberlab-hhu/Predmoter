# Testdata
The training and validation test data was created out of valid data from the
species *Arabidopsis thaliana*. The "coordinates" in ``start_ends`` are therefore
incorrect, but not needed for the test cases. The prediction testdata h5 files
were generated from the corresponding artificial fasta file. They contain
artificial but coherent ``start_ends``, as they are necessary to create coherent
predictions files. Almost all files have the default chunk sequence length 21384 bp.
The last file ``pred_data_2.h5`` has the chunk sequence length 42768 bp. Some files
have artificial special cases to test: gaps (large stretches of Ns with no
corresponding NGS coverage, these are masked by default), Ns with corresponding NGS
coverage (these are not masked), padding (chromosome end, when the chromosome is not
exactly divisible by the sequence length, so that arrays of equal size are created),
MOA-seq (random values to test the code on three datasets), and blacklisting information
(added data/blacklist dataframe to test blacklisting). the file ``train_data_5.h5`` comes
with a corresponding sequence report file ``train_data_5_report.jsonl``, which was used
to create the data/blacklist dataframe via
[add_blacklist.py](../../side_scripts/add_blacklist.py).
     
| File           | Artificial datasets         | Chunks | Artificial special cases                         |
|:---------------|:----------------------------|:-------|:-------------------------------------------------|
| train_data_1   | ATAC-seq, ChIP-seq          | 25     | /                                                |
| train_data_2   | ATAC-seq, ChIP-seq, MOA-seq | 10     | gap, padding                                     |
| train_data_3   | MOA-seq                     | 7      | /                                                |
| train_data_4   | ATAC-seq, ChIP-seq          | 6      | Ns in the middle                                 |
| train_data_5   | ATAC-seq                    | 11     | gap, blacklist information (``data/blacklist``)  |
| val_data_1     | ATAC-seq, ChIP-seq          | 5      | /                                                |
| val_data_2     | ATAC-seq, ChIP-seq, MOA-seq | 4      | /                                                |
| val_data_3     | ATAC-seq                    | 6      | gap (Ns stretching beyond chunk border)          |
| predict_data_1 | /                           | 16     | gap (Ns stretching beyond chunk border), padding |
| predict_data_2 | /                           | 10     | gap (Ns stretching beyond chunk border), padding |