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
a corresponding sequence report file ``<file>_sequence_report.jsonl``, which was
used to create the ``data/blacklist`` dataframe for ``train_data_4.h5`` and
``train_data_5.h5``. The sequence report for ``train_data_2.h5`` is used to test
adding this dataframe to it via [add_blacklist.py](../../side_scripts/add_blacklist.py).

     
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
    
Tests for Predmoter are performed using [test.py](../test/test.py). Instructions
on how to execute and customize tests can be found [here](../test/README.md).
