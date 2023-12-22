# Tests
The tests are run from the ``Predmoter/predmoter`` directory. The tests are run with
``pytest``. A basic test can be run like this:
```bash
pytest --verbose test/test.py -W ignore::DeprecationWarning
# the verbose option is recommended to get a more detailed output
# the option -W ignore::DeprecationWarning is recommended to not see
# eventual deprecation warnings that have nothing to do with Predmoter itself
```
     
The output should look like this (Predmoter's version/current commit can change.):
```bash
Testing Predmoter v0.3.2. The current commit is 3ddadf9.
====================================================================================== test session starts =======================================================================================
platform linux -- Python 3.8.10, pytest-7.4.0, pluggy-1.2.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: <rootdir>/Predmoter/predmoter/test
configfile: pytest.ini
plugins: anyio-3.6.2
collected 26 items

test/test.py::test_get_available_datasets PASSED                                                                                                                                           [  3%]
test/test.py::test_finding_and_sorting_h5_input_files PASSED                                                                                                                               [  7%]
test/test.py::test_prepping_predict_input_file PASSED                                                                                                                                      [ 11%]
test/test.py::test_helixer_hook_fasta2h5_conversion PASSED                                                                                                                                 [ 15%]
test/test.py::test_extracting_h5_file_meta_data PASSED                                                                                                                                     [ 19%]
test/test.py::test_getting_file_basename PASSED                                                                                                                                            [ 23%]
test/test.py::test_padding_formulas PASSED                                                                                                                                                 [ 26%]
test/test.py::test_compute_masking PASSED                                                                                                                                                  [ 30%]
test/test.py::test_custom_metrics PASSED                                                                                                                                                   [ 34%]
test/test.py::test_tensor_unpacking PASSED                                                                                                                                                 [ 38%]
test/test.py::test_chromosome_map_creation PASSED                                                                                                                                          [ 42%]
test/test.py::test_return_unique_duplicates PASSED                                                                                                                                         [ 46%]
test/test.py::test_extracting_start_and_end_coordinates PASSED                                                                                                                             [ 50%]
test/test.py::test_dataset_classes_without_filtering PASSED                                                                                                                                [ 53%]
test/test.py::test_dataset_classes_with_filtering PASSED                                                                                                                                   [ 57%]
test/test.py::test_dataset_classes_predict_mode PASSED                                                                                                                                     [ 61%]
test/test.py::test_train_mode PASSED                                                                                                                                                       [ 65%]
test/test.py::test_test_mode PASSED                                                                                                                                                        [ 69%]
test/test.py::test_test_mode_without_ram_efficiency PASSED                                                                                                                                 [ 73%]
test/test.py::test_test_mode_with_filtering PASSED                                                                                                                                         [ 76%]
test/test.py::test_predict_mode PASSED                                                                                                                                                     [ 80%]
test/test.py::test_convert_predictions_2coverage PASSED                                                                                                                                    [ 84%]
test/test.py::test_convert_experimental_data_2coverage PASSED                                                                                                                              [ 88%]
test/test.py::test_write_blacklist_text_file PASSED                                                                                                                                        [ 92%]
test/test.py::test_add_blacklist PASSED                                                                                                                                                    [ 96%]
test/test.py::test_overwrite_blacklist PASSED                                                                                                                                              [100%]
Teardown tests. Deleted the temporary directory in Predmoter/predmoter.


================================================================================= 26 passed in 166.37s (0:02:46) =================================================================================
```
    
The tests will create a temporary directory in the ``Predmoter/predmoter`` directory
that will be deleted as soon as the tests are finished running, even if an error
occurred. It sets up the necessary input directory structure for testing:
```raw
tmp_<random_number> 
| 
|
├── train 
|     ├── train_data_1.h5
|     ├── train_data_2.h5
|     ├── train_data_3.h5
|     ├── train_data_4.h5
|     └── train_data_5.h5
|
├── val 
|     ├── val_data_1.h5
|     ├── val_data_2.h5
|     └── val_data_3.h5
|
└── test
      ├── train_data_1.h5
      ├── train_data_2.h5
      ├── train_data_3.h5
      ├── train_data_4.h5
      └── train_data_5.h5
```
The temporary directory also functions as the output directory for tests. In total,
the maximum size this temporary directory reached was 34 Mb.
    
## Options
There are customizable options/arguments for the tests executed to accommodate for
different systems/setups.
### Modes
First, there are two sets of tests: unittests and command line tests. They are marked
with custom pytest markers and can be selected/unselected like so:
```bash
# just execute unittests
pytest -m unittest --verbose test/test.py -W ignore::DeprecationWarning
# just execute command line tests
pytest -m cmdlinetest --verbose test/test.py -W ignore::DeprecationWarning
```
The command line tests will take a little while (around 2 minutes). By default, the
command line tests will be executed on the CPU and the GPU.    
The output of just executing the unittests should look like this (Predmoter's
version/current commit can change.):
```bash
Testing Predmoter v0.3.2. The current commit is 3ddadf9.
====================================================================================== test session starts =======================================================================================
platform linux -- Python 3.8.10, pytest-7.4.0, pluggy-1.2.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: <rootdir>/Predmoter/predmoter/test
configfile: pytest.ini
plugins: anyio-3.6.2
collected 26 items / 10 deselected / 16 selected

test/test.py::test_get_available_datasets PASSED                                                                                                                                           [  6%]
test/test.py::test_finding_and_sorting_h5_input_files PASSED                                                                                                                               [ 12%]
test/test.py::test_prepping_predict_input_file PASSED                                                                                                                                      [ 18%]
test/test.py::test_helixer_hook_fasta2h5_conversion PASSED                                                                                                                                 [ 25%]
test/test.py::test_extracting_h5_file_meta_data PASSED                                                                                                                                     [ 31%]
test/test.py::test_getting_file_basename PASSED                                                                                                                                            [ 37%]
test/test.py::test_padding_formulas PASSED                                                                                                                                                 [ 43%]
test/test.py::test_compute_masking PASSED                                                                                                                                                  [ 50%]
test/test.py::test_custom_metrics PASSED                                                                                                                                                   [ 56%]
test/test.py::test_tensor_unpacking PASSED                                                                                                                                                 [ 62%]
test/test.py::test_chromosome_map_creation PASSED                                                                                                                                          [ 68%]
test/test.py::test_return_unique_duplicates PASSED                                                                                                                                         [ 75%]
test/test.py::test_extracting_start_and_end_coordinates PASSED                                                                                                                             [ 81%]
test/test.py::test_dataset_classes_without_filtering PASSED                                                                                                                                [ 87%]
test/test.py::test_dataset_classes_with_filtering PASSED                                                                                                                                   [ 93%]
test/test.py::test_dataset_classes_predict_mode PASSED                                                                                                                                     [100%]
Teardown tests. Deleted the temporary directory in Predmoter/predmoter.


=============================================================================== 16 passed, 10 deselected in 10.21s ===============================================================================
```
     
### Arguments
| Argument               | Explanaition                                                                                                                                                                                                                                                                             | Default |
|:-----------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------|
| --exclude-device       | by default the command line tests will be executed on the CPU and the GPU, if one of these should not be used (e.g. no GPU is available) it can be excluded                                                                                                                              | None    |
| --num-workers          | number of CPU cores doing the computations (not associated with device number when training on the CPU), see [Predmoter options](../../docs/Predmoter_options.md) for more information                                                                                                   | 4       |
| --num-devices          | only tests multiple device training when >1 (test multi device reproducibility by adding '--reproducibility-test'), train on multiple GPUs (if available) or multiple CPUs (number of CPU cores not CPUs), see [Predmoter options](../../docs/Predmoter_options.md) for more information | 1       |
| --reproducibility-test | additionally tests command line training reproducibility                                                                                                                                                                                                                                 | False   |
     
The arguments are intended to help customize the tests, e.g. if specific hardware,
like multiple GPUs, isn't available. The training reproducibility test might not be
a test most users need, so it is toggled off by default. Examples on how tests can
be customized are:
```bash
# add reproducibility tests (extends the test time by around one minute)
pytest --verbose test/test.py --reproducibility-test -W ignore::DeprecationWarning
# test two devices
pytest --verbose test/test.py --num-devices 2 -W ignore::DeprecationWarning
# exclude the gpu
pytest --verbose test/test.py --exclude-device gpu -W ignore::DeprecationWarning
# only do command line tests without testing on the CPU
pytest -m cmdlinetest --verbose test/test.py --exclude-device cpu -W ignore::DeprecationWarning
```
