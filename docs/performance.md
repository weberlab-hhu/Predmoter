# Performance
Performance of Predmoter depends on hardware, batch size, number of datasets used
(ATAC-seq, ChIP-seq, etc.), model configurations, number of devices/workers and the
device itself. In general training, testing and predicting is way faster on one or
multiple GPUs than on CPUs and faster with more workers in ram efficient mode and
with zero workers when ``--ram-effiecient false``. A general rule of thumb is to
use ``4 workers * number of devices`` when ``--ram-effiecient true``.
    
## The ram-efficient argument
There are two dataset classes, that can be used by Predmoter:    
1. PredmoterSequence:
   - argument: ``--ram-efficient false``
   - compresses all data and stores it in RAM
   - takes time, depending on the dataset size a significant amount of time
     (the longest tested around 2 h), before training to process all the data
   - the data processing time and memory consumption is multiplied by the 
     number of devices used to train on
   - training is a lot faster afterwards, since the data was already processed
     
     
2. PredmoterSequence2:
    - argument: ``--ram-efficient true`` (default, since it's good for testing and predicting) 
    - reads data directly from the hard-drive/file for each chunk
    - takes less time (the longest tested around 15 min) before the training to process the data
    - slows down training as the data is always reprocessed at each get_item call
    - very effective for testing and predicting as the data will need to be processed only once
    - extremely RAM efficient
    - **Warning:** Don't move the input data while Predmoter is running.

    
## Recommendations
### Training
**Arguments:** ``--ram-efficient false`` (not the default), ``--num-workers 0``
(default) for one and multiple GPUs
    
**Explanation:** Data loading takes a longer time  when not using the RAM
efficient method. The data read-in time gets multiplied by the number of devices.
The training time when not using RAM efficient method was 4 times shorter than
when using it (both time tests were run on 2 GPUs), since the data, when using the
RAM efficient method, needs to be read-in and processed each epoch. But, the RAM
inefficient method consumed 4 times as much RAM (``--ram-efficient true``: ~ 20
Gb; ``--ram-efficient false``: ~ 85 Gb).
    
### Testing
**Arguments**: ``--ram-efficient true`` (default), ``--num-workers 4``
(not the default), only possible on 1 GPU/CPU (recommended by Lightning)
    
> It is recommended to use
> ``Trainer(devices=1, num_nodes=1)``/``--num-devices 1`` to ensure each
> sample/batch gets evaluated exactly once. Otherwise, multi-device settings
> use `DistributedSampler` that replicates some samples to make sure all devices
> have same batch size in case of uneven inputs.
    
**Explanation**: The data loading takes a lot of time when not using the RAM
efficient method. As the test data needs to be processed just once, no matter
which dataset class is used, the RAM efficient method is ideal.
    
### Predicting
**Arguments**: ``--ram-efficient true`` (default), ``--num-workers 4``
(not the default), only possible on 1 GPU/CPU
    
>**Warning**: It isn't possible to predict on multiple devices, since the
> DistributedDataParallel (DDP) strategy initializes the script
> on each device, resulting in Predmoter trying to write the predictions h5
> file multiple times, resulting in IO errors. It is also not possible to
> use multiple devices for prediction, as the multi-device settings
> use `DistributedSampler` that replicates some samples to make sure all devices
> have same batch size in case of uneven inputs, which could result in faulty
> predictions.
    
**Explanation**: The data loading takes a lot of time when not using the RAM
efficient method. As the genome you want to predict ATAC-/ChIP-seq coverage for
needs to be processed just once, no matter which dataset class is used,
the RAM efficient method is ideal here as well. When using multiple workers
the RAM usage increases a little. When predicting on the GPU the RAM usage
(using ``--ram-efficient true``) was found to be around 16 GB, predicting on
genomes in a range of ~120 Mbp to ~2.2 Gbp. When predicting on the CPU instead
(also using ``--ram-efficient true``) the RAM usage was found to be around 4 GB,
also predicting on genomes in a range of ~120 Mbp to ~2.2 Gbp. This is a result
of PyTorch's and Lightning's method to push the model to the GPU. Predicting on
the CPU takes around 3 times longer than on the GPU.
    
## Benchmarking (prediction time)
Benchmarking was performed on a machine with an Intel(R) Xeon(R) CPU W-2125
@ 4.00 GHz and an Nvidia GeForce GTX 1050 Ti GPU (4 Gb memory). The software
versions were CUDA 11.5, cuDNN 8.9.5 and Python 3.10.12. The relevant Python
package versions were PyTorch 2.0.1, Lightning 2.0.8, Helixer version 0.3.2 and
Predmoter version 0.3.2. The exact versions of the other packages used can be found
[here](../benchmarking_package_versions_freeze.txt).    
Depending on the model used, there is always a slight fluctuation in the prediction
and conversion to bigWig or bedGraph files. Two different models BiHybrid_04 and
the Combined model were used (see
[here](https://github.com/weberlab-hhu/predmoter_models) to download these models).
The BiHybrid_04 model can predict ATAC-seq coverage and the Combined and
Combined_02 models ATAC- and ChIP-seq coverage. Some genome assemblies were
highly fragmented, on contig or scaffold level, increasing the number of
subsequences. Since inference and conversion to bigWig or bedGraph files is
dependent on the amount of data, so the number of subsequences, that was used to
quantify the wall time.
     
### Conversion from fasta to h5 files (Helixer)
![](../images/fasta2h5_benchmarking.png)
### Predicting one dataset
![](../images/predicting_one_dataset_benchmarking.png)
### Predicting two datasets simultaneously
![](../images/predicting_two_datasets_benchmarking.png)
