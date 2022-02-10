# PromotorsPreds

## Reading
Just to get started

 - promotors and enhancers: 
   - https://www.nature.com/articles/s41576-019-0173-8
 - promotor prediction in animals: 
   - https://www.biorxiv.org/content/10.1101/2020.12.04.410795v3
   - https://genome.cshlp.org/content/31/6/1097.short
   - https://www.nature.com/articles/s41592-021-01252-x

## Tutorials
Pytorch lightening
 - https://exxactcorp.com/blog/Deep-Learning/getting-started-with-pytorch-lightning
Pytorch, intro on up
 - https://www.tutorialspoint.com/pytorch/index.htm

## ML course
https://www.coursera.org/learn/machine-learning/home/welcome (there should somewhere be a free/without certificate version)


## Next Steps / While Ali is Out

### Mini / guided pytorch intro

Move towards getting _a_ project setup in pytorch. 

Any tutorial you like is fine. Ideally,
 - try out an LSTM
 - try out a CNN 

### Look at a workshop on some promotor-related data analysis

https://nbis-workshop-epigenomics.readthedocs.io/en/latest/content/tutorials.html

You can definitely do this at a "high level", you do not need to follow 
all the code (and without their setup/backend, I'm not sure how feasible this would even be).
But look at the intro to where the data comes from and the major analysis steps. 

### Actually look at some demo data
This is pre-processed ATAC-seq data for two species under /mnt/data/feli/prototyping_data

You can get a feel for it by visualizing the <species>/raw/seqdata/*/mapping/*.bam files
with a browser such as IGV (https://software.broadinstitute.org/software/igv/) 
or Tablet (https://ics.hutton.ac.uk/tablet/).

In the files <sp>/h5/test_data.h5, these reads have been quantified as coverage per bp. 
This is a reasonable format (at least for starts) to directly feed into a NN.
(input data/x, predict evaluation/atacseq_coverage). 

So if you run out of other stuff todo, you can then start trying to get basically
any network setup in pytorch with this as input output. 

### Questions
collect questions & feel free to e-mail me. I will check in the evenings, and 
respond if I can, but no promises. Also, you can always try the HelixerTeam chat.
