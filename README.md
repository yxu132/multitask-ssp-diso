# multitask-ssp-diso

This repository provides the source codes for our developed multi-task deep learning framework for simultaneous prediction of protein secondary structure population (SSP) and intrinsically disordered proteins (IDPs) and regions (IDRs). The related manuscript *Simultaneous prediction of protein secondary structure population and intrinsic disorder us-ing multi-task deep learning* is submitted to *Bioinformations*. 

The deep learning implementation is based on the framework [TensorFlow](https://www.tensorflow.org/install/). Please refer to the original manuscript for detailed scription of the singletask framework and the multitask framework for prediction SSPs and IDP/IDRs. 

The input feature，the position-based scoring matrix, is generated using the [PSI-BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE=Proteins&PROGRAM=blastp&RUN_PSIBLAST=on) and parsed using a module 'chkparse' from [the s2D method](http://www-mvsoftware.ch.cam.ac.uk/index.php/s2D). The blast database uniref90filt.fasta.zip is downloaded from [the online server](http://www-mvsoftware.ch.cam.ac.uk/index.php/s2D) of the s2D method. 

## Basic requirement

To run the scripts in this repository, you need to install 

* Python >= 2.7.x

* [Numpy](http://www.numpy.org) >= 1.3

* [Tensorflow](https://www.tensorflow.org/install/) >= 1.3

* [matplotlib](https://matplotlib.org) >= 2.1.0

* [subprocess](https://docs.python.org/2/library/subprocess.html) >= 2.4

* [argparse](https://docs.python.org/3/library/argparse.html) >= 3.2

## System compatibility

This program is tested on MasOS and GNU/Linux. 

## Steps for setting up the scripts

1, Please set up the parameter *blast_path* in dist/config.py to the binary path of your psiblast program. 

2, Please set up the parameter *uniref90_psi_blast_database* in dist/config.py to the path of the compiled psiblast database, if any

3, If the psiblast database is not compiled yet, please compile the database before step 2. 

4, please set up the parameter *gcc_path* if you have a different command for running *gcc*. 

5, Please set up the parameter *tmp_path* if you want to save the genearated .chk files from [PSI-BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE=Proteins&PROGRAM=blastp&RUN_PSIBLAST=on) to a different directory. 

6, For detailed instruction of the input and output parameters, please enter the directory dist and run,  

```
 $ python run_prediction.py -h
```
This command will describe the input parameters and the output specifications in detail. 

3, An brief introduction to the input and output parameters are given below. 

```
  -i INPUT    Input file in FASTA format, required. You will need to prepare a input file of protein sequences in the format of FASTA.

  -o OUTPUT   Output file for storing generated results, optional. If set, the predicted results will be written to the destination file specified here. 

  -v          Visualise the results, optional. If set, a graph demonstrating the predicted results will be generated and saved default to ../figure/visualisation.pdf. 

  -f          The output path of generated visualisation, only available when -v is set, optional. If set, the generated graph will be saved to the destination path set here. 

  -s          Genearate additional results from single task framework (DeepS2D-D) for IDP/IDR prediction as a comparison, optional. 
```


## Examples of running the scripts

1, To predict the SSPs and IDP/IDRs by using the pretrained multitask framework, please run
```
  $ python run_predictin.py -i test.fasta
```

2, To predict the SSPs and IDP/IDRs by using the pretrained multitask framework and save the predicted results to output.txt, please run
```
  $ python run_predictin.py -i test.fasta -o output.txt
```

3, To predict the SSPs and IDP/IDRs by using the pretrained multitask framework, save the predicted results to output.txt and generated the visualisation of the results, please run
```
  $ python run_predictin.py -i test.fasta -o output.txt -v
```

4, To predict the SSPs and IDP/IDRs by using the pretrained multitask framework and IDP/IDRs by using the pretrained singletask framework, please run
```
  $ python run_predictin.py -i test.fasta -s
```

5, To predict the SSPs and IDP/IDRs by using the pretrained multitask framework and IDP/IDRs by using the pretrained singletask framework, and to visualise the compared results in graphs, please run
```
  $ python run_predictin.py -i test.fasta -s -v
```

## License
This project is licensed under GNU GPLv3
