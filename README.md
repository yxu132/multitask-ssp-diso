# multitask-ssp-diso

This repository is for predict protein secondary structure population and intrisically disordered proteins and regions simultaneously. 

## Basic requirement

To run the scripts in this repository, you need to install 

* Python >= 2.7.x

* [Numpy](http://www.numpy.org) >= 1.3

* [Tensorflow](https://www.tensorflow.org/install/) >= 1.3

* [matplotlib](https://matplotlib.org) >= 2.1.0

* [subprocess](https://docs.python.org/2/library/subprocess.html) >= 2.4

* [argparse](https://docs.python.org/3/library/argparse.html) >= 3.2

## Steps for running the scripts

1, Please set up the blast_path and the compiled blast database (if any) in dist/config.py before running the following scripts. 

2, To predict protein secondary structure population and intrinsically disordered proteins and regions, please enter the directory dist and run,  

```
> python run_prediction.py -h
```
This command will describe the input parameters and the output specifications in detail. 

3, An brief introduction to the input and output parameters are given below. 
```
  -i INPUT    Input file in FASTA format, required. You will need to prepare a input file of protein sequences in the format of [FASTA](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp).

  -o OUTPUT   Output file for storing generated results, optional. If set, the predicted results will be written to the destination file specified here. 

  -v          Visualise the results, optional. If set, a graph demonstrating the predicted results will be generated and saved default to ../figure/visualisation.pdf. 

  -f          The output path of generated visualisation, only available when -v is set, optional. If set, the generated graph will be saved to the destination path set here. 

  -s          Genearate additional results from single task framework (DeepS2D-D) for IDP/IDR prediction as a comparison, optional. 

```

