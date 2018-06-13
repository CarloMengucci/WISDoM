# WISDoM
*Wishart Distributed Matrices Multiple Order Classification Method, Pipeline and Utilities*


In this work we introduce the Wishart Distributed Matrices Multiple Order Classification (WISDoM) method.
The WISDoM Classification method consists of a pipeline for single feature analysis, supervised learning, cross validation and classification for any problems whose elements can be tied to a symmetric positive-definite matrix representation.
The general idea is for information about properties of a certain system contained in a symmetric positive-definite matrix representation (i.e covariance and correlation matrices) to be extracted by modelling an estimated distribution for the expected classes of a given problem.
The application to fMRI data classification and clustering processing follows naturally: the WISDoM classification method has been tested on the ADNI2 (Alzheimer's Disease Neuroimaging Initiative) database.
The goal was to achieve good classification performances between Alzheimer's Disease diagnosed patients (AD) and Normal Control (NC) subjects, while retaining information on which features were the most informative decision-wise.
In our work, the information about topological properties contained in ADNI2 functional correlation matrices are extracted by modelling an estimated Wishart distribution for the expected diagnostical groups AD and NC, and allowed a complete separation between the two groups.

**The Method**

The main idea for the WISDoM Classifier is to use the free parameters of the Wishart distribution to compute an estimation of the distribution for a certain class of elements, represented via positive symetric-definite matrices, and then assign a single element to a given class by computing some sort of "distance" between the element being analyzed and the classes.
Furhermore, if we assume that the matrices are somehow representative of the features of the system studied (i.e. covariance matrices might be taken into account), a score can be assigned to each feature by estimating the weight of said feature in terms of Log Likelihood Ratio.
In other words, a score can be assigned to each feature by analyzing the variation in terms of LogLikelihood caused by the deletion of it. If the deletion of a feature causes significant increase (or decrease) in the LogLikelihood computed with respect to the estimated distributions for the classes, it can be stated that said feature is highly representative of the system analyzed.

For a complete mathematical description of WISDoM and the Wishart Distribution see: [WISDoM-Complete](https://github.com/CarloMengucci/WISDoM/blob/master/WISDoM/WISDoM-Complete.pdf)


**Snakemake Pipeline**

The main tool used to develope a parallel and optimized pipeline is the Snakemake Workflow Management System, a Python-based interface created to build reproducible and scalable data analyses and machine-learning routines.
To briefly sum up the the advantages of using such tools and structures, the Snakemake Workflow can be described as rules that denote how to create output files from input files. The workflow is implied by dependencies between the rules that arise from one rule needing an output file of another as an input file.

*General Pipeline Summary Visualization*:

<img src="https://github.com/CarloMengucci/WISDoM/blob/master/WISDoM/General-Pipeline.png" alt="General-Pipeline" width="500px"/>

*Sample Pipeline Rules Execution DAG*:

<img src="https://github.com/CarloMengucci/WISDoM/blob/master/WISDoM/Sample-pipeline-DAG.png" alt="General-Pipeline" width="750px"/>

To see the complete pipeline for the ADNI Database analysis developed with Snakemake please look at: [ADNI_Snakefile](https://github.com/CarloMengucci/WISDoM/blob/master/WISDoM/Code/ADNI_Snakefile)

**Data Formats**

The WISDoM pipeline is compatible with .hdf tabulated data. Each row must be an entry of the database and each column an element of the upper (or lower) triangle of the symmetric positive-definite matrix associated to said entry, excluding the diagonal elements.
Module [head_wrap.py](https://github.com/CarloMengucci/WISDoM/blob/master/WISDoM/Code/Modules/head_wrap.py) contains useful functions and wrappers to associate labels to entries from existing .csv files, while module [gen_fs.py](https://github.com/CarloMengucci/WISDoM/blob/master/WISDoM/Code/Modules/gen_fs.py) contains functions to reconstruct the NumPy tensor from a Pandas Dataframe obtained by reading the .hdf file.

**Requisites**

In order to successfully use the WISDoM pipeline, the [Snakemake Environment](https://snakemake.readthedocs.io/en/stable/) must be correctly set up.
Furthermore, the modules relies on **scipy.stats** for Wishart sampling generation and **scikit.learn** for training and classification; **Pandas** is also required for Dataframe creation and handling.


