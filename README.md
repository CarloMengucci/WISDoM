# WISDoM
Wishart Distributed Matrices Multiple Order Classification Method, Pipeline and Utilities


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

For a complete mathematical description of WISDoM and the Wishart Distribution see: https://github.com/CarloMengucci/WISDoM/blob/master/WISDoM/WISDoM-Complete.pdf


**Snakemake Pipeline**

The main tool used to develope a parallel and optimized pipeline is the Snakemake Workflow Management System, a Python-based interface created to build reproducible and scalable data analyses and machine-learning routines.
To briefly some up the the advantages of using such tools and structures, the Snakemake Workflow can be described as rules that denote how to create output files from input files. The workflow is implied by dependencies between the rules that arise from one rule needing an output file of another as an input file.

![Pipeline Rules Visual Summary](https://github.com/CarloMengucci/WISDoM/blob/master/WISDoM/General-Pipeline.png)

