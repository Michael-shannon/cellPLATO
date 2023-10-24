# cellPLATO: cell PLasticity Analysis TOol

On the 14th of May 2023 we presented cellPLATO's UMAP and HDBSCAN module to produce a fingerprint of cell behaviours in a heterogeneous population. Go [here](https://drive.google.com/drive/folders/1_f2GmdqbaF15FyesgxnsotuAu_XGh10o?usp=sharing) to see the poster!

[![DOI](https://zenodo.org/badge/588728402.svg)](https://zenodo.org/badge/latestdoi/588728402)

A Python data analysis package for time-lapse cell migration experiments written in collaboration with [Tyler Sloan](https://github.com/tsloan1377) at [Quorumetrix](https://github.com/Quorumetrix). Used after segmention (eg. [Cellpose](https://github.com/MouseLand/cellpose)) and tracking (eg. [Bayesian Tracker](https://github.com/quantumjot/btrack)) of large timelapse microscopy datasets, cellPLATO measures morphokinetic information about each cell per timepoint and automatically makes statistical plots (plots of difference in python, inspired by those in R by [Joachim Goedhart](https://github.com/JoachimGoedhart)). Users can pool/compare multiple replicates from multiple experimental conditions. Next, dimensionality reduction and cluster analysis is used to segregate cells into behavioural subtypes and produce a fingerprint for each condition (cells per behaviour subtype). Finally, exemplar cells are automatically selected and graphically displayed to disambiguate the nature of each quanitfied cell behavioural subtype.

![GraphicalAbstractForGithub](https://github.com/Michael-shannon/cellPLATO/assets/37793157/c8f28934-18ae-4e36-8853-563c39391b53)

## Installation instructions

1. Using anaconda terminal, cd to a directory where you want to install the software
2. Clone the repository onto your local machine: git clone 
3. cd to the folder that contains 'environment.yml' and type: conda env create -f environment.yml
4. Activate the environment: conda activate cellPLATO
5. Install the rest of the packages: pip install -e .

Known issues with installation:

If you get the ERROR: Could not build wheels for hdbscan, which is required to install pyproject.toml-based projects

Please check 1) you have C++ installed, 2) install hdbscan using 'conda install -c conda-forge hdbscan'

If matplotlib fails to install via pip for the same reason, please use:

conda install -c conda-forge matplotlib

## How to use cellPLATO:

cellPLATO is made to be used downstream of cell segmentation and tracking. We used cellpose and then bayesian tracker, with files organized as below in the 'file organization' section.

With jupyter notebook installed, type jupyter notebook in the terminal to begin, and select one of the notebooks to begin running cellPLATO.

## Description: 

A collection of Jupyter notebooks allows user to process through the analysis step by step, or using pre-assembled pipelines.

All experimental constants and filepaths are contained within the config.py file. This will inform the active Python kernel where to find the data files (.h5) files, where to export plots, and ey parameters to control the analysis. Each time the analysis is run, it generates a time-stamped analysis output folder, with a copy of the config file as a record for future verification.

The experimental conditions and replicates are indicated in the config.py file as the EXPERIMENTS_TO_INCLUDE = []. The data_processing module will automatically extract the replicates from the following file folder structure:

my_data_path
    condition 1
        Replicate 1
          Replicate 1.h5
        Replicate 2
          Replicate 2.h5
        ...
        Replicate n
          Replicate n.h5
     Condition 2
        Replicate 1
          Replicate 1.h5
        Replicate 2
          Replicate 2.h5
        ...
        Replicate n
          Replicate n.h5
     ...
     Condition N
        Replicate 1
          Replicate 1.h5
        ...
        Replicate n
          Replicate n.h5
       

The data_processing submodule is designed to sequentially process the cell tracks and shape measurements from the btracker-generated h5 files, and combine them into a Pandas dataframe for further processing, filtering and visualization. 

The functionality of the subsequent processing steps are defined below;

Pre-preprocessed data is are combined into a single dataframe (comb_df), maininging labels for the Condition and replicate_ID. For plotting, optionally Condition_shortlabel is also used to have more succinct plot labels. The comb_df both cell shape and cell migration-related factors. 

At this stage, additional measurements are performed, such as the aspect ratio and Ripleys L and K. The factors are calibrated according to the micron_per_pixel ratio defined in the config.py file. Optionally, data are filtered upstream of dimensionality reduction. 

Next, the combined dataframe undergoes dimensionality reduction: initially PCA, followed by both tSNE and UMAP low-dimension embeddings. The lowD representations contain information about both the cell migration and shape characteristics of each cell, at each timepoint, and additional filtering steps following the dimensionality reduction steps are possible.

The low-dimensional embeddings are then clustered using hdbscan to automatically extract density-based clusters from the selected embedding. Cells at a given timepoint are clustered into distinct groups and provided a label for their group. 




