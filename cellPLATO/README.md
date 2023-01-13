# cellPLATO: cell PLasticity Analysis TOol

A Python data analysis package for time-lapse cell migration experiments. Used in conjunction with Bayesian Tracker for automated cell tracking and segmentation, cellPLATO adds additional layers of analysis and visualization. This tool allows users to pool/compare multiple replicates from multiple experimental conditions, perform dimensionality reduction, and explore cell behavioural trajectories through physical and low-dimensional space.

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




