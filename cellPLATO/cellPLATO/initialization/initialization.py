#initialization.py

from initialization.config import *

import os
import shutil
import datetime
import warnings
warnings.filterwarnings("ignore")

TIMESTAMP = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '_')

# print('Dataset in current notebook: ',DATASET_SHORTNAME)

print('Initializing: ', DATASET_SHORTNAME)
print('Hypthesis testing using: ',STAT_TEST)


'''
Make the folders for exporting
'''

TEMP_OUTPUT = os.path.join(OUTPUT_PATH,DATASET_SHORTNAME,TIMESTAMP,'tmp/')
ANIM_OUTPUT = os.path.join(OUTPUT_PATH,DATASET_SHORTNAME,TIMESTAMP,'animations/')

SAVED_DATA_PATH = os.path.join(OUTPUT_PATH,DATASET_SHORTNAME,'saved_data/')

# Create timestamped folders to contain data and plot from this analysis
# Main level:
DATA_OUTPUT = os.path.join(OUTPUT_PATH,DATASET_SHORTNAME,TIMESTAMP,'data/')
PLOT_OUTPUT = os.path.join(OUTPUT_PATH,DATASET_SHORTNAME,TIMESTAMP,'plots/')
print('Plots will be exported to: ', PLOT_OUTPUT)


if not os.path.exists(TEMP_OUTPUT):
     os.makedirs(TEMP_OUTPUT)

if not os.path.exists(ANIM_OUTPUT):
     os.makedirs(ANIM_OUTPUT)

if not os.path.exists(SAVED_DATA_PATH):
     os.makedirs(SAVED_DATA_PATH)

if not os.path.exists(PLOT_OUTPUT):
  os.makedirs(PLOT_OUTPUT)

if not os.path.exists(DATA_OUTPUT):
  os.makedirs(DATA_OUTPUT)

if not os.path.exists(SAVED_DATA_PATH):
  os.makedirs(SAVED_DATA_PATH)


print('Using unique embedding per dataset shortname: ',DATASET_SHORTNAME)
EMBEDDING_PATH = os.path.join(OUTPUT_PATH,DATASET_SHORTNAME,'tsne_embedding/')



# Sub folders for analysis components:
COMP_DIR = os.path.join(PLOT_OUTPUT,'Comparative_analysis/')
DR_DIR = os.path.join(PLOT_OUTPUT,'Dimensionality_Reduction/')
CLUST_DIR = os.path.join(PLOT_OUTPUT,'Clustering/')

# Sub-directories for parameter sweeping:
DR_PARAMS_DIR = os.path.join(DR_DIR,'Parameter_sweep/')
CLUST_PARAMS_DIR = os.path.join(CLUST_DIR,'Parameter_sweep/')

if not os.path.exists(COMP_DIR):
     os.makedirs(COMP_DIR)

if DIMENSION_REDUCTION and not os.path.exists(DR_DIR):
     os.makedirs(DR_DIR)

if CLUSTERING and not os.path.exists(CLUST_DIR):
     os.makedirs(CLUST_DIR)

if PARAM_SWEEP and not os.path.exists(DR_PARAMS_DIR):
     os.makedirs(DR_PARAMS_DIR)

if PARAM_SWEEP and not os.path.exists(CLUST_PARAMS_DIR):
     os.makedirs(CLUST_PARAMS_DIR)


# Sub folders for plot types (Comparative)
SUPERPLOT_DIR = os.path.join(COMP_DIR,'Superplots/')
SUPERPLOT_grays_DIR = os.path.join(COMP_DIR,'Superplots_grays/')
DIFFPLOT_DIR = os.path.join(COMP_DIR,'Plots_of_differences/')
MARGSCAT_DIR = os.path.join(COMP_DIR,'Marginal_scatterplots/')
TIMEPLOT_DIR = os.path.join(COMP_DIR,'Timeplots/')
BAR_DIR = os.path.join(COMP_DIR,'Bar_plots/')
BAR_SNS_DIR = os.path.join(COMP_DIR,'SNS_Gray_Bar_plots/')


if DRAW_SUPERPLOTS and not os.path.exists(SUPERPLOT_DIR):
    print('Exporting static Superplots')
    os.makedirs(SUPERPLOT_DIR)

if DRAW_SUPERPLOTS_grays and not os.path.exists(SUPERPLOT_grays_DIR):
    print('Exporting static Superplots')
    os.makedirs(SUPERPLOT_grays_DIR)

if DRAW_DIFFPLOTS and not os.path.exists(DIFFPLOT_DIR):
    print('Exporting static Plots of Differences')
    os.makedirs(DIFFPLOT_DIR)

if DRAW_MARGSCAT and not os.path.exists(MARGSCAT_DIR):
    print('Exporting static Marginal scatterplots')
    os.makedirs(MARGSCAT_DIR)

if DRAW_TIMEPLOTS and not os.path.exists(TIMEPLOT_DIR):
    print('Exporting static Timeplots')
    os.makedirs(TIMEPLOT_DIR)

if DRAW_BARPLOTS and not os.path.exists(BAR_DIR):
    print('Exporting Bar plots')
    os.makedirs(BAR_DIR)

if DRAW_SNS_BARPLOTS and not os.path.exists(BAR_SNS_DIR):
    print('Exporting SNS Bar plots')
    os.makedirs(BAR_SNS_DIR)    


# Create the folder where the subgroup cluster outputs will go:

CLUST_TSNE_DIR = os.path.join(CLUST_DIR,'tSNE/')
CLUST_PCA_DIR = os.path.join(CLUST_DIR,'PCA/')
CLUST_XY_DIR = os.path.join(CLUST_DIR,'xy/')
CLUST_DISAMBIG_DIR = os.path.join(CLUST_DIR,'Cluster_Disambiguation/')
CLUST_DISAMBIG_DIR_TAVG = os.path.join(CLUST_DIR,'Cluster_Disambiguation_tavg/')
CLUSTERING_DIR = os.path.join(CLUST_DIR,'Clustering/')

if not os.path.exists(CLUSTERING_DIR):
     os.makedirs(CLUSTERING_DIR)

if not os.path.exists(CLUST_DISAMBIG_DIR):
     os.makedirs(CLUST_DISAMBIG_DIR)

if not os.path.exists(CLUST_DISAMBIG_DIR_TAVG):
     os.makedirs(CLUST_DISAMBIG_DIR_TAVG)

if CLUSTERING and CLUSTER_TSNE and not os.path.exists(CLUST_TSNE_DIR):
     os.makedirs(CLUST_TSNE_DIR)

if CLUSTERING and CLUSTER_PCA and not os.path.exists(CLUST_PCA_DIR):
     os.makedirs(CLUST_PCA_DIR)

if CLUSTERING and CLUSTER_XY and not os.path.exists(CLUST_XY_DIR):
     os.makedirs(CLUST_XY_DIR)



# Some assert statements as sanity checks:
assert CTL_LABEL in CONDITIONS_TO_INCLUDE, 'Be sure that CTL_LABEL in config is within the CONDITIONS_TO_INCLUDE list'

if(USE_SHORTLABELS):
    this_cond_ind = CONDITIONS_TO_INCLUDE.index(CTL_LABEL)
    CTL_SHORTLABEL = CONDITION_SHORTLABELS[this_cond_ind]
    print('Using corresponding CTL_SHORTLABEL: ',CTL_SHORTLABEL,
    ' for condition: ', CTL_LABEL)

# Archive a copy of this config file for future reference
if(ARCHIVE_CONFIG):

    # Also save copy as a .py file so it is easy to re-run later
    path_to_config = 'initialization/config.py'
    export_path = DATA_OUTPUT + 'config_' + '.txt' #+ TIMESTAMP (removed because folder already created with timestamp in name)
    shutil.copyfile(path_to_config, export_path)


print('Dataset in current notebook: ',DATASET_SHORTNAME)
