'''
Configuration file. 
Fill out this file then run the jupyter notebook to analyze your data

'''
#############################################

'''
Experiment-specific constants
'''

DATA_PATH = 'D:/PATH/'
OUTPUT_PATH = 'D:/PATH_OUTPUT/'
CTL_LABEL = 'CONTROL_CONDITION'

# Input here the folder names of the conditions you want to include in the analysis
# Note: the order of the conditions here will be the order of the conditions in the plots

CONDITIONS_TO_INCLUDE = ['CONTROL_CONDITION',
                        'CONDITION_2',
                         'CONDITION_3',
                         'CONDITION_4']
                         

CONDITION_SHORTLABELS = ['Ctrl','One','Two','Three',] # Short labels for the conditions, for plotting purposes
DATASET_SHORTNAME = 'EXAMPLE_DATASET_NAME'    

INPUT_FMT = 'btrack' # 'usiigaci'#btrack
TRACK_FILENAME = '.h5'

MICRONS_PER_PIXEL = 0.537
MICRONS_PER_PIXEL_LIST = [0.537,0.537,0.537, 0.537,] # For mixed spatial scaling
MICRONS_PER_PIXEL = MICRONS_PER_PIXEL_LIST[0]

SAMPLING_INTERVAL = 40/60 # time between frames in minutes
SAMPLING_INTERVAL_LIST= [40/60,40/60,40/60,40/60,] # For mixed temporal scaling
SAMPLING_INTERVAL = SAMPLING_INTERVAL_LIST[0]

IMAGE_HEIGHT = 1024 # pixels
IMAGE_WIDTH = 1024 # pixels
Z_SCALE = 1.00

MigrationTimeWindow_minutes = 5 # Here, set the length of the time window in minutes
MIG_T_WIND = round(MigrationTimeWindow_minutes / SAMPLING_INTERVAL)

CLUSTER_CMAP = 'tab20' # Define colormap used for clustering plots
CONDITION_CMAP = 'Dark2' #'Define colormap used for condition maps. Dark2 is good for 7 conditions, tab20 > 20 conditions.
# Note: use paired for groups of 2

ARREST_THRESHOLD = 3 * SAMPLING_INTERVAL # Here, user can define threshold in MICRONS PER MINUTE, because we multiply by the sampling interval to convert it to microns per frame.
RIP_R = 140 # Radius to search when calculating Ripleys L in pixels. 1.5 * the size of a cell = 12+6=18

DATA_FILTERS = {
  "area": (10, 10000), # Debris removal
  "ntpts": (8,1800) # Remove cells that are tracked for less than 8 frames

}

# Booleans to draw or not specific plots.
DRAW_SUPERPLOTS = True
DRAW_DIFFPLOTS = True
DRAW_MARGSCAT = True
DRAW_TIMEPLOTS = True
DRAW_BARPLOTS = True
DRAW_SUPERPLOTS_grays = True
DRAW_SNS_BARPLOTS = True


'''
Measurements to make
'''

# Cell migration factors calculated in migration_calcs()

MIG_FACTORS = ['euclidean_dist',     
                'cumulative_length', 
                'speed',
                'orientedness', 
                'directedness',
                'turn_angle',
                'endpoint_dir_ratio',
                'dir_autocorr',
                'outreach_ratio',
                'MSD',                
                'max_dist',           
                'glob_turn_deg',
                'arrest_coefficient']

# Region property factors to be extracted from the cell contours
# This list must match with props from regionprops

REGIONPROPS_LIST = ['area',
                    'bbox_area',
                    'eccentricity',
                    'equivalent_diameter',
                    'extent',
                    'filled_area',
                    'major_axis_length',
                    'minor_axis_length',
                    'orientation',
                    'perimeter',
                     'solidity']

SHAPE_FACTORS = ['area',
                    'bbox_area',
                    'eccentricity',
                    'equivalent_diameter',
                    'extent',
                    'filled_area',
                    'major_axis_length',
                    'minor_axis_length',
                    'orientation',
                    'perimeter',
                     'solidity']

ADDITIONAL_FACTORS = ['aspect', 'rip_p', 'rip_K', 'rip_L']

DR_FACTORS = REGIONPROPS_LIST + MIG_FACTORS + ADDITIONAL_FACTORS
ALL_FACTORS = REGIONPROPS_LIST + MIG_FACTORS + ADDITIONAL_FACTORS


NUM_FACTORS = DR_FACTORS + ['tSNE1', 'tSNE2', 'PC1', 'PC2']

'''
Advanced parameters (can stay default)
'''

MIXED_SCALING = False # Not used yet, for futureproofing
SELF_STANDARDIZE = False #STANDARDIZES ACROSS factors within a cell df.
AVERAGE_TIME_WINDOWS = False #This does two things. 1) provides a time window averaged value for every metric (_tmean). 
# 2)  gives also a ratio of the time window averaged value to the first timepoint in the time window (_tmean_ratio). 

CALIBRATED_POS = False # Does the data need to be calibrated?
OVERWRITE = True # Overwrite the pre-processed data.
USE_INPUT_REGIONPROPS = True
CALCULATE_REGIONPROPS = False
USE_SHORTLABELS = True
PERFORM_RIPLEYS = True
ARCHIVE_CONFIG = True

'''
Everything below does not need to be changed by the user
'''

N_COMPONENTS = 3 #this is for UMAP
UMAPS = ['UMAP1','UMAP2','UMAP3'] 
FRAME_START = 0 # Start frame for analysis (deprecated)
FRAME_END = 180 # End frame for analysis (deprecated)
MIN_CELLS_PER_TPT = 1 # used in: average_per_timepoint()

CLUSTER_BY = 'umap' # temp
PALETTE = 'colorblind'
PX_COLORS = 'px.colors.qualitative.Safe' # Choose between discrete colors from https://plotly.com/python/discrete-color/

STATIC_PLOTS = True
PLOTS_IN_BROWSER = False

ANIMATE_TRAJECTORIES = True
DEBUG = False

# Booleans for Analysis components:
'''(Only run pipelines if true)'''
DIMENSION_REDUCTION = True
PARAM_SWEEP = True
CLUSTERING = True

CLUSTER_TSNE = True
CLUSTER_PCA = True
CLUSTER_XY = True

###############################################
# tSNE/UMAP parameters and embedding:
###############################################

SCALING_METHOD = 'choice' # minmax powertransformer log2minmax choice
TSNE_PERP = 185#230 # Perplexity
TSNE_R_S = 11 # Random seed
USE_SAVED_EMBEDDING = False#True
EMBEDDING_FILENAME = 'saved_embedding.npy'
TRAINX_FILENAME = 'saved_x_train.npy'
UMAP_NN = 10 # Nearest-neighbors
UMAP_MIN_DIST = 0.2 #0.5
MIN_SAMPLES = 10 # DBScan
EPS = 0.06 # DBScan

############################################### 
# Factor wrangling - no need to change these
###############################################

# Factors to display on the animated plots
MIG_DISPLAY_FACTORS=['speed', 'euclidean_dist', 'arrest_coefficient', 'turn_angle','directedness', 'dir_autocorr','orientedness']
SHAPE_DISPLAY_FACTORS = ['area','aspect','orientation']

# Factor to standardize to themselves over time (to look at self-relative instead of absolute values.)
FACTORS_TO_STANDARDIZE = ['area',
                          'bbox_area',
                          'equivalent_diameter',
                          'filled_area',
                          'major_axis_length',
                          'minor_axis_length',
                          'perimeter']

FACTORS_TO_CONVERT = ['area', 'bbox_area', 'equivalent_diameter', 'extent', 'filled_area',
       'major_axis_length', 'minor_axis_length', 'perimeter']

###############################################
# Plotting parameters
###############################################

AXES_LIMITS = '2-sigma' #'min-max' #'2-sigma' # Currently only implemented in marginal_xy contour plots.
STAT_TEST = 'st.ttest_ind'
# Plot display Parameters
PLOT_TEXT_SIZE = 30
DIFF_PLOT_TYPE = 'violin' # 'swarm', 'violin', 'box'

# Pre-defined pairs of factors for generating comparison plots
FACTOR_PAIRS = [['tSNE1', 'tSNE2'],
                ['area', 'speed'],
                ['directedness', 'speed'],
                ['orientedness', 'speed'],
                ['endpoint_dir_ratio', 'speed'],
                ['orientation', 'speed'],
                ['turn_angle', 'speed'], # These are identical
                ['major_axis_length', 'speed'],
                ['major_axis_length', 'minor_axis_length'],
                ['euclidean_dist','cumulative_length'],
                ['euclidean_dist','speed'],
                ['PC1', 'PC2']]

# No need to change these #

DIS_REGIONPROPS_LIST = ['area',
            # 'bbox_area',
            'eccentricity',
            'equivalent_diameter',
            # 'extent',
            # 'filled_area',
            'major_axis_length',
            'minor_axis_length',
            'orientation',
            'perimeter',
             'solidity']
DIS_MIG_FACTORS = ['euclidean_dist',     # Valid?
                'cumulative_length', # Valid?
                'speed',
                # 'orientedness', # name changed from orientation
                # 'directedness',
                # 'turn_angle',
                'endpoint_dir_ratio',
                'dir_autocorr',
                'outreach_ratio',
                'MSD',                # Valid?
                # 'max_dist',           # Valid?
                'glob_turn_deg',
                'arrest_coefficient']

DIS_ADDITIONAL_FACTORS = ['aspect', 'rip_L']

T_WIND_DR_FACTORS = ['MSD',

#                      'MSD_ratio',
#                      'MSD_tmean',
                     'area',
#                      'area_ratio', # Doesn't work in DR if using self-standardized because min (0) becomes inf.
                     'area_tmean',
                     'arrest_coefficient',
#                      'arrest_coefficient_ratio',
                     'arrest_coefficient_tmean',
                     'aspect',
                     'aspect_ratio',
                     'aspect_tmean',
                     'bbox_area',
#                      'bbox_area_ratio',
                     'bbox_area_tmean',
                     'cumulative_length',
#                      'cumulative_length_ratio',
#                      'cumulative_length_tmean',
                     'dir_autocorr',
                     'dir_autocorr_ratio',
                     'dir_autocorr_tmean',
                     'directedness',
                     'directedness_ratio',
                     'directedness_tmean',
                     'eccentricity',
                     'eccentricity_ratio',
                     'eccentricity_tmean',
                     'endpoint_dir_ratio',
                     'endpoint_dir_ratio_ratio',
                     'endpoint_dir_ratio_tmean',
                     'equivalent_diameter',
                     'equivalent_diameter_ratio',
                     'equivalent_diameter_tmean',
                     'euclidean_dist',
                     'euclidean_dist_ratio',
                     'euclidean_dist_tmean',
                     'extent',
                     'extent_ratio',
                     'extent_tmean',
                     'filled_area',
#                      'filled_area_ratio', # Doesn't work in DR if using self-standardized because min (0) becomes inf.
                     'filled_area_tmean',
                     'glob_turn_deg',
#                      'glob_turn_deg_ratio',
#                      'glob_turn_deg_tmean',
                     'major_axis_length',
#                      'major_axis_length_ratio',
#                      'major_axis_length_tmean',
                     'max_dist',
                     'max_dist_ratio',
                     'max_dist_tmean',
                     'minor_axis_length',
#                      'minor_axis_length_ratio',
#                      'minor_axis_length_tmean',
                     'orientation',
                     'orientation_ratio',
                     'orientation_tmean',
                     'orientedness',
                     'orientedness_ratio',
                     'orientedness_tmean',
                     'outreach_ratio',
                     'outreach_ratio_ratio',
                     'outreach_ratio_tmean',
                     'perimeter',
                     'perimeter_ratio',
                     'perimeter_tmean',
                     'rip_K',
#                      'rip_K_ratio',
#                      'rip_K_tmean',
                     'rip_L',
#                      'rip_L_ratio',
#                      'rip_L_tmean',
                     'rip_p',
#                      'rip_p_ratio',
#                      'rip_p_tmean',
                     'solidity',
                     'solidity_ratio',
                     'solidity_tmean',
                     'speed',
                     'speed_ratio',
                     'speed_tmean',
                     'turn_angle',
                     'turn_angle_ratio',
                     'turn_angle_tmean']




