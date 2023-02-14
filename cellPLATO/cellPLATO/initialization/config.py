#config.py

'''
Experiment-specific constants
'''


INPUT_FMT = 'btrack' # 'usiigaci'#btrack
MICRONS_PER_PIXEL = 0.537
SAMPLING_INTERVAL = 10/60 # time between frames in minutes
TRACK_FILENAME = '.h5'
Z_SCALE = 1.00
CALIBRATED_POS = False

N_COMPONENTS = 3 #this is for UMAP!!!!

UMAPS = ['UMAP1','UMAP2','UMAP3'] #this is for umap clustering and isnt used yet

##################
# Small test set of Btracker data
#################

DATA_PATH = 'D:/Michael_Shannon/CELLPLATO_MASTER/OCTOBERTESTING_/ThreeConditions_Go_Stopping_Stopped/'
CTL_LABEL = 'Condition_Migrating'
CONDITIONS_TO_INCLUDE = ['Condition_Migrating', 'Condition_Slowing','Condition_Stopped'] #'Condition_Stopped_artificial'

CONDITION_SHORTLABELS = ['Untreated', 'CytoD 0.5 uM','CytoD 2 uM']

USE_SHORTLABELS = True
PERFORM_RIPLEYS = True
DATASET_SHORTNAME = 'ThreeConditions_Go_Stopping_Stopped_1-12-2023'
# ATASET_SHORTNAME = 'CellPlatoFigure_20x100x_July20_mod1'
# ATASET_SHORTNAME = 'CellPlatoFigure_20x100x_July20_mod2'
# ATASET_SHORTNAME = 'CellPlatoFigure_20x100x_July20_mod3'

IMAGE_HEIGHT = 1024 # pixels
IMAGE_WIDTH = 1024 # pixels
OVERWRITE = True # Overwrite the pre-processed data.

USE_INPUT_REGIONPROPS = True
CALCULATE_REGIONPROPS = False


MICRONS_PER_PIXEL_LIST = [0.537,0.537,0.537]
MICRONS_PER_PIXEL = MICRONS_PER_PIXEL_LIST[0] # Default value
SAMPLING_INTERVAL_LIST= [10/60,10/60,10/60]#[1,1, 1/60,1/60]
SAMPLING_INTERVAL = SAMPLING_INTERVAL_LIST[0] # Default value

# Timecourse analysis parameters
FRAME_START = 0 # Start frame for analysis
FRAME_END = 180 # End frame for analysis

MIXED_SCALING = False # Not used yet, for futureproofing
SELF_STANDARDIZE = False #STANDARDIZES ACROSS factors within a cell df, not across the whole dataframe like the umap standardization. Experiment with it turned on and off for DR.

AVERAGE_TIME_WINDOWS = False #This does two things. 1) provides a time window averaged value for every metric (_tmean). 
#                                                     2)  gives also a ratio of the time window averaged value to the first timepoint in the time window (_tmean_ratio). 
'''
Non-experiment specific constants
'''
MIG_T_WIND = 12#6 # in frames
# MIG_T_WIND = ?? * SAMPLING_INTERVAL ''' To convert into seconds'''

MIN_CELLS_PER_TPT = 1 # used in: average_per_timepoint()

OUTPUT_PATH = 'D:/Michael_Shannon/CELLPLATO_MASTER/OCTOBERTESTING_/ThreeConditions_Go_Stopping_Stopped_1-27-2023_NOT_AVERAGE_TIME_WINDOWS/'

CLUSTER_CMAP = 'tab20'
CONDITION_CMAP = 'dark'
CLUSTER_BY = 'umap' # TEMP - already in config
# STATIC_PLOTS
PALETTE = 'colorblind'
PX_COLORS = 'px.colors.qualitative.Safe' # Choose between discrete colors from https://plotly.com/python/discrete-color/
ARCHIVE_CONFIG = True
STATIC_PLOTS = True
PLOTS_IN_BROWSER = False

ANIMATE_TRAJECTORIES = True
DEBUG = False

#tSNE parameters and embedding:
SCALING_METHOD = 'log2minmax' # minmax powertransformer log2minmax
TSNE_PERP = 185#230 # Perplexity
TSNE_R_S = 11 # Random seed
USE_SAVED_EMBEDDING = False#True
EMBEDDING_FILENAME = 'saved_embedding.npy'
TRAINX_FILENAME = 'saved_x_train.npy'

# UMAP parameters:
UMAP_NN = 10 # Nearest-neighbors
UMAP_MIN_DIST = 0.2 #0.5

# DBScan
MIN_SAMPLES = 10
EPS = 0.06

# Factors to display on the animateed plots
MIG_DISPLAY_FACTORS=['speed', 'euclidean_dist', 'arrest_coefficient', 'turn_angle','directedness', 'dir_autocorr','orientedness']
SHAPE_DISPLAY_FACTORS = ['area','aspect','orientation']

STAT_TEST = 'st.ttest_ind'

# Plot display Parameters
PLOT_TEXT_SIZE = 16
DIFF_PLOT_TYPE = 'violin' # 'swarm', 'violin', 'box'


# Measurement constants
ARREST_THRESHOLD = 3 * SAMPLING_INTERVAL# microns - distance per step for arrest coefficient. Default 2 microns
RIP_R = 140#140#34 # Radius to search when calculating Ripleys K. 1.5 * the size of a cell = 12+6=18

# Factor to standardize to themselves over time (to look at self-relative instead of absolute values.)
FACTORS_TO_STANDARDIZE = ['area',
                          'bbox_area',
                          'equivalent_diameter',
                          'filled_area',
                          'major_axis_length',
                          'minor_axis_length',
                          'perimeter']


# Add fine control over certain plots axis limits, allows: '3-sigma','min-max', '2-sigma'
# Currently only implemented in marginal_xy contour plots.
AXES_LIMITS = '2-sigma' #'min-max' #'2-sigma'

FACTORS_TO_CONVERT = ['area', 'bbox_area', 'equivalent_diameter', 'extent', 'filled_area',
       'major_axis_length', 'minor_axis_length', 'perimeter']






# Cell migration factors calculated in migration_calcs()
MIG_FACTORS = ['euclidean_dist',     # Valid?
                'cumulative_length', # Valid?
                'speed',
                'orientedness', # name changed from orientation
                'directedness',
                'turn_angle',
                'endpoint_dir_ratio',
                'dir_autocorr',
                'outreach_ratio',
                'MSD',                # Valid?
                'max_dist',           # Valid?
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



# Additional factors calculated after the migration and segmentation
# ADDITIONAL_FACTORS = ['aspect']
ADDITIONAL_FACTORS = ['aspect', 'rip_p', 'rip_K', 'rip_L']

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

DR_FACTORS = REGIONPROPS_LIST + MIG_FACTORS + ADDITIONAL_FACTORS
ALL_FACTORS = REGIONPROPS_LIST + MIG_FACTORS + ADDITIONAL_FACTORS

# DR_FACTORS = ['area',

#             # 'bbox_area',
#             'eccentricity',
#             'equivalent_diameter',
#             # 'extent',
#             # 'filled_area',
#             # 'major_axis_length',
#             # 'minor_axis_length',
#             # 'orientation',
#             'perimeter',
#             'solidity',
#             'cumulative_length',
#            # 'euclidean_dist',
#             'speed',
#             'orientedness',
#             'directedness',
#             # 'turn_angle',
#             'endpoint_dir_ratio',
#             'dir_autocorr',
#             'outreach_ratio',
#             'MSD',
#             'max_dist',
#             # 'glob_turn_deg',
#             'arrest_coefficient',
#             'aspect',
#             'rip_L']

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

# DR_FACTORS = DIS_REGIONPROPS_LIST + DIS_MIG_FACTORS + DIS_ADDITIONAL_FACTORS
# Numerical factors for plotting.
NUM_FACTORS = DR_FACTORS + ['tSNE1', 'tSNE2', 'PC1', 'PC2']

# Optionally define your data filters here.
DATA_FILTERS = {
  "area": (10, 10000), # Warning: range will change if self-normalized. Do we need an if statement? Or to standardscale the filter settings, by matching a row of the df?
  "ntpts": (12,1800)

}

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



# Booleans to draw or not specific plots.
DRAW_SUPERPLOTS = True
DRAW_DIFFPLOTS = True
DRAW_MARGSCAT = True
DRAW_TIMEPLOTS = True
DRAW_BARPLOTS = True
DRAW_SUPERPLOTS_grays = True
DRAW_SNS_BARPLOTS = True

# Booleans for Analysis components:
'''(Only run pipelines if true)'''
DIMENSION_REDUCTION = True
PARAM_SWEEP = True
CLUSTERING = True

CLUSTER_TSNE = True
CLUSTER_PCA = True
CLUSTER_XY = True
