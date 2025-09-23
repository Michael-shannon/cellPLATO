# pipelines.py

from initialization.initialization import *
from initialization.config import *

from data_processing.cell_identifier import *
from data_processing.cleaning_formatting_filtering import *
from data_processing.clustering import *
from data_processing.data_wrangling import *
from data_processing.dimensionality_reduction import *
from data_processing.measurements import *
from data_processing.statistics import *
from data_processing.time_calculations import *
from data_processing.trajectory_clustering import *

from visualization.cluster_visualization import *
from visualization.comparative_visualization import *
from visualization.low_dimension_visualization import *
from visualization.plots_of_differences import *
from visualization.superplots import *
from visualization.timecourse_visualization import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
from tqdm import tqdm

'''
This script is exceptional among the data_processing submodule as it combines
functions from data_processing and data_visualization submodules.
Any functions that mix these two submodules should be included here.
'''


def measurement_pipeline(comb_df, mixed=MIXED_SCALING, factors_to_timeaverage = ALL_FACTORS):

    # Calculate the clusteredness in xy-space
    if (PERFORM_RIPLEYS):
        comb_df = calc_ripleys_xy(comb_df, plot=False) #'''This is still done in pixels'''

    # Calculate the cells aspect ratio
    if INPUT_FMT != 'trackmate':
        calc_aspect_ratio(comb_df, drop=True)# if there are resulting nans
    
    # Clean up remaining dataframe, and calibrate the micron-dependent values
    cleaned_df = clean_comb_df(comb_df, deduplicate=False)
    
    if INPUT_FMT != 'trackmate':
        comb_df = factor_calibration(cleaned_df,mixed_calibration=mixed)
    else:
        comb_df = cleaned_df
        
    if INPUT_FMT != 'trackmate':
        apply_unique_id(comb_df)

    if(SELF_STANDARDIZE):
        comb_df = standardize_factors_per_cell(comb_df,FACTORS_TO_STANDARDIZE)

    add_fac_list = None
    if(AVERAGE_TIME_WINDOWS):
        # Calculate time-windowed metrics for specified factors
        comb_df, add_fac_list = t_window_metrics(comb_df,factor_list=factors_to_timeaverage)

    # Reset index since aspect calc may have dropped rows.
    comb_df.reset_index(inplace=True, drop=True)

    return comb_df, add_fac_list #, add_fac_list to be output in order to integrate into dr downstream

def dr_pipeline(df, dr_factors=DR_FACTORS, dr_input='factors', tsne_perp=TSNE_PERP,umap_nn=UMAP_NN,min_dist=UMAP_MIN_DIST):

    '''
    An updated dimensionality reduction that performs PCA, tSNE and UMAP,
    and returns the combined dataframe.
    Function intended to be useful as a single call with default values (from config)
    A single call with user-defined input parameters (i.e. following a sweep or optimization)
    OR as a part of a parameter sweep.

    Input:
        df
        dr_factors: list of factors to use when extracting the data matrix
                    default: DR_FACTORS constant from the config.
        dr_input: string indicating what to use as input to the tSNE and UMAP functions.
                default: 'factors' standardized factors directly. Alternatively 'PCs' will use PCA output.

        tsne_perp: default = TSNE_PERP
        umap_nn:default = UMAP_NN
        min_dist:default = UMAP_MIN_DIST


    '''

    print('Running dr_pipeline...')
    print('tSNE perplexity = ',tsne_perp)
    print('UMAP nearest neighbors = ', umap_nn, ' min distance = ',min_dist)

    # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
    x = get_data_matrix(df,dr_factors)

    # Principal component analysis
    pca_df, _, _ = do_pca(x)

    if dr_input == 'factors':

        x_ = StandardScaler().fit_transform(x)
        print('Using standardized factors for dimensionality reduction, matrix shape: ', x_.shape)

    elif dr_input == 'PCs':

        x_ = pca_df.values
        print('Using Principal Components for dimensionality reduction, matrix shape: ', x_.shape)

    # openTSNE using default vaues
    '''
    This should be replaced by a version of tSNE that allows us to set
    the perplexity value right here, or use the default.
    Not just the default as currently.

    Otherwsie need to use other function test_tsne()
    '''
    tsne_x, flag = do_open_tsne(x_,perplexity=tsne_perp)

    # # openTSNE using default vaues (Previously...)
    # tsne_x, flag = do_open_tsne(x)

    # Format tSNE results into a dataframe
    tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])
    # tsne_df['used_existing'] = flag


    # Use standrard scaler upstream of tSNE.
    umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist)
    umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])

    # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
    dr_df = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)

    assert list(df.index) == list(dr_df.index), 'dr_df should be the same length as input dataframe. Check indexing of input dataframe.'

    return dr_df

def dr_pipeline_dev(df, dr_factors=DR_FACTORS, dr_input='factors', tsne_perp=TSNE_PERP,umap_nn=UMAP_NN,min_dist=UMAP_MIN_DIST, n_components=N_COMPONENTS):


    print('Running dr_pipeline...')
    print('tSNE perplexity = ',tsne_perp)
    print('UMAP nearest neighbors = ', umap_nn, ' min distance = ',min_dist)

    # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
    x = get_data_matrix(df,dr_factors)

    # Principal component analysis
    pca_df, _, _ = do_pca(x)

    if dr_input == 'factors':

        g = StandardScaler().fit_transform(x)
        x_ = MinMaxScaler().fit_transform(g)
        print('Using standardized factors for dimensionality reduction, matrix shape: ', x_.shape)

    elif dr_input == 'PCs':

        x_ = pca_df.values
        print('Using Principal Components for dimensionality reduction, matrix shape: ', x_.shape)

    # openTSNE using default vaues
    '''
    This should be replaced by a version of tSNE that allows us to set
    the perplexity value right here, or use the default.
    Not just the default as currently.

    Otherwsie need to use other function test_tsne()
    '''
    tsne_x, flag = do_open_tsne(x_,perplexity=tsne_perp)

    # # openTSNE using default vaues (Previously...)
    # tsne_x, flag = do_open_tsne(x)

    # Format tSNE results into a dataframe
    tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2','tSNE3', 'tSNE4','tSNE5'])
    # tsne_df['used_existing'] = flag


    # Use standrard scaler upstream of tSNE.
    umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist, n_components=N_COMPONENTS)
    umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2','UMAP3', 'UMAP4','UMAP5'])

    # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
    dr_df = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)

    assert list(df.index) == list(dr_df.index), 'dr_df should be the same length as input dataframe. Check indexing of input dataframe.'

    return dr_df

def dr_pipeline_multiUMAPandTSNE(df, dr_factors=DR_FACTORS, tsne_perp=TSNE_PERP, umap_nn=UMAP_NN,min_dist=UMAP_MIN_DIST, n_components=N_COMPONENTS, scalingmethod=SCALING_METHOD, do_tsne = True, factors_to_transform=None, factors_not_to_transform=None, verbose=False): 
    """
    Dimensionality reduction pipeline with UMAP and optional tSNE.
    
    Parameters:
    -----------
    factors_to_transform : list, optional
        For 'choice' scaling: List of factors that should receive log2 + minmax scaling.
        If None, uses default hardcoded list.
        
    factors_not_to_transform : list, optional
        For 'choice' scaling: List of factors that should receive only minmax scaling.
        If None, uses default hardcoded list.
        
    verbose : bool, optional
        Whether to print detailed progress and debugging information. Default is False.
        
    Notes:
    ------
    For 'choice' scaling method, you can specify custom factor lists:
    - Provide both lists for full control
    - Provide only factors_not_to_transform to specify what NOT to log-transform
    - Provide only factors_to_transform to specify what TO log-transform
    - Provide neither to use defaults
    
    Example:
    --------
    # Custom factor lists for 'choice' scaling
    factors_to_log = ['AREA', 'SPEED', 'MAX_INTENSITY_CH1'] 
    factors_to_keep = ['CIRCULARITY', 'ASPECT_RATIO', 'DIRECTEDNESS']
    
    dr_df = cp.dr_pipeline_multiUMAPandTSNE(
        df, 
        scalingmethod='choice',
        factors_to_transform=factors_to_log,
        factors_not_to_transform=factors_to_keep
    )
    """

    component_list=np.arange(1, n_components+1,1).tolist()
    from sklearn.preprocessing import PowerTransformer
    savedir = CLUST_DISAMBIG_DIR

    umap_components=([f'UMAP{i}' for i in component_list])
    # tsne_components=([f'tSNE{i}' for i in component_list])

    if verbose:
        print('Running dr_pipeline for multi dimension UMAP and tSNE...')
        print('tSNE perplexity = ',tsne_perp)
        print('UMAP nearest neighbors = ', umap_nn, ' min distance = ',min_dist)
        print('Number of UMAP components = ', n_components)

    # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.

    #### DEV part - adding tmeans functionality ####

    def create_histogram_plot(data_df, columns, title, save_path, bins=160):
        """Create properly formatted histogram plot that fits A4 page."""
        # Calculate optimal subplot layout
        n_factors = len(columns)
        if n_factors <= 4:
            ncols = 2
        elif n_factors <= 9:
            ncols = 3
        elif n_factors <= 16:
            ncols = 4
        elif n_factors <= 25:
            ncols = 5
        else:
            ncols = 6
        
        nrows = int(np.ceil(n_factors / ncols))
        
        # Set figure size for A4 page (8.27 x 11.69 inches)
        fig_width = 8.27
        fig_height = min(11.69, nrows * 1.5 + 1)  # Limit height to A4
        
        plt.rcParams.update({'font.size': 10})
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        
        # Handle single subplot case
        if n_factors == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, factor in enumerate(columns):
            if i < len(axes):
                axes[i].hist(data_df[factor].dropna(), bins=bins, color="black", ec="black", alpha=0.7)
                axes[i].set_title(factor, fontsize=10, pad=5)
                axes[i].tick_params(axis='both', which='major', labelsize=8)
                axes[i].tick_params(axis='both', which='minor', labelsize=8)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title, fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for title
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory

    if AVERAGE_TIME_WINDOWS:
        tmeansdrfactors = [f'{i}_tmean' for i in dr_factors]
        if verbose:
            print("DR factors used were" + str(tmeansdrfactors))
            print("THIS IS THE UNTRANSFORMED DATA ")
        sub_df = df[tmeansdrfactors]
        create_histogram_plot(sub_df, tmeansdrfactors, 'Untransformed data', savedir + 'UntransformedData.svg')
    else:
        if verbose:
            print("DR factors used were" + str(dr_factors))
            print("THIS IS THE UNTRANSFORMED DATA ")
        sub_df = df[dr_factors]
        create_histogram_plot(sub_df, dr_factors, 'Untransformed data', savedir + 'UntransformedData.svg')
        
    # x = get_data_matrix(df,dr_factors)

    #### End Dev part ####

    # print("DR factors used were" + str(dr_factors))
    # sub_df = df[dr_factors]

    x= sub_df.values
    # rs = RobustScaler(quantile_range=(0,95)) #Check usage of this scalar
    ## THIS IS WHAT YOU HAD ##
    # g = StandardScaler().fit_transform(x)
    if scalingmethod == 'minmax': #log2minmax minmax powertransformer
        x_ = MinMaxScaler().fit_transform(x)
        if verbose:
            print(f"Applied {scalingmethod} scaling (MinMaxScaler)")
    elif scalingmethod in ['standardscaler', 'standard']:  # Support both naming conventions
        x_ = StandardScaler().fit_transform(x)
        if verbose:
            print(f"Applied {scalingmethod} scaling (StandardScaler)")
    elif scalingmethod == 'log2minmax':
        negative_FACTORS = []
        positive_FACTORS = []
        if verbose:
            print('Using log2 and then minmax scaling for factors with positive values, and minmax scaling for factors with negative values')
        for factor in dr_factors:
            if np.min(df[factor]) < 0:
                # print('factor ' + factor + ' has negative values')
                negative_FACTORS.append(factor)
            else:
                # print('factor ' + factor + ' has no negative values')
                positive_FACTORS.append(factor)
        
        pos_df = df[positive_FACTORS]
        pos_x = pos_df.values
        neg_df = df[negative_FACTORS]
        neg_x = neg_df.values
        # neg_x_ = MinMaxScaler().fit_transform(neg_x)
        if len(neg_x[0]) == 0: #This controls for an edge case in which there are no negative factors - must be implemented in the other transforms as well (pipelines and clustering)
            if verbose:
                print('No negative factors present at all!')
            neg_x_ = neg_x
        else:
            neg_x_ = MinMaxScaler().fit_transform(neg_x) 
        pos_x_constant = pos_x + 0.000001
        # pos_x_log = np.log2(pos_x + pos_x_constant) #possible mistake.
        pos_x_log = np.log2(pos_x_constant)
        pos_x_ = MinMaxScaler().fit_transform(pos_x_log)
        x_ = np.concatenate((pos_x_, neg_x_), axis=1)
        newcols=positive_FACTORS + negative_FACTORS

        scaled_df_here = pd.DataFrame(x_, columns = newcols)
        if verbose:
            print("THIS IS THE " + str(scalingmethod) + " TRANSFORMED DATA ")
        create_histogram_plot(scaled_df_here, newcols, 'Transformed data', savedir + str(scalingmethod) + '.svg')

    elif scalingmethod == 'choice':
        if verbose:
            print('Factors to be scaled using log2 and then minmax:')

        # Use custom factor lists if provided, otherwise fall back to defaults
        if factors_to_transform is not None and factors_not_to_transform is not None:
            # User provided custom lists
            FactorsToTransform_actual = [f for f in dr_factors if f in factors_to_transform]
            FactorsNottotransform_actual = [f for f in dr_factors if f in factors_not_to_transform]
            
            # Check for factors that aren't in either list
            unassigned_factors = [f for f in dr_factors if f not in factors_to_transform and f not in factors_not_to_transform]
            if unassigned_factors and verbose:
                print(f"Warning: These factors are not assigned to either list and will be log-transformed by default: {unassigned_factors}")
                FactorsToTransform_actual.extend(unassigned_factors)
                
        elif factors_not_to_transform is not None:
            # User only provided factors NOT to transform
            FactorsNottotransform_actual = [f for f in dr_factors if f in factors_not_to_transform]
            FactorsToTransform_actual = [f for f in dr_factors if f not in factors_not_to_transform]
            
        elif factors_to_transform is not None:
            # User only provided factors TO transform
            FactorsToTransform_actual = [f for f in dr_factors if f in factors_to_transform]
            FactorsNottotransform_actual = [f for f in dr_factors if f not in factors_to_transform]
            
        else:
            # Use default hardcoded lists
            if AVERAGE_TIME_WINDOWS:
                FactorsNOTtotransform = ['arrest_coefficient_tmean', 'rip_L_tmean', 'rip_p_tmean', 'rip_K_tmean', 'eccentricity_tmean', 'orientation_tmean', 'directedness_tmean', 'turn_angle_tmean', 'dir_autocorr_tmean', 'glob_turn_deg_tmean']
                FactorsNottotransform_actual=[]
                FactorsToTransform_actual=[]
                for factor in tmeansdrfactors:
                    if factor in FactorsNOTtotransform:
                       if verbose:
                           print('Factor: ' + factor + ' will not be transformed')
                       FactorsNottotransform_actual.append(factor) 
                    else:
                        if verbose:
                            print('Factor: ' + factor + ' will be transformed')
                        FactorsToTransform_actual.append(factor)                   
            else:
                FactorsNOTtotransform = ['arrest_coefficient', 'rip_L', 'rip_p', 'rip_K', 'eccentricity', 'orientation', 'directedness', 'turn_angle', 'dir_autocorr', 'glob_turn_deg']
                FactorsNottotransform_actual=[]
                FactorsToTransform_actual=[]
                for factor in dr_factors:
                    if factor in FactorsNOTtotransform:
                        if verbose:
                            print('Factor: ' + factor + ' will not be transformed')
                        FactorsNottotransform_actual.append(factor)
                    else:
                        if verbose:
                            print('Factor: ' + factor + ' will be transformed')
                        FactorsToTransform_actual.append(factor)
        
        # Print what will be transformed vs not
        if verbose:
            print(f"Factors TO transform (log2 + minmax): {FactorsToTransform_actual}")
            print(f"Factors NOT to transform (minmax only): {FactorsNottotransform_actual}")

        # Get data for transformation with detailed error checking
        trans_df = df[FactorsToTransform_actual]
        trans_x = trans_df.values
        nontrans_df = df[FactorsNottotransform_actual]
        nontrans_x = nontrans_df.values
        
        if verbose:
            print("\n=== DETAILED TRANSFORMATION DEBUGGING ===")
            
            # Check original data for problems
            print("Checking original data...")
            for i, factor in enumerate(FactorsToTransform_actual):
                factor_data = trans_x[:, i]
                n_nan = np.isnan(factor_data).sum()
                n_inf = np.isinf(factor_data).sum()
                min_val = np.nanmin(factor_data)
                max_val = np.nanmax(factor_data)
                n_zero = (factor_data == 0).sum()
                n_negative = (factor_data < 0).sum()
                
                print(f"  {factor}: NaN={n_nan}, Inf={n_inf}, Min={min_val:.6f}, Max={max_val:.6f}, Zeros={n_zero}, Negatives={n_negative}")
                
                if n_nan > 0 or n_inf > 0:
                    print(f"    ⚠️  {factor} has problematic values before transformation!")
        
        # Add constant and check
        trans_x_constant = trans_x + 0.000001
        if verbose:
            print(f"\nAfter adding constant 0.000001:")
            for i, factor in enumerate(FactorsToTransform_actual):
                factor_data = trans_x_constant[:, i]
                min_val = np.nanmin(factor_data)
                max_val = np.nanmax(factor_data)
                print(f"  {factor}: Min={min_val:.6f}, Max={max_val:.6f}")
        
        # Apply log2 transformation with error checking
        if verbose:
            print(f"\nApplying log2 transformation...")
        try:
            trans_x_log = np.log2(trans_x_constant)
            
            # Check for problems after log transformation
            if verbose:
                for i, factor in enumerate(FactorsToTransform_actual):
                    factor_data = trans_x_log[:, i]
                    n_nan = np.isnan(factor_data).sum()
                    n_inf = np.isinf(factor_data).sum()
                    min_val = np.nanmin(factor_data)
                    max_val = np.nanmax(factor_data)
                    
                    print(f"  {factor} after log2: NaN={n_nan}, Inf={n_inf}, Min={min_val:.6f}, Max={max_val:.6f}")
                    
                    if n_nan > 0 or n_inf > 0:
                        print(f"    ❌ {factor} has NaN/Inf values after log2 transformation!")
                        # Show some actual values for debugging
                        print(f"    Original sample values: {trans_x[:5, i]}")
                        print(f"    After +constant: {trans_x_constant[:5, i]}")
                        print(f"    After log2: {trans_x_log[:5, i]}")
                    
        except Exception as e:
            if verbose:
                print(f"Error during log2 transformation: {e}")
            raise
        
        # Apply MinMaxScaler to log-transformed data
        if verbose:
            print(f"\nApplying MinMaxScaler to log-transformed data...")
        try:
            trans_x_ = MinMaxScaler().fit_transform(trans_x_log)
            
            # Check for problems after scaling
            if verbose:
                for i, factor in enumerate(FactorsToTransform_actual):
                    factor_data = trans_x_[:, i]
                    n_nan = np.isnan(factor_data).sum()
                    n_inf = np.isinf(factor_data).sum()
                    min_val = np.nanmin(factor_data)
                    max_val = np.nanmax(factor_data)
                    
                    print(f"  {factor} after MinMax: NaN={n_nan}, Inf={n_inf}, Min={min_val:.6f}, Max={max_val:.6f}")
                    
                    if n_nan > 0 or n_inf > 0:
                        print(f"    ❌ {factor} has NaN/Inf values after MinMax scaling!")
                    
        except Exception as e:
            if verbose:
                print(f"Error during MinMax scaling of log-transformed data: {e}")
            raise
        
        # Apply MinMaxScaler to non-transformed data
        if verbose:
            print(f"\nApplying MinMaxScaler to non-transformed data...")
        try:
            nontrans_x_ = MinMaxScaler().fit_transform(nontrans_x)
            
            # Check for problems
            if verbose:
                for i, factor in enumerate(FactorsNottotransform_actual):
                    factor_data = nontrans_x_[:, i]
                    n_nan = np.isnan(factor_data).sum()
                    n_inf = np.isinf(factor_data).sum()
                    min_val = np.nanmin(factor_data)
                    max_val = np.nanmax(factor_data)
                    
                    print(f"  {factor} after MinMax: NaN={n_nan}, Inf={n_inf}, Min={min_val:.6f}, Max={max_val:.6f}")
                    
                    if n_nan > 0 or n_inf > 0:
                        print(f"    ❌ {factor} has NaN/Inf values after MinMax scaling!")
                    
        except Exception as e:
            if verbose:
                print(f"Error during MinMax scaling of non-transformed data: {e}")
            raise
        
        # Concatenate arrays
        if verbose:
            print(f"\nConcatenating arrays...")
            print(f"  Log-transformed data shape: {trans_x_.shape}")
            print(f"  Non-transformed data shape: {nontrans_x_.shape}")
        
        x_ = np.concatenate((trans_x_, nontrans_x_), axis=1)
        newcols = FactorsToTransform_actual + FactorsNottotransform_actual
        
        # Final check on concatenated data
        if verbose:
            print(f"\nFinal concatenated data check:")
            print(f"  Shape: {x_.shape}")
            total_nan = np.isnan(x_).sum()
            total_inf = np.isinf(x_).sum()
            print(f"  Total NaN values: {total_nan}")
            print(f"  Total Inf values: {total_inf}")
            
            if total_nan > 0 or total_inf > 0:
                print("  ❌ Final data contains NaN or Inf values!")
                # Find which columns have problems
                for i, col in enumerate(newcols):
                    col_data = x_[:, i]
                    col_nan = np.isnan(col_data).sum()
                    col_inf = np.isinf(col_data).sum()
                    if col_nan > 0 or col_inf > 0:
                        print(f"    Problem column: {col} (NaN={col_nan}, Inf={col_inf})")
            else:
                print("  ✅ Final data looks clean!")
            
            print("=== END DEBUGGING ===\n")
            
        scaled_df_here = pd.DataFrame(x_, columns = newcols)
        create_histogram_plot(scaled_df_here, newcols, 'Transformed data', savedir + str(scalingmethod) + '.svg')

    elif scalingmethod == 'edelblum_choice':
        if verbose:
            print('Factors to be scaled using log2 and then minmax:')

        
        FactorsNOTtotransform = ['Displacement_X','Displacement_Y','Displacement_Z','Velocity_X','Velocity_Y','Velocity_Z',]
        FactorsNottotransform_actual=[]
        FactorsToTransform_actual=[]
        for factor in dr_factors:
            if factor in FactorsNOTtotransform:
                if verbose:
                    print('Factor: ' + factor + ' will not be transformed')
                FactorsNottotransform_actual.append(factor)
            else:
                if verbose:
                    print('Factor: ' + factor + ' will be transformed')
                FactorsToTransform_actual.append(factor)

        trans_df = df[FactorsToTransform_actual]
        trans_x=trans_df.values
        nontrans_df = df[FactorsNottotransform_actual]
        nontrans_x=nontrans_df.values
        trans_x_constant=trans_x + 0.000001
        # trans_x_log = np.log2(trans_x + trans_x_constant) # 
        trans_x_log = np.log2(trans_x_constant) # This is what it should be.
        trans_x_=MinMaxScaler().fit_transform(trans_x_log)
        nontrans_x_=MinMaxScaler().fit_transform(nontrans_x)

        x_=np.concatenate((trans_x_, nontrans_x_), axis=1)
        newcols=FactorsToTransform_actual + FactorsNottotransform_actual
        scaled_df_here = pd.DataFrame(x_, columns = newcols)
        create_histogram_plot(scaled_df_here, newcols, 'Transformed data', savedir + str(scalingmethod) + '.svg')

    elif scalingmethod == 'powertransformer':    
        
        pt = PowerTransformer(method='yeo-johnson')
        x_ = pt.fit_transform(x)
        scaled_df_here = pd.DataFrame(x_, columns = sub_df.columns)
        if verbose:
            print("THIS IS THE " + str(scalingmethod) + " DATA ")
        create_histogram_plot(scaled_df_here, dr_factors, f'{scalingmethod} data', savedir + str(scalingmethod) + '.svg')
    
    elif scalingmethod in ['robust', 'robustscaler']:
        from sklearn.preprocessing import RobustScaler
        x_ = RobustScaler().fit_transform(x)
        if verbose:
            print(f"Applied {scalingmethod} scaling (RobustScaler)")
        
    elif scalingmethod in ['normalize', 'normalizer']:
        from sklearn.preprocessing import Normalizer
        x_ = Normalizer().fit_transform(x)
        if verbose:
            print(f"Applied {scalingmethod} scaling (Normalizer)")
        
    elif scalingmethod in ['quantile', 'quantileuniform']:
        from sklearn.preprocessing import QuantileTransformer
        x_ = QuantileTransformer(output_distribution='uniform').fit_transform(x)
        if verbose:
            print(f"Applied {scalingmethod} scaling (QuantileTransformer)")
        
    elif scalingmethod in ['maxabs', 'maxabsscaler']:
        from sklearn.preprocessing import MaxAbsScaler
        x_ = MaxAbsScaler().fit_transform(x)
        if verbose:
            print(f"Applied {scalingmethod} scaling (MaxAbsScaler)")
        
    elif scalingmethod in ['yeo-johnson', 'box-cox']:
        from sklearn.preprocessing import PowerTransformer
        method = 'yeo-johnson' if scalingmethod == 'yeo-johnson' else 'box-cox'
        pt = PowerTransformer(method=method)
        x_ = pt.fit_transform(x)
        if verbose:
            print(f"Applied {scalingmethod} scaling (PowerTransformer with {method})")
        
    else:
        # Catch-all for unknown scaling methods
        available_methods = ['minmax', 'standard', 'standardscaler', 'log2minmax', 'choice', 
                           'edelblum_choice', 'powertransformer', 'robust', 'normalize', 
                           'quantile', 'maxabs', 'yeo-johnson', 'box-cox']
        raise ValueError(f"Unknown scaling method: '{scalingmethod}'. "
                        f"Available methods: {available_methods}")

        ########

    # x_ = MinMaxScaler().fit_transform(x)
    # Principal component analysis ?? Not needed here right now.


    pca_df, _, _ = do_pca(x_)

    if verbose:
        print('Using standardized factors for dimensionality reduction, matrix shape: ', x_.shape)

#     elif dr_input == 'PCs':

#         x_ = pca_df.values
#         print('Using Principal Components for dimensionality reduction, matrix shape: ', x_.shape)

    # Do tSNE and insert into dataframe:
    if do_tsne == True:
        tsne_x, flag = do_open_tsne(x_,perplexity=tsne_perp)
        tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1','tSNE2'])
    else:
        if verbose:
            print('Skipping tSNE')
        tsne_df = pd.DataFrame()

    # Do UMAP and insert into dataframe:
    umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist, n_components=n_components) #aubergine
    umap_df = pd.DataFrame(data = umap_x, columns = umap_components)

    # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
    dr_df = pd.concat([df, pca_df, tsne_df, umap_df], axis=1)

    # assert list(df.index) == list(dr_df.index), 'dr_df should be the same length as input dataframe. Check indexing of input dataframe.' #trackmate removed

    return dr_df


def dr_pipeline_multiUMAPandTSNE_v2(
    df,
    dr_factors=DR_FACTORS,
    tsne_perp=TSNE_PERP,
    umap_nn=UMAP_NN,
    min_dist=UMAP_MIN_DIST,
    n_components=N_COMPONENTS,
    scalingmethod=SCALING_METHOD,
    do_tsne=True,
    factors_to_transform=None,
    factors_not_to_transform=None,
    verbose=False,
):
    """
    New version of the DR pipeline that delegates all scaling to
    data_processing.scaling.scale_features so scaling is consistent
    with clustering.
    """
    component_list = np.arange(1, n_components + 1, 1).tolist()
    umap_components = ([f'UMAP{i}' for i in component_list])

    if verbose:
        print('Running dr_pipeline_multiUMAPandTSNE_v2...')
        print('tSNE perplexity = ', tsne_perp)
        print('UMAP nearest neighbors = ', umap_nn, ' min distance = ', min_dist)
        print('Number of UMAP components = ', n_components)

    # Select factors (respect AVERAGE_TIME_WINDOWS is handled inside scale_features)
    # Apply shared scaling
    from data_processing.scaling import scale_features
    X_scaled, used_cols = scale_features(
        df=df,
        factors=dr_factors,
        method=scalingmethod,
        average_time_windows=AVERAGE_TIME_WINDOWS,
        factors_to_transform=factors_to_transform,
        factors_not_to_transform=factors_not_to_transform,
        verbose=verbose,
    )

    # PCA on scaled features (kept for consistency with v1 behavior)
    pca_df, _, _ = do_pca(X_scaled)

    if do_tsne:
        tsne_x, flag = do_open_tsne(X_scaled, perplexity=tsne_perp)
        tsne_df = pd.DataFrame(data=tsne_x, columns=['tSNE1', 'tSNE2'])
    else:
        if verbose:
            print('Skipping tSNE')
        tsne_df = pd.DataFrame()

    umap_x = do_umap(X_scaled, n_neighbors=umap_nn, min_dist=min_dist, n_components=n_components)
    umap_df = pd.DataFrame(data=umap_x, columns=umap_components)

    dr_df = pd.concat([df, pca_df, tsne_df, umap_df], axis=1)
    return dr_df

def analyze_factors_for_choice_scaling(df, factors_list, show_distributions=True):
    """
    Helper function to analyze factors and suggest which should be log-transformed vs not.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your data
    factors_list : list
        List of factor column names to analyze
    show_distributions : bool
        Whether to show histograms of factor distributions
        
    Returns:
    --------
    dict : Dictionary with suggested factor lists and analysis
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    results = {
        'suggested_to_transform': [],      # Should get log2 + minmax
        'suggested_not_to_transform': [],  # Should get minmax only
        'analysis': {}
    }
    
    print("=== FACTOR ANALYSIS FOR 'CHOICE' SCALING ===\n")
    
    for factor in factors_list:
        if factor not in df.columns:
            print(f"Warning: {factor} not found in dataframe")
            continue
            
        data = df[factor].dropna()
        
        # Calculate statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        
        # Check for different characteristics
        has_zeros = (min_val == 0)
        has_negatives = (min_val < 0)
        is_ratio_like = (min_val >= 0) and (max_val <= 1)
        is_bounded = (min_val >= -1) and (max_val <= 1)
        high_dynamic_range = (max_val / (min_val + 0.001)) > 100
        is_skewed = abs(mean_val - np.median(data)) / std_val > 0.5 if std_val > 0 else False
        
        # Decision logic
        should_transform = True
        reasons = []
        
        if is_ratio_like:
            should_transform = False
            reasons.append("appears to be ratio/proportion (0-1 range)")
        elif is_bounded and not high_dynamic_range:
            should_transform = False  
            reasons.append("appears bounded and well-scaled")
        elif has_negatives:
            should_transform = False
            reasons.append("has negative values (log not applicable)")
        elif 'ratio' in factor.lower() or 'coefficient' in factor.lower():
            should_transform = False
            reasons.append("name suggests it's already a ratio/coefficient")
        elif 'angle' in factor.lower() or 'direction' in factor.lower():
            should_transform = False
            reasons.append("appears to be angular/directional data")
        elif high_dynamic_range or is_skewed:
            should_transform = True
            reasons.append("high dynamic range or skewed distribution")
        elif 'area' in factor.lower() or 'length' in factor.lower() or 'speed' in factor.lower():
            should_transform = True
            reasons.append("appears to be measurement that benefits from log scaling")
        
        # Store results
        if should_transform:
            results['suggested_to_transform'].append(factor)
        else:
            results['suggested_not_to_transform'].append(factor)
            
        results['analysis'][factor] = {
            'min': min_val, 'max': max_val, 'mean': mean_val, 'std': std_val,
            'transform': should_transform, 'reasons': reasons
        }
        
        # Print analysis
        transform_text = "LOG-TRANSFORM" if should_transform else "MINMAX ONLY"
        print(f"{factor}: {transform_text}")
        print(f"  Range: {min_val:.3f} to {max_val:.3f}")
        print(f"  Reasons: {', '.join(reasons)}")
        print()
    
    print(f"\n=== SUMMARY ===")
    print(f"Factors TO log-transform: {len(results['suggested_to_transform'])}")
    print(f"  {results['suggested_to_transform']}")
    print(f"\nFactors NOT to log-transform: {len(results['suggested_not_to_transform'])}")  
    print(f"  {results['suggested_not_to_transform']}")
    
    if show_distributions:
        # Show histograms
        n_factors = len(factors_list)
        if n_factors > 0:
            n_cols = min(4, n_factors)
            n_rows = (n_factors + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            elif n_cols == 1:
                axes = [[ax] for ax in axes]
            
            for i, factor in enumerate(factors_list):
                if factor in df.columns:
                    row, col = i // n_cols, i % n_cols
                    ax = axes[row][col] if n_rows > 1 else axes[col]
                    
                    data = df[factor].dropna()
                    ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_title(factor, fontsize=10)
                    ax.set_ylabel('Frequency')
                    
                    # Color code the title based on recommendation
                    color = 'red' if factor in results['suggested_to_transform'] else 'blue'
                    ax.set_title(factor, color=color, fontsize=10)
            
            # Hide empty subplots
            for i in range(len(factors_list), n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row][col].set_visible(False)
                
            plt.tight_layout()
            plt.suptitle('Factor Distributions\n(Red=Log-transform, Blue=MinMax only)', y=1.02)
            plt.show()
    
    return results


def handle_nan_for_dr(df, dr_factors, method='auto', nan_threshold=30.0, verbose=True):
    """
    Handle NaN values in dataframe before dimensionality reduction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your data
    dr_factors : list
        List of factors to be used for dimensionality reduction
    method : str
        How to handle NaN values:
        - 'auto': Drop factors with >nan_threshold% NaN, then drop remaining NaN rows
        - 'drop_rows': Drop all rows containing any NaN values in DR factors
        - 'drop_factors': Drop all factors containing any NaN values
        - 'analyze_only': Just analyze and report, don't modify data
    nan_threshold : float
        For 'auto' method: percentage threshold for dropping factors (default 30%)
    verbose : bool
        Print detailed information
        
    Returns:
    --------
    tuple : (cleaned_df, cleaned_dr_factors, report_dict)
    """
    import pandas as pd
    import numpy as np
    
    if verbose:
        print(f"\n=== NaN HANDLING FOR DIMENSIONALITY REDUCTION ({method}) ===")
        print(f"Original data shape: {df.shape}")
        print(f"Original DR factors: {len(dr_factors)}")
    
    # Check which DR factors are actually in the dataframe
    dr_factors_available = [col for col in dr_factors if col in df.columns]
    missing_factors = [col for col in dr_factors if col not in df.columns]
    
    if missing_factors and verbose:
        print(f"Warning: {len(missing_factors)} DR factors not found in dataframe:")
        for col in missing_factors:
            print(f"  - {col}")
    
    # Analyze NaN values in available DR factors
    nan_summary = {}
    for col in dr_factors_available:
        nan_count = df[col].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        nan_summary[col] = {'count': nan_count, 'pct': nan_pct}
        
    problematic_factors = [col for col, info in nan_summary.items() if info['count'] > 0]
    high_nan_factors = [col for col, info in nan_summary.items() if info['pct'] > nan_threshold]
    
    if verbose and problematic_factors:
        print(f"\nFound {len(problematic_factors)} factors with NaN values:")
        for col in problematic_factors:
            info = nan_summary[col]
            marker = "⚠️" if info['pct'] > nan_threshold else "ℹ️"
            print(f"  {marker} {col}: {info['count']}/{len(df)} ({info['pct']:.1f}%) are NaN")
    
    # Apply the chosen method
    df_result = df.copy()
    dr_factors_result = dr_factors_available.copy()
    
    if method == 'auto':
        # Step 1: Drop factors with high NaN percentage
        if high_nan_factors:
            if verbose:
                print(f"\nStep 1: Dropping {len(high_nan_factors)} factors with >{nan_threshold}% NaN:")
                for col in high_nan_factors:
                    print(f"  - {col} ({nan_summary[col]['pct']:.1f}% NaN)")
            dr_factors_result = [col for col in dr_factors_result if col not in high_nan_factors]
        
        # Step 2: Drop rows with NaN in remaining factors
        if dr_factors_result:
            df_clean = df_result.dropna(subset=dr_factors_result, how='any')
            rows_dropped = len(df_result) - len(df_clean)
            if rows_dropped > 0 and verbose:
                print(f"\nStep 2: Dropped {rows_dropped} rows with NaN in remaining factors")
            df_result = df_clean
            
    elif method == 'drop_rows':
        # Drop all rows with NaN in any DR factor
        df_clean = df_result.dropna(subset=dr_factors_available, how='any')
        rows_dropped = len(df_result) - len(df_clean)
        if rows_dropped > 0 and verbose:
            print(f"Dropped {rows_dropped} rows with NaN values")
        df_result = df_clean
        
    elif method == 'drop_factors':
        # Drop all factors with any NaN values
        if problematic_factors and verbose:
            print(f"Dropping {len(problematic_factors)} factors with NaN values:")
            for col in problematic_factors:
                print(f"  - {col} ({nan_summary[col]['pct']:.1f}% NaN)")
        dr_factors_result = [col for col in dr_factors_result if col not in problematic_factors]
        
    elif method == 'analyze_only':
        # Just analyze, don't modify anything
        if verbose:
            print("Analysis only - no data modifications made")
    
    # Prepare report
    report = {
        'original_shape': df.shape,
        'final_shape': df_result.shape,
        'original_factors': len(dr_factors),
        'final_factors': len(dr_factors_result),
        'removed_factors': [col for col in dr_factors_available if col not in dr_factors_result],
        'nan_summary': nan_summary,
        'problematic_factors': problematic_factors,
        'high_nan_factors': high_nan_factors
    }
    
    if verbose:
        print(f"\n=== SUMMARY ===")
        print(f"Data shape: {df.shape} → {df_result.shape}")
        print(f"DR factors: {len(dr_factors)} → {len(dr_factors_result)}")
        if report['removed_factors']:
            print(f"Removed factors: {report['removed_factors']}")
        print("=== END NaN HANDLING ===\n")
    
    return df_result, dr_factors_result, report


def investigate_nan_causes(df, migration_factors=None, verbose=True):
    """
    Investigate why there are NaN values in migration calculations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your data
    migration_factors : list, optional
        List of migration-related factors to investigate. 
        If None, uses common migration metrics.
    verbose : bool
        Print detailed analysis
        
    Returns:
    --------
    dict : Investigation results
    """
    if migration_factors is None:
        migration_factors = [
            'euclidean_dist', 'max_dist', 'speed', 'cumulative_length', 'MSD',
            'turn_angle', 'directedness', 'arrest_coefficient', 'dir_autocorr', 'glob_turn_deg'
        ]
    
    # Filter to factors that actually exist in the dataframe
    available_factors = [f for f in migration_factors if f in df.columns]
    
    if verbose:
        print("=== INVESTIGATING NaN CAUSES IN MIGRATION METRICS ===\n")
        print(f"Dataset shape: {df.shape}")
        print(f"Investigating factors: {available_factors}\n")
    
    results = {}
    
    # 1. Check track lengths
    if 'TRACK_ID' in df.columns and 'ntpts' in df.columns:
        print("1. TRACK LENGTH ANALYSIS:")
        
        # Get one example migration factor to identify NaN rows
        example_factor = available_factors[0] if available_factors else None
        if example_factor:
            nan_mask = df[example_factor].isna()
            
            print(f"   Cells with NaN {example_factor}: {nan_mask.sum()}")
            print(f"   Cells with valid {example_factor}: {(~nan_mask).sum()}")
            
            # Compare track lengths
            if 'ntpts' in df.columns:
                valid_track_lengths = df[~nan_mask]['ntpts']
                nan_track_lengths = df[nan_mask]['ntpts']
                
                print(f"\n   Track lengths for VALID migration data:")
                print(f"     Min: {valid_track_lengths.min()}, Max: {valid_track_lengths.max()}")
                print(f"     Mean: {valid_track_lengths.mean():.1f}, Median: {valid_track_lengths.median():.1f}")
                
                print(f"\n   Track lengths for NaN migration data:")
                print(f"     Min: {nan_track_lengths.min()}, Max: {nan_track_lengths.max()}")
                print(f"     Mean: {nan_track_lengths.mean():.1f}, Median: {nan_track_lengths.median():.1f}")
                
                # Check if NaNs correlate with short tracks
                short_tracks = df['ntpts'] <= 2
                print(f"\n   Cells with ≤2 timepoints: {short_tracks.sum()}")
                print(f"   Overlap with NaN migration: {(short_tracks & nan_mask).sum()}")
                
                results['track_length_analysis'] = {
                    'valid_lengths': valid_track_lengths.describe(),
                    'nan_lengths': nan_track_lengths.describe(),
                    'short_tracks_count': short_tracks.sum(),
                    'short_tracks_overlap_nan': (short_tracks & nan_mask).sum()
                }
    
    # 2. Check by condition
    if 'Condition_shortlabel' in df.columns:
        print(f"\n2. CONDITION-SPECIFIC ANALYSIS:")
        
        for factor in available_factors[:3]:  # Check first 3 factors
            print(f"\n   NaN counts by condition for {factor}:")
            condition_nan_counts = df.groupby('Condition_shortlabel')[factor].apply(lambda x: x.isna().sum())
            condition_total_counts = df.groupby('Condition_shortlabel')[factor].count()
            condition_nan_pct = (condition_nan_counts / (condition_nan_counts + condition_total_counts) * 100)
            
            for condition in df['Condition_shortlabel'].unique():
                nan_count = condition_nan_counts.get(condition, 0)
                total = condition_nan_counts.get(condition, 0) + condition_total_counts.get(condition, 0)
                pct = condition_nan_pct.get(condition, 0)
                print(f"     {condition}: {nan_count}/{total} ({pct:.1f}%) are NaN")
    
    # 3. Check temporal distribution
    if 'frame' in df.columns and available_factors:
        print(f"\n3. TEMPORAL DISTRIBUTION ANALYSIS:")
        
        example_factor = available_factors[0]
        nan_mask = df[example_factor].isna()
        
        print(f"   Frame range: {df['frame'].min()} to {df['frame'].max()}")
        print(f"   Frames with NaN {example_factor}:")
        
        nan_frames = df[nan_mask]['frame'].unique()
        valid_frames = df[~nan_mask]['frame'].unique()
        
        print(f"     NaN present in {len(nan_frames)} frames: {sorted(nan_frames)[:10]}{'...' if len(nan_frames) > 10 else ''}")
        print(f"     Valid data in {len(valid_frames)} frames")
        
        # Check if all frames have some NaN
        all_frames = set(df['frame'].unique())
        frames_with_nan = set(nan_frames)
        frames_without_nan = all_frames - frames_with_nan
        
        print(f"     Frames with NO NaN values: {len(frames_without_nan)} frames")
        if len(frames_without_nan) < 10:
            print(f"       {sorted(frames_without_nan)}")
    
    # 4. Check for specific track examples
    if 'TRACK_ID' in df.columns and available_factors:
        print(f"\n4. EXAMPLE PROBLEMATIC TRACKS:")
        
        example_factor = available_factors[0]
        nan_mask = df[example_factor].isna()
        
        # Get some example tracks with NaN values
        nan_tracks = df[nan_mask]['TRACK_ID'].unique()[:5]
        
        for track_id in nan_tracks:
            track_data = df[df['TRACK_ID'] == track_id]
            print(f"\n   Track {track_id}:")
            print(f"     Frames: {track_data['frame'].min()} to {track_data['frame'].max()}")
            print(f"     Length: {len(track_data)} timepoints")
            print(f"     NaN factors: {[f for f in available_factors if track_data[f].isna().all()]}")
            
            # Check if ANY migration metrics are valid for this track
            any_valid = any(~track_data[f].isna().all() for f in available_factors)
            print(f"     Any valid migration data: {any_valid}")
    
    # 5. Check for patterns in the data processing
    print(f"\n5. DATA PROCESSING CLUES:")
    
    # Check if included column exists and correlates
    if 'included' in df.columns:
        included_counts = df['included'].value_counts()
        print(f"   'included' column values: {dict(included_counts)}")
        
        if available_factors:
            example_factor = available_factors[0]
            nan_mask = df[example_factor].isna()
            
            # Cross-tabulate included vs NaN status
            crosstab = pd.crosstab(df['included'], nan_mask, margins=True)
            print(f"   Cross-tab 'included' vs NaN {example_factor}:")
            print(f"     {crosstab}")
    
    # Check time window settings
    print(f"\n6. CONFIGURATION CHECK:")
    try:
        from cellPLATO.initialization.config import MIG_T_WIND, SAMPLING_INTERVAL
        print(f"   Migration time window: {MIG_T_WIND} frames")
        print(f"   Sampling interval: {SAMPLING_INTERVAL} minutes/frame")
        print(f"   Time window in minutes: {MIG_T_WIND * SAMPLING_INTERVAL}")
        
        if 'ntpts' in df.columns:
            max_track_length = df['ntpts'].max()
            print(f"   Max track length: {max_track_length} frames")
            if MIG_T_WIND >= max_track_length:
                print(f"   ⚠️  WARNING: Time window ({MIG_T_WIND}) >= max track length ({max_track_length})!")
    except ImportError:
        print("   Could not import config values")
    
    # 7. Deep dive into windowing logic
    if 'TRACK_ID' in df.columns and 'frame' in df.columns and available_factors:
        print(f"\n7. WINDOWING ANALYSIS:")
        
        try:
            from cellPLATO.initialization.config import MIG_T_WIND
            half_window = MIG_T_WIND // 2
            print(f"   Half window size: {half_window} frames")
            
            example_factor = available_factors[0]
            
            # Check a few example tracks in detail
            print(f"\n   Detailed track analysis:")
            
            # Get a track with NaN and a track without NaN
            nan_mask = df[example_factor].isna()
            nan_track_ids = df[nan_mask]['TRACK_ID'].unique()[:3]
            valid_track_ids = df[~nan_mask]['TRACK_ID'].unique()[:3]
            
            for track_type, track_ids in [("NaN", nan_track_ids), ("Valid", valid_track_ids)]:
                print(f"\n   {track_type} tracks:")
                for track_id in track_ids:
                    track_data = df[df['TRACK_ID'] == track_id].sort_values('frame')
                    frames = track_data['frame'].values
                    
                    print(f"     Track {track_id}:")
                    print(f"       Frames: {frames.min():.0f} to {frames.max():.0f} (length: {len(frames)})")
                    print(f"       Frame sequence: {frames[:10].astype(int).tolist()}{'...' if len(frames) > 10 else ''}")
                    
                    # Check if frames are consecutive
                    consecutive = all(frames[i+1] - frames[i] == 1 for i in range(len(frames)-1))
                    print(f"       Consecutive frames: {consecutive}")
                    
                    # Check which frames could theoretically have migration data
                    valid_for_migration = []
                    for frame in frames:
                        # Check if this frame can have a full window around it
                        window_start = frame - half_window
                        window_end = frame + half_window
                        frames_in_window = frames[(frames >= window_start) & (frames <= window_end)]
                        if len(frames_in_window) == MIG_T_WIND:
                            valid_for_migration.append(frame)
                    
                    print(f"       Frames that SHOULD have migration data: {len(valid_for_migration)}/{len(frames)}")
                    if len(valid_for_migration) > 0:
                        print(f"         Examples: {valid_for_migration[:5]}")
                    
                    # Check actual migration data for first few frames
                    migration_values = track_data[example_factor].values
                    nan_count = np.isnan(migration_values).sum()
                    print(f"       Actual NaN count in {example_factor}: {nan_count}/{len(migration_values)}")
                    
                    if len(track_data) > 0:
                        break  # Just check first track of each type
                        
        except ImportError:
            print("   Could not import MIG_T_WIND for detailed analysis")
    
    print(f"\n=== END INVESTIGATION ===")
    
    return results


def debug_dr_pipeline_nan_creation(df_before, df_after, verbose=True):
    """
    Debug what's causing wholesale NaN creation in the DR pipeline.
    
    Parameters:
    -----------
    df_before : pandas.DataFrame
        Dataframe before DR pipeline
    df_after : pandas.DataFrame  
        Dataframe after DR pipeline with NaN issues
    verbose : bool
        Print detailed debugging info
        
    Returns:
    --------
    dict : Analysis of what went wrong
    """
    import pandas as pd
    import numpy as np
    
    if verbose:
        print("=== DEBUGGING DR PIPELINE NaN CREATION ===\n")
        
        print(f"BEFORE DR pipeline:")
        print(f"  Shape: {df_before.shape}")
        print(f"  Total NaN values: {df_before.isna().sum().sum()}")
        
        print(f"\nAFTER DR pipeline:")
        print(f"  Shape: {df_after.shape}")
        print(f"  Total NaN values: {df_after.isna().sum().sum()}")
        
        # Check if shapes match
        if df_before.shape[0] != df_after.shape[0]:
            print(f"⚠️  ROW COUNT MISMATCH: {df_before.shape[0]} → {df_after.shape[0]}")
        
        # Identify completely NaN rows
        completely_nan_rows_after = df_after.isna().all(axis=1)
        nan_row_count = completely_nan_rows_after.sum()
        
        if nan_row_count > 0:
            print(f"\n🚨 FOUND {nan_row_count} completely NaN rows in output!")
            
            # Get indices of NaN rows
            nan_indices = df_after[completely_nan_rows_after].index.tolist()
            print(f"   NaN row indices: {nan_indices[:10]}{'...' if len(nan_indices) > 10 else ''}")
            
            # Check if these indices existed in original data
            if hasattr(df_before, 'index'):
                original_indices = df_before.index.tolist()
                missing_from_original = [idx for idx in nan_indices if idx not in original_indices]
                if missing_from_original:
                    print(f"   ⚠️  {len(missing_from_original)} NaN row indices weren't in original data!")
                    print(f"      New indices: {missing_from_original[:10]}")
            
            # Check if we can match rows by position
            if len(df_before) == len(df_after):
                print(f"\n   Checking row-by-row alignment...")
                
                # Compare a few key columns that should never change
                key_cols = [col for col in ['TRACK_ID', 'frame', 'x', 'y'] if col in df_before.columns and col in df_after.columns]
                
                if key_cols:
                    print(f"   Comparing key columns: {key_cols}")
                    
                    for col in key_cols[:2]:  # Check first 2 key columns
                        before_vals = df_before[col].values
                        after_vals = df_after[col].values
                        
                        # Count mismatches (excluding both being NaN)
                        both_valid = ~(pd.isna(before_vals) | pd.isna(after_vals))
                        mismatches = np.sum(before_vals[both_valid] != after_vals[both_valid])
                        
                        print(f"     {col}: {mismatches} mismatches in non-NaN values")
                        
                        # Check if NaN rows in after correspond to valid rows in before
                        nan_mask_after = pd.isna(after_vals)
                        valid_in_before_nan_in_after = np.sum(~pd.isna(before_vals[nan_mask_after]))
                        
                        print(f"     {col}: {valid_in_before_nan_in_after} rows were valid before but NaN after")
            
            # Check for specific patterns
            print(f"\n   Pattern analysis:")
            
            # Check if NaN rows are clustered
            nan_positions = np.where(completely_nan_rows_after)[0]
            if len(nan_positions) > 1:
                gaps = np.diff(nan_positions)
                consecutive_groups = np.sum(gaps == 1)
                print(f"     Consecutive NaN row groups: {consecutive_groups}")
                print(f"     NaN row position range: {nan_positions.min()} to {nan_positions.max()}")
            
            # Check if certain tracks are affected
            if 'TRACK_ID' in df_before.columns:
                # Get track IDs from before that correspond to NaN rows
                if len(df_before) == len(df_after):
                    affected_tracks = df_before.iloc[nan_positions]['TRACK_ID'].unique()
                    print(f"     Affected tracks: {len(affected_tracks)} unique track IDs")
                    print(f"     Example affected tracks: {affected_tracks[:5].tolist()}")
        
        # Check for partial NaN patterns
        partial_nan_cols = []
        for col in df_after.columns:
            nan_count = df_after[col].isna().sum()
            if 0 < nan_count < len(df_after):
                partial_nan_cols.append((col, nan_count))
        
        if partial_nan_cols:
            print(f"\n   Columns with partial NaN values:")
            for col, count in partial_nan_cols[:10]:
                pct = (count / len(df_after)) * 100
                print(f"     {col}: {count} ({pct:.1f}%)")
    
    # Check if this might be an indexing/merging issue
    analysis = {
        'shape_change': df_before.shape != df_after.shape,
        'completely_nan_rows': completely_nan_rows_after.sum() if 'completely_nan_rows_after' in locals() else 0,
        'total_nan_increase': df_after.isna().sum().sum() - df_before.isna().sum().sum(),
    }
    
    if verbose:
        print(f"\n=== END DEBUGGING ===")
    
    return analysis


def clean_factors_for_scaling(df, factors_list, remove_nan=True, remove_inf=True, remove_constant=True, 
                              remove_high_nan_fraction=0.5, verbose=True):
    """
    Clean factor list by removing problematic factors that can cause scaling issues.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your data
    factors_list : list
        Original list of factors
    remove_nan : bool
        Remove factors with any NaN values
    remove_inf : bool
        Remove factors with any infinite values
    remove_constant : bool
        Remove factors with constant values (no variation)
    remove_high_nan_fraction : float
        Remove factors where more than this fraction of values are NaN
    verbose : bool
        Print detailed cleaning report
        
    Returns:
    --------
    list : Cleaned factor list
    """
    import numpy as np
    
    cleaned_factors = []
    removed_factors = {}
    
    if verbose:
        print("=== CLEANING FACTORS FOR SCALING ===\n")
    
    for factor in factors_list:
        if factor not in df.columns:
            removed_factors[factor] = "not found in dataframe"
            if verbose:
                print(f"❌ {factor}: not found in dataframe")
            continue
            
        data = df[factor]
        
        # Check for various problems
        n_total = len(data)
        n_nan = data.isna().sum()
        n_inf = np.isinf(data.fillna(0)).sum()  # fillna to avoid NaN issues with isinf
        n_constant = (data.nunique() <= 1)
        nan_fraction = n_nan / n_total
        
        # Check specific conditions
        reasons = []
        should_remove = False
        
        if remove_nan and n_nan > 0:
            should_remove = True
            reasons.append(f"has {n_nan} NaN values")
            
        if remove_inf and n_inf > 0:
            should_remove = True
            reasons.append(f"has {n_inf} infinite values")
            
        if remove_constant and n_constant:
            should_remove = True
            reasons.append("has constant values (no variation)")
            
        if nan_fraction > remove_high_nan_fraction:
            should_remove = True
            reasons.append(f"has {nan_fraction:.1%} NaN values (>{remove_high_nan_fraction:.1%})")
        
        if should_remove:
            removed_factors[factor] = "; ".join(reasons)
            if verbose:
                print(f"❌ {factor}: {'; '.join(reasons)}")
        else:
            cleaned_factors.append(factor)
            if verbose:
                stats = f"Range: {data.min():.3f} to {data.max():.3f}"
                if n_nan > 0:
                    stats += f", NaN: {n_nan}"
                print(f"✅ {factor}: {stats}")
    
    if verbose:
        print(f"\n=== CLEANING SUMMARY ===")
        print(f"Original factors: {len(factors_list)}")
        print(f"Cleaned factors: {len(cleaned_factors)}")
        print(f"Removed factors: {len(removed_factors)}")
        
        if removed_factors:
            print(f"\nRemoved factors and reasons:")
            for factor, reason in removed_factors.items():
                print(f"  {factor}: {reason}")
    
    return cleaned_factors


def trajectory_clustering_pipeline(dr_df, traj_factor=CLUSTER_BY, zeroed=False, dist_metric='hausdorff', filename_out='trajectory_clustering_test'):

    cell_df_list = get_cell_df_list(dr_df,length_threshold=10)


    traj_list = get_trajectories(cell_df_list,traj_factor=traj_factor, interp_pts=20, zeroed=zeroed)

    '''
    A few alternative processing steps that aren't currently in use,
    kept here for potentially working them back in at a later date.
    '''

    # traj_list = get_trajectories(cell_df_list,traj_factor='tSNE',
    #                              interp_pts=None, zeroed=False, method='segment')

    # traj_list = simplify_trajectories(traj_list, method='rdp', param=1)
    # traj_list = simplify_trajectories(traj_list, method='vw', param=5)

    # seg_list = get_trajectory_segments(traj_list)
    # D = trajectory_distances(seg_list, method='frechet')# 'hausdorff', 'dtw'



    D = trajectory_distances(traj_list, method=dist_metric)# 'hausdorff', 'dtw'
    eps = find_max_clusters(D)

    # Cluster the distance matrix for the trajectories list
    cluster_lst = cluster_trajectories(traj_list, D, eps, filename_out+'_'+traj_factor)


    # Add the trajectory id (label) back into the original dataframe.
    traj_clust_df = traj_clusters_2_df(dr_df, cell_df_list, cluster_lst)

    return traj_clust_df, traj_list, cluster_lst


def comparative_visualization_pipeline(df, num_factors=NUM_FACTORS, factor_pairs=FACTOR_PAIRS):


    '''
    A pipeline that produces graphics for each of the requested numerical factors
    and pairs of factors

    Input:
        df: DataFrame containing multiple conditions to compare
        num_factors: local list of numerical factors to generate plots from
            default: NUM_FACTORS defined in config
        factor_pairs: local list of factor pairs to generate plots from
            default: FACTOR_PAIRS defined in config

    '''

    # Be sure there are no NaNs before starting the visualization pipeline.
    df.dropna(inplace=True)

    assert len(df['Condition'].unique()) > 1, 'comparative_visualization_pipeline() must have >1 unique conditions in input dataframe'

    # Process a time-averaged DataFrame
    tavg_df = time_average(df)

    # Make summary calculations from time-averaged dataframe
    #Per condition:
    avg_df = average_per_condition(tavg_df)

    # Per replicate
    repavg_df = average_per_condition(tavg_df, avg_per_rep=True)

    '''
    A hypothesis-testing, p-value calculation function would go here,
    and run its own loop over factors, generating a single .txt file

    ** May be better WITHIN average_per_condition() **
    '''


    # Create comparative plots for each of the numerical factors
    for factor in tqdm(num_factors, desc="Generating comparative visualizations"):
    # for factor in num_factors:

        '''
        Summary comparison of means between conditions and replicates

        Note:
            - Displaying error bars for SDEV or SEM will be more complicated, will involve creating other dataframes
            or adding another nested set of measurements to the existing avg_df or repavg_df, in the functions where
            they are being generated, average_per_condition().

        For now, the comparative_bar() plots in comparative_visualization.py exists as a convenience function and placeholder
        for something potentially more elaborate in the future.

        '''

        stats_table(tavg_df, factor)

        if DRAW_BARPLOTS:
            cond_stats = average_per_condition(tavg_df, avg_per_rep=False)
            comparative_bar(cond_stats, x='Condition', y=factor, to_plot='avg',title='_per_condition_')

            rep_stats = average_per_condition(tavg_df, avg_per_rep=True)
            comparative_bar(rep_stats, x='Replicate_ID', y=factor, to_plot='avg', title='_per_replicate_')

            # Make sure to only output the N's once:
            if factor == 'area':
                comparative_bar(cond_stats, x='Condition', y=factor, to_plot='n',title='_per_condition_')
                comparative_bar(rep_stats, x='Replicate_ID', y=factor, to_plot='n', title='_per_replicate_')

        if DRAW_SNS_BARPLOTS:
            comparative_SNS_bar(tavg_df, save_path=BAR_SNS_DIR)

        if DRAW_SUPERPLOTS:
            # Time-averaged superplots
            superplots_plotly(tavg_df, factor, t='timeaverage')
            # superplots(tavg_df,factor , t='timeaverage')

        if DRAW_SUPERPLOTS_grays:
            # Time-averaged superplots
            superplots_plotly_grays(tavg_df, factor, t='timeaverage')
            # superplots(tavg_df,factor , t='timeaverage')

        if DRAW_DIFFPLOTS:
           # Time-averaged plots-of-differences
            plots_of_differences_plotly(tavg_df, factor=factor, ctl_label=CTL_LABEL)
            plots_of_differences_sns(tavg_df, factor=factor, ctl_label=CTL_LABEL)

        if DRAW_TIMEPLOTS:
            # print('Time superplots..')
            # time_superplot(df, factor)
            multi_condition_timeplot(df, factor)
            timeplots_of_differences(df, factor=factor)

    if DRAW_MARGSCAT:
        # for pair in tqdm(factor_pairs):
        for pair in factor_pairs:

            '''
            Marginal xy scatterplots (including hex plots and contours)
            Exports plots with all conditions combined, then per condition.
            '''

            if(AXES_LIMITS == 'min-max'):
                # Calculate bounds of entire set:
                x_min = np.min(tavg_df[pair[0]])
                x_max = np.max(tavg_df[pair[0]])
                y_min = np.min(tavg_df[pair[1]])
                y_max = np.max(tavg_df[pair[1]])

            elif(AXES_LIMITS == '2-sigma'):
                # Set the axes limits custom (2 sigma)
                x_min = np.mean(tavg_df[pair[0]]) - 2 * np.std(tavg_df[pair[0]])
                x_max = np.mean(tavg_df[pair[0]]) + 2 * np.std(tavg_df[pair[0]])
                y_min = np.mean(tavg_df[pair[1]]) - 2 * np.std(tavg_df[pair[1]])
                y_max = np.mean(tavg_df[pair[1]]) + 2 * np.std(tavg_df[pair[1]])

            elif(AXES_LIMITS == '3-sigma'):
                # Set the axes limits custom (3 sigma)
                x_min = np.mean(tavg_df[pair[0]]) - 3 * np.std(tavg_df[pair[0]])
                x_max = np.mean(tavg_df[pair[0]]) + 3 * np.std(tavg_df[pair[0]])
                y_min = np.mean(tavg_df[pair[1]]) - 3 * np.std(tavg_df[pair[1]])
                y_max = np.mean(tavg_df[pair[1]]) + 3 * np.std(tavg_df[pair[1]])

            bounds = x_min, x_max, y_min, y_max

            marginal_xy(tavg_df, pair, plot_type = 'scatter', renderer='seaborn') # All conditions shown.
            marginal_xy(tavg_df, pair, plot_type = 'hex', renderer='seaborn') # All conditions shown.
            marginal_xy(tavg_df, pair, plot_type = 'contour', renderer='seaborn') # All conditions shown.

            # Generate separate plots for each condition.
            for cond in tavg_df['Condition'].unique():
                cond_sub_df = tavg_df[tavg_df['Condition'] == cond]
                if USE_SHORTLABELS:
                    this_label = cond_sub_df['Condition_shortlabel'].unique()[0]
                else:
                    this_label = cond

                marginal_xy(cond_sub_df, pair, plot_type = 'contour', renderer='seaborn', bounds=bounds, supp_label=this_label)
                marginal_xy(cond_sub_df, pair, plot_type = 'hex', renderer='seaborn', bounds=bounds,supp_label=this_label)
                marginal_xy(cond_sub_df, pair, plot_type = 'scatter', renderer='seaborn', bounds=bounds, supp_label=this_label)

def cluster_analysis_pipeline(df,cluster_by, eps=None):#=CLUSTER_BY):

    '''
    A <pipeline> function that groups together elements of the workflow that involve clustering on tSNE, PCA or XY data,
    generating a labelled DataFrame (lab_dr_df), and generating plots that compare subgroup behaviors.


    Input:
        df: DataFrame - containing dimension reduced columns. (tSNE1, tSNE2, PC1...)
        cluster_by: 'tSNE', 'PCA' or 'xy'
        ***Also allow overriding of clustering values EPS and min_samples??**
            defaults to CLUSTER_BY global defined in the config.

    Returns:
        lab_dr_df: DataFrame containing an extra column, 'label' for each cells group ID at a given timepoint,
        where a value of -1 is for uncategorized cells.

    '''

    print('Clustering by: ' + cluster_by)
    print('Plot output folder :',CLUST_DIR)

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'
        curr_save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
        print('DBScan clustering by x,y position...')

    elif cluster_by == 'pca':
        x_name = 'PC1'
        y_name = 'PC2'
        curr_save_path = CLUST_PCA_DIR
        print('DBScan clustering by principal components...')

    elif cluster_by == 'tsne':
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        curr_save_path = CLUST_TSNE_DIR
        print('DBScan clustering by tSNE...')

    elif cluster_by == 'umap':
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        curr_save_path = CLUST_TSNE_DIR
        print('DBScan clustering by tSNE...')

    # Allow override of the EPS from config in dbscan_clustering
    if eps is not None:
        # Cluster the input dataframe by the specified factors
        lab_dr_df = dbscan_clustering(df,cluster_by=cluster_by, eps=eps,
                                      plot=True, save_path=curr_save_path)
    else:
        # Cluster the input dataframe by the specified factors
        lab_dr_df = hdbscan_clustering(df,cluster_by=cluster_by,
                                      plot=True, save_path=curr_save_path)

    lab_count_df = get_label_counts(lab_dr_df, per_rep=True)


    # # Plot the counts per subgroup in a swarm plot
    # fig = px.strip(lab_count_df, x="label", y="count", color="Condition")
    #
    # if STATIC_PLOTS:
    #     fig.write_image(curr_save_path+'cluster_counts_' + cluster_by + '.png')
    #
    # if PLOTS_IN_BROWSER:
    #     fig.show()

    for factor in NUM_FACTORS:

        plots_of_differences_plotly(lab_dr_df, factor, ctl_label=-1, cust_txt=cluster_by+'_clustered_', save_path=curr_save_path)


    # For each label, create a subfolder and export superplots (seaborn)
    for label in lab_dr_df['label'].unique():

        print('label: ', label)
        curr_label_path = os.path.join(curr_save_path,'label_'+str(label)+'/')
        print('curr_label_path: ', curr_label_path)
        if not os.path.exists(curr_label_path):
             os.makedirs(curr_label_path)

        # Get the sub_df for this label
        this_lab_df = lab_dr_df[lab_dr_df['label'] == label]

        # print('Reminder, cannot time-average labelled datasets, producing plots without time-averaging')

        for factor in NUM_FACTORS:

            if(DRAW_DIFFPLOTS):
                plots_of_differences_plotly(this_lab_df, factor=factor,
                                            ctl_label=CTL_LABEL, save_path=curr_label_path)

    return lab_dr_df



def cluster_switching_pipeline(lab_dr_df):

    assert 'label' in lab_dr_df.columns, 'dataframe passed to cluster_switching_pipeline() must have cluster <labels>'

    # Count the cluster changes
    sum_labels, tptlabel_dr_df = count_cluster_changes(lab_dr_df)

    time_superplot(tptlabel_dr_df, 'n_changes',t_window=None)

    clust_sum_df = cluster_purity(lab_dr_df)

    trajclust_sum_df = cluster_purity(lab_dr_df, cluster_label='traj_id')

    purity_plots(lab_dr_df, clust_sum_df,lab_dr_df,trajclust_sum_df)

    clust_sum_t_df = cluster_composition_timecourse(lab_dr_df)
    cluster_timeplot(clust_sum_t_df)

    # Count the number of cells that fall into each cluster - show on per condition and replicate basis.
    lab_count_df = get_label_counts(lab_dr_df, per_rep=True)

    # # Plot the counts per subgroup in a swarm plot
    # fig = px.strip(lab_count_df, x="label", y="count", color="Condition")
    # fig.show()

def sweep_dbscan(df,eps_vals,cluster_by):

    '''
    Convenience function to parameter sweep values of eps using dbscan
    while showing a progress bar.

        Input:
        perp_range: tuple (start, end, number)
    '''

    # Parameter sweep values of eps
    for eps in tqdm(eps_vals):
        _ = cluster_vis_sns(df,float(eps), cluster_by)

def sweep_tsne_umap(df_in, perp_min_max=(10,300), nn_min_max=(2, 100),n_vals=11,nested_dbscan_sweep=False, dr_factors=DR_FACTORS, eps_vals=None):

    df = df_in.copy()

    perp_low, perp_high = perp_min_max
    nn_low, nn_high = nn_min_max

    perp_vals = np.round(np.linspace(perp_low,perp_high,n_vals))
    nn_vals = np.round(np.linspace(nn_low,nn_high,n_vals))

    # OPTIONAL: nested DBscan sweep.
    methods = ['umap', 'tsne']


    for (perp, nn) in list(zip(perp_vals, nn_vals)):
        print('---')
        dr_df = dr_pipeline(df, dr_factors=dr_factors,tsne_perp=perp,umap_nn=nn)  #, dr_input='PCs'
        pca_df, components, expl = do_pca(df[dr_factors])

        f = dimension_reduction_subplots(dr_df,pca_tuple=[pca_df, components, expl])

        if STATIC_PLOTS:
            # Export to parameter sweeping folder:
            identifier = 'sweep_perp_'+str(perp)+'_nn_' + str(nn)
            plt.savefig(DR_PARAMS_DIR+identifier+'.png', dpi=300)


        # Optional nested loop of DBscan clustering.
        if(nested_dbscan_sweep):
            for dr_method in methods:
                for eps in eps_vals:

                    lab_dr_df = dbscan_clustering(dr_df,eps=eps,cluster_by=dr_method, plot=False)
                    print('dbscan sweep of ',dr_method,' with eps: ', eps, 'generated cluster count of: ', len(lab_dr_df['label'].unique()))
                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[30,10])
                    draw_cluster_hulls(lab_dr_df,cluster_by=dr_method, color_by='condition',legend=False,ax=ax1,draw_pts=True,save_path=CLUST_PARAMS_DIR+'condition_')
                    draw_cluster_hulls(lab_dr_df,cluster_by=dr_method, color_by='PCs',ax=ax2,draw_pts=True,save_path=CLUST_PARAMS_DIR+'pca_')
                    draw_cluster_hulls(lab_dr_df,cluster_by=dr_method, color_by='cluster',ax=ax3,draw_pts=True,save_path=CLUST_PARAMS_DIR+'cluster_')

                    if STATIC_PLOTS:
                        plt.savefig(CLUST_PARAMS_DIR+dr_method+'_sweep_eps_'+str(eps)+'nclusters_'+str(len(lab_dr_df['label'].unique()))+'.png', dpi=300)

    print('Finished dimensionality-reduction sweep, see output in directory: ',DR_PARAMS_DIR)


def compare_mig_shape_factors(df_in,dr_factors_mig,dr_factors_shape, dr_factors_all,perp=TSNE_PERP,umap_nn=UMAP_NN, min_dist=UMAP_MIN_DIST):

    df = df_in.copy()

    ''' Migration Calculations'''

    # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
    x = get_data_matrix(df,dr_factors_mig)
    x_ = StandardScaler().fit_transform(x)
    pca_df, _, _ = do_pca(x_)
    tsne_x, flag = do_open_tsne(x_,perplexity=perp)

    # Format tSNE results into a dataframe
    tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])

    # Use standrard scaler upstream of tSNE.
    umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist)
    umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])

    # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
    dr_df_mig = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)


    '''Shape Metrics'''
    # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
    x = get_data_matrix(df,dr_factors_shape)
    x_ = StandardScaler().fit_transform(x)
    pca_df, _, _ = do_pca(x_)

    tsne_x, flag = do_open_tsne(x_,perplexity=perp)

    # Format tSNE results into a dataframe
    tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])

    # Use standrard scaler upstream of tSNE.
    umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist)
    umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])

    # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
    dr_df_shape = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)



    '''All metrics together'''

    # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
    x = get_data_matrix(df,dr_factors_all)

    x_ = StandardScaler().fit_transform(x)
    pca_df, _, _ = do_pca(x_)
    tsne_x, flag = do_open_tsne(x_,perplexity=perp)

    # Format tSNE results into a dataframe
    tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])

    # Use standrard scaler upstream of tSNE.
    umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist)
    umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])

    # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
    dr_df_all = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)

    # return dr_df_mig, dr_df_shape, dr_df_all
    # plot_mig_shape_factors(dr_df_mig,dr_df_shape,dr_df_all)

    plt.clf()
    main_title =  'Compare tSNE & UMAP input groups,  '
    # subtitle = ' cmapping: ' + fac1 + ', ' + fac2
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,15))
    # fig1.suptitle(main_title+subtitle,fontsize=20)
    # cmap='cmy'
    ax1.set_title('Migration tSNE')
    ax2.set_title('Cell shape tSNE')
    ax3.set_title('Ensemble tSNE')

    ax4.set_title('Migration UMAP')
    ax5.set_title('Cell shape UMAP')
    ax6.set_title('Ensemble UMAP')

    ax1.scatter(x=dr_df_mig['tSNE1'],y=dr_df_mig['tSNE2'], s=0.5, c=colormap_pcs(dr_df_mig))#, cmap='viridis')
    ax2.scatter(x=dr_df_shape['tSNE1'],y=dr_df_shape['tSNE2'], s=0.5, c=colormap_pcs(dr_df_shape))#, cmap='viridis')
    ax3.scatter(x=dr_df_all['tSNE1'],y=dr_df_all['tSNE2'], s=0.5, c=colormap_pcs(dr_df_all))#, cmap='viridis')

    ax4.scatter(x=dr_df_mig['UMAP1'],y=dr_df_mig['UMAP2'], s=0.5, c=colormap_pcs(dr_df_mig))#dr_df_mig[fac2], cmap='plasma')
    ax5.scatter(x=dr_df_shape['UMAP1'],y=dr_df_shape['UMAP2'], s=0.5, c=colormap_pcs(dr_df_shape))#=dr_df_shape[fac2], cmap='plasma')
    ax6.scatter(x=dr_df_all['UMAP1'],y=dr_df_all['UMAP2'], s=0.5, c=colormap_pcs(dr_df_all))#=dr_df_all[fac2], cmap='plasma')

    _df1 = hdbscan_clustering(dr_df_mig,cluster_by='tSNE', plot=False)
    _df2 = hdbscan_clustering(dr_df_shape,cluster_by='tSNE', plot=False)
    _df3 = hdbscan_clustering(dr_df_all,cluster_by='tSNE', plot=False)
    _df4 = hdbscan_clustering(dr_df_mig,cluster_by='UMAP', plot=False)
    _df5 = hdbscan_clustering(dr_df_shape,cluster_by='UMAP', plot=False)
    _df6 = hdbscan_clustering(dr_df_all,cluster_by='UMAP', plot=False)

    draw_cluster_hulls(_df1,cluster_by='tSNE', color_by='condition',legend=False,ax=ax1,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
    draw_cluster_hulls(_df2,cluster_by='tSNE', color_by='condition',legend=False,ax=ax2,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
    draw_cluster_hulls(_df3,cluster_by='tSNE', color_by='condition',legend=False,ax=ax3,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
    draw_cluster_hulls(_df4,cluster_by='UMAP', color_by='condition',legend=False,ax=ax4,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
    draw_cluster_hulls(_df5,cluster_by='UMAP', color_by='condition',legend=False,ax=ax5,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
    draw_cluster_hulls(_df6,cluster_by='UMAP', color_by='condition',legend=False,ax=ax6,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')


    fig.show()

    if STATIC_PLOTS:
        fig.savefig(DR_PARAMS_DIR+'umap_'+str(umap_nn)+'_tSNE_'+str(perp)+'_mig_shape.png', dpi=300)
