# pipelines.py

from initialization.config import *
from initialization.initialization import *

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
    calc_aspect_ratio(comb_df, drop=True)# if there are resulting nans

    # Clean up remaining dataframe, and calibrate the micron-dependent values
    cleaned_df = clean_comb_df(comb_df, deduplicate=False)
    # comb_df = factor_calibration(cleaned_df)

    print('Calibrating with mixed_scaling = ', MIXED_SCALING)
    comb_df = factor_calibration(cleaned_df,mixed_calibration=mixed)
    apply_unique_id(comb_df)

    if(SELF_STANDARDIZE):
        print('Self-standardizing factors: ',FACTORS_TO_STANDARDIZE)
        comb_df = standardize_factors_per_cell(comb_df,FACTORS_TO_STANDARDIZE)

    add_fac_list = None
    if(AVERAGE_TIME_WINDOWS): #this is averaging across time windows, but lets change name to factor time window or something. CHANGE NAME. FACTOR_TIMEAVERAGE AVERAGE_TIME_WINDOWS

        # Which factors should we calculate acorss the cells time window.
        # factors_to_timeaverage = DR_FACTORS
        print('Time-averaging CHANGE THIS NAME factors: ', factors_to_timeaverage)
        # comb_df, add_fac_list = t_window_metrics(std_comb_df,factor_list=factors_to_timeaverage)
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

def dr_pipeline_multiUMAPandTSNE(df, dr_factors=DR_FACTORS, tsne_perp=TSNE_PERP, umap_nn=UMAP_NN,min_dist=UMAP_MIN_DIST, n_components=N_COMPONENTS, scalingmethod=SCALING_METHOD):

    component_list=np.arange(1, n_components+1,1).tolist()
    from sklearn.preprocessing import PowerTransformer
    savedir = CLUST_DISAMBIG_DIR

    umap_components=([f'UMAP{i}' for i in component_list])
    # tsne_components=([f'tSNE{i}' for i in component_list])

    print('Running dr_pipeline for multi dimension UMAP and tSNE...')
    print('tSNE perplexity = ',tsne_perp)
    print('UMAP nearest neighbors = ', umap_nn, ' min distance = ',min_dist)
    print('Number of UMAP components = ', n_components)

    # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
    # x = get_data_matrix(df,dr_factors)
    print("DR factors used were" + str(dr_factors))
    sub_df = df[dr_factors]

    print("THIS IS THE UNTRANSFORMED DATA ")
    sub_df.hist(column=dr_factors, bins = 160, figsize=(20, 10),color = "black", ec="black")
    plt.tight_layout()
    # plt.show()
    plt.savefig(savedir+ 'UntransformedData.png')

    x= sub_df.values
    # rs = RobustScaler(quantile_range=(0,95)) #Check usage of this scalar
    ## THIS IS WHAT YOU HAD ##
    # g = StandardScaler().fit_transform(x)
    if scalingmethod == 'minmax': #log2minmax minmax powertransformer
        x_ = MinMaxScaler().fit_transform(x)
    elif scalingmethod == 'log2minmax':

        negative_FACTORS = []
        positive_FACTORS = []
        for factor in dr_factors:
            if np.min(df[factor]) < 0:
                print('factor ' + factor + ' has negative values')
                negative_FACTORS.append(factor)
                
            else:
                print('factor ' + factor + ' has no negative values')
                positive_FACTORS.append(factor)
        
        
        pos_df = df[positive_FACTORS]
        pos_x = pos_df.values
        neg_df = df[negative_FACTORS]
        neg_x = neg_df.values
        # neg_x_ = MinMaxScaler().fit_transform(neg_x)
        if len(neg_x[0]) == 0: #This controls for an edge case in which there are no negative factors - must be implemented in the other transforms as well (pipelines and clustering)
            print('No negative factors at all!')
            neg_x_ = neg_x
        else:
            neg_x_ = MinMaxScaler().fit_transform(neg_x) 
        pos_x_constant = pos_x + 0.000001
        pos_x_log = np.log2(pos_x + pos_x_constant)
        pos_x_ = MinMaxScaler().fit_transform(pos_x_log)
        x_ = np.concatenate((pos_x_, neg_x_), axis=1)
        newcols=positive_FACTORS + negative_FACTORS

        scaled_df_here = pd.DataFrame(x_, columns = newcols)
        print("THIS IS THE " + str(scalingmethod) + " TRANSFORMED DATA ")
        scaled_df_here.hist(column=newcols, bins = 160, figsize=(20, 10),color = "black", ec="black")
        plt.tight_layout()
        # plt.show()
        plt.savefig(savedir+ str(scalingmethod) +'.png')

    elif scalingmethod == 'powertransformer':    
        
        pt = PowerTransformer(method='yeo-johnson')
        x_ = pt.fit_transform(x)
        scaled_df_here = pd.DataFrame(x_, columns = sub_df.columns)
        print("THIS IS THE " + str(scalingmethod) + " DATA ")
        scaled_df_here.hist(column=dr_factors, bins = 160, figsize=(20, 10),color = "black", ec="black")
        plt.tight_layout()
        plt.show()
        plt.savefig(savedir+ str(scalingmethod) +'.png')


        ########

    # x_ = MinMaxScaler().fit_transform(x)
    # Principal component analysis ?? Not needed here right now.


    pca_df, _, _ = do_pca(x_)

    print('Using standardized factors for dimensionality reduction, matrix shape: ', x_.shape)

#     elif dr_input == 'PCs':

#         x_ = pca_df.values
#         print('Using Principal Components for dimensionality reduction, matrix shape: ', x_.shape)

    # Do tSNE and insert into dataframe:
    tsne_x, flag = do_open_tsne(x_,perplexity=tsne_perp)
    tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1','tSNE2'])

    # Do UMAP and insert into dataframe:
    umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist, n_components=N_COMPONENTS)
    umap_df = pd.DataFrame(data = umap_x, columns = umap_components)

    # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
    dr_df = pd.concat([df, pca_df, tsne_df, umap_df], axis=1)

    assert list(df.index) == list(dr_df.index), 'dr_df should be the same length as input dataframe. Check indexing of input dataframe.'

    return dr_df

# def dr_pipeline_multiUMAPandTSNE_DEV(df, dr_factors=DR_FACTORS, tsne_perp=TSNE_PERP, umap_nn=UMAP_NN,min_dist=UMAP_MIN_DIST, n_components=N_COMPONENTS, scalingmethod='arcsinh', positive_FACTORS=DR_FACTORS, negative_FACTORS=DR_FACTORS ):

#     component_list=np.arange(1, n_components+1,1).tolist()
#     from sklearn.preprocessing import RobustScaler
#     from sklearn.preprocessing import MaxAbsScaler
#     from sklearn.preprocessing import PowerTransformer
#     savedir = CLUST_DEV_DIR

#     umap_components=([f'UMAP{i}' for i in component_list])
#     # tsne_components=([f'tSNE{i}' for i in component_list])

#     print('Running dr_pipeline for multi dimension UMAP and tSNE...')
#     print('tSNE perplexity = ',tsne_perp)
#     print('UMAP nearest neighbors = ', umap_nn, ' min distance = ',min_dist)
#     print('Number of UMAP components = ', n_components)

#     # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
#     # x = get_data_matrix(df,dr_factors)
#     print("DR factors used were" + str(dr_factors))
#     sub_df = df[dr_factors]
#     print("THIS IS THE non TRANSFORMED DATA")
#     sub_df.hist(column=dr_factors, bins = 160, figsize=(20, 10),color = "black", ec="black")
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(savedir+'NOTTRANSFORMED'+'.png')

#     x= sub_df.values
#     # rs = RobustScaler(quantile_range=(0,95)) #Check usage of this scalar
#     ## THIS IS WHAT YOU HAD ##
#     # g = StandardScaler().fit_transform(x)

#     # This is what WAS there
#     # x_ = MinMaxScaler().fit_transform(x)

#     if scalingmethod == 'arcsinh':
#         x_ = np.arcsinh(x)
#     elif scalingmethod == 'minmax':
#         x_ = MinMaxScaler().fit_transform(x)
#     elif scalingmethod == 'standard':
#         x_ = StandardScaler().fit_transform(x)    
#     elif scalingmethod == 'arcsinhminmax':
#         x_ = np.arcsinh(x)
#         x_ = MinMaxScaler().fit_transform(x_)
#     elif scalingmethod == 'minmaxarcsinh':
#         x_ = MinMaxScaler().fit_transform(x)
#         x_ = np.arcsinh(x_)
#     elif scalingmethod == 'arcsinhstandard':
#         x_ = np.arcsinh(x)
#         x_ = StandardScaler().fit_transform(x_)
#     elif scalingmethod =='robust':
#         x_ = RobustScaler().fit_transform(x)
#     elif scalingmethod =='robustarcsinh':
#         x_ = RobustScaler().fit_transform(x)
#         # x_ = np.arcsinh(x_)
#     elif scalingmethod == 'log2':
#         # factorswithnonegatives = 
#         # sub_df = df[dr_factors]
#         # x= sub_df.values
#         print('number of nan values is ' + str(sub_df.isnull().sum()))
        
        
#         count = -np.isinf(sub_df).values.sum()
#         print("DF contains " + str(count) + " infinite values")
#         x=sub_df.values
#         count2=-np.isinf(x).sum()
#         print("x contains " + str(count2) + " infinite values")

#         zerocount = (sub_df == 0).sum(axis=0)
#         print('Count of zeros is as follows')
#         display(zerocount)

#         sub_df_onlyfinites=sub_df[np.isfinite(sub_df)]
#         count = -np.isinf(sub_df_onlyfinites).values.sum()
#         print("sub_df_onlyfinites contains " + str(count) + " infinite values")
#         # x2=sub_df_onlyfinites.values
#         # count3=np.sum(-np.isinf(x2))
#         # print("x2 from the non finite contains " + str(count3) + " infinite values")


#         # x_ = np.log2(x - (np.min(x) - 1))
#         x_ = np.log2(x+0.00001)
#     elif scalingmethod == 'log2minmax':
#         xplusconstant = x + 0.00001
#         log2xplusconstant = np.log2(xplusconstant)
#         x_ = MinMaxScaler().fit_transform(log2xplusconstant)


#         print("x contains " + str(-np.isinf(x).sum()) + " negative infinite values")
#         print("xplusconstant contains " + str(-np.isinf(xplusconstant).sum()) + " negative infinite values")
#         print("log2xplusconstant contains " + str(-np.isinf(log2xplusconstant).sum()) + " negative infinite values")
#         print("MinMaxScaledlog2xplusconstant data contains " + str(-np.isinf(x_).sum()) + " negative infinite values")

#         print("x contains " + str(np.isnan(x).sum()) + " NaN values")
#         print("xplusconstant contains " + str(np.isnan(xplusconstant).sum()) + " NaN values")
#         print("log2xplusconstant contains " + str(np.isnan(log2xplusconstant).sum()) + " NaN values")
#         print("MinMaxScaledlog2xplusconstant data contains " + str(np.isnan(x_).sum()) + " NaN values")

#     elif scalingmethod =='minmaxlog2':
#         # xplusconstant = x + 0.00001
#         minmaxscaled = MinMaxScaler().fit_transform(x)
#         minmaxscaledplusconstant = minmaxscaled + 0.00001
#         x_ = np.log2(minmaxscaledplusconstant)
        
#         print("x contains " + str(-np.isinf(x).sum()) + " negative infinite values")
#         print("minmaxscaled contains " + str(-np.isinf(minmaxscaled).sum()) + " negative infinite values")
#         print("minmaxscaledplusconstant contains " + str(-np.isinf(minmaxscaledplusconstant).sum()) + " negative infinite values")
#         print("MinMaxScaledlog2xplusconstant data contains " + str(-np.isinf(x_).sum()) + " negative infinite values")

#         print("x contains " + str(np.isnan(x).sum()) + " NaN values")
#         print("minmaxscaled contains " + str(np.isnan(minmaxscaled).sum()) + " NaN values")
#         print("minmaxscaledplusconstant contains " + str(np.isnan(minmaxscaledplusconstant).sum()) + " NaN values")
#         print("MinMaxScaledlog2xplusconstant data contains " + str(np.isnan(x_).sum()) + " NaN values")

#         print('x contains  ' + str(np.count_nonzero(x==0)) + ' zero values')
#         print('minmaxscaled contains  ' + str(np.count_nonzero(minmaxscaled==0)) + ' zero values')
#         print('minmaxscaledplusconstant contains  ' + str(np.count_nonzero(minmaxscaledplusconstant==0)) + ' zero values')
#         print('MinMaxScaledlog2xplusconstant contains  ' + str(np.count_nonzero(x_==0)) + ' zero values')


#     elif scalingmethod == 'maxabs':
#         x_ = MaxAbsScaler().fit_transform(x)

#     elif scalingmethod == 'log2minmaxorjustminmax':
#         negative_FACTORS = []
#         positive_FACTORS = []
#         for factor in dr_factors:
#             if np.min(df[factor]) < 0:
#                 print('factor ' + factor + ' has negative values')
#                 negative_FACTORS.append(factor)
                
#             else:
#                 print('factor ' + factor + ' has no negative values')
#                 positive_FACTORS.append(factor)
        
#         display(positive_FACTORS)
#         display(negative_FACTORS)
#         pos_df = df[positive_FACTORS]
#         pos_x = pos_df.values
#         print("pos_x contains " + str(-np.isinf(pos_x).sum()) + " negative infinite values")
#         print("pos_x contains " + str(np.isnan(pos_x).sum()) + " NaN values")
#         print('pos_x contains  ' + str(np.count_nonzero(pos_x==0)) + ' zero values')

        
#         neg_df = df[negative_FACTORS]
#         neg_x = neg_df.values
#         print("neg_x contains " + str(-np.isinf(neg_x).sum()) + " negative infinite values")
#         print("neg_x contains " + str(np.isnan(neg_x).sum()) + " NaN values")
#         print('neg_x contains  ' + str(np.count_nonzero(neg_x==0)) + ' zero values')
#         neg_x_ = MinMaxScaler().fit_transform(neg_x)


#         pos_x_constant = pos_x + 0.000001
#         pos_x_log = np.log2(pos_x + pos_x_constant)
#         pos_x_ = MinMaxScaler().fit_transform(pos_x_log)

        

#         # notransform_df = df[notransform_FACTORS]
#         # notransform_x = notransform_df.values
#         # print("notransform_x contains " + str(-np.isinf(notransform_x).sum()) + " negative infinite values")
#         # print("notransform_x contains " + str(np.isnan(notransform_x).sum()) + " NaN values")
#         # print('notransform_x contains  ' + str(np.count_nonzero(notransform_x==0)) + ' zero values')


#         x_ = np.concatenate((pos_x_, neg_x_), axis=1)
#         newcols=positive_FACTORS + negative_FACTORS

#         scaled_df_here = pd.DataFrame(x_, columns = newcols)
#         print("THIS IS THE " + str(scalingmethod) + " DATA ACTUALLY")
#         scaled_df_here.hist(column=newcols, bins = 160, figsize=(20, 10),color = "black", ec="black")
#         plt.tight_layout()
#         plt.show()
#         plt.savefig(savedir+ str(scalingmethod) +'.png')

#     elif scalingmethod == 'powertransformer':    
        
#         pt = PowerTransformer(method='yeo-johnson')
#         x_ = pt.fit_transform(x)



#         scaled_df_here = pd.DataFrame(x_, columns = sub_df.columns)
#         print("THIS IS THE " + str(scalingmethod) + " DATA ")
#         scaled_df_here.hist(column=dr_factors, bins = 160, figsize=(20, 10),color = "black", ec="black")
#         plt.tight_layout()
#         plt.show()
#         plt.savefig(savedir+ str(scalingmethod) +'.png')
        
    

#     # minmaxscaled = MinMaxScaler().fit_transform(x)

#     # print("THIS IS THE " + str(scalingmethod) + " DATA")
#     # minmaxscaled_df = pd.DataFrame(x_, columns = sub_df.columns)
#     # minmaxscaled_df.hist(column=dr_factors, bins = 160, figsize=(20, 10),color = "black", ec="black")
#     # plt.tight_layout()
#     # plt.show()
#     # plt.savefig(savedir+ str(scalingmethod) +'.png')

#     # Perform arcsinh transformation on the data
#     # x_ = np.arcsinh(x)

#     # if dominmax == True:
#     #     x_ = MinMaxScaler().fit_transform(x_)


#     # print("THIS IS THE TRANSFORMED DATA")
#     # scaled_subset_df = pd.DataFrame(x_, columns = sub_df.columns)
#     # scaled_subset_df.hist(column=dr_factors, bins = 160, figsize=(20, 10),color = "black", ec="black")
#     # plt.tight_layout()
#     # plt.show()
#     # plt.savefig(savedir+'ArcSinhTransformed'+'.png')

#     ##
#     # Principal component analysis ?? Not needed here right now.
#     pca_df, _, _ = do_pca(x_)

#     print('Using standardized factors for dimensionality reduction, matrix shape: ', x_.shape)

# #     elif dr_input == 'PCs':

# #         x_ = pca_df.values
# #         print('Using Principal Components for dimensionality reduction, matrix shape: ', x_.shape)



#     # Do UMAP and insert into dataframe:
#     umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist, n_components=N_COMPONENTS)
#     umap_df = pd.DataFrame(data = umap_x, columns = umap_components)

#     # Do tSNE and insert into dataframe:
#     if do_tsne == True:
#         tsne_x, flag = do_open_tsne(x_,perplexity=tsne_perp)
#         tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1','tSNE2'])

#     # tsne_x, flag = do_open_tsne(x_,perplexity=tsne_perp)
#     # tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1','tSNE2'])

#     # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
#     # dr_df = pd.concat([df, pca_df, tsne_df, umap_df], axis=1)
#     dr_df = pd.concat([df, umap_df], axis=1)

#     assert list(df.index) == list(dr_df.index), 'dr_df should be the same length as input dataframe. Check indexing of input dataframe.'

#     return dr_df



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
    for factor in tqdm(num_factors):
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
        print('Processing factor: ', factor)

        print('Processing statistics...')
        stats_table(tavg_df, factor)


        if DRAW_BARPLOTS:
            print('Exporting Comparative bar charts... ')
            
            cond_stats = average_per_condition(tavg_df, avg_per_rep=False)
            comparative_bar(cond_stats, x='Condition', y=factor, to_plot='avg',title='_per_condition_')

            rep_stats = average_per_condition(tavg_df, avg_per_rep=True)
            comparative_bar(rep_stats, x='Replicate_ID', y=factor, to_plot='avg', title='_per_replicate_')


            # Make sure to only output the N's once:
            if factor == 'area':
                comparative_bar(cond_stats, x='Condition', y=factor, to_plot='n',title='_per_condition_')
                comparative_bar(rep_stats, x='Replicate_ID', y=factor, to_plot='n', title='_per_replicate_')

        if DRAW_SNS_BARPLOTS:
            print('Exporting gray SNS barplots with points...')
            comparative_SNS_bar(tavg_df, save_path=BAR_SNS_DIR)

        if DRAW_SUPERPLOTS:
            print('Exporting static Superplots...')

            # Time-averaged superplots
            superplots_plotly(tavg_df, factor, t='timeaverage')
            # superplots(tavg_df,factor , t='timeaverage')

        if DRAW_SUPERPLOTS_grays:
            print('Exporting static gray Superplots...')

            # Time-averaged superplots
            superplots_plotly_grays(tavg_df, factor, t='timeaverage')
            # superplots(tavg_df,factor , t='timeaverage')

        if DRAW_DIFFPLOTS:
            print('Exporting static Plots of Differences')
           # Time-averaged plots-of-differences
            plots_of_differences_plotly(tavg_df, factor=factor, ctl_label=CTL_LABEL)
            plots_of_differences_sns(tavg_df, factor=factor, ctl_label=CTL_LABEL)

        if DRAW_TIMEPLOTS:
            print('Exporting static Timeplots')
            # print('Time superplots..')
            # time_superplot(df, factor)
            multi_condition_timeplot(df, factor)
            timeplots_of_differences(df, factor=factor)


    if DRAW_MARGSCAT:
        print('Exporting static Marginal scatterplots')
        print('Processing factor pairs: ')

        # for pair in tqdm(factor_pairs):
        for pair in factor_pairs:

            print('Currently generating scatter, hex, contour plots for pair: ', pair)

            '''
            Consider refactoring the section below into a more concise function.
            '''

            if(AXES_LIMITS == 'min-max'):
                # Calculate bounds of entire set:
                x_min = np.min(tavg_df[pair[0]])
                x_max = np.max(tavg_df[pair[0]])
                y_min = np.min(tavg_df[pair[1]])
                y_max = np.max(tavg_df[pair[1]])

            elif(AXES_LIMITS == '2-sigma'):
                # Set the axes limits custom (3 sigma)
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


            bounds = x_min, x_max,y_min, y_max
            print(AXES_LIMITS, ' bounds: ', x_min, x_max, y_min, y_max)

            # Make the combined versions of all plots first.
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
