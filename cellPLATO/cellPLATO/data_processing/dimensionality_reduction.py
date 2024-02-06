#dimensionality_reduction.py

from initialization.initialization import *
from initialization.config import *

from data_processing.clustering import *
from data_processing.data_wrangling import get_data_matrix
# from data_processing.pipelines import *

# from visualization.cluster_visualization import *
# from visualization.low_dimension_visualization import *


import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as skTSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from openTSNE.callbacks import ErrorLogger
'''
openTSNE: a modular Python library for t-SNE dimensionality reduction and embedding
https://www.biorxiv.org/content/early/2019/08/13/731877
'''
import umap

'''
McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
'''
import matplotlib.pyplot as plt
import plotly.graph_objects as go




'''
UMAP functions
'''

def do_umap(datax, n_neighbors=15, min_dist=0.0, n_components=3, metric='euclidean', plot=False): #changed variable name to datax

    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=42,
                        n_components=n_components,
                        low_memory=True)
    # print("UMAP input data contains " + str(-np.isinf(datax).sum()) + " negative infinite values")
    # print("UMAP input data contains " + str(np.isinf(datax).sum()) + " positive infinite values")
    # print("UMAP input data contains " + str(np.isnan(datax).sum()) + " NaN values")
    

    max = datax.max(axis=0, keepdims=True)
    min = datax.min(axis=0, keepdims=True)
    print('max:', max)
    print('min:', min)
    # reducer.fit(datax)

    # embedding = reducer.transform(datax)
    embedding = reducer.fit_transform(datax)
    print("UMAP output data contains " + str(-np.isinf(embedding).sum()) + " negative infinite values")
    assert(np.all(embedding == reducer.embedding_))
    print('Embedding shape: ',embedding.shape)

    if plot:
        plt.scatter(embedding[:, 0], embedding[:, 1],s=2)#, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title('UMAP projection', fontsize=24)
        plt.show()

    return embedding

def test_umap_rembedding(x, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', plot=False):


    x_train, x_test = train_test_split(x, test_size=0.5,
                                       random_state=42)

    print('test, train:', x_train.shape, x_test.shape)

    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=42,
                        n_components=n_components)

    reducer.fit(x_train) # Fit on the training data.

    # First, visualize embedding of the training data, this should be normal
    embedding = reducer.transform(x_train)
    assert(np.all(embedding == reducer.embedding_))
    print('Embedding shape: ',embedding.shape)

    if plot:
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title('x_train', fontsize=24);
        plt.show()

    # Test that error thrown if trying to re-embed data of different input dimensionality.
    x_test = x_test[:,:-3]
    print(x_test.shape)

    # Then, create a new embedding with the test data
    _embedding = reducer.transform(x_test)

    if plot:
        plt.scatter(_embedding[:, 0], _embedding[:, 1], cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title('x_test', fontsize=24);
        plt.show()

    return embedding


def umap_sweep(x, n_neighbors_list=[10,15,20], min_dist_list=[0.1,0.25,0.5,0.8,0.99]):

    for n_neighbors in [10,15,20]:
        for min_dist in min_dist_list:

            print(n_neighbors,min_dist)

            umap_x = do_umap(x,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist)

            rip = ripley_K(umap_x,0.1)
            print(np.mean(rip[:,2]))

            plt.scatter(umap_x[:, 0], umap_x[:, 1], c=rip[:,2], s=2) # Colormap by ripleys L
            plt.title('UMAP n_neighbors: '+ str(n_neighbors), fontsize=12)
            plt.show()

# def do_umap(x, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', plot=False): #Duplicate function - removed 2/2023

#     reducer = umap.UMAP(n_neighbors=n_neighbors,
#                         min_dist=min_dist,
#                         random_state=42,
#                         n_components=n_components)

#     reducer.fit(x)

#     embedding = reducer.transform(x)
#     assert(np.all(embedding == reducer.embedding_))
#     print('Embedding shape: ',embedding.shape)

#     if plot:
#         plt.scatter(embedding[:, 0], embedding[:, 1],s=2)#, cmap='Spectral', s=5)
#         plt.gca().set_aspect('equal', 'datalim')
#         plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
#         plt.title('UMAP projection', fontsize=24);
#         plt.show()

#     return embedding

def umap_array(df_in, fac1, fac2, fac3,main_title=' UMAP matrix', tag='umap_sweep_'):

    plt.clf()

    subtitle = ' cmapping: ' + fac1 + ', ' + fac2 + ', ' + fac3
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15,15))
    fig.suptitle(main_title+subtitle,fontsize=20)

    ax1.set_title('Unstandardized UMAP')
    ax2.set_title('UMAP on PCs 1-10')
    ax3.set_title('Standardized UMAP')

    ax1.scatter(x=df_in['UMAP1'],y=df_in['UMAP2'], s=0.5, c=df_in[fac1], cmap='viridis')
    ax2.scatter(x=df_in['pca_UMAP1'],y=df_in['pca_UMAP2'], s=0.5, c=df_in[fac1], cmap='viridis')
    ax3.scatter(x=df_in['ft_UMAP1'],y=df_in['ft_UMAP2'], s=0.5, c=df_in[fac1], cmap='viridis')

    ax4.scatter(x=df_in['UMAP1'],y=df_in['UMAP2'], s=0.5, c=df_in[fac2], cmap='plasma')
    ax5.scatter(x=df_in['pca_UMAP1'],y=df_in['pca_UMAP2'], s=0.5, c=df_in[fac2], cmap='plasma')
    ax6.scatter(x=df_in['ft_UMAP1'],y=df_in['ft_UMAP2'], s=0.5, c=df_in[fac2], cmap='plasma')

    if(fac3) == 'label':
        ax7.scatter(x=df_in['UMAP1'],y=df_in['UMAP2'], s=0.5,
                    c=df_in['labels'],cmap='tab20c')
        ax8.scatter(x=df_in['pca_UMAP1'],y=df_in['pca_UMAP2'], s=0.5,
                    c=df_in['pca_labels'],cmap='tab20c')
        ax9.scatter(x=df_in['ft_UMAP1'],y=df_in['ft_UMAP2'], s=0.5,
                    c=df_in['ft_labels'],cmap='tab20c')

    else:
        ax7.scatter(x=df_in['UMAP1'],y=df_in['UMAP2'], s=0.5,
                    c=df_in[fac3],cmap='cool')
        ax8.scatter(x=df_in['pca_UMAP1'],y=df_in['pca_UMAP2'], s=0.5,
                    c=df_in[fac3],cmap='cool')
        ax9.scatter(x=df_in['ft_UMAP1'],y=df_in['ft_UMAP2'], s=0.5,
                    c=df_in[fac3],cmap='cool')

    fig.show()

    if STATIC_PLOTS:
        fig.savefig(DR_PARAMS_DIR+tag+'.png', dpi=300)



def sweep_umap_params(df_in, n_neighbors_list, min_dist_list):


    # for perp in perp_vals:
    for n_neighbors in n_neighbors_list:
        min_dist_list=[0.1,0.25,0.5,0.8,0.99]

    #     n_neighbors = 10
        for min_dist in min_dist_list:

            print(n_neighbors,min_dist)

            # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
            x = get_data_matrix(df)

            # Principal component analysis
            pca_df, components, expl = do_pca(x)

            # Do UMAP on the un-scaled input data
            umap_x = do_umap(x, n_neighbors=n_neighbors, min_dist=min_dist)
            umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])


            # Run UMAP on PCA values only.
            x_ = pca_df.values
            pca_umap_x =  do_umap(x_, n_neighbors=n_neighbors, min_dist=min_dist)
            pca_umap_df = pd.DataFrame(data = pca_umap_x, columns = ['pca_UMAP1', 'pca_UMAP2'])

            # Use standrard scaler upstream of tSNE.
            x__ = StandardScaler().fit_transform(x)
            ft_umap_x = do_umap(x__, n_neighbors=n_neighbors, min_dist=min_dist)
            ft_umap_df = pd.DataFrame(data = ft_umap_x, columns = ['ft_UMAP1', 'ft_UMAP2'])

            # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
            dr_df = pd.concat([df,pca_df,umap_df,pca_umap_df,ft_umap_df], axis=1)



            # Plots
            umap_array(dr_df, 'area', 'eccentricity', 'aspect',
                        main_title='UMAP array, perplexity=' + str(min_dist)+'_nneigh_'+str(n_neighbors),
                        tag='umap_shape_mindist_'+ str(min_dist)+'_nneigh_'+str(n_neighbors))

            umap_array(dr_df, 'speed', 'cumulative_length', 'rip_L',
                        main_title='UMAP array, perplexity=' + str(min_dist)+'_nneigh_'+str(n_neighbors),
                        tag='umap_migm_indist_'+ str(min_dist)+'_nneigh_'+str(n_neighbors))

            umap_array(dr_df, 'PC1', 'PC2', 'PC3',
                        main_title='UMAP array, perplexity=' + str(min_dist)+'_nneigh_'+str(n_neighbors),
                        tag='umap_PCs_mindist_'+ str(min_dist)+'_nneigh_'+str(n_neighbors))




'''
PCA functions
'''
def do_pca(x,n_comp=10):

    '''
    Perform principal component analysis on matrix x.

    input
        x: data matrix containing features to be reduced
        n_comp: number of components to include

    '''

    # Standardizing the features
    # x = StandardScaler().fit_transform(x) # This is done upstream of PCA 2023 in the multiumap function.

    pca = PCA(n_components=n_comp)

    principalComponents = pca.fit_transform(x)

    # Name the columns as PC1, PC2 ...
    col_names = []
    for i in range(1,n_comp+1):
        col_names.append('PC'+str(i))

    # Put into a dataframe
    pca_df = pd.DataFrame(data=principalComponents, columns=col_names)

    return pca_df, pca.components_.T, pca.explained_variance_ratio_


def pca_contrib_factors(x,n):

    '''
    Return a list of the n factors that contribute most to the variance
    '''

    assert n <= x.shape[1], 'requested of factors greater than number included in pca'

    pca_df, components, expl = do_pca(x)
    factor_variance = np.sum(components*expl,axis=1)
    sorted_factors = [x for _,x in sorted(zip(factor_variance,DR_FACTORS), reverse=True)]

    return sorted_factors[:n]






'''
tSNE functions:
'''

def do_tsne(x):

    '''
    Perform tSNE analysis on matrix x.

    **Returns only the first 3  components.

    '''
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(x)

    # Format tSNE results into a dataframe
    tsne_df = pd.DataFrame(data = tsne_results, columns = ['tSNE1', 'tSNE2', 'tSNE3'])

    return tsne_df

def test_embedding_split(x):

    # Split the set into two parts for training and testing
    x_train, x_test = train_test_split(x,test_size=0.3, random_state=11)

    #openTSNE: define the parameters
    tsne = TSNE(
        perplexity=TSNE_PERP,
        n_jobs=-1,
        random_state=TSNE_R_S,
    )

    # Get the tSNE embedding of the training data, and save as np.array
    embedding_train = tsne.fit(x_train)

    # Apply tSNE embedding from the training set to the test set.
    embedding_test = embedding_train.transform(x_test)

    return embedding_train, embedding_test


def do_open_tsne(x, perplexity=TSNE_PERP):

    '''
    Based on the tSNE parameters from config.py, and whether there is an existing
    embedding file, this function determines whether to perform a fresh tSNE embedding
    on the input data, load and apply an existing embedding, when USE_SAVED_EMBEDDING is True.

    Returns:
        embedded_x: may be directly trained tSNE data OR fitted into an existing embedding.
        embed_flag: Boolean, True if used an existing embedding

    '''
    print('Using openTSNE with perplexity = ', perplexity)
    # Decide what to do with the data array x:
    if(USE_SAVED_EMBEDDING and os.path.exists(EMBEDDING_PATH+EMBEDDING_FILENAME)):

        print('Loading existing openTSNE embedding to fit input data.')
        # Load the embedding as a numpy array
        loaded_embedd_arr = np.load(EMBEDDING_PATH+EMBEDDING_FILENAME)
        loaded_x_train = np.load(EMBEDDING_PATH+TRAINX_FILENAME)

        # Build the affinities needed for the embedding.
        affinities = affinity.PerplexityBasedNN(
            loaded_x_train,
            # To be consistent with other tsne implementations
            perplexity=perplexity,
            n_jobs=-1,
            random_state=TSNE_R_S,
            # Previous values:
            # perplexity=30,
            # n_jobs=8,
            # random_state=0,
        )

        # Reconstruct the tSNEEmbedding.
        saved_embedding = TSNEEmbedding(
            loaded_embedd_arr,
            affinities,
        #     learning_rate=1000,
        #     negative_gradient_method="fft",
        #     n_jobs=8,
        #     callbacks=ErrorLogger(),
        #     random_state=42,
        )

        # Apply the saved embedding to the existing current testing data x.
        embedded_x = saved_embedding.transform(x)
        embedding_array = np.asarray(embedded_x) # as array for easy visualization
        embed_flag = True

    else:

        print('Using openTSNE to calculate new embedding for input data.')
        #openTSNE: define the parameters
        tsne = TSNE(
            perplexity=perplexity,
            n_jobs=-1,
            random_state=TSNE_R_S,
        )

        # Get the tSNE embedding of the training data, and save as np.array
        # embedding_train = tsne.fit(x)
        # embedding_array = np.asarray(embedding_train) # as array for easy visualization
        embedded_x = tsne.fit(x)
        embedding_array = np.asarray(embedded_x) # as array for easy visualization

        embed_flag = False # False if the file didn't already exist and new tsne was calculated

        if(USE_SAVED_EMBEDDING): # This only saves if the file doesn't already exist
            print('Saving new embedding to file:')
            print(EMBEDDING_PATH+EMBEDDING_FILENAME)

            if not os.path.exists(EMBEDDING_PATH):
                os.makedirs(EMBEDDING_PATH)

            np.save(EMBEDDING_PATH+EMBEDDING_FILENAME,embedding_array)
            np.save(EMBEDDING_PATH+TRAINX_FILENAME,x)

    return embedding_array, embed_flag



def test_tsne(x, perplexity=TSNE_PERP):
    #openTSNE: define the parameters
    tsne = TSNE(
        perplexity=perplexity,
        n_jobs=-1,
        random_state=TSNE_R_S,
    )

    # Create the tSNE embedding of the training data, and save as np.array
    embedded_x = tsne.fit(x)
    tsne_x = np.asarray(embedded_x) # as array for easy visualization

    return tsne_x



'''
This should replace what is presently being done in the parameter sweep for tSNE.
'''
def tsne_array(df_in, fac1, fac2, fac3,main_title=' tSNE matrix', tag='tsne_sweep_'):

    plt.clf()

    subtitle = ' cmapping: ' + fac1 + ', ' + fac2 + ', ' + fac3
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15,15))
    fig.suptitle(main_title+subtitle,fontsize=20)

    ax1.set_title('Unstandardized tSNE')
    ax2.set_title('tSNE on PCs 1-10')
    ax3.set_title('Standardized tSNE')

    ax1.scatter(x=df_in['tSNE1'],y=df_in['tSNE2'], s=0.5, c=df_in[fac1], cmap='viridis')
    ax2.scatter(x=df_in['pca_tSNE1'],y=df_in['pca_tSNE2'], s=0.5, c=df_in[fac1], cmap='viridis')
    ax3.scatter(x=df_in['ft_tSNE1'],y=df_in['ft_tSNE2'], s=0.5, c=df_in[fac1], cmap='viridis')

    ax4.scatter(x=df_in['tSNE1'],y=df_in['tSNE2'], s=0.5, c=df_in[fac2], cmap='plasma')
    ax5.scatter(x=df_in['pca_tSNE1'],y=df_in['pca_tSNE2'], s=0.5, c=df_in[fac2], cmap='plasma')
    ax6.scatter(x=df_in['ft_tSNE1'],y=df_in['ft_tSNE2'], s=0.5, c=df_in[fac2], cmap='plasma')

    if(fac3) == 'label':
        ax7.scatter(x=df_in['tSNE1'],y=df_in['tSNE2'], s=0.5,
                    c=df_in['labels'],cmap='tab20c')
        ax8.scatter(x=df_in['pca_tSNE1'],y=df_in['pca_tSNE2'], s=0.5,
                    c=df_in['pca_labels'],cmap='tab20c')
        ax9.scatter(x=df_in['ft_tSNE1'],y=df_in['ft_tSNE2'], s=0.5,
                    c=df_in['ft_labels'],cmap='tab20c')

    else:
        ax7.scatter(x=df_in['tSNE1'],y=df_in['tSNE2'], s=0.5,
                    c=df_in[fac3],cmap='cool')
        ax8.scatter(x=df_in['pca_tSNE1'],y=df_in['pca_tSNE2'], s=0.5,
                    c=df_in[fac3],cmap='cool')
        ax9.scatter(x=df_in['ft_tSNE1'],y=df_in['ft_tSNE2'], s=0.5,
                    c=df_in[fac3],cmap='cool')

    fig.show()

    if STATIC_PLOTS:
        fig.savefig(DR_PARAMS_DIR+tag+str(perp)+'.png', dpi=300)


# def compare_mig_shape_factors(df_in,dr_factors_mig,dr_factors_shape, dr_factors_all,perp=TSNE_PERP,umap_nn=UMAP_NN, min_dist=UMAP_MIN_DIST):
#
#     df = df_in.copy()
#
#     ''' Migration Calculations'''
#
#     # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
#     x = get_data_matrix(df,dr_factors_mig)
#     x_ = StandardScaler().fit_transform(x)
#     pca_df, _, _ = do_pca(x_)
#     tsne_x, flag = do_open_tsne(x_,perplexity=perp)
#
#     # Format tSNE results into a dataframe
#     tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])
#
#     # Use standrard scaler upstream of tSNE.
#     umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist)
#     umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])
#
#     # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
#     dr_df_mig = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)
#
#
#     '''Shape Metrics'''
#     # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
#     x = get_data_matrix(df,dr_factors_shape)
#     x_ = StandardScaler().fit_transform(x)
#     pca_df, _, _ = do_pca(x_)
#
#     tsne_x, flag = do_open_tsne(x_,perplexity=perp)
#
#     # Format tSNE results into a dataframe
#     tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])
#
#     # Use standrard scaler upstream of tSNE.
#     umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist)
#     umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])
#
#     # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
#     dr_df_shape = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)
#
#
#
#     '''All metrics together'''
#
#     # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
#     x = get_data_matrix(df,dr_factors_all)
#
#     x_ = StandardScaler().fit_transform(x)
#     pca_df, _, _ = do_pca(x_)
#     tsne_x, flag = do_open_tsne(x_,perplexity=perp)
#
#     # Format tSNE results into a dataframe
#     tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])
#
#     # Use standrard scaler upstream of tSNE.
#     umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist)
#     umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])
#
#     # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
#     dr_df_all = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)
#
#     return dr_df_mig, dr_df_shape, dr_df_all
#     # plot_mig_shape_factors(dr_df_mig,dr_df_shape,dr_df_all)
#
#     # plt.clf()
#     # main_title =  'Compare tSNE & UMAP input groups,  '
#     # # subtitle = ' cmapping: ' + fac1 + ', ' + fac2
#     # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,15))
#     # # fig1.suptitle(main_title+subtitle,fontsize=20)
#     # # cmap='cmy'
#     # ax1.set_title('Migration tSNE')
#     # ax2.set_title('Cell shape tSNE')
#     # ax3.set_title('Ensemble tSNE')
#     #
#     # ax4.set_title('Migration UMAP')
#     # ax5.set_title('Cell shape UMAP')
#     # ax6.set_title('Ensemble UMAP')
#     #
#     # ax1.scatter(x=dr_df_mig['tSNE1'],y=dr_df_mig['tSNE2'], s=0.5, c=colormap_pcs(dr_df_mig))#, cmap='viridis')
#     # ax2.scatter(x=dr_df_shape['tSNE1'],y=dr_df_shape['tSNE2'], s=0.5, c=colormap_pcs(dr_df_shape))#, cmap='viridis')
#     # ax3.scatter(x=dr_df_all['tSNE1'],y=dr_df_all['tSNE2'], s=0.5, c=colormap_pcs(dr_df_all))#, cmap='viridis')
#     #
#     # ax4.scatter(x=dr_df_mig['UMAP1'],y=dr_df_mig['UMAP2'], s=0.5, c=colormap_pcs(dr_df_mig))#dr_df_mig[fac2], cmap='plasma')
#     # ax5.scatter(x=dr_df_shape['UMAP1'],y=dr_df_shape['UMAP2'], s=0.5, c=colormap_pcs(dr_df_shape))#=dr_df_shape[fac2], cmap='plasma')
#     # ax6.scatter(x=dr_df_all['UMAP1'],y=dr_df_all['UMAP2'], s=0.5, c=colormap_pcs(dr_df_all))#=dr_df_all[fac2], cmap='plasma')
#     #
#     # _df1 = hdbscan_clustering(dr_df_mig,cluster_by='tSNE', plot=False)
#     # _df2 = hdbscan_clustering(dr_df_shape,cluster_by='tSNE', plot=False)
#     # _df3 = hdbscan_clustering(dr_df_all,cluster_by='tSNE', plot=False)
#     # _df4 = hdbscan_clustering(dr_df_mig,cluster_by='UMAP', plot=False)
#     # _df5 = hdbscan_clustering(dr_df_shape,cluster_by='UMAP', plot=False)
#     # _df6 = hdbscan_clustering(dr_df_all,cluster_by='UMAP', plot=False)
#     #
#     # draw_cluster_hulls(_df1,cluster_by='tSNE', color_by='condition',legend=False,ax=ax1,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
#     # draw_cluster_hulls(_df2,cluster_by='tSNE', color_by='condition',legend=False,ax=ax2,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
#     # draw_cluster_hulls(_df3,cluster_by='tSNE', color_by='condition',legend=False,ax=ax3,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
#     # draw_cluster_hulls(_df4,cluster_by='UMAP', color_by='condition',legend=False,ax=ax4,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
#     # draw_cluster_hulls(_df5,cluster_by='UMAP', color_by='condition',legend=False,ax=ax5,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
#     # draw_cluster_hulls(_df6,cluster_by='UMAP', color_by='condition',legend=False,ax=ax6,draw_pts=False,save_path=CLUST_PARAMS_DIR+'condition')
#     #
#     #
#     # fig.show()
#     #
#     # if STATIC_PLOTS:
#     #     fig.savefig(DR_PARAMS_DIR+'umap_'+str(umap_nn)+'_tSNE_'+str(perp)+'_mig_shape.png', dpi=300)
#

'''
Parameter sweeping functions
'''

# tSNE function for parameter sweeping perplexity.
def tsne_pipeline(df_in,perplexity, dr_factors=DR_FACTORS, overwrite_embedding=False, force_show=False):

    '''
    FROM param_sweep.py - May not be currently used!

    Apply tSNE to the input dataframe, with provided perplexity,
    and return results as a dataframe
    '''

    df = df_in.copy()

    # Extract data matrix for DR.
    x = df[dr_factors].values   # Matrix to be used in the dimensionality reduction

    #openTSNE: define the parameters
    tsne = TSNE(
        perplexity=perplexity,
        n_jobs=-1,
        random_state=TSNE_R_S,
    )

    # Create the tSNE embedding of the training data, and save as np.array
    embedded_x = tsne.fit(x)
    tsne_x = np.asarray(embedded_x) # as array for easy visualization

    if(overwrite_embedding):

        print('Overwriting embedding file:')
        print(EMBEDDING_PATH+EMBEDDING_FILENAME)

        if not os.path.exists(EMBEDDING_PATH):
            print('EMBEDDING_PATH ',EMBEDDING_PATH, ' Does not exist, creating folder and embedding files')
            os.makedirs(EMBEDDING_PATH)

        np.save(EMBEDDING_PATH+EMBEDDING_FILENAME,tsne_x)
        np.save(EMBEDDING_PATH+TRAINX_FILENAME,x)

    # Assemble a dataframe from the results
    tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])

    # Plot the resulting tsne in a scatter with marginal histograms
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

    g = sns.jointplot(data=tsne_df,x="tSNE1", y="tSNE2", kind='scatter',
               joint_kws={'alpha': 0.4,'s': 5}, height=10)#, legend=False)


    if STATIC_PLOTS:
        plt.savefig(DR_PARAMS_DIR+'sweep_perp_'+str(perplexity)+'.png', dpi=300)

    if PLOTS_IN_BROWSER or force_show:
        plt.show()

    plt.clf() # Clear figure because it will be run in a loop

    return tsne_x, tsne_df




def sweep_perplexity(df,perp_vals):

    '''
    Convenience function to parameter sweep values of perplexity using the tsne_pipeline
    while showing a progress bar.

    Input:
        df
        perp_range: tuple (start, end, number)

    '''


    for perp in tqdm(perp_vals):

        print('Perplexity: ',perp)
        _, __ = tsne_pipeline(df,perp)
