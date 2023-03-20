#clustering.py

from initialization.config import *
from initialization.initialization import *


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import hdbscan
from matplotlib import cm

'''
K. Chaudhuri and S. Dasgupta. “Rates of convergence for the cluster tree.”
In Advances in Neural Information Processing Systems, 2010.
'''


def dbscan_clustering(df_in, eps=EPS, min_samples=MIN_SAMPLES,cluster_by='tsne', plot=False, save_path=CLUST_DIR):

    print('dbscan_clustering() with eps = ', eps)
    # Determine how to cluster
    x_name,y_name = ' ', ' '

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'
        # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
        print('DBScan clustering by x,y position...')

    elif (cluster_by == 'pca' or cluster_by == 'PCA'):
        x_name = 'PC1'
        y_name = 'PC2'
        # save_path = CLUST_PCA_DIR
        print('DBScan clustering by principal components...')

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        # save_path = CLUST_TSNE_DIR
        print('DBScan clustering by tSNE...')

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
        x_name = 'UMAP1'
        y_name = 'UMAP2'
        # save_path = CLUST_TSNE_DIR
        print('DBScan clustering by UMAP...')


    #DBScan
    sub_set = df_in[[x_name, y_name]] # self.df
    X = StandardScaler().fit_transform(sub_set)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = ['label'])
    lab_dr_df = pd.concat([df_in,lab_df], axis=1)

    if(plot):
        # import seaborn as sns
        # import matplotlib.pyplot as plt

        ''' Eventually this plotting function should probably be in another script.'''
        scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
                          kind='scatter',
                          palette=CLUSTER_CMAP,
                          joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
        if STATIC_PLOTS:
            # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
            plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
        if PLOTS_IN_BROWSER:
            plt.show()


    print('Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')

    return lab_dr_df


# def hdbscan_clustering(df_in, min_cluster_size=20,cluster_by='tsne', plot=False, save_path=CLUST_DIR):
#
#     print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
#     # Determine how to cluster
#     x_name,y_name = ' ', ' '
#
#     if cluster_by == 'xy':
#         x_name = 'x'
#         y_name = 'y'
#         # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
#         print('DBScan clustering by x,y position...')
#
#     elif (cluster_by == 'pca' or cluster_by == 'PCA' or cluster_by == 'PCs'):
#         x_name = 'PC1'
#         y_name = 'PC2'
#         # save_path = CLUST_PCA_DIR
#         print('DBScan clustering by principal components...')
#
#     elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
#         x_name = 'tSNE1'
#         y_name = 'tSNE2'
#         # save_path = CLUST_TSNE_DIR
#         print('DBScan clustering by tSNE...')
#
#     elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
#         x_name = 'UMAP1'
#         y_name = 'UMAP2'
#         # save_path = CLUST_TSNE_DIR
#         print('DBScan clustering by UMAP...')
#
#
#     #hDBScan
#     sub_set = df_in[[x_name, y_name]] # self.df
#     X = StandardScaler().fit_transform(sub_set)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
#     labels = clusterer.fit_predict(X)
#
#     # Assemble a dataframe from the results
#     lab_df = pd.DataFrame(data = labels, columns = ['label'])
#     lab_dr_df = pd.concat([df_in,lab_df], axis=1)
#
#     if(plot):
#         # import seaborn as sns
#         # import matplotlib.pyplot as plt
#
#         ''' Eventually this plotting function should probably be in another script.'''
#         scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
#                           kind='scatter',
#                           palette=CLUSTER_CMAP,
#                           joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
#         if STATIC_PLOTS:
#             # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
#             plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
#         if PLOTS_IN_BROWSER:
#             plt.show()
#
#
#     # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')
#
#     return lab_dr_df

# def hdbscan_clustering(df_in, min_cluster_size=20,min_samples=10,cluster_by='NDIM',  metric='manhattan', plot=False): #deprecated 11-8-2022
#
#     print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
#     # Determine how to cluster
#     x_name,y_name = ' ', ' '
#
#     if cluster_by == 'xy':
#         x_name = 'x'
#         y_name = 'y'
#         # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
#         print('DBScan clustering by x,y position...')
#
#     elif (cluster_by == 'pca' or cluster_by == 'PCA' or cluster_by == 'PCs'):
#         x_name = 'PC1'
#         y_name = 'PC2'
#         # save_path = CLUST_PCA_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by principal components...')
#
#     elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
#         x_name = 'tSNE1'
#         y_name = 'tSNE2'
#         # save_path = CLUST_TSNE_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by tSNE...')
#
#     elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
#         x_name = 'UMAP1'
#         y_name = 'UMAP2'
#         CLUSTERON=[x_name, y_name]
#         # save_path = CLUST_TSNE_DIR
#         print('DBScan clustering by UMAP...')
#     elif (cluster_by == 'NDIM' or cluster_by == 'ndim'):
#         CLUSTERON = DR_FACTORS
#
#     # cluster_selection_method ='eom'
#     # epsilon=1
#
#
#     #hDBScan
#     sub_set = df_in[CLUSTERON] # self.df
#     X = StandardScaler().fit_transform(sub_set)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric)
#     labels = clusterer.fit_predict(X)
#
#     # Assemble a dataframe from the results
#     lab_df = pd.DataFrame(data = labels, columns = ['label'])
#     lab_dr_df = pd.concat([df_in,lab_df], axis=1)
#
#     if(plot):
#         CLUSTER_CMAP = 'tab20'
#         clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#         # import seaborn as sns
#         # import matplotlib.pyplot as plt
#
#         ''' Eventually this plotting function should probably be in another script.'''
#         scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
#                           kind='scatter',
#                           palette=CLUSTER_CMAP,
#                           joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
#         if STATIC_PLOTS:
#             # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
#             plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
#         if PLOTS_IN_BROWSER:
#             plt.show()
#
#
#     # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')
#
#     return lab_dr_df

# def hdbscan_clustering(df_in, min_cluster_size=20,min_samples=10,cluster_by='NDIM',  metric='manhattan', plot=False):
#
#     print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
#     # Determine how to cluster
#     x_name,y_name = ' ', ' '
#
#     if cluster_by == 'xy':
#         x_name = 'x'
#         y_name = 'y'
#         # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
#         print('DBScan clustering by x,y position...')
#
#     elif (cluster_by == 'pca' or cluster_by == 'PCA' or cluster_by == 'PCs'):
#         x_name = 'PC1'
#         y_name = 'PC2'
#         # save_path = CLUST_PCA_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by principal components...')
#
#     elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
#         x_name = 'tSNE1'
#         y_name = 'tSNE2'
#         # save_path = CLUST_TSNE_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by tSNE...')
#
#     elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
#         x_name = 'UMAP1'
#         y_name = 'UMAP2'
#         CLUSTERON=[x_name, y_name]
#         # save_path = CLUST_TSNE_DIR
#         print('DBScan clustering by UMAP...')
#     elif (cluster_by == 'NDIM' or cluster_by == 'ndim'):
#         CLUSTERON = DR_FACTORS
#
#     # cluster_selection_method ='eom'
#     # epsilon=1
#
#
#     #hDBScan
#     sub_set = df_in[CLUSTERON] # self.df
#     X = StandardScaler().fit_transform(sub_set)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric)
#     labels = clusterer.fit_predict(X)
#
#     # Assemble a dataframe from the results
#     lab_df = pd.DataFrame(data = labels, columns = ['label'])
#     lab_dr_df = pd.concat([df_in,lab_df], axis=1)
#
#
#     # Exemplars
#     if clusterer._prediction_data is None:
#         clusterer.generate_prediction_data()
#
#     selected_clusters = clusterer.condensed_tree_._select_clusters()
#     raw_condensed_tree = clusterer.condensed_tree_._raw_tree
#
#     exemplars = []
#     for cluster in selected_clusters:
#
#         cluster_exemplars = np.array([], dtype=np.int64)
#         for leaf in clusterer._prediction_data._recurse_leaf_dfs(cluster):
#             leaf_max_lambda = raw_condensed_tree['lambda_val'][
#                 raw_condensed_tree['parent'] == leaf].max()
#             points = raw_condensed_tree['child'][
#                 (raw_condensed_tree['parent'] == leaf) &
#                 (raw_condensed_tree['lambda_val'] == leaf_max_lambda)]
#             cluster_exemplars = np.hstack([cluster_exemplars, points])
#         exemplars.append(cluster_exemplars)
#
#     if(plot):
#         CLUSTER_CMAP = 'tab20'
#         clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#         # import seaborn as sns
#         # import matplotlib.pyplot as plt
#
#         ''' Eventually this plotting function should probably be in another script.'''
#         scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
#                           kind='scatter',
#                           palette=CLUSTER_CMAP,
#                           joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
#         if STATIC_PLOTS:
#             # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
#             plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
#         if PLOTS_IN_BROWSER:
#             plt.show()
#
#
#     # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')
#
#     return lab_dr_df, exemplars

    ################################################################

# def hdbscan_clustering__DEV(df_in, min_cluster_size=20,min_samples=10,cluster_by='NDIM',  metric='manhattan', plot=False, savedir = CLUST_DEV_DIR):
#
#     print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
#     from sklearn.preprocessing import MinMaxScaler
#     from pandas.plotting import scatter_matrix
#     from numpy import inf
#
#     # from matplotlib import pyplot
#     # Determine how to cluster
#     x_name,y_name = ' ', ' '
#
#     if cluster_by == 'xy':
#         x_name = 'x'
#         y_name = 'y'
#         # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
#         print('DBScan clustering by x,y position...')
#
#     elif (cluster_by == 'pca' or cluster_by == 'PCA' or cluster_by == 'PCs'):
#         x_name = 'PC1'
#         y_name = 'PC2'
#         # save_path = CLUST_PCA_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by principal components...')
#
#     elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
#         x_name = 'tSNE1'
#         y_name = 'tSNE2'
#         # save_path = CLUST_TSNE_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by tSNE...')
#
#     elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
#         x_name = 'UMAP1'
#         y_name = 'UMAP2'
#         CLUSTERON=[x_name, y_name]
#         # save_path = CLUST_TSNE_DIR
#         print('DBScan clustering by UMAP...')
#     elif (cluster_by == 'NDIM' or cluster_by == 'ndim'):
#         CLUSTERON = DR_FACTORS
#
#     # cluster_selection_method ='eom'
#     # epsilon=1
#
#     # df["log_2"] = np.log(df["col2])
#     #hDBScan
#     # plt.hist(df[np.isfinite(df['distance'])].values)
#
#     sub_set = df_in[CLUSTERON] # self.df #here, you don't do 'values' function. Therefore is this actually a numpy array?
#     Z = StandardScaler().fit_transform(sub_set)
#     X = MinMaxScaler().fit_transform(Z)
#     # X = np.log2(Y)
#     # X[X == -inf] = NaN
#     # X=np.nan_to_num(X)
#     # sub_set[CLUSTERON] = np.log2(sub_set[CLUSTERON])
#     # X=sub_set
#     print(X)
#
#     # Z = StandardScaler().fit_transform(sub_set)
#     # X = MinMaxScaler().fit_transform(Z)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric)
#     labels = clusterer.fit_predict(X)
#
#     # Assemble a dataframe from the results
#     lab_df = pd.DataFrame(data = labels, columns = ['label'])
#     lab_dr_df = pd.concat([df_in,lab_df], axis=1)
#
#     # summarize
#
#     scaled_subset_df=pd.DataFrame(data=X,columns=CLUSTERON)
#     print(scaled_subset_df.describe())
#     # histograms of the variables
#     # sub_set.hist(figsize=(20, 10))
#     # plt.tight_layout()
#     # plt.show()
#
#     scaled_subset_df.hist(figsize=(20, 10),color = "black", ec="black")
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(savedir+'scaledHISTOGRAMS'+'.png')
#
#
#
#
#     # Exemplars
#     if clusterer._prediction_data is None:
#         clusterer.generate_prediction_data()
#
#     selected_clusters = clusterer.condensed_tree_._select_clusters()
#     raw_condensed_tree = clusterer.condensed_tree_._raw_tree
#
#     exemplars = []
#     for cluster in selected_clusters:
#
#         cluster_exemplars = np.array([], dtype=np.int64)
#         for leaf in clusterer._prediction_data._recurse_leaf_dfs(cluster):
#             leaf_max_lambda = raw_condensed_tree['lambda_val'][
#                 raw_condensed_tree['parent'] == leaf].max()
#             points = raw_condensed_tree['child'][
#                 (raw_condensed_tree['parent'] == leaf) &
#                 (raw_condensed_tree['lambda_val'] == leaf_max_lambda)]
#             cluster_exemplars = np.hstack([cluster_exemplars, points])
#         exemplars.append(cluster_exemplars)
#
#     if(plot):
#         CLUSTER_CMAP = 'tab20'
#         clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#         # import seaborn as sns
#         # import matplotlib.pyplot as plt
#
#         ''' Eventually this plotting function should probably be in another script.'''
#         scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
#                           kind='scatter',
#                           palette=CLUSTER_CMAP,
#                           joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
#         if STATIC_PLOTS:
#             # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
#             plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
#         if PLOTS_IN_BROWSER:
#             plt.show()
#
#
#     # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')
#
#     return lab_dr_df, exemplars

# def hdbscan_clustering__DEV_DEV(df_in, min_cluster_size=20,min_samples=10,cluster_by='UMAPNDIM',  metric='manhattan', plot=False, savedir = CLUST_DEV_DIR, n_components=N_COMPONENTS):
#
#     print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
#     from sklearn.preprocessing import MinMaxScaler
#     from pandas.plotting import scatter_matrix
#     from numpy import inf
#
#     component_list=np.arange(1, n_components+1,1).tolist()
#     umap_components=([f'UMAP{i}' for i in component_list])
#
#     # from matplotlib import pyplot
#     # Determine how to cluster
#     x_name,y_name = ' ', ' '
#     # UMAPS = ['UMAP1','UMAP2','UMAP3','UMAP4','UMAP5']
#
#     if cluster_by == 'xy':
#         x_name = 'x'
#         y_name = 'y'
#         # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
#         print('DBScan clustering by x,y position...')
#
#     elif (cluster_by == 'pca' or cluster_by == 'PCA' or cluster_by == 'PCs'):
#         x_name = 'PC1'
#         y_name = 'PC2'
#         # save_path = CLUST_PCA_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by principal components...')
#
#     elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
#         x_name = 'tSNE1'
#         y_name = 'tSNE2'
#         # save_path = CLUST_TSNE_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by tSNE...')
#
#     elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
#         x_name = 'UMAP1'
#         y_name = 'UMAP2'
#         CLUSTERON=[x_name, y_name]
#         # save_path = CLUST_TSNE_DIR
#         print('DBScan clustering by UMAP...')
#     elif (cluster_by == 'NDIM' or cluster_by == 'ndim'):
#         CLUSTERON = DR_FACTORS
#     elif (cluster_by == 'UMAPNDIM' or cluster_by == 'umapndim'):
#         CLUSTERON = umap_components
#
#
#
#     sub_set = df_in[CLUSTERON] # self.df #here, you don't do 'values' function. Therefore this is a df
#     # X = sub_set.values
#     X = sub_set.values
#     #####
#     # Z = StandardScaler().fit_transform(sub_set)
#     # X = MinMaxScaler().fit_transform(Z)
#     ######
#     print(X)
#
#     # Z = StandardScaler().fit_transform(sub_set)
#     # X = MinMaxScaler().fit_transform(Z)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric)
#     labels = clusterer.fit_predict(X)
#
#     # Assemble a dataframe from the results
#     lab_df = pd.DataFrame(data = labels, columns = ['label'])
#     lab_dr_df = pd.concat([df_in,lab_df], axis=1)
#
#     # summarize
#
#     scaled_subset_df=pd.DataFrame(data=X,columns=CLUSTERON)
#     print(scaled_subset_df.describe())
#     # histograms of the variables
#     # sub_set.hist(figsize=(20, 10))
#     # plt.tight_layout()
#     # plt.show()
#
#     scaled_subset_df.hist(figsize=(20, 10),color = "black", ec="black")
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(savedir+'scaledHISTOGRAMS'+'.png')
#
#
#
#
#     # Exemplars
#     if clusterer._prediction_data is None:
#         clusterer.generate_prediction_data()
#
#     selected_clusters = clusterer.condensed_tree_._select_clusters()
#     raw_condensed_tree = clusterer.condensed_tree_._raw_tree
#
#     exemplars = []
#     for cluster in selected_clusters:
#
#         cluster_exemplars = np.array([], dtype=np.int64)
#         for leaf in clusterer._prediction_data._recurse_leaf_dfs(cluster):
#             leaf_max_lambda = raw_condensed_tree['lambda_val'][
#                 raw_condensed_tree['parent'] == leaf].max()
#             points = raw_condensed_tree['child'][
#                 (raw_condensed_tree['parent'] == leaf) &
#                 (raw_condensed_tree['lambda_val'] == leaf_max_lambda)]
#             cluster_exemplars = np.hstack([cluster_exemplars, points])
#         exemplars.append(cluster_exemplars)
#
#
#         exemplar_df = pd.DataFrame()
#         lengthlistexemplar=np.arange(0, (len(exemplars)), 1).tolist()
#         for exemp in lengthlistexemplar:
#             cluster1_rows=exemplars[exemp]
#             cluster1_rows=cluster1_rows.tolist()
#             iterablelist2=np.arange(0, (len(cluster1_rows)), 1).tolist()
#             for preciseexemp in iterablelist2:
#                 preciserowID=cluster1_rows[preciseexemp]
#                 row_from_df=lab_dr_df.iloc[[preciserowID]]
#                 exemplar_df = pd.concat([exemplar_df, row_from_df], axis=0)
#
#
#     if(plot):
#         CLUSTER_CMAP = 'tab20'
#         clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#         # import seaborn as sns
#         # import matplotlib.pyplot as plt
#
#         ''' Eventually this plotting function should probably be in another script.'''
#         scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
#                           kind='scatter',
#                           palette=CLUSTER_CMAP,
#                           joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
#         if STATIC_PLOTS:
#             # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
#             plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
#         if PLOTS_IN_BROWSER:
#             plt.show()
#
#
#     # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')
#
#     return lab_dr_df, exemplar_df
### USE THIS ONE AS OF 12-20-2022 ##### old name hdbscan_clustering__DEV_DEV_SPEEDUP
def hdbscan_clustering(df_in, min_cluster_size=20,min_samples=10,cluster_by='UMAPNDIM',  metric='manhattan', doeps = True, epsilon=0.5,plot=False, savedir = CLUSTERING_DIR, n_components=N_COMPONENTS, scalingmethod=None, DR_FACTORS=DR_FACTORS):

    print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
    from sklearn.preprocessing import MinMaxScaler
    from pandas.plotting import scatter_matrix
    from numpy import inf

    component_list=np.arange(1, n_components+1,1).tolist()
    umap_components=([f'UMAP{i}' for i in component_list])

    # from matplotlib import pyplot
    # Determine how to cluster
    x_name,y_name = ' ', ' '
    # UMAPS = ['UMAP1','UMAP2','UMAP3','UMAP4','UMAP5']

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'
        # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
        print('DBScan clustering by x,y position...')

    elif (cluster_by == 'pca' or cluster_by == 'PCA' or cluster_by == 'PCs'):
        x_name = 'PC1'
        y_name = 'PC2'
        # save_path = CLUST_PCA_DIR
        CLUSTERON=[x_name, y_name]
        print('DBScan clustering by principal components...')

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        # save_path = CLUST_TSNE_DIR
        CLUSTERON=[x_name, y_name]
        print('DBScan clustering by tSNE...')

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
        x_name = 'UMAP1'
        y_name = 'UMAP2'
        CLUSTERON=[x_name, y_name]
        # save_path = CLUST_TSNE_DIR
        print('DBScan clustering by UMAP...')
    elif (cluster_by == 'NDIM' or cluster_by == 'ndim'):
        CLUSTERON = DR_FACTORS
    elif (cluster_by == 'UMAPNDIM' or cluster_by == 'umapndim'):
        CLUSTERON = umap_components



    sub_set = df_in[CLUSTERON] # self.df #here, you don't do 'values' function. Therefore this is a df
    # X = sub_set.values
    Z = sub_set.values
    #####

    if scalingmethod == 'minmax': #log2minmax minmax powertransformer
        X = MinMaxScaler().fit_transform(Z)
        correctcolumns = CLUSTERON
    elif scalingmethod == 'log2minmax':

        negative_FACTORS = []
        positive_FACTORS = []
        for factor in DR_FACTORS:
            if np.min(df_in[factor]) < 0:
                print('factor ' + factor + ' has negative values')
                negative_FACTORS.append(factor)
                
            else:
                print('factor ' + factor + ' has no negative values')
                positive_FACTORS.append(factor)
        
        
        pos_df = df_in[positive_FACTORS]
        pos_x = pos_df.values
        neg_df = df_in[negative_FACTORS]
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
        X = np.concatenate((pos_x_, neg_x_), axis=1)
        correctcolumns=positive_FACTORS + negative_FACTORS

    elif scalingmethod == 'powertransformer':    
        
        pt = PowerTransformer(method='yeo-johnson')
        X = pt.fit_transform(x)
        correctcolumns = CLUSTERON

        #########
    elif scalingmethod == None:
        X = Z
        correctcolumns = CLUSTERON


    # X = StandardScaler().fit_transform(Z)
    # X=Z
    # X = np.arcsinh(Z)
    # X = MinMaxScaler().fit_transform(Z)
    ######
    # print(X)

    # Z = StandardScaler().fit_transform(sub_set)
    # X = MinMaxScaler().fit_transform(Z)
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric) ### SPIDERMAN ###
    if doeps == True:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric, cluster_selection_epsilon=epsilon, cluster_selection_method = 'eom')
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric)
    labels = clusterer.fit_predict(X)

    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = ['label'])
    lab_dr_df = pd.concat([df_in,lab_df], axis=1)

    # summarize

    scaled_subset_df=pd.DataFrame(data=X,columns=correctcolumns)
    print(scaled_subset_df.describe())
    # histograms of the variables
    # sub_set.hist(figsize=(20, 10))
    # plt.tight_layout()
    # plt.show()

    scaled_subset_df.hist(figsize=(20, 10), bins=160, color = "black", ec="black")
    plt.tight_layout()
    plt.show()
    plt.savefig(savedir+'scaledHISTOGRAMS'+'.png')




    # Exemplars
    if clusterer._prediction_data is None:
        clusterer.generate_prediction_data()

    selected_clusters = clusterer.condensed_tree_._select_clusters()
    raw_condensed_tree = clusterer.condensed_tree_._raw_tree

    exemplars = []
    for cluster in selected_clusters:

        cluster_exemplars = np.array([], dtype=np.int64)
        for leaf in clusterer._prediction_data._recurse_leaf_dfs(cluster):
            leaf_max_lambda = raw_condensed_tree['lambda_val'][
                raw_condensed_tree['parent'] == leaf].max()
            points = raw_condensed_tree['child'][
                (raw_condensed_tree['parent'] == leaf) &
                (raw_condensed_tree['lambda_val'] == leaf_max_lambda)]
            cluster_exemplars = np.hstack([cluster_exemplars, points])
        exemplars.append(cluster_exemplars)


    exemplararray = []
    # exemplar_df = pd.DataFrame()
    lengthlistexemplar=np.arange(0, (len(exemplars)), 1).tolist()
    for exemp in lengthlistexemplar:
        cluster1_rows=exemplars[exemp]
        cluster1_rows=cluster1_rows.tolist()
        iterablelist2=np.arange(0, (len(cluster1_rows)), 1).tolist()
        for preciseexemp in iterablelist2:
            preciserowID=cluster1_rows[preciseexemp]
            row_from_df=lab_dr_df.iloc[[preciserowID]].to_numpy()
            exemplararray.append(row_from_df)
    squeezed_exemplararray = np.squeeze(exemplararray)

    colsare = lab_dr_df.columns.tolist()
    exemplar_df = pd.DataFrame(squeezed_exemplararray, columns=colsare)


                # exemplar_df = pd.concat([exemplar_df, row_from_df], axis=0)


    if(plot):
        CLUSTER_CMAP = 'tab20'
        clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
        # import seaborn as sns
        # import matplotlib.pyplot as plt

        ''' Eventually this plotting function should probably be in another script.'''
        scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
                          kind='scatter',
                          palette=CLUSTER_CMAP,
                          joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
        if STATIC_PLOTS:
            # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
            plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
        if PLOTS_IN_BROWSER:
            plt.show()


    # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')

    return lab_dr_df, exemplar_df

# def hdbscan_clustering_debug(df_in, min_cluster_size=20,min_samples=10,cluster_by='UMAPNDIM',  metric='manhattan', plot=False, savedir = CLUST_DEV_DIR, n_components=N_COMPONENTS, scalingmethod=None, DR_FACTORS=DR_FACTORS):

#     print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
#     from sklearn.preprocessing import MinMaxScaler
#     from pandas.plotting import scatter_matrix
#     from numpy import inf

#     component_list=np.arange(1, n_components+1,1).tolist()
#     umap_components=([f'UMAP{i}' for i in component_list])
#     print('the umap components are')
#     print(umap_components)
#     # from matplotlib import pyplot
#     # Determine how to cluster
#     x_name,y_name = ' ', ' '
#     # UMAPS = ['UMAP1','UMAP2','UMAP3','UMAP4','UMAP5']

#     if cluster_by == 'xy':
#         x_name = 'x'
#         y_name = 'y'
#         # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
#         print('DBScan clustering by x,y position...')

#     elif (cluster_by == 'pca' or cluster_by == 'PCA' or cluster_by == 'PCs'):
#         x_name = 'PC1'
#         y_name = 'PC2'
#         # save_path = CLUST_PCA_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by principal components...')

#     elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
#         x_name = 'tSNE1'
#         y_name = 'tSNE2'
#         # save_path = CLUST_TSNE_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by tSNE...')

#     elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
#         x_name = 'UMAP1'
#         y_name = 'UMAP2'
#         CLUSTERON=[x_name, y_name]
#         # save_path = CLUST_TSNE_DIR
#         print('DBScan clustering by UMAP...')
#     elif (cluster_by == 'NDIM' or cluster_by == 'ndim'):
#         CLUSTERON = DR_FACTORS
#     elif (cluster_by == 'UMAPNDIM' or cluster_by == 'umapndim'):
#         CLUSTERON = umap_components



#     sub_set = df_in[CLUSTERON] # self.df #here, you don't do 'values' function. Therefore this is a df
#     # X = sub_set.values
#     Z = sub_set.values
#     X = Z
#     correctcolumns = CLUSTERON
#     print('these are the correct columns')
#     print(correctcolumns)
#     altcols = sub_set.columns
#     print('these are the altcols')
#     print(altcols)


#     #####

#     # if scalingmethod == 'minmax': #log2minmax minmax powertransformer
#     #     X = MinMaxScaler().fit_transform(Z)
#     #     correctcolumns = CLUSTERON
#     # elif scalingmethod == 'log2minmax':

#     #     negative_FACTORS = []
#     #     positive_FACTORS = []
#     #     for factor in DR_FACTORS:
#     #         if np.min(df_in[factor]) < 0:
#     #             print('factor ' + factor + ' has negative values')
#     #             negative_FACTORS.append(factor)
                
#     #         else:
#     #             print('factor ' + factor + ' has no negative values')
#     #             positive_FACTORS.append(factor)
        
        
#     #     pos_df = df_in[positive_FACTORS]
#     #     pos_x = pos_df.values
#     #     neg_df = df_in[negative_FACTORS]
#     #     neg_x = neg_df.values
#     #     # neg_x_ = MinMaxScaler().fit_transform(neg_x)

#     #     if len(neg_x[0]) == 0: #This controls for an edge case in which there are no negative factors - must be implemented in the other transforms as well (pipelines and clustering)
#     #         print('No negative factors at all!')
#     #         neg_x_ = neg_x
#     #     else:
#     #         neg_x_ = MinMaxScaler().fit_transform(neg_x) 
#     #     pos_x_constant = pos_x + 0.000001
#     #     pos_x_log = np.log2(pos_x + pos_x_constant)
#     #     pos_x_ = MinMaxScaler().fit_transform(pos_x_log)
#     #     X = np.concatenate((pos_x_, neg_x_), axis=1)
#     #     correctcolumns=positive_FACTORS + negative_FACTORS

#     # elif scalingmethod == 'powertransformer':    
        
#     #     pt = PowerTransformer(method='yeo-johnson')
#     #     X = pt.fit_transform(x)
#     #     correctcolumns = CLUSTERON

#     #     #########
#     # elif scalingmethod == None:
#     #     X = Z
#     #     correctcolumns = CLUSTERON
   

#     # X = StandardScaler().fit_transform(Z)
#     # X=Z
#     # X = np.arcsinh(Z)
#     # X = MinMaxScaler().fit_transform(Z)
#     ######
#     # print(X)

#     # Z = StandardScaler().fit_transform(sub_set)
#     # X = MinMaxScaler().fit_transform(Z)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric)
#     labels = clusterer.fit_predict(X)
#     print('It worked and did the clustering')
#     # Assemble a dataframe from the results
#     lab_df = pd.DataFrame(data = labels, columns = ['label'])
#     lab_dr_df = pd.concat([df_in,lab_df], axis=1)
#     print('It worked and made the labelled dataframe')
#     # summarize

#     scaled_subset_df=pd.DataFrame(data=X,columns=correctcolumns)
#     print(scaled_subset_df.describe())
#     # histograms of the variables
#     # sub_set.hist(figsize=(20, 10))
#     # plt.tight_layout()
#     # plt.show()

#     scaled_subset_df.hist(figsize=(20, 10), bins=160, color = "black", ec="black")
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(savedir+'scaledHISTOGRAMS'+'.png')




#     # Exemplars
#     if clusterer._prediction_data is None:
#         clusterer.generate_prediction_data()

#     selected_clusters = clusterer.condensed_tree_._select_clusters()
#     raw_condensed_tree = clusterer.condensed_tree_._raw_tree

#     exemplars = []
#     for cluster in selected_clusters:

#         cluster_exemplars = np.array([], dtype=np.int64)
#         for leaf in clusterer._prediction_data._recurse_leaf_dfs(cluster):
#             leaf_max_lambda = raw_condensed_tree['lambda_val'][
#                 raw_condensed_tree['parent'] == leaf].max()
#             points = raw_condensed_tree['child'][
#                 (raw_condensed_tree['parent'] == leaf) &
#                 (raw_condensed_tree['lambda_val'] == leaf_max_lambda)]
#             cluster_exemplars = np.hstack([cluster_exemplars, points])
#         exemplars.append(cluster_exemplars)


#     exemplararray = []
#     print('It finished the first exemplar loop')
#     # exemplar_df = pd.DataFrame()
#     lengthlistexemplar=np.arange(0, (len(exemplars)), 1).tolist()
#     for exemp in lengthlistexemplar:
#         cluster1_rows=exemplars[exemp]
#         cluster1_rows=cluster1_rows.tolist()
#         iterablelist2=np.arange(0, (len(cluster1_rows)), 1).tolist()
#         for preciseexemp in iterablelist2:
#             preciserowID=cluster1_rows[preciseexemp]
#             row_from_df=lab_dr_df.iloc[[preciserowID]].to_numpy()
#             exemplararray.append(row_from_df)
#     squeezed_exemplararray = np.squeeze(exemplararray)

#     colsare = lab_dr_df.columns.tolist()
#     exemplar_df = pd.DataFrame(squeezed_exemplararray, columns=colsare)


#                 # exemplar_df = pd.concat([exemplar_df, row_from_df], axis=0)


#     if(plot):
#         CLUSTER_CMAP = 'tab20'
#         clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#         # import seaborn as sns
#         # import matplotlib.pyplot as plt

#         ''' Eventually this plotting function should probably be in another script.'''
#         scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
#                           kind='scatter',
#                           palette=CLUSTER_CMAP,
#                           joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
#         if STATIC_PLOTS:
#             # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
#             plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
#         if PLOTS_IN_BROWSER:
#             plt.show()


#     # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')

#     return lab_dr_df, exemplar_df    

# def hdbscan_clustering__DEV_DEV_NOSCALING(df_in, min_cluster_size=20,min_samples=10,cluster_by='UMAPNDIM',  metric='manhattan', plot=False, savedir = CLUST_DEV_DIR, umaps=UMAPS):
#
#     print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
#     from sklearn.preprocessing import MinMaxScaler
#     from pandas.plotting import scatter_matrix
#     from numpy import inf
#
#     # from matplotlib import pyplot
#     # Determine how to cluster
#     x_name,y_name = ' ', ' '
#     # UMAPS = ['UMAP1','UMAP2','UMAP3','UMAP4','UMAP5']
#
#     if cluster_by == 'xy':
#         x_name = 'x'
#         y_name = 'y'
#         # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
#         print('DBScan clustering by x,y position...')
#
#     elif (cluster_by == 'pca' or cluster_by == 'PCA' or cluster_by == 'PCs'):
#         x_name = 'PC1'
#         y_name = 'PC2'
#         # save_path = CLUST_PCA_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by principal components...')
#
#     elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
#         x_name = 'tSNE1'
#         y_name = 'tSNE2'
#         # save_path = CLUST_TSNE_DIR
#         CLUSTERON=[x_name, y_name]
#         print('DBScan clustering by tSNE...')
#
#     elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
#         x_name = 'UMAP1'
#         y_name = 'UMAP2'
#         CLUSTERON=[x_name, y_name]
#         # save_path = CLUST_TSNE_DIR
#         print('DBScan clustering by UMAP...')
#     elif (cluster_by == 'NDIM' or cluster_by == 'ndim'):
#         CLUSTERON = DR_FACTORS
#     elif (cluster_by == 'UMAPNDIM' or cluster_by == 'umapndim'):
#         CLUSTERON = umaps
#
#     # cluster_selection_method ='eom'
#     # epsilon=1
#
#     # df["log_2"] = np.log(df["col2])
#     #hDBScan
#     # plt.hist(df[np.isfinite(df['distance'])].values)
#
#     sub_set = df_in[CLUSTERON].values # self.df #here, you don't do 'values' function. Therefore is this actually a numpy array?
#
#     X = StandardScaler().fit_transform(sub_set)
#     # X = MinMaxScaler().fit_transform(Z)
#     # X = np.log2(Y)
#     # X[X == -inf] = NaN
#     # X=np.nan_to_num(X)
#     # sub_set[CLUSTERON] = np.log2(sub_set[CLUSTERON])
#     # X=sub_set
#     print(X)
#
#     # Z = StandardScaler().fit_transform(sub_set)
#     # X = MinMaxScaler().fit_transform(Z)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric=metric)
#     labels = clusterer.fit_predict(X)
#
#     # Assemble a dataframe from the results
#     lab_df = pd.DataFrame(data = labels, columns = ['label'])
#     lab_dr_df = pd.concat([df_in,lab_df], axis=1)
#
#     # summarize
#
#     scaled_subset_df=pd.DataFrame(data=X,columns=CLUSTERON)
#     print(scaled_subset_df.describe())
#     # histograms of the variables
#     # sub_set.hist(figsize=(20, 10))
#     # plt.tight_layout()
#     # plt.show()
#
#     scaled_subset_df.hist(figsize=(20, 10),color = "black", ec="black")
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(savedir+'scaledHISTOGRAMS'+'.png')
#
#
#
#
#     # Exemplars
#     if clusterer._prediction_data is None:
#         clusterer.generate_prediction_data()
#
#     selected_clusters = clusterer.condensed_tree_._select_clusters()
#     raw_condensed_tree = clusterer.condensed_tree_._raw_tree
#
#     exemplars = []
#     for cluster in selected_clusters:
#
#         cluster_exemplars = np.array([], dtype=np.int64)
#         for leaf in clusterer._prediction_data._recurse_leaf_dfs(cluster):
#             leaf_max_lambda = raw_condensed_tree['lambda_val'][
#                 raw_condensed_tree['parent'] == leaf].max()
#             points = raw_condensed_tree['child'][
#                 (raw_condensed_tree['parent'] == leaf) &
#                 (raw_condensed_tree['lambda_val'] == leaf_max_lambda)]
#             cluster_exemplars = np.hstack([cluster_exemplars, points])
#         exemplars.append(cluster_exemplars)
#
#     if(plot):
#         CLUSTER_CMAP = 'tab20'
#         clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#         # import seaborn as sns
#         # import matplotlib.pyplot as plt
#
#         ''' Eventually this plotting function should probably be in another script.'''
#         scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
#                           kind='scatter',
#                           palette=CLUSTER_CMAP,
#                           joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
#         if STATIC_PLOTS:
#             # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
#             plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
#         if PLOTS_IN_BROWSER:
#             plt.show()
#
#
#     # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')
#
#     return lab_dr_df, exemplars

################################################################

def add_tavglabel_todf(df, lab_tavg_dr_df_p): #At the moment, this is a slow function that adds cols to a PD df

    lab_list_tpt=[]

    for rep in df['Replicate_ID'].unique():

            for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

                cell_df = df[(df['Replicate_ID']==rep) &
                                (df['particle']==cell_id)]
                tavg_cell_df=lab_tavg_dr_df_p[(lab_tavg_dr_df_p['Replicate_ID']==rep) &
                                (lab_tavg_dr_df_p['particle']==cell_id)]

                # If you've already attributed a unique id, then use it.
                if 'uniq_id' in cell_df.columns:
                    assert len(cell_df['uniq_id'].unique()) == 1
                    uniq_id = cell_df['uniq_id'].unique()[0]
                else:
                    uniq_id = cell_id

                s = cell_df['label']

                # lengthofs=len(s)
                [tavg_label]=tavg_cell_df['label']

                # tavg_label_list=[tavg_label]*lengthofs


                for t in cell_df['frame'].unique():

                    lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, tavg_label))
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'tavg_label'])
    display(lab_list_tpt_df[lab_list_tpt_df['Cell_ID'] == '0_22'])

    # Need to re-sort it by cell_id and frame
    lab_list_tpt_df.sort_values(["Cell_ID", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    # Resort the main dataframe too to be sure they're aligned before contatenating
    df.sort_values(["uniq_id", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    df.reset_index(inplace=True, drop=True)
    lab_list_tpt_df.reset_index(inplace=True, drop=True)

     # Test that this lines up with the original dataframe
    for i in range(len(lab_list_tpt_df)):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        id_1 = df.iloc[i]['uniq_id']
        id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        assert id_1 == id_2

     # Insert the timepoint label counts back into the input dataframe
    fresh_df = pd.concat([df,lab_list_tpt_df['tavg_label']], axis=1)

    return(fresh_df)


##### DEV VERSION: #####
def add_tavglabel_todf_DEV(df, lab_tavg_dr_df_p): #At the moment, this is a slow function that adds cols to a PD df

    lab_list_tpt=[]

    for rep in df['Replicate_ID'].unique():

            for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

                cell_df = df[(df['Replicate_ID']==rep) &
                                (df['particle']==cell_id)]
                tavg_cell_df=lab_tavg_dr_df_p[(lab_tavg_dr_df_p['Replicate_ID']==rep) &
                                (lab_tavg_dr_df_p['particle']==cell_id)]

                # If you've already attributed a unique id, then use it.
                if 'uniq_id' in cell_df.columns:
                    assert len(cell_df['uniq_id'].unique()) == 1
                    uniq_id = cell_df['uniq_id'].unique()[0]
                else:
                    uniq_id = cell_id

                s = cell_df['label']

                # lengthofs=len(s)
                [tavg_label]=tavg_cell_df['label']

                # tavg_label_list=[tavg_label]*lengthofs


                for t in cell_df['frame'].unique():

                    lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, tavg_label))
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'tavg_label'])
    # display(lab_list_tpt_df[lab_list_tpt_df['Cell_ID'] == '0_22'])

    # Need to re-sort it by cell_id and frame
    lab_list_tpt_df.sort_values(["Cell_ID", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    # Resort the main dataframe too to be sure they're aligned before contatenating
    df.sort_values(["uniq_id", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    df.reset_index(inplace=True, drop=True)
    lab_list_tpt_df.reset_index(inplace=True, drop=True)

     # Test that this lines up with the original dataframe
    for i in range(len(lab_list_tpt_df)):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        id_1 = df.iloc[i]['uniq_id']
        id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        assert id_1 == id_2

     # Insert the timepoint label counts back into the input dataframe
    fresh_df = pd.concat([df,lab_list_tpt_df['tavg_label']], axis=1)

    return(fresh_df)

def exemplar_df_check(df, exemp_df):

    #This creates a dummy df for plotting so you can highlight the points and they have the same colours as the originals!
    # It creates some fake exemplars at if required.

    # exemplar_df
    # Insert dummy rows with missing labels?
    # Get list of labels in original df
    originallabels=set(df['label'].unique())
    # originallabels
    #Which labels are missing of these in the other list?
    exemplarlabellist=set(exemp_df['label'].unique())
    # exemplarlabellist
    # Find which are missing in the original

    missing = list(sorted(originallabels - exemplarlabellist))
    print('missing:', missing)
    f=len(missing)
    f
    # make a fake df that has the rows of the missing values and the cols of the regular one but is all zeros....
    if len(missing) != 0:

        numcols = len(df.axes[1])
        numrows=len(missing)
        fakearray=np.zeros((numrows,numcols))
        originalcolumns=df.columns
        fake_df=pd.DataFrame(fakearray, columns=originalcolumns)
        #Plonk in the fake labels
        fake_df['label'] = missing

        #merge this fake_df with exemplar_df

        exemplar_df_f=pd.concat([exemp_df, fake_df])

    else:
        print('Every cluster is represented by an exemplar')

    return(exemplar_df_f)
############################################

########## Added 12-2-2022. Deprecated, sped up and built into main clustering part 12-20-2022 ####

# def get_exemplar_df(df, exemplars):
#     exemplar_df = pd.DataFrame()
#     lengthlistexemplar=np.arange(0, (len(exemplars)), 1).tolist()
#     for exemp in lengthlistexemplar:
#         cluster1_rows=exemplars[exemp]
#         cluster1_rows=cluster1_rows.tolist()
#         iterablelist2=np.arange(0, (len(cluster1_rows)), 1).tolist()
#         for preciseexemp in iterablelist2:
#             preciserowID=cluster1_rows[preciseexemp]
#             row_from_df=df.iloc[[preciserowID]]
#             exemplar_df = pd.concat([exemplar_df, row_from_df], axis=0)
#     return exemplar_df

##########

### VARIANCE THRESHOLDER #######
def variance_threshold(df_in, threshold_value, dr_factors=ALL_FACTORS): #Added 12-14-2022
    # df_in = comb_df
    # threshold_value = 0.06

    from sklearn.feature_selection import VarianceThreshold
    # Made a subset df containing only the metrics
    CLUSTERON=dr_factors
    subset_df=df_in[CLUSTERON]
    subset_df.head()

    #Variance threshold
    vt = VarianceThreshold(threshold_value) #make the thresholder
    _ = vt.fit(subset_df)
    mask = vt.get_support() #Give a boolean array: True if the variance of each column exceeds the threshold.
    masked_df=subset_df.iloc[:,mask]


    ###

    # threshold_low = threshold_value
    # threshold_mid = threshold_value * 1.25
    # threshold_high = threshold_value * 1.5 #A big threshold number, such that only the most variable columns are include in the dr
    #
    # vt = VarianceThreshold(threshold_low) #make the thresholder
    # _ = vt.fit(subset_df)
    # lowmask = vt.get_support() #Give a boolean array: True if the variance of each column exceeds the threshold.
    # lowmasked_df=subset_df.iloc[:,lowmask]
    # inverse_lowmask = np.invert(lowmask)
    # inverse_lowmasked_df=subset_df.iloc[0,inverse_mask]
    # print("Metrics that vary a the most and were kept are " + str(lowmasked_df.columns.tolist()))
    # print("Metrics that are very constant and so were removed are " + str(inverse_lowmasked_df.index.tolist()))
    ####

    # Print the metrics that are kept and the metrics that are removed
    print("Metrics that vary are " + str(masked_df.columns.tolist()))
    inverse_mask = np.invert(mask)
    inverse_masked_df=subset_df.iloc[0,inverse_mask]
    inverse_masked_df
    print("Metrics that are constant and so were removed are " + str(inverse_masked_df.index.tolist()))

    # Make a new DR_FACTORS to be used down the pipeline

    DR_FACTORS = masked_df.columns.tolist()

    return DR_FACTORS

def optics_clustering(df_in, min_samples=MIN_SAMPLES,cluster_by='tsne', plot=False, save_path=CLUST_DIR):


    # Determine how to cluster
    x_name,y_name = ' ', ' '

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'
        # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
        print('OPTICS clustering by x,y position...')

    elif (cluster_by == 'pca' or cluster_by == 'PCA'):
        x_name = 'PC1'
        y_name = 'PC2'
        # save_path = CLUST_PCA_DIR
        print('OPTICS clustering by principal components...')

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        # save_path = CLUST_TSNE_DIR
        print('OPTICS clustering by tSNE...')

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
        x_name = 'UMAP1'
        y_name = 'UMAP2'
        # save_path = CLUST_TSNE_DIR
        print('OPTICS clustering by UMAP...')


    # Optics
    sub_set = df_in[[x_name, y_name]] # self.df
    X = StandardScaler().fit_transform(sub_set)
    clustering = OPTICS(min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    labels = clustering.labels_


    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = ['label'])
    lab_dr_df = pd.concat([df_in,lab_df], axis=1)

    if(plot):
        # import seaborn as sns
        # import matplotlib.pyplot as plt

        ''' Eventually this plotting function should probably be in another script.'''
        scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
                          kind='scatter',
                          palette=CLUSTER_CMAP,
                          joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
        if STATIC_PLOTS:
            plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
        if PLOTS_IN_BROWSER:
            plt.show()

    return lab_dr_df


def cluster_purity(lab_dr_df, cluster_label='label'):

    '''
    Calculate purity of input dataframe clusters with respect to the experimental condition.

    Input:
        lab_dr_df: pd.DataFrame containing cluster ids in the 'label' column.

    '''

    assert cluster_label in lab_dr_df.columns, 'Dataframe must contain cluster labels'
    assert 'Condition_shortlabel' in lab_dr_df.columns, 'For now, assuming shortlabels in use.'

    cond_list = lab_dr_df['Condition_shortlabel'].unique()


    # Create a new dataframe to hold the cluster summary info.
    clust_sum_df = pd.DataFrame()

    clusters = list(set(lab_dr_df[cluster_label].dropna())) # DropNA to also handle trajectory_ids where some are NaN

    for cluster_id in clusters[:-1]: # Skip last one that is noise (-1)


        clust_sum_df.at[cluster_id,'cluster_id'] = cluster_id

        # Get the dataframe for this cluster.
        clust_sub_df = lab_dr_df[lab_dr_df[cluster_label] == cluster_id]


        for cond in cond_list:


            cond_clust_sub_df = clust_sub_df[clust_sub_df['Condition_shortlabel'] == cond]

            # Count the number of timepoints for this condition in the cluster
            clust_sum_df.at[cluster_id,cond+'_ntpts'] = len(cond_clust_sub_df)
            clust_sum_df.at[cluster_id,cond+'_ntpts_%'] = len(cond_clust_sub_df) / len(clust_sub_df) * 100

            # Count the number of unique cells for this condition in the cluster
            clust_sum_df.at[cluster_id,cond+'_ncells'] = len(cond_clust_sub_df['uniq_id'].unique())
            clust_sum_df.at[cluster_id,cond+'_ncells_%'] = len(cond_clust_sub_df['uniq_id'].unique()) / len(clust_sub_df['uniq_id'].unique()) * 100

    return clust_sum_df







def count_cluster_changes_old(df):

    '''
    Count the number of changes to the cluster ID
    '''

    assert 'label' in df.columns, 'count_cluster_changes() must be provided with label-containing dataframe.'

    lab_list = []
    lab_list_tpt = []

    for rep in df['Replicate_ID'].unique():

        for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            # If you've already attributed a unique id, then use it.
            if 'uniq_id' in cell_df.columns:
                assert len(cell_df['uniq_id'].unique()) == 1
                uniq_id = cell_df['uniq_id'].unique()[0]
            else:
                uniq_id = cell_id

            # Calculate at the per-cell level
            s = cell_df['label']
            label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
            n_labels = len(s.unique()) # Count the number of times it changes
            lab_list.append((cell_df['Condition'].unique()[0],rep, uniq_id, label_changes,n_labels))

            # The extra step of doing these calculations cumulatively, per timepoint.
            for t in cell_df['frame'].unique():

                # Get the dataframe including all timepoints upto and including t
                cumulative_df = cell_df[(cell_df['frame']<=t)]

                # Count the label changes and total numbers.
                s = cumulative_df['label']
                label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
                n_labels = len(s.unique()) # Count the number of times it changes
                lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, label_changes,n_labels))

    # Turn the lists to dataframes
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'n_changes', 'n_labels'])
    lab_list_df = pd.DataFrame(lab_list, columns=['Condition','Replicate', 'Cell_ID', 'n_changes', 'n_labels'])

    assert len(lab_list_tpt_df.index) == len(df.index), 'Label count list doesnt add up'

    # Test that this lines up with the original dataframe
    for i in range(len(lab_list_tpt_df)):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        # id_1 = df.iloc[i]['particle']
        # id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        # assert id_1 == id_2

    # Insert the timepoint label counts back into the input dataframe
    new_df = pd.concat([df,lab_list_tpt_df[['n_changes','n_labels']]], axis=1)

    return lab_list_df, new_df

#########
#########
def count_cluster_changesnew(df): #This one allows us to add the total_label_changes and total_n_changes to the df

    '''
    Count the number of changes to the cluster ID
    '''

    assert 'label' in df.columns, 'count_cluster_changes() must be provided with label-containing dataframe.'

    lab_list = []
    lab_list_tpt = []

    for rep in df['Replicate_ID'].unique():

        for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            # If you've already attributed a unique id, then use it.
            if 'uniq_id' in cell_df.columns:
                assert len(cell_df['uniq_id'].unique()) == 1
                uniq_id = cell_df['uniq_id'].unique()[0]
            else:
                uniq_id = cell_id


            # Calculate at the per-cell level
            s = cell_df['label']
            total_label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
            total_n_labels = len(s.unique()) # Count the number of times it changes
            lab_list.append((cell_df['Condition'].unique()[0],rep, uniq_id, total_label_changes,total_n_labels))
            total_n_changes_list=[total_label_changes]*(len(s))
            total_n_labels_list=[total_n_labels]*(len(s))


            #Need to make a list the length of x.
            #Need to

            # The extra step of doing these calculations cumulatively, per timepoint.
            for t in cell_df['frame'].unique():

                # Get the dataframe including all timepoints upto and including t
                cumulative_df = cell_df[(cell_df['frame']<=t)]
                # Count the label changes and total numbers.
                s = cumulative_df['label']
                label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
                n_labels = len(s.unique()) # Count the number of times it changes
                # total_n_changes_list[1]
                # display(t)
                lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, label_changes,n_labels,total_n_changes_list[0],total_n_labels_list[0]))

    # Turn the lists to dataframes
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'n_changes', 'n_labels', 'total_n_changes', 'total_n_labels'])
    lab_list_df = pd.DataFrame(lab_list, columns=['Condition','Replicate', 'Cell_ID', 'n_changes', 'n_labels'])
    print('     lab_list_tpt_df')
    display(lab_list_tpt_df[lab_list_tpt_df['Cell_ID'] == '0_22'])
#     display(lab_list_tpt_df.head(100))
    assert len(lab_list_tpt_df.index) == len(df.index), 'Label count list doesnt add up'

    # While the label list isthe same length, it's not sorted in the exact same way as the df input
    # Because the list is created on a cell-by-cell basis

    # Need to re-sort it by cell_id and frame
    lab_list_tpt_df.sort_values(["Cell_ID", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    # Resort the main dataframe too to be sure they're aligned before contatenating
    df.sort_values(["uniq_id", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    df.reset_index(inplace=True, drop=True)
    lab_list_tpt_df.reset_index(inplace=True, drop=True)

    # Test that this lines up with the original dataframe
    for i in range(len(lab_list_tpt_df)):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        id_1 = df.iloc[i]['uniq_id']
        id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        assert id_1 == id_2


    # Insert the timepoint label counts back into the input dataframe
    new_df = pd.concat([df,lab_list_tpt_df[['n_changes','n_labels','total_n_changes','total_n_labels']]], axis=1)

    return lab_list_df, new_df

#########
#########

def count_cluster_changes_deprecated(df): #This one incorporates tavg_df clustering calculations and is what is used now. But is not time windowed

    '''
    Count the number of changes to the cluster ID
    '''

    assert 'label' in df.columns, 'count_cluster_changes() must be provided with label-containing dataframe.'
    assert 'tavg_label' in df.columns, 'count_cluster_changes() must be provided with tavg_label containing dataframe such as lab_tavg_lab_dr_df'

    lab_list = []
    lab_list_tpt = []

    for rep in df['Replicate_ID'].unique():

        for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            # If you've already attributed a unique id, then use it.
            if 'uniq_id' in cell_df.columns:
                assert len(cell_df['uniq_id'].unique()) == 1
                uniq_id = cell_df['uniq_id'].unique()[0]
            else:
                uniq_id = cell_id


            # Calculate at the per-cell level
            s = cell_df['label']
            total_label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
            total_n_labels = len(s.unique()) # Count the number of times it changes


            total_n_changes_list=[total_label_changes]*(len(s))
            total_n_labels_list=[total_n_labels]*(len(s))

            g = cell_df['tavg_label']
            total_tavg_label_changes = (np.diff(g)!=0).sum() # Count the number of times it changes
            total_tavg_labels = len(g.unique()) # Count the number of times it changes

            lab_list.append((cell_df['Condition'].unique()[0],rep, uniq_id, total_label_changes,total_n_labels,total_tavg_label_changes,total_tavg_labels))

            total_tavg_label_changes_list=[total_tavg_label_changes]*(len(g))
            total_tavg_labels_list=[total_tavg_labels]*(len(g))

            # The extra step of doing these calculations cumulatively, per timepoint.
            for t in cell_df['frame'].unique():

                # Get the dataframe including all timepoints upto and including t
                cumulative_df = cell_df[(cell_df['frame']<=t)]
                # Count the label changes and total numbers.
                s = cumulative_df['label']
                g=cumulative_df['tavg_label']

                label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
                tavg_label_changes = (np.diff(g)!=0).sum() # Count the number of times it changes
                n_labels = len(s.unique()) # Count the number of times it changes
                tavg_n_labels = len(g.unique()) # Count the number of times it changes
                # total_n_changes_list[1]
                # display(t)
                lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, label_changes,n_labels,total_n_changes_list[0],total_n_labels_list[0],
                                     tavg_label_changes,tavg_n_labels,total_tavg_label_changes_list[0],total_tavg_labels_list[0] ))

    # Turn the lists to dataframes
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'n_changes', 'n_labels', 'total_n_changes', 'total_n_labels', 'tavg_n_changes', 'tavg_n_labels', 'tavg_total_n_changes','tavg_total_n_labels'])
    lab_list_df = pd.DataFrame(lab_list, columns=['Condition','Replicate', 'Cell_ID', 'n_changes', 'n_labels', 'tavg_n_changes', 'tavg_n-labels'])
    print('     lab_list_tpt_df')
    display(lab_list_tpt_df[lab_list_tpt_df['Cell_ID'] == '0_22'])
#     display(lab_list_tpt_df.head(100))
    assert len(lab_list_tpt_df.index) == len(df.index), 'Label count list doesnt add up'

    # While the label list isthe same length, it's not sorted in the exact same way as the df input
    # Because the list is created on a cell-by-cell basis

    # Need to re-sort it by cell_id and frame
    lab_list_tpt_df.sort_values(["Cell_ID", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    # Resort the main dataframe too to be sure they're aligned before contatenating
    df.sort_values(["uniq_id", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    df.reset_index(inplace=True, drop=True)
    lab_list_tpt_df.reset_index(inplace=True, drop=True)

    # Test that this lines up with the original dataframe
    for i in range(len(lab_list_tpt_df)):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        id_1 = df.iloc[i]['uniq_id']
        id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        assert id_1 == id_2


    # Insert the timepoint label counts back into the input dataframe
    new_df = pd.concat([df,lab_list_tpt_df[['n_changes','n_labels','total_n_changes','total_n_labels','tavg_n_changes', 'tavg_n_labels', 'tavg_total_n_changes','tavg_total_n_labels']]], axis=1)

    return lab_list_df, new_df

########## DEV VERSION 1-4-2023 #########
def count_cluster_changes_DEV_deprecated(df, t_window=MIG_T_WIND): #This one incorporates tavg_df clustering calculations and is what is used now. Not time windowed

    '''
    Count the number of changes to the cluster ID in time windows
    '''

    assert 'label' in df.columns, 'count_cluster_changes() must be provided with label-containing dataframe.'
    assert 'tavg_label' in df.columns, 'count_cluster_changes() must be provided with tavg_label containing dataframe such as lab_tavg_lab_dr_df'

    lab_list = []
    lab_list_tpt = []



    for rep in df['Replicate_ID'].unique():

        for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            # If you've already attributed a unique id, then use it.
            if 'uniq_id' in cell_df.columns:
                assert len(cell_df['uniq_id'].unique()) == 1
                uniq_id = cell_df['uniq_id'].unique()[0]
            else:
                uniq_id = cell_id

            # At this point, you have a cell_df. Now, get a time window cell_df?

            init_f = int(np.min(cell_tarray[:,0]))
            final_f = int(np.max(cell_tarray[:,0]))

            t_window_arr = np.squeeze(cell_tarray[np.where((cell_tarray[:,0] >= t - t_window/2) &
                                                              (cell_tarray[:,0] < t + t_window/2))])
            init_frame_arr = t_window_arr[0,:] # Use the first row of the window

            # Calculate at the per-cell level
            s = cell_df['label']
            total_label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
            total_n_labels = len(s.unique()) # Count the number of times it changes


            total_n_changes_list=[total_label_changes]*(len(s))
            total_n_labels_list=[total_n_labels]*(len(s))

            g = cell_df['tavg_label']
            total_tavg_label_changes = (np.diff(g)!=0).sum() # Count the number of times it changes
            total_tavg_labels = len(g.unique()) # Count the number of times it changes

            lab_list.append((cell_df['Condition'].unique()[0],rep, uniq_id, total_label_changes,total_n_labels,total_tavg_label_changes,total_tavg_labels))

            total_tavg_label_changes_list=[total_tavg_label_changes]*(len(g))
            total_tavg_labels_list=[total_tavg_labels]*(len(g))

            # The extra step of doing these calculations cumulatively, per timepoint.
            for t in cell_df['frame'].unique():

                # Get the dataframe including all timepoints upto and including t
                cumulative_df = cell_df[(cell_df['frame']<=t)]
                # Count the label changes and total numbers.
                s = cumulative_df['label']
                g=cumulative_df['tavg_label']

                label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
                tavg_label_changes = (np.diff(g)!=0).sum() # Count the number of times it changes
                n_labels = len(s.unique()) # Count the number of times it changes
                tavg_n_labels = len(g.unique()) # Count the number of times it changes
                # total_n_changes_list[1]
                # display(t)
                lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, label_changes,n_labels,total_n_changes_list[0],total_n_labels_list[0],
                                     tavg_label_changes,tavg_n_labels,total_tavg_label_changes_list[0],total_tavg_labels_list[0] ))

    # Turn the lists to dataframes
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'n_changes', 'n_labels', 'total_n_changes', 'total_n_labels', 'tavg_n_changes', 'tavg_n_labels', 'tavg_total_n_changes','tavg_total_n_labels'])
    lab_list_df = pd.DataFrame(lab_list, columns=['Condition','Replicate', 'Cell_ID', 'n_changes', 'n_labels', 'tavg_n_changes', 'tavg_n-labels'])
    print('     lab_list_tpt_df')
    display(lab_list_tpt_df[lab_list_tpt_df['Cell_ID'] == '0_22'])
#     display(lab_list_tpt_df.head(100))
    assert len(lab_list_tpt_df.index) == len(df.index), 'Label count list doesnt add up'

    # While the label list isthe same length, it's not sorted in the exact same way as the df input
    # Because the list is created on a cell-by-cell basis

    # Need to re-sort it by cell_id and frame
    lab_list_tpt_df.sort_values(["Cell_ID", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    # Resort the main dataframe too to be sure they're aligned before contatenating
    df.sort_values(["uniq_id", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    df.reset_index(inplace=True, drop=True)
    lab_list_tpt_df.reset_index(inplace=True, drop=True)

    # Test that this lines up with the original dataframe
    for i in range(len(lab_list_tpt_df)):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        id_1 = df.iloc[i]['uniq_id']
        id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        assert id_1 == id_2


    # Insert the timepoint label counts back into the input dataframe
    new_df = pd.concat([df,lab_list_tpt_df[['n_changes','n_labels','total_n_changes','total_n_labels','tavg_n_changes', 'tavg_n_labels', 'tavg_total_n_changes','tavg_total_n_labels']]], axis=1)

    return lab_list_df, new_df

##### 1-10-2023 ### With time windows

# To include time-windowed versions of the cluster-change metrics.
def count_cluster_changes(df_in,t_window=MIG_T_WIND, min_frames=MIG_T_WIND):

    '''
    Count the number of changes to the cluster ID
    '''

    assert 'label' in df_in.columns, 'count_cluster_changes() must be provided with label-containing dataframe.'

    lab_list = []
    lab_list_tpt = []

    df = df_in.copy() # Make a copy so as not to modify the original input
    df.reset_index(inplace=True, drop=True)

    for rep in df['Replicate_ID'].unique():

        for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            # If you've already attributed a unique id, then use it.
            if 'uniq_id' in cell_df.columns:
                assert len(cell_df['uniq_id'].unique()) == 1
                uniq_id = cell_df['uniq_id'].unique()[0]
            else:
                uniq_id = cell_id


            # Calculating cluster changes at the per-cell level
            s = cell_df['label'] # label is the cluster number.
            label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
            n_labels = len(s.unique()) # Count the number of times it changes
            lab_list.append((cell_df['Condition'].unique()[0],rep, uniq_id, label_changes,n_labels))

            # The extra step of doing these calculations cumulatively, per timepoint.
            for t in cell_df['frame'].unique():

                # Cumulative measurements
                # Get the dataframe including all timepoints upto and including t
                cumulative_df = cell_df[(cell_df['frame']<=t)]
                # Count the label changes and total numbers.
                s = cumulative_df['label']
                label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
                n_labels = len(s.unique()) # Count the number of times it changes

                # Time-windowed measurements
                '''
                Think of the time-windowed measurements like a cumulative measurement
                with a moving lower bound. Therefore, calculations can be made the same way.
                '''
                # Get the dataframe for a window of timepoints centered at t

                t_wind_df = cell_df[(cell_df['frame']>=t - t_window/2) &
                              (cell_df['frame']<t + t_window/2)]

                # Apply a cutoff for number of frames, make calculations if satisfies
                if(len(t_wind_df) >= min_frames):
#                     print('Length sufficient')
                    # Count the label changes and total numbers for this window
                    tw_s = t_wind_df['label']
                    twind_label_changes = (np.diff(tw_s)!=0).sum() # Count the number of times it changes
                    twind_n_labels = len(tw_s.unique()) # Count the number of times it changes

                else:
#                     print('Length insufficient')
                    # Otherwise set the values to NaN
                    twind_label_changes = np.nan
                    twind_n_labels = np.nan

                # This effectively builds a new dataframe up from the list of lists
                lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, label_changes,n_labels,twind_label_changes,twind_n_labels))

    # Turn the lists to dataframes
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'n_changes', 'n_labels', 'twind_n_changes', 'twind_n_labels']) # Added the new columns here.
    lab_list_df = pd.DataFrame(lab_list, columns=['Condition','Replicate', 'Cell_ID', 'n_changes', 'n_labels'])
    assert len(lab_list_tpt_df.index) == len(df.index), 'Label count list doesnt add up'

    # Need to re-sort it by cell_id and frame
    lab_list_tpt_df.sort_values(["Cell_ID", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    # Resort the main dataframe too to be sure they're aligned before contatenating
    df.sort_values(["uniq_id", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    df.reset_index(inplace=True, drop=True)
    lab_list_tpt_df.reset_index(inplace=True, drop=True)

    # Test that this lines up with the original dataframe
    for i in range(len(lab_list_tpt_df)):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        id_1 = df.iloc[i]['uniq_id']
        id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        assert id_1 == id_2


    # Insert the timepoint label counts back into the input dataframe
    new_df = pd.concat([df,lab_list_tpt_df[['n_changes','n_labels', 'twind_n_changes', 'twind_n_labels']]], axis=1)

    return new_df

##########

def count_cluster_changes_with_tavg(df_in,t_window=MIG_T_WIND, min_frames=MIG_T_WIND):

    '''
    Count the number of changes to the cluster ID
    '''

    assert 'label' in df_in.columns, 'count_cluster_changes() must be provided with label-containing dataframe.'

    lab_list = []
    lab_list_tpt = []

    df = df_in.copy() # Make a copy so as not to modify the original input
    df.reset_index(inplace=True, drop=True)

    for rep in df['Replicate_ID'].unique():
        print("Processing replicate: " + str(rep))
        for cell_id in tqdm(df[df['Replicate_ID'] == rep]['particle'].unique()):
            # print("Processing cell number: " + str(cell_id) + ' and replicate: ' + str(rep))
            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            # If you've already attributed a unique id, then use it.
            if 'uniq_id' in cell_df.columns:
                assert len(cell_df['uniq_id'].unique()) == 1
                uniq_id = cell_df['uniq_id'].unique()[0]
            else:
                uniq_id = cell_id

            '''
            This part (now deprecated) calculated the n_changes and n_labels per cell track non-cumulatively
            '''
            # Calculating cluster changes at the per-cell level
            s = cell_df['label'] # label is the cluster number.
            label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
            n_labels = len(s.unique()) # Count the number of times it changes

            g = cell_df['tavg_label']
            tavg_label_changes = (np.diff(g)!=0).sum() # Count the number of times it changes
            tavg_n_labels = len(g.unique()) # Count the number of times it changes

            lab_list.append((cell_df['Condition'].unique()[0],rep, uniq_id, label_changes,n_labels, tavg_label_changes, tavg_n_labels))

            # The extra step of doing these calculations cumulatively, per timepoint.
            for t in cell_df['frame'].unique():

                '''
                This part calculates the n_changes and n_labels cumulatively per frame (expanding the 'window' by 1 each time)
                '''
                # Cumulative measurements
                # Get the dataframe including all timepoints upto and including t
                # print('Processing frame ' + str(t))
                cumulative_df = cell_df[(cell_df['frame']<=t)] #Is this part taking a long time?
                # Count the label changes and total numbers.
                s = cumulative_df['label']
                cum_label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
                cum_n_labels = len(s.unique()) # Count the number of times it changes

                g = cumulative_df['tavg_label']
                tavg_cum_label_changes = (np.diff(g)!=0).sum() # Count the number of times it changes
                tavg_cum_n_labels = len(g.unique()) # Count the number of times it changes

                # Time-windowed measurements
                '''
                This part calculated the time windowed n changes and n labels. It isn't cumulative.
                '''
                # Get the dataframe for a window of timepoints centered at t

                t_wind_df = cell_df[(cell_df['frame']>=t - t_window/2) &
                              (cell_df['frame']<t + t_window/2)]

                # Apply a cutoff for number of frames, make calculations if satisfies
                if(len(t_wind_df) >= min_frames):
#                     print('Length sufficient')
                    # Count the label changes and total numbers for this window
                    tw_s = t_wind_df['label']
                    twind_label_changes = (np.diff(tw_s)!=0).sum() # Count the number of times it changes
                    twind_n_labels = len(tw_s.unique()) # Count the number of times it changes

                    tw_g = t_wind_df['label']
                    tavg_twind_label_changes = (np.diff(tw_g)!=0).sum() # Count the number of times it changes
                    tavg_twind_n_labels = len(tw_g.unique()) # Count the number of times it changes

                    #
                    #
                    #

                else:
#                     print('Length insufficient')
                    # Otherwise set the values to NaN
                    twind_label_changes = np.nan
                    twind_n_labels = np.nan
                    tavg_twind_label_changes = np.nan
                    tavg_twind_n_labels = np.nan
                # This effectively builds a new dataframe up from the list of lists
                lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, cum_label_changes,cum_n_labels,twind_label_changes,twind_n_labels,
                tavg_cum_label_changes,tavg_cum_n_labels,tavg_twind_label_changes,tavg_twind_n_labels))





    # Turn the lists to dataframes
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'cum_n_changes', 'cum_n_labels', 'twind_n_changes', 'twind_n_labels',
    'tavg_cum_n_changes', 'tavg_cum_n_labels', 'tavg_twind_n_changes', 'tavg_twind_n_labels']) # Added the new columns here.
    lab_list_df = pd.DataFrame(lab_list, columns=['Condition','Replicate', 'Cell_ID', 'n_changes', 'n_labels', 'tavg_n_changes', 'tavg_n_labels'])
    assert len(lab_list_tpt_df.index) == len(df.index), 'Label count list doesnt add up'

    # Need to re-sort it by cell_id and frame
    lab_list_tpt_df.sort_values(["Cell_ID", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    # Resort the main dataframe too to be sure they're aligned before contatenating
    df.sort_values(["uniq_id", "frame"],
               axis = 0, ascending = True,
               inplace = True)

    df.reset_index(inplace=True, drop=True)
    lab_list_tpt_df.reset_index(inplace=True, drop=True)

    # Test that this lines up with the original dataframe
    print('Checking dataframe')
    for i in tqdm(range(len(lab_list_tpt_df))):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        id_1 = df.iloc[i]['uniq_id']
        id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        assert id_1 == id_2


    # Insert the timepoint label counts back into the input dataframe
    # new_df = pd.concat([df,lab_list_tpt_df[['cum_n_changes','n_labels', 'twind_n_changes', 'twind_n_labels']]], axis=1)
    new_df = pd.concat([df,lab_list_tpt_df[['cum_n_changes', 'cum_n_labels', 'twind_n_changes', 'twind_n_labels',
    'tavg_cum_n_changes', 'tavg_cum_n_labels', 'tavg_twind_n_changes', 'tavg_twind_n_labels']]], axis=1)


    return new_df

##########

#########

def count_time_in_label(df):


    from collections import Counter

    

    timinglist=[]
    timinglistmaster=[]
    # df=tptlabel_dr_df
    # rep='PBMCeNK_untreated_1__tracks'
    # cell_id=25
    # df['Replicate_ID'].unique()

    for rep in df['Replicate_ID'].unique():

            for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

                cell_df = df[(df['Replicate_ID']==rep) &
                                (df['particle']==cell_id)]

                # If you've already attributed a unique id, then use it.
                if 'uniq_id' in cell_df.columns:
                    assert len(cell_df['uniq_id'].unique()) == 1
                    uniq_id = cell_df['uniq_id'].unique()[0]
                else:
                    uniq_id = cell_id

                TimeSpentinEachlabeldict=Counter(cell_df['label']) #Counts the number of unique values and makes a dictionary object
                TimeSpentinEachlabel_labels=list(TimeSpentinEachlabeldict.keys()) #makes a list out of the dictionary
                TimeSpentinEachlabel_frames=list(TimeSpentinEachlabeldict.values()) #makes a list out of the dictionary

                condition_list=[cell_df['Condition_shortlabel'].unique()[0]]*(len(TimeSpentinEachlabel_labels))#makes a list of the same length
                rep_list=[rep]*(len(TimeSpentinEachlabel_labels)) # makes a list of the same length
                uniq_id_list=[uniq_id]*(len(TimeSpentinEachlabel_labels)) #makes a list of the same length

                timinglist=list(zip(condition_list,rep_list,uniq_id_list, TimeSpentinEachlabel_labels,TimeSpentinEachlabel_frames))

                timinglistmaster.extend(timinglist) #extends timinglistmaster each time

    timinglist_df = pd.DataFrame(timinglistmaster, columns=['Condition','Replicate','Cell_ID', 'label', 'time_in_label'])

    df=timinglist_df
    import seaborn as sns
    import matplotlib.pyplot as plt
    # CONDITION_CMAP='viridis'
    # cmap = cm.get_cmap(CONDITION_CMAP, len(sum_labels['Condition'].unique())) #wally
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(15, 10))
    # ax.set_xscale("log")
    time = df['time_in_label']
    SAMPLING_INTERVAL=10/60
    timeminutes=time*SAMPLING_INTERVAL
    x_lab = "label"
    y_lab = "time_in_label"
    plottitle = ""
    # ax= sns.boxplot(x=x_lab, y=timeminutes, data=df,
    #             whis=[25, 75], width=.6, hue='Condition', ax=ax, palette=CONDITION_CMAP)
    # sns.stripplot(x= x_lab, y= timeminutes, data=df,
    #               size=4, hue='Condition', dodge=True, ax=ax, palette=CONDITION_CMAP,ec='k', linewidth=1, alpha=.5)

    # ax = sns.boxenplot(x=x_lab, y=timeminutes, data=df, hue='Condition',ax=ax, palette=CONDITION_CMAP) #bw=.2, orient = 'v'
    # sns.stripplot(x= x_lab, y= timeminutes, data=df,
    #               size=4, hue='Condition', dodge=True, ax=ax, palette=CONDITION_CMAP, ec='k', linewidth=1, alpha=.5) 
   
    
    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition'].unique()))
    for i in range(cmap.N):
        colors.append(cmap(i))
        

    ax = sns.barplot(x=x_lab, y=timeminutes, data=df, hue='Condition',ax=ax, palette=colors, estimator='median', errorbar=('ci', 95), capsize=.1, errwidth=2, edgecolor='k', linewidth=2, alpha=0.9) #bw=.2, orient = 'v'
    sns.stripplot(x= x_lab, y= timeminutes, data=df,
                  size=4, hue='Condition', dodge=True, ax=ax, palette=colors,ec='k', linewidth=1, alpha=.5)
    
    # ax= sns.violinplot(x=x_lab, y=timeminutes, data=df, hue='Condition',ax=ax, palette=CONDITION_CMAP,  cut=1, linewidth=2, inner='quartile', scale = 'count') #bw=.2, orient = 'v'
    #inner{“box”, “quartile”, “point”, “stick”, None}, optional
    #scale{“area”, “count”, “width”}, optional
    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set_title(plottitle, fontsize=36)
    ax.set_xlabel("Cluster ID", fontsize=36)
    ax.set_ylabel("Time in cluster (min)", fontsize=36)
    ax.tick_params(axis='both', labelsize=36)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=5,fancybox=True)
    f.tight_layout()
    # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
    f.savefig(CLUST_DISAMBIG_DIR+'\TIMEINCLUSTERS.png', dpi=300)#plt.

    return timinglist_df

def get_exemplar_df(df, exemplars):
    exemplar_df = pd.DataFrame()
    lengthlistexemplar=np.arange(0, (len(exemplars)), 1).tolist()
    for exemp in lengthlistexemplar:
        cluster1_rows=exemplars[exemp]
        cluster1_rows=cluster1_rows.tolist()
        iterablelist2=np.arange(0, (len(cluster1_rows)), 1).tolist()
        for preciseexemp in iterablelist2:
            preciserowID=cluster1_rows[preciseexemp]
            row_from_df=df.iloc[[preciserowID]]
            exemplar_df = pd.concat([exemplar_df, row_from_df], axis=0)
    return exemplar_df


def dbscan(df_in, x_name, y_name, eps, cust_label = 'label'):

    '''
    Pretty sure this is a clustering accessory function
    that is now obeselete
    '''
    print('dbscan with eps=', eps)
    # DBScan
    sub_set = df_in[[x_name, y_name]] # self.df
    X = StandardScaler().fit_transform(sub_set)
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    print(np.unique(labels))
    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = [cust_label])

    return lab_df


def get_label_counts(df, per_rep=False):

    '''
    Takes a DBscan labelled dataframe as input, and computes the number of cells that has each of the labels,
    separated by condition or replicate.

    input:
        df: dataframe that must contain a 'label' column (i.e. output from  dbscan_clustering())

        per_rep: Boolean, if True - each row in the summary dataframe corresponds to an individual replicate,
            otherwise defaults to one row per conditon.

    returns:
        label_counts_df: dataframe with the columns: Condition, Replicate_ID, label, count


    Note: Output easily visualized by plotly express stripplot.
        fig = px.strip(lab_count_df, x="label", y="count", color="Condition")

    '''


    assert 'label' in df.columns, 'No label in dataframe'
    labels = df['label'].unique() #IMportantly computed for the whole set.
    label_counts_df = pd.DataFrame(columns=['Condition', 'Replicate_ID', 'label', 'count', 'percent'])

    # Per condition OR per replicate
    cond_list = df['Condition_shortlabel'].unique()
    rep_list = df['Replicate_ID'].unique()

    i=0

    if(per_rep):


        for this_rep in rep_list:

            this_rep_df = df[df['Replicate_ID'] == this_rep]

            assert len(this_rep_df['Condition'].unique()) == 1, 'Condition not unique in rep_df'

            # Count how many rows for each label
            for label in labels:

                 # Keep this dataframe being made for when we want to look at distributions
                this_lab_df = this_rep_df[this_rep_df['label'] == label]

                label_counts_df.loc[i] = [this_rep_df['Condition'].unique()[0], this_rep, label, len(this_lab_df.index) ]
                i+=1

    else:

        for cond in cond_list:

            this_cond_df = df[df['Condition_shortlabel'] == cond]
            totalforthiscondition=len(this_cond_df.index)
            # print(totalforthiscondition)

            # Count how many rows for each label
            for label in labels:

                 # Keep this dataframe being made for when we want to look at distributions
                this_lab_df = this_cond_df[this_cond_df['label'] == label]                
                fraction_in_label = len(this_lab_df.index)/totalforthiscondition
                percent_in_label = fraction_in_label*100
                label_counts_df.loc[i] = [cond, 'NA', label, len(this_lab_df.index), percent_in_label]
                i+=1
    # label_counts_df
    return label_counts_df        

def plot_label_counts(df_in, colors=CONDITION_CMAP):
    x_label = 'label'
    y_label='count'
    y_label2='percent'

    fig, ax = plt.subplots(figsize=[15,10])
    sns.barplot(x = x_label,
                y = y_label,
                hue = 'Condition',
                palette = colors,
                data = df_in, ax=ax
               )
    ax.legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=30,markerscale=20,fancybox=True)
    # ax.legend(fontsize=24)
    ax.xaxis.grid(True)
    # ax.set(ylabel=y)
    ax.set_title("", fontsize=36)
    ax.set_xlabel('Cluster label', fontsize=36)
    ax.set_ylabel('Absolute frequency', fontsize=36)
    ax.tick_params(axis='both', labelsize=36)
    # sns.despine(left=True)

    # ax.set_yticklabels(['eNK','eNK+CytoD'])
    fig.tight_layout()
    # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
    fig.savefig(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png', dpi=300)#plt.
    # Show the plot
    plt.show()

    fig2, ax2 = plt.subplots(figsize=[15,10])
    sns.barplot(x = x_label,
                y = y_label2,
                hue = 'Condition',
                palette = colors,
                data = df_in, ax=ax2
               )
    ax2.legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=30,markerscale=20,fancybox=True)
    # ax.legend(fontsize=24)
    ax2.xaxis.grid(True)
    # ax.set(ylabel=y)
    ax2.set_title("", fontsize=36)
    ax2.set_xlabel('Cluster label', fontsize=36)
    ax2.set_ylabel('Percent frequency', fontsize=36)
    ax2.tick_params(axis='both', labelsize=36)
    # sns.despine(left=True)

    # ax.set_yticklabels(['eNK','eNK+CytoD'])
    fig2.tight_layout()
    # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
    fig2.savefig(CLUST_DISAMBIG_DIR+'\cluster_label_percents.png', dpi=300)#plt.
    # Show the plot
    plt.show()



    return   


# def get_label_counts(df, per_rep=False): #deprecated 2-2023


#     '''
#     Takes a DBscan labelled dataframe as input, and computes the number of cells that has each of the labels,
#     separated by condition or replicate.

#     input:
#         df: dataframe that must contain a 'label' column (i.e. output from  dbscan_clustering())

#         per_rep: Boolean, if True - each row in the summary dataframe corresponds to an individual replicate,
#             otherwise defaults to one row per conditon.

#     returns:
#         label_counts_df: dataframe with the columns: Condition, Replicate_ID, label, count


#     Note: Output easily visualized by plotly express stripplot.
#         fig = px.strip(lab_count_df, x="label", y="count", color="Condition")

#     '''


#     assert 'label' in df.columns, 'No label in dataframe'
#     labels = df['label'].unique() #IMportantly computed for the whole set.
#     label_counts_df = pd.DataFrame(columns=['Condition', 'Replicate_ID', 'label', 'count'])

#     # Per condition OR per replicate
#     cond_list = df['Condition'].unique()
#     rep_list = df['Replicate_ID'].unique()

#     i=0

#     if(per_rep):


#         for this_rep in rep_list:

#             this_rep_df = df[df['Replicate_ID'] == this_rep]

#             assert len(this_rep_df['Condition'].unique()) == 1, 'Condition not unique in rep_df'

#             # Count how many rows for each label
#             for label in labels:

#                  # Keep this dataframe being made for when we want to look at distributions
#                 this_lab_df = this_rep_df[this_rep_df['label'] == label]

#                 label_counts_df.loc[i] = [this_rep_df['Condition'].unique()[0], this_rep, label, len(this_lab_df.index) ]
#                 i+=1

#     else:

#         for cond in cond_list:

#             this_cond_df = df[df['Condition'] == cond]


#             # Count how many rows for each label
#             for label in labels:

#                  # Keep this dataframe being made for when we want to look at distributions
#                 this_lab_df = this_cond_df[this_cond_df['label'] == label]

#                 label_counts_df.loc[i] = [cond, 'NA', label, len(this_lab_df.index) ]
#                 i+=1



#     return label_counts_df

# def plot_label_counts(df_in): #deprecated 2-2023
#     x_label = 'label'
#     y_label='count'

#     fig, ax = plt.subplots(figsize=[20,10])
#     sns.barplot(x = x_label,
#                 y = y_label,
#                 hue = 'Condition',
#                 palette = CONDITION_CMAP,
#                 data = df_in, ax=ax
#                )
#     ax.legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
#     # ax.legend(fontsize=24)
#     ax.xaxis.grid(True)
#     # ax.set(ylabel=y)
#     ax.set_title("", fontsize=36)
#     ax.set_xlabel('Cluster label', fontsize=36)
#     ax.set_ylabel('Total count', fontsize=36)
#     ax.tick_params(axis='both', labelsize=36)
#     # sns.despine(left=True)

#     # ax.set_yticklabels(['eNK','eNK+CytoD'])
#     fig.tight_layout()
#     # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
#     fig.savefig(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png', dpi=300)#plt.
#     # Show the plot
#     plt.show()

#     return
