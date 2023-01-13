#trajectory_clustering.py
from initialization.config import *
from initialization.initialization import *

# from visualization.trajectory_visualization import *

import os
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN

import similaritymeasures
import simplification

# from simplification.cutil import (
#     simplify_coords,
#     simplify_coords_idx,
#     simplify_coords_vw,
#     simplify_coords_vw_idx,
#     simplify_coords_vwp,
# )

def hausdorff( u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d


def lineterp_traj(traj,npts=20):

    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(traj, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    alpha = np.linspace(0, 1, npts)

    method = 'slinear'
    interpolator =  interp1d(distance, traj, kind=method, axis=0)
    new_pts = interpolator(alpha)

    return new_pts





def get_trajectories(cell_df_list,traj_factor='tSNE',interp_pts=20, zeroed=False, method='trajectory'):

    x_lab = None
    y_lab = None

    if traj_factor == 'xy':
        # x,y
        x_lab = 'x'
        y_lab = 'y'

    elif traj_factor == 'pca':
        x_lab = 'PC1'
        y_lab = 'PC2'

    elif traj_factor == 'tSNE':
        # tSNE
        x_lab = 'tSNE1'
        y_lab = 'tSNE2'

    elif traj_factor == 'umap':

        x_lab = 'UMAP1'
        y_lab = 'UMAP2'

    print('Defining trajectories using: (', x_lab, y_lab, ')')

    traj_list = []

    #
    # Standard case - get entire tracjectory
    #

    if(method=='trajectory'):

        for i, traj in enumerate(cell_df_list):#traj_list:
            this_traj = cell_df_list[i]
            this_traj = this_traj[[x_lab,y_lab]].values

            if(zeroed):
                this_traj = this_traj - this_traj[0,:]


            if interp_pts is not None:
                new_pts = lineterp_traj(this_traj,npts=interp_pts)
                traj_list.append(new_pts)

            else:
                traj_list.append(this_traj)

    #
    # Trajectory segment clustering
    #
    elif(method=='segment'):

        if interp_pts is not None:

            print('WARNING: Using interp_pts = ',interp_pts, ' with method = ', method)
            print(' Curently no interpolation being done for segment method.')


        for i, traj in enumerate(cell_df_list):#traj_list:
            this_traj = cell_df_list[i]
            this_traj = this_traj[[x_lab,y_lab]].values

            if(zeroed):
                this_traj = this_traj - this_traj[0,:]

            for j in range(len(this_traj)-1):

                seg = this_traj[j:j+2,:]
                assert seg.shape == (2,2), 'expecting single step of trajectory'
                traj_list.append(seg)

    return traj_list




def trajectory_distances(traj_list, method='hausdorff'):

    '''
    Compute the distance matrix for trajectories in the list, using designated method.
    '''

    traj_count = len(traj_list)
    D = np.zeros((traj_count, traj_count))

    for i in range(traj_count):
        for j in range(i + 1, traj_count):

            if (method=='hausdorff'):
                distance = hausdorff(traj_list[i], traj_list[j])

            elif(method=='frechet'):
                distance = similaritymeasures.frechet_dist(traj_list[i], traj_list[j])

            elif(method=='area'):
                area = similaritymeasures.frechet_dist(traj_list[i], traj_list[j])
                distance = area

            elif(method=='dtw'):
                distance = similaritymeasures.frechet_dist(traj_list[i], traj_list[j])

            D[i, j] = distance
            D[j, i] = distance

    print('Completed distance matrix, shape: ', D.shape)
    return D


# Compute and plot the clusters with these settings.
def find_max_clusters(D):

    # Find the cluster parameters that maximizes the number of clusters.
    epsrange = np.linspace(1,100,100)
    eps_count = []
    for eps in epsrange:

        mdl = DBSCAN(eps=eps, min_samples=10,metric='precomputed')
        cluster_lst = mdl.fit_predict(D)
        eps_count.append([eps,len(np.unique(np.asarray(cluster_lst)))])

    arr=np.asarray(eps_count)
    mask = (arr[:, 1] == np.max(arr[:,1]))
    filt_arr = arr[mask, :]

    eps=filt_arr[0,0] # This takes the first one.
    print('Determined maximum number of clusters where eps =  ', eps)

    return eps

def cluster_trajectories(traj_list, D, eps, label=''):

    # Plot the result.
    mdl = DBSCAN(eps=eps, min_samples=10,metric='precomputed')
    cluster_lst = mdl.fit_predict(D)

    # # plot_traj_cluster(interp_traj_list, cluster_lst)
    # plot_traj_cluster_avg(traj_list, cluster_lst, label)

    return cluster_lst


def simplify_trajectories(traj_list, method='rdp', param=1.0):

    '''
    method: str;
                'rdp' Ramer–Douglas–Peucker
                'vw' Visvalingam-Whyatt
    '''

    simpl_traj_list = []

    for traj in traj_list:

        traj = traj.copy(order='C') # Otherwise ValueError: ndarray is not C-contiguous

        if(method == 'rdp'):

            # Using Ramer–Douglas–Peucker
            simpl_traj = simplify_coords(traj, param) # param=eps

        elif (method == 'vw'):
            # Using Visvalingam-Whyatt
            simpl_traj = simplify_coords_vw(traj, param)

        simpl_traj_list.append(simpl_traj)

    return simpl_traj_list


def get_trajectory_segments(traj_list):

    seg_list = []
    for traj in traj_list:

        for j in range(len(traj)-1):

            seg = traj[j:j+2,:]
            assert seg.shape == (2,2), 'expecting single step of trajectory'

            seg_list.append(seg)

    return seg_list


def traj_clusters_2_df(df_in, cell_df_list, cluster_lst):

    '''
    Add a column for the trajectory label from cluster_lst into the original dataframe
    Using the unique id from  cell_df_list
    '''

    df = df_in.copy()

    assert len(cell_df_list) == len(cluster_lst)
    df['traj_id'] = np.nan # Initialize all, some will stay NaN if not long enough to anylse

    for i, cell_df in enumerate(cell_df_list):

        this_cell_id = cell_df['uniq_id'].unique()[0] # index [0] because it's a list
        this_traj_id = cluster_lst[i]


        # Now find the same unique id in the input dataframe (full-length)
        df_inds = list(df.index[df['uniq_id']==this_cell_id])

        df.at[df_inds,'traj_id'] = this_traj_id


    return df
