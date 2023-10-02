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

#######################
## Adding DTW here ####
#######################

from fastdtw import fastdtw

# def dtw_distance(list1, list2, use_levenshtein=True):
#     """
#     Calculate the DTW distance between two sequences.
    
#     Parameters:
#     - sequence1: The first sequence (list of cluster IDs).
#     - sequence2: The second sequence (list of cluster IDs).
#     - use_levenshtein: If True, use Levenshtein distance; if False, use Hamming distance.
    
#     Returns:
#     - distance: The DTW distance between the sequences.
#     """
#     if use_levenshtein:
#         # Define a custom distance function for Levenshtein distance
#         def levenshtein_distance(list1, list2):
#             if len(list1) != len(list2):
#                 raise ValueError("Both lists should have the same length")
            
#             matrix = [[0] * (len(list2) + 1) for _ in range(len(list1) + 1)]
            
#             for i in range(len(list1) + 1):
#                 matrix[i][0] = i
#             for j in range(len(list2) + 1):
#                 matrix[0][j] = j
            
#             for i in range(1, len(list1) + 1):
#                 for j in range(1, len(list2) + 1):
#                     cost = 0 if list1[i - 1] == list2[j - 1] else 1
#                     matrix[i][j] = min(
#                         matrix[i - 1][j] + 1,      # Deletion
#                         matrix[i][j - 1] + 1,      # Insertion
#                         matrix[i - 1][j - 1] + cost  # Substitution
#                     )
            
#             return matrix[len(list1)][len(list2)]
        
#         # Calculate DTW distance using Levenshtein distance
#         distance, _ = fastdtw(list1, list2, dist=levenshtein_distance)
    
#     else:
#         # Define a custom distance function for Hamming distance
#         def hamming_distance(list1, list2):
#             if len(list1) != len(list2):
#                 raise ValueError("Both lists should have the same length")
            
#             distance = sum(el1 != el2 for el1, el2 in zip(list1, list2))
#             return distance
        
#         # Calculate DTW distance using Hamming distance
#         distance, _ = fastdtw(list1, list2, dist=hamming_distance)
    
#     return distance

# # Example usage:
# sequence1 = [1, 2, 3, 4, 5]
# sequence2 = [1, 3, 4, 5, 6, 7]

# # Calculate DTW distance using Levenshtein distance
# distance_levenshtein = dtw_distance(sequence1, sequence2, use_levenshtein=True)
# print(f"DTW distance (Levenshtein) between sequence1 and sequence2: {distance_levenshtein:.2f}")

# # Calculate DTW distance using Hamming distance
# distance_hamming = dtw_distance(sequence1, sequence2, use_levenshtein=False)
# print(f"DTW distance (Hamming) between sequence1 and sequence2: {distance_hamming}")


# from fastdtw import fastdtw
# import numpy as np

# def dtw_distance(sequence1, sequence2, use_levenshtein=True):
#     """
#     Calculate the DTW distance between two sequences.
    
#     Parameters:
#     - sequence1: The first sequence (matrix of cluster IDs).
#     - sequence2: The second sequence (matrix of cluster IDs).
#     - use_levenshtein: If True, use Levenshtein distance; if False, use Hamming distance.
    
#     Returns:
#     - distance: The DTW distance between the sequences.
#     """
#     if use_levenshtein:
#         # Define a custom distance function for Levenshtein distance
#         def levenshtein_distance(list1, list2):
#             if len(list1) != len(list2):
#                 raise ValueError("Both lists should have the same length")
            
#             matrix = [[0] * (len(list2) + 1) for _ in range(len(list1) + 1)]
            
#             for i in range(len(list1) + 1):
#                 matrix[i][0] = i
#             for j in range(len(list2) + 1):
#                 matrix[0][j] = j
            
#             for i in range(1, len(list1) + 1):
#                 for j in range(1, len(list2) + 1):
#                     cost = 0 if list1[i - 1] == list2[j - 1] else 1
#                     matrix[i][j] = min(
#                         matrix[i - 1][j] + 1,      # Deletion
#                         matrix[i][j - 1] + 1,      # Insertion
#                         matrix[i - 1][j - 1] + cost  # Substitution
#                     )
            
#             return matrix[len(list1)][len(list2)]
        
#         # Convert matrix-based sequences to lists of lists
#         sequence1 = sequence1.tolist()
#         sequence2 = sequence2.tolist()
        
#         # Calculate DTW distance using Levenshtein distance
#         distance, _ = fastdtw(sequence1, sequence2, dist=levenshtein_distance)
    
#     else:
#         # Define a custom distance function for Hamming distance
#         def hamming_distance(list1, list2):
#             if len(list1) != len(list2):
#                 raise ValueError("Both lists should have the same length")
            
#             distance = sum(el1 != el2 for el1, el2 in zip(list1, list2))
#             return distance
        
#         # Convert matrix-based sequences to lists of lists
#         sequence1 = sequence1.tolist()
#         sequence2 = sequence2.tolist()
        
#         # Calculate DTW distance using Hamming distance
#         distance, _ = fastdtw(sequence1, sequence2, dist=hamming_distance)
    
#     return distance

import textdistance
# import numpy as np
# import pandas as pd
import tqdm

def damerau_levenshtein_distance(seq1, seq2): 
    return 1.0 - textdistance.damerau_levenshtein.normalized_similarity(seq1, seq2)

# Calculate DTW distance between two sequences
def fastdtw_distance(seq1, seq2):
    distance, _ = fastdtw(seq1, seq2)
    return distance

def calculate_edit_distances(df, distancemetric = 'dameraulev', sequence_column='label', uniq_id_column='uniq_id', frame_column='frame', print_interval=1000):
    print(f'Using {distancemetric} distance metric')
    # Sort the DataFrame by 'uniq_id' and 'frame'
    df_sorted = df.sort_values([uniq_id_column, frame_column])
    
    # Group sequences by 'uniq_id' and collect them in a list
    grouped_sequences = df_sorted.groupby(uniq_id_column)[sequence_column].apply(list).tolist()
    
    # Convert sequences to NumPy arrays
    sequences_array = np.array(grouped_sequences)
    
    # Calculate the number of sequences
    num_sequences = len(grouped_sequences)
    
    # Initialize the distance matrix
    distance_matrix = np.zeros((num_sequences, num_sequences))
    
    # Calculate pairwise distances using broadcasting
    for i in tqdm.tqdm(range(num_sequences)):
        for j in range(i, num_sequences):
            if distancemetric == 'dameraulev':
                distance = damerau_levenshtein_distance(sequences_array[i], sequences_array[j])
            elif distancemetric == 'fastdtw':
                distance = fastdtw_distance(sequences_array[i], sequences_array[j])
            # distance = damerau_levenshtein_distance(sequences_array[i], sequences_array[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

            # Check if the current comparison index is a multiple of print_interval
            if (i * num_sequences + j) % print_interval == 0:
                print(f"Pairwise comparison ({i}, {j}):")
                print("Sequence 1:", sequences_array[i])
                print("Sequence 2:", sequences_array[j])
                print("Distance:", distance)

    np.save(SAVED_DATA_PATH + f'distance_matrix_{distancemetric}.npy', distance_matrix)            
    
    return distance_matrix

from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.lines import Line2D

def cluster_sequences(df, distance_matrix, do_umap = True, eps=0.1, min_samples=5, min_cluster_size = 5, n_neighbors = 5):


    
    if do_umap:
        print('Performing UMAP')
        # Apply UMAP for dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        )
        # reduced_data = reducer.fit_transform(distance_matrix)
        out_data = reducer.fit_transform(distance_matrix)

        # Create new columns in the dataframe for UMAP dimensions
        print(out_data[:, 0])
        print(out_data[:, 1])

            
        # # Create a dictionary to map uniq_id to umap traj 1
        UMAP1_mapping = dict(zip(df['uniq_id'].unique(), out_data[:, 0]))
        # Add 'umap traj 1' to the DataFrame 
        df['UMAP_traj_1'] = df['uniq_id'].map(UMAP1_mapping)
        # # Create a dictionary to map uniq_id to umap traj 1
        UMAP2_mapping = dict(zip(df['uniq_id'].unique(), out_data[:, 1]))
        # Add 'umap traj 1' to the DataFrame 
        df['UMAP_traj_2'] = df['uniq_id'].map(UMAP2_mapping)


    else:
        print('Skipping UMAP')

    # Cluster using HDBSCAN with adjusted parameters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    print(f'Using min_cluster_size = {min_cluster_size} and min_samples = {min_samples}')
    cluster_labels = clusterer.fit_predict(out_data)

    # Calculate the number of clusters
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    print(f'The number of clusters is {n_clusters}')



    # # Create a dictionary to map uniq_id to cluster labels
    cluster_mapping = dict(zip(df['uniq_id'].unique(), cluster_labels))

    # Add 'cluster_id' to the DataFrame based on cluster labels
    df['trajectory_id'] = df['uniq_id'].map(cluster_mapping)

    if n_clusters <= 1:
        print("Only one cluster was found. Skipping silhouette score calculation.")
        silhouette_avg = 'none'
    else:
        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(out_data, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg}")
        # Calculate Adjusted Rand Index 
        adjusted_rand = adjusted_rand_score(df['label'], df['trajectory_id'])
        print(f"Adjusted Rand Index: {adjusted_rand}")
        # Calculate Adjusted Mutual Information
        adjusted_mutual_info = adjusted_mutual_info_score(df['label'], df['trajectory_id'])
        print(f"Adjusted Mutual Information: {adjusted_mutual_info}")

        # Visualize the clusters as a bar plot
        cluster_counts = df.groupby('trajectory_id')['uniq_id'].count()

        # Sort the DataFrame by 'uniq_id' and 'frame'
        df = df.sort_values(['trajectory_id', 'uniq_id', 'frame'])

        ################# PLOTTING CHAOS BEGINS ####################
        colors = []
        cmap = cm.get_cmap(CLUSTER_CMAP) 
        numcolors=len(df['trajectory_id'].unique())
        for i in range(numcolors):
            colors.append(cmap(i))    

        # First plot:

        fig = plt.figure(figsize=(9, 6))
        fig.suptitle("Trajectory clusters", fontsize=PLOT_TEXT_SIZE)
        mpl.rcParams['font.size'] = PLOT_TEXT_SIZE
        plt.bar(cluster_counts.index, cluster_counts.values, color = 'black')
        plt.xlabel('Cluster (Trajectory ID)')
        plt.ylabel('Number of Trajectories')

        # Second plot: UMAP
        fig2 = plt.figure(figsize=(10, 10))
        scatter = plt.scatter(out_data[:, 0], out_data[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP, s=25, alpha=0.6)
        plt.xlabel('UMAP Dimension 1', fontsize=PLOT_TEXT_SIZE)
        plt.ylabel('UMAP Dimension 2', fontsize=PLOT_TEXT_SIZE)
        plt.title('UMAP: trajectory ID', fontsize=PLOT_TEXT_SIZE)

        # Create a custom legend
        legend_labels = np.unique(cluster_labels)
        legend_elements = []

        for label in sorted(legend_labels):
            indices = np.where(cluster_labels == label)[0]
            color = cmap(label / (len(legend_labels) - 1))  # Normalize the color by dividing by the number of unique labels
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Trajectory {label}', markerfacecolor=color, markersize=10))

        legend = plt.legend(handles=legend_elements, title="Trajectory ID", fontsize=PLOT_TEXT_SIZE, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().add_artist(legend)

        # Save the figure to the chosen directory with the specified name
        figure_name = "TrajectoryClustersOnUMAP.png"
        output_filename = os.path.join(TRAJECTORY_DISAMBIG_DIR, figure_name)
        fig2.savefig(output_filename, dpi=300)  # Adjust dpi as needed
        plt.show()


###########################################

        # Print DataFrame with trajectory IDs
        print("DataFrame with Trajectory IDs:")
        print(df[['uniq_id', 'frame', 'label', 'trajectory_id']])

        # Save the DataFrame to a CSV file
        df.to_csv(SAVED_DATA_PATH + f'tptlabel_dr_df_{min_cluster_size}_{min_samples}_silhouette_{silhouette_avg}.csv') #SAVE OPTION
        return df

# Using this, you will get UNIQUE examples of each trajectory_id to make a fake exemplar df that looks like the one you usually make

import pandas as pd
import random

def make_exemplar_df_basedon_trajectories(df, cells_per_traj=4):

    # Define the number of unique 'uniq_id' values to select for each 'trajectory_id'
    n = 4

    # Initialize an empty list to store the selected 'uniq_id' values
    selected_uniq_ids = []

    # Get a list of unique 'trajectory_id' values
    unique_trajectory_ids = df['trajectory_id'].unique()

    # Iterate over each unique 'trajectory_id'
    for traj_id in unique_trajectory_ids:
        # Filter the DataFrame to select rows for the current 'trajectory_id'
        traj_df = df[df['trajectory_id'] == traj_id]
        
        # Get a list of unique 'uniq_id' values within the current 'trajectory_id'
        unique_uniq_ids = traj_df['uniq_id'].unique().tolist()
        
        # Shuffle the list of unique 'uniq_id' values
        random.shuffle(unique_uniq_ids)
        
        # Take the first n 'uniq_id' values and add them to the selected_uniq_ids list
        selected_uniq_ids.extend(unique_uniq_ids[:cells_per_traj])

    # Initialize an empty DataFrame to store the selected rows
    exemplar_df_trajectories = pd.DataFrame()

    # Iterate over the selected 'uniq_id' values and select a random row for each 'uniq_id'
    for uniq_id in selected_uniq_ids:
        # Filter the DataFrame to get rows with the current 'uniq_id'
        selected_rows = df[df['uniq_id'] == uniq_id]
        
        # Randomly select one row from the filtered rows
        random_row = selected_rows.sample(n=1, random_state=42)  # You can change the random_state for reproducibility
        
        # Append the selected row to the exemplar DataFrame
        exemplar_df_trajectories = exemplar_df_trajectories.append(random_row)

        # Reset the index of the exemplar DataFrame
        exemplar_df_trajectories.reset_index(drop=True, inplace=True)
    
    # After that, output a 'full tracks' version of the exemplar df

    # Extract the full tracks from the df using the uniq_id's specified in the exemplar_df
    
    # List of unique 'uniq_id' values from the 'exemplar_df'
    unique_uniq_ids = exemplar_df_trajectories['uniq_id'].unique()

    # Extract full tracks for each 'uniq_id' in a list
    full_tracks = []

    for uniq_id in unique_uniq_ids:
        # Filter the DataFrame to get the rows with the current 'uniq_id'
        track = df[df['uniq_id'] == uniq_id]
        full_tracks.append(track)

    # Combine the full tracks into a single DataFrame if needed
    exemplar_df_trajectories_fulltrack = pd.concat(full_tracks)

    # 'full_tracks_df' now contains the full tracks for the 'uniq_id' values specified in 'exemplar_df'

    return exemplar_df_trajectories, exemplar_df_trajectories_fulltrack




