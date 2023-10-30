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

# def monge_elkan_distance(seq1, seq2):
#     similarity_score = textdistance.MongeElkan().similarity(seq1, seq2)
#     return 1.0 - similarity_score

# def monge_elkan_distance(seq1, seq2):
#     # Convert integer sequences to strings
#     seq1_str = " ".join(map(str, seq1))
#     seq2_str = " ".join(map(str, seq2))
    
#     # Calculate Monge-Elkan similarity score
#     similarity_score = textdistance.MongeElkan().similarity(seq1_str, seq2_str)
    
#     # Transform similarity into distance
#     distance = 1.0 - similarity_score
    
#     return distance

def monge_elkan_distance_batch(seq_batch1, seq_batch2):
    seq_batch1_str = [" ".join(map(str, seq)) for seq in seq_batch1]
    seq_batch2_str = [" ".join(map(str, seq)) for seq in seq_batch2]
    
    similarity_scores = textdistance.MongeElkan().similarity(seq_batch1_str, seq_batch2_str)
    
    # Transform similarities into distances
    distances = 1.0 - np.array(similarity_scores)
    
    return distances

# Calculate DTW distance between two sequences
def fastdtw_distance(seq1, seq2):
    distance, _ = fastdtw(seq1, seq2)
    return distance


import textdistance
import numpy as np
import tqdm
from fastdtw import fastdtw

# Define a function to calculate Monge-Elkan distance in batch
def monge_elkan_distance_batch(seq_batch1, seq_batch2):
    seq_batch1_str = [" ".join(map(str, seq)) for seq in seq_batch1]
    seq_batch2_str = [" ".join(map(str, seq)) for seq in seq_batch2]
    
    similarity_scores = textdistance.MongeElkan().similarity(seq_batch1_str, seq_batch2_str)
    
    # Transform similarities into distances
    distances = 1.0 - np.array(similarity_scores)
    
    return distances

def calculate_edit_distances_dev(df, distancemetric='dameraulev', sequence_column='label', uniq_id_column='uniq_id', frame_column='frame', print_interval=1000):
    print(f'Using {distancemetric} distance metric')
    df_sorted = df.sort_values([uniq_id_column, frame_column])
    grouped_sequences = df_sorted.groupby(uniq_id_column)[sequence_column].apply(list).tolist()
    sequences_array = np.array(grouped_sequences)
    num_sequences = len(grouped_sequences)
    distance_matrix = np.zeros((num_sequences, num_sequences))

    # Define batch size (adjust as needed)
    batch_size = 100

    for i in tqdm.tqdm(range(0, num_sequences, batch_size)):
        for j in range(i, num_sequences, batch_size):
            batch1 = sequences_array[i:i+batch_size]
            batch2 = sequences_array[j:j+batch_size]

            if distancemetric == 'dameraulev':
                # Process Damerau-Levenshtein distance (you can modify this part if needed)
                for i_batch in range(len(batch1)):
                    for j_batch in range(len(batch2)):
                        distance = damerau_levenshtein_distance(batch1[i_batch], batch2[j_batch])
                        distance_matrix[i+i_batch, j+j_batch] = distance
                        distance_matrix[j+j_batch, i+i_batch] = distance

            elif distancemetric == 'fastdtw':
                # Process FastDTW distance (you can modify this part if needed)
                distances = fastdtw_distance_batch(batch1, batch2)
                distance_matrix[i:i+batch_size, j:j+batch_size] = distances
                distance_matrix[j:j+batch_size, i:i+batch_size] = distances

            elif distancemetric == 'mongeelkan':
                # Calculate distances using batch processing
                distances = monge_elkan_distance_batch(batch1, batch2)
                distance_matrix[i:i+batch_size, j:j+batch_size] = distances
                distance_matrix[j:j+batch_size, i:i+batch_size] = distances

    return distance_matrix




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

            elif distancemetric == 'mongeelkan':
                distance = monge_elkan_distance(sequences_array[i], sequences_array[j])  # Use Monge-Elkan distance
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

# def cluster_sequences_deprecated(df, distance_matrix, do_umap = True, eps=0.1, min_samples=5, min_cluster_size = 5, n_neighbors = 5, dotsize = 10):


    
#     if do_umap:
#         print('Performing UMAP')
#         # Apply UMAP for dimensionality reduction
#         reducer = umap.UMAP(
#             n_neighbors=n_neighbors,
#             min_dist=0.0,
#             n_components=2,
#             random_state=42,
#         )
#         # reduced_data = reducer.fit_transform(distance_matrix)
#         out_data = reducer.fit_transform(distance_matrix)

#         # Create new columns in the dataframe for UMAP dimensions
#         print(out_data[:, 0])
#         print(out_data[:, 1])

            
#         # # Create a dictionary to map uniq_id to umap traj 1
#         UMAP1_mapping = dict(zip(df['uniq_id'].unique(), out_data[:, 0]))
#         # Add 'umap traj 1' to the DataFrame 
#         df['UMAP_traj_1'] = df['uniq_id'].map(UMAP1_mapping)
#         # # Create a dictionary to map uniq_id to umap traj 1
#         UMAP2_mapping = dict(zip(df['uniq_id'].unique(), out_data[:, 1]))
#         # Add 'umap traj 1' to the DataFrame 
#         df['UMAP_traj_2'] = df['uniq_id'].map(UMAP2_mapping)


#     else:
#         print('Skipping UMAP')
#         out_data = distance_matrix

#     # Cluster using HDBSCAN with adjusted parameters
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
#     print(f'Using min_cluster_size = {min_cluster_size} and min_samples = {min_samples}')
#     cluster_labels = clusterer.fit_predict(out_data)

#     # Calculate the number of clusters
#     n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
#     print(f'The number of clusters is {n_clusters}')



#     # # Create a dictionary to map uniq_id to cluster labels
#     cluster_mapping = dict(zip(df['uniq_id'].unique(), cluster_labels))

#     # Add 'cluster_id' to the DataFrame based on cluster labels
#     df['trajectory_id'] = df['uniq_id'].map(cluster_mapping)

#     if n_clusters <= 1:
#         print("Only one cluster was found. Skipping silhouette score calculation.")
#         silhouette_avg = 'none'
#     else:
#         # Calculate Silhouette Score
#         silhouette_avg = silhouette_score(out_data, cluster_labels)
#         print(f"Silhouette Score: {silhouette_avg}")
#         # Calculate Adjusted Rand Index 
#         adjusted_rand = adjusted_rand_score(df['label'], df['trajectory_id'])
#         print(f"Adjusted Rand Index: {adjusted_rand}")
#         # Calculate Adjusted Mutual Information
#         adjusted_mutual_info = adjusted_mutual_info_score(df['label'], df['trajectory_id'])
#         print(f"Adjusted Mutual Information: {adjusted_mutual_info}")

#         # Visualize the clusters as a bar plot
#         cluster_counts = df.groupby('trajectory_id')['uniq_id'].count()

#         # Sort the DataFrame by 'uniq_id' and 'frame'
#         df = df.sort_values(['trajectory_id', 'uniq_id', 'frame'])

#         ################# PLOTTING CHAOS BEGINS ####################
#         colors = []
#         cmap = cm.get_cmap(CLUSTER_CMAP) 
#         numcolors=len(df['trajectory_id'].unique())
#         for i in range(numcolors):
#             colors.append(cmap(i))    

#         # First plot:

#         fig = plt.figure(figsize=(9, 6))
#         fig.suptitle("Trajectory clusters", fontsize=PLOT_TEXT_SIZE)
#         mpl.rcParams['font.size'] = PLOT_TEXT_SIZE
#         plt.bar(cluster_counts.index, cluster_counts.values, color = 'black')
#         plt.xlabel('Cluster (Trajectory ID)')
#         plt.ylabel('Number of Trajectories')

#         # Create a dictionary that maps each unique trajectory_id to a color
#         color_dict = {id: color for id, color in zip(df['trajectory_id'].unique(), colors)}

#         # Create a new column 'color' in the DataFrame that maps each trajectory_id to its color
#         df['color'] = df['trajectory_id'].map(color_dict) #picasso

#         # Second plot: UMAP
#         fig2 = plt.figure(figsize=(10, 10))
#         # scatter = plt.scatter(out_data[:, 0], out_data[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP, s=25, alpha=0.6)
#         scatter = plt.scatter(df['UMAP_traj_1'], df['UMAP_traj_2'], c=df['color'], cmap=CLUSTER_CMAP, s=dotsize, alpha=0.6)
#         plt.xlabel('UMAP Dimension 1', fontsize=PLOT_TEXT_SIZE)
#         plt.ylabel('UMAP Dimension 2', fontsize=PLOT_TEXT_SIZE)
#         plt.title('UMAP: trajectory ID', fontsize=PLOT_TEXT_SIZE)

#         # Create a custom legend
#         legend_labels = np.unique(df['trajectory_id'])
#         legend_elements = []

#         for label in sorted(legend_labels):
#             color = color_dict[label]  # Use the same color mapping as the scatter plot
#             legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Trajectory {label}', markerfacecolor=color, markersize=10))

#         legend = plt.legend(handles=legend_elements, title="Trajectory ID", fontsize=PLOT_TEXT_SIZE, bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.gca().add_artist(legend)

#         # Save the figure to the chosen directory with the specified name
#         figure_name = "TrajectoryClustersOnUMAP.png"
#         output_filename = os.path.join(TRAJECTORY_DISAMBIG_DIR, figure_name)
#         fig2.savefig(output_filename, dpi=300)  # Adjust dpi as needed
#         plt.show()


# ###########################################

#         # Print DataFrame with trajectory IDs
#         print("DataFrame with Trajectory IDs:")
#         print(df[['uniq_id', 'frame', 'label', 'trajectory_id']])

#         # Save the DataFrame to a CSV file
#         df.to_csv(SAVED_DATA_PATH + f'tptlabel_dr_df_{min_cluster_size}_{min_samples}_silhouette_{silhouette_avg}.csv') #SAVE OPTION
#         return df
    
def cluster_sequences(df, distance_matrix, do_umap = True, eps=0.1, min_samples=5, min_cluster_size = 5, n_neighbors = 5, dotsize = 10):


    
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
        # print(out_data[:, 0])
        # print(out_data[:, 1])

            
        # # Create a dictionary to map uniq_id to umap traj 1
        UMAP1_mapping = dict(zip(df['uniq_id'].unique(), out_data[:, 0]))
        # Add 'umap traj 1' to the DataFrame 
        df['UMAP_traj_1'] = df['uniq_id'].map(UMAP1_mapping)
        # # Create a dictionary to map uniq_id to umap traj 1
        UMAP2_mapping = dict(zip(df['uniq_id'].unique(), out_data[:, 1]))
        # Add 'umap traj 1' to the DataFrame 
        df['UMAP_traj_2'] = df['uniq_id'].map(UMAP2_mapping)


    else:
        print('Skipping UMAP and clustering on n dimensions')

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


    # Cluster using HDBSCAN with adjusted parameters

        out_data = distance_matrix

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

        # Create a dictionary that maps each unique trajectory_id to a color
        color_dict = {id: color for id, color in zip(df['trajectory_id'].unique(), colors)}

        # Create a new column 'color' in the DataFrame that maps each trajectory_id to its color
        df['color'] = df['trajectory_id'].map(color_dict) #picasso

        # Second plot: UMAP
        fig2 = plt.figure(figsize=(10, 10))
        # scatter = plt.scatter(out_data[:, 0], out_data[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP, s=25, alpha=0.6)
        scatter = plt.scatter(df['UMAP_traj_1'], df['UMAP_traj_2'], c=df['color'], cmap=CLUSTER_CMAP, s=dotsize, alpha=0.6)
        plt.xlabel('UMAP Dimension 1', fontsize=PLOT_TEXT_SIZE)
        plt.ylabel('UMAP Dimension 2', fontsize=PLOT_TEXT_SIZE)
        plt.title('UMAP: trajectory ID', fontsize=PLOT_TEXT_SIZE)

        # Create a custom legend
        legend_labels = np.unique(df['trajectory_id'])
        legend_elements = []

        for label in sorted(legend_labels):
            color = color_dict[label]  # Use the same color mapping as the scatter plot
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



def plot_cumulative_plasticity_changes_trajectories(df, identifier='\_allcells', miny=None, maxy=None, t_window_multiplier = T_WINDOW_MULTIPLIER, plotametric=None, plotallcells = False): #spidey
    
    if plotallcells:
        f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False)
    else:
        f, axes = plt.subplots(2, 1, figsize=(15, 20), sharex=False)
    # f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False) #sharex=True

    if plotametric != None:
        whattoplot=[plotametric, plotametric]
    else:
        whattoplot=['cum_n_changes','cum_n_labels',]

    time = df['frame']

    timeminutes=time*SAMPLING_INTERVAL

    ##
    if miny != None or maxy != None:
        minimumy=miny
        maximumy1=maxy
        maximumy2=maxy

    else:
        minimumy=0
        maximumy1=np.nanmax(df[whattoplot[0]])
        maximumy2=np.nanmax(df[whattoplot[1]])

    ##
    import seaborn as sns
    sns.set_theme(style="ticks")

    colors=[]
    if CLUSTER_CMAP != 'Dark24':

        cmap = cm.get_cmap(CLUSTER_CMAP)
        numcolors= len(df['trajectory_id'].unique())
        sns.set_palette(CONDITION_CMAP)
        for i in range(numcolors):
            colors.append(cmap(i))
    else:
        colors = plotlytomatplotlibcolors()
        colors=colors[:len(df['trajectory_id'].unique())]

    df=df.dropna(how='any')

    conditionsindf = df['trajectory_id'].unique()


    if plotallcells == False:
            
        # Plot the responses for different events and regions
        sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
                    hue="trajectory_id",
                    data=df, palette=colors)

        sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
                    hue="trajectory_id",
                    data=df, palette=colors)
    else:
        # Create a dictionary to map condition_shortlabel to colors
        condition_colors = {condition: color for condition, color in zip(df['trajectory_id'].unique(), colors)}
        # Make timeminuteslist and get the last element for the annotations
        timeminuteslist = timeminutes.tolist()
        # x_final = timeminuteslist[-1] # Final entry in x_data
        max_x = np.max(timeminutes)
        # Create a set to keep track of plotted condition_shortlabels
        plotted_labels = set()
        # This will be used to offset the x position of the annotations
        # x_offsets = np.arange(20, len(df['uniq_id'].unique()), 1)
        x_offsets = [20, 100, 180, 260, 340, 420, 500] * int(len(df['uniq_id'].unique()) / 2 + 1) #used for placing labels in readable positions

        y_offsets = [0, -40, -80, -120] * int(len(df['uniq_id'].unique()) / 2 + 1) #used for placing labels in readable positions

        for offset, uniq_id in enumerate(df['uniq_id'].unique()):
            df_uniq = df[df['uniq_id'] == uniq_id]
            condition_label = df_uniq['trajectory_id'].iloc[0]  # Get the condition_shortlabel for this 'uniq_id'
            line_color = condition_colors[condition_label]  # Get the corresponding color for the condition_shortlabel
            if condition_label not in plotted_labels:
                # Plot the line with a label for the legend only if it hasn't been plotted before
                singleline = sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], data=df_uniq, color=line_color, label=condition_label)
                singleline2 = sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], data=df_uniq, color=line_color, label=condition_label)
                plotted_labels.add(condition_label)  # Add the condition_shortlabel to the set of plotted labels
            else:
                # If the label has already been plotted, don't add it to the legend again
                singleline = sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], data=df_uniq, color=line_color)
                singleline2 = sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], data=df_uniq, color=line_color) #, label=condition_label

            x_final_frames = df_uniq['frame'].iloc[-1]  # Final entry in x_data
            x_final = x_final_frames*SAMPLING_INTERVAL

            y_final = df_uniq[whattoplot[0]].iloc[-1]  # Final entry in y_data
            y_final1 = df_uniq[whattoplot[1]].iloc[-1]  # Final entry in y_data
            
            uniq_id_label = f'Cell_ID: {uniq_id}'
            axes[0].annotate(uniq_id_label, xy=(max_x + x_offsets[offset], y_final), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center', arrowprops=dict(arrowstyle="<-", color='black'))
            axes[0].plot([x_final, max_x + x_offsets[offset]], [y_final, y_final], linestyle='dotted', color='gray', transform=axes[0].transData, zorder=4)

            axes[1].annotate(uniq_id_label, xy=(max_x + x_offsets[offset], y_final1 ), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center',arrowprops=dict(arrowstyle="<-", color='black'))
            axes[1].plot([x_final, max_x + x_offsets[offset]], [y_final1, y_final1], linestyle='dotted', color='gray', transform=axes[1].transData, zorder=4)            
        
    timewindowmins = (MIG_T_WIND*t_window_multiplier) * SAMPLING_INTERVAL #Here is just a superficial matching of the time window specified in the config for plotting purposes
    # The actual calculation is done in count_cluster_changes_with_tavg function
    timewindowmins = round(timewindowmins, 1)

    print('Time window mins: ', timewindowmins)
    text1 = "Cumulative cluster switches"
    text2 = "Cumulative new clusters"
    # text3 = "New clusters / " + str(timewindowmins) + " min"

    x_lab = "Distinct Behaviors"
    plottitle = ""

    # get the max value of the whattoplot[1] column of df
    max1=np.nanmax(df[whattoplot[0]])
    max2=np.nanmax(df[whattoplot[1]])
    tickfrequency1 = int(max1/5)
    tickfrequency2 = int(max2/5)

    the_yticks = np.arange(0, len(df[whattoplot[0]].unique()), tickfrequency1)
    the_yticks = [int(x) for x in the_yticks]
    axes[0].set_yticks(the_yticks) # set new tick positions
    axes[0].margins(y=0) # set tight margins
    the_yticks = np.arange(0, len(df[whattoplot[1]].unique()),tickfrequency2 ) #tickfrequency1)
    the_yticks = [int(x) for x in the_yticks]


    axes[1].set_yticks(the_yticks) # set new tick positions
    axes[1].margins(y=0) # set tight margins

    # Tweak the visual presentation
    axes[0].xaxis.grid(True)
    axes[1].xaxis.grid(True)

    axes[0].set_ylabel(text1, fontsize=36)
    axes[1].set_ylabel(text2, fontsize=36)

    axes[0].set_title("", fontsize=36)
    axes[1].set_title("", fontsize=36)

    axes[0].set_xlabel("Time (min)", fontsize=36)
    axes[1].set_xlabel("Time (min)", fontsize=36)

    axes[0].set_ylim(0, max1+1)
    axes[1].set_ylim(0, max2+2)
  
    axes[0].tick_params(axis='both', labelsize=36)
    axes[1].tick_params(axis='both', labelsize=36)

    axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
    axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)

    f.tight_layout()
    f.savefig(TRAJECTORY_DISAMBIG_DIR+identifier+'_cumulative_plasticity_cluster_changes_over_time.png', dpi=300)#plt.

    return

def plot_plasticity_changes_trajectories(df, identifier='\_allcells', miny=None, maxy=None, t_window_multiplier = T_WINDOW_MULTIPLIER): #spidey

    # f, axes = plt.subplots(1, 3, figsize=(15, 5)) #sharex=True
    # f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=True) #sharex=True
    f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=False) #sharex=True

    whattoplot=['label','twind_n_changes', 'twind_n_labels']

    # CLUSTER_CMAP = 'tab20'
    # CONDITION_CMAP = 'dark'

    time = df['frame']
    # SAMPLING_INTERVAL=10/60 #This shouldn't be hardcoded!
    timeminutes=time*SAMPLING_INTERVAL

    # dfnumericals = df.select_dtypes('number')

    # extracted_col = df["Condition_shortlabel"]

    # df=dfnumericals.join(extracted_col)

    ##
    if miny != None or maxy != None:
        minimumy=miny
        maximumy1=maxy
        maximumy2=maxy
        maximumy3=maxy
    else:
        minimumy=0
        maximumy1=np.nanmax(df[whattoplot[0]])
        maximumy2=np.nanmax(df[whattoplot[1]])
        maximumy3=np.nanmax(df[whattoplot[2]])

    ##
    import seaborn as sns
    sns.set_theme(style="ticks")
    # sns.set_palette(CONDITION_CMAP) #removed
    # colors=[]
    # cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
    # for i in range(cmap.N):
    #     colors.append(cmap(i))

    colors=[]
    if CLUSTER_CMAP != 'Dark24':

        cmap = cm.get_cmap(CLUSTER_CMAP)
        numcolors= len(df['trajectory_id'].unique())
        sns.set_palette(CLUSTER_CMAP)
        for i in range(numcolors):
            colors.append(cmap(i))
    else:
        colors = plotlytomatplotlibcolors()
        colors=colors[:len(df['trajectory_id'].unique())]



    # display(df)
    df=df.dropna(how='any')
    # display(df)
    # Plot the responses for different events and regions
    sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
                 hue="trajectory_id",
                 data=df, palette=colors)

    sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
                 hue="trajectory_id",
                 data=df, palette=colors)

    sns.lineplot(ax=axes[2], x=timeminutes, y=whattoplot[2], #n_labels #n_changes #label
                 hue="trajectory_id",
                 data=df, palette=colors)

    timewindowmins = (MIG_T_WIND * t_window_multiplier)*SAMPLING_INTERVAL #Here is just a superficial matching of the time window specified in the config for plotting purposes
    # The actual calculation is done in count_cluster_changes_with_tavg function
    timewindowmins = round(timewindowmins, 1)

    print('Time window mins: ', timewindowmins)
    text1 = "Cluster ID"
    text2 = "Cluster switches / " + str(timewindowmins) + " min"
    text3 = "New clusters / " + str(timewindowmins) + " min"

    x_lab = "Distinct Behaviors"
    plottitle = ""

    # get the max value of the whattoplot[1] column of df
    max1=np.nanmax(df[whattoplot[1]])
    max2=np.nanmax(df[whattoplot[2]])
    tickfrequency1 = int(max1/5)
    tickfrequency2 = int(max2/5)

    the_yticks = np.arange(0, len(df[whattoplot[0]].unique()), 1)
    the_yticks = [int(x) for x in the_yticks]
    axes[0].set_yticks(the_yticks) # set new tick positions
    axes[0].margins(y=0) # set tight margins
    the_yticks = np.arange(0, len(df[whattoplot[1]].unique()), tickfrequency1)
    the_yticks = [int(x) for x in the_yticks]


    axes[1].set_yticks(the_yticks) # set new tick positions
    axes[1].margins(y=0) # set tight margins
    the_yticks = np.arange(0, len(df[whattoplot[2]].unique()), tickfrequency2)
    the_yticks = [int(x) for x in the_yticks]
    axes[2].set_yticks(the_yticks) # set new tick positions
    axes[2].margins(y=0) # set tight margins

    # the_xticks = np.arange(0, len(timeminutes), 1)
    # the_xticks = [int(x) for x in the_xticks]
    # axes[0].set_xticks(the_xticks) # set new tick positions
    # axes[0].margins(x=0) # set tight margins
    # axes[1].set_xticks(the_xticks) # set new tick positions
    # axes[1].margins(x=0) # set tight margins
    # axes[2].set_xticks(the_xticks) # set new tick positions
    # axes[2].margins(x=0) # set tight margins


    # Tweak the visual presentation
    axes[0].xaxis.grid(True)
    axes[1].xaxis.grid(True)
    axes[2].xaxis.grid(True)

    # axes[0].set_ylabel(whattoplot[0], fontsize=36)
    # axes[1].set_ylabel(whattoplot[1], fontsize=36)
    # axes[2].set_ylabel(whattoplot[2], fontsize=36)
    axes[0].set_ylabel(text1, fontsize=36)
    axes[1].set_ylabel(text2, fontsize=36)
    axes[2].set_ylabel(text3, fontsize=36)

    axes[0].set_title("", fontsize=36)
    axes[1].set_title("", fontsize=36)
    axes[2].set_title("", fontsize=36)

    axes[0].set_xlabel("Time (min)", fontsize=36)
    axes[1].set_xlabel("Time (min)", fontsize=36)
    axes[2].set_xlabel("Time (min)", fontsize=36)

    # axes[0].set_ylim(0, np.nanmax(df[whattoplot[0]]))
    # axes[1].set_ylim(0, np.nanmax(df[whattoplot[1]]))
    # axes[2].set_ylim(0, np.nanmax(df[whattoplot[2]]))
    axes[0].set_ylim(0, maximumy1)
    axes[1].set_ylim(0, maximumy2)
    axes[2].set_ylim(0, maximumy3)


    

    # ax.set_ylabel(y_lab, fontsize=36)
    axes[0].tick_params(axis='both', labelsize=36)
    axes[1].tick_params(axis='both', labelsize=36)
    axes[2].tick_params(axis='both', labelsize=36)

    axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
    axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
    axes[2].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)

    f.tight_layout()
    # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
    f.savefig(TRAJECTORY_DISAMBIG_DIR+identifier+'_trajectories_plasticity_cluster_changes_over_time.png', dpi=300)#plt.

    return