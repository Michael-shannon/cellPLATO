#measurements.py
from initialization.config import *
from initialization.initialization import *

import os
import numpy as np
import pandas as pd

from tqdm import tqdm

def calc_aspect_ratio(df, drop=False):

    df['aspect'] = df['major_axis_length']/df['minor_axis_length']

    # Remove NaNs
    if (drop):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["aspect"], how="all", inplace=True)



def ripley_K(X,r):

    '''
    Calculate Ripleys K for a given radius r

    '''

    # Extract the number of other points, p, within a distance r.
    rip = []

    for i,x in enumerate(X):

            # Get the distance matrix for this point.
            Xd = np.sqrt((X[:,0]-x[0])**2 + (X[:,1]-x[1])**2)

            Xd = np.delete(Xd, (i), axis=0) # Delete self.

            # Count the number of points within radius
            n = len(X) # Number of points total
            A = np.pi * r ** 2 # Area of circle with radius r
            p = sum(Xd < r) # Number of points within radius r
            K = p * A / n # Ripley's K - number of points within radius r per unit area
            L = (K / np.pi) ** 0.5 # Ripley's L - radius of circle with same density as K

            rip.append([p,K,L]) # Append tuple containing count, Ripley's K and L

    rip = np.asarray(rip)

    return rip




def calc_ripleys_xy(df_in, r=RIP_R, plot=False, inplace=False):

    '''
    Calculate ripleys p, K and L for a given radius r.
        Create a dataframe with these measurements.

    '''

    print('Calculating ripleys p, K and L with radius: ', r, ' (pixels)')
    df = df_in.copy()

    df_list = []

    for rep in df['Replicate_ID'].unique():

        rep_df = df[df['Replicate_ID'] == rep]


        for frame in rep_df['frame'].unique():

            t_df = rep_df[rep_df['frame'] == frame]
            pos = t_df[['x_um', 'y_um']].values
            rip = ripley_K(pos,r)


            t_df['rip_p'] = rip[:,0] # Number of points within radius
            t_df['rip_K'] = rip[:,1]
            t_df['rip_L'] = rip[:,2]

            df_list.append(t_df)


            if plot:

                '''
                Plot should be made to create animation, gif??
                '''

                plt.scatter(pos[:, 0], pos[:, 1], c=rip[:,2], s=t_df['area']/5) # Colormap by ripleys L

                plt.show()

    df_out = pd.concat(df_list)
    df_out.sort_index(inplace=True)

    return df_out



def standardize_factors_per_cell(df_in, factor_list=['area', 'perimeter']):

    from sklearn.preprocessing import StandardScaler

    df = df_in.copy()
    cell_df_list = []

    unique_id = 0 # Create a unique cell id
    rep_list = df['Replicate_ID'].unique()

    # For each replicate
    for i_rep, this_rep in enumerate(rep_list):

        rep_df = df[df['Replicate_ID']==this_rep]
        cell_ids = rep_df['particle'].unique() # Particle ids only unique for replicate, not between.

        # For each cell, calculate the average value and add to new DataFrame
        print('Replicate ',  i_rep+1, ' out of ', len(rep_list))
        for cid in tqdm(cell_ids):

            cell_df = rep_df[rep_df['particle'] == cid]

            # A test to ensure there is only one replicate label included.
            assert len(cell_df['Rep_label'].unique()) == 1, 'check reps'

            # x = get_data_matrix(cell_df, dr_factors=factor_list)
            x = cell_df[factor_list].values
            x_ = StandardScaler().fit_transform(x)


            cell_df[factor_list] = x_
            cell_df_list.append(cell_df)

    df_out = pd.concat(cell_df_list)
    df_out.sort_index(inplace=True)

    return df_out




def t_window_metrics(df_in, t_window=MIG_T_WIND,min_frames=MIG_T_WIND/2,factor_list=DR_FACTORS):

    '''
    Create measurements average and ratio measurements for each.
    '''

    df = df_in.copy()
    df_list = []

    time_avg_df = pd.DataFrame()
    unique_id = 0 # Create a unique cell id
    rep_list = df['Replicate_ID'].unique()
    new_factor_list = []

    # For each replicate
    for i_rep, this_rep in enumerate(rep_list):

        rep_df = df[df['Replicate_ID']==this_rep]
        cell_ids = rep_df['particle'].unique() # Particle ids only unique for replicate, not between.

        # For each cell, calculate the average value and add to new DataFrame (akin to making the tavg_df)
        print('Replicate ',  i_rep, ' out of ', len(rep_list))
        for cid in tqdm(cell_ids):

            cell_df = rep_df[rep_df['particle'] == cid]

            # A test to ensure there is only one replicate label included.
            assert len(cell_df['Rep_label'].unique()) == 1, 'check reps'

            # Unique list of frames for this cell
            frame_list = cell_df['frame'].unique()

            for frame in frame_list:

                # get a subset of the dataframe across the range of frames
                t_wind_df = cell_df[(cell_df['frame']>=frame - t_window/2) &
                                (cell_df['frame']<frame + t_window/2)]

                tpt_df = cell_df[(cell_df['frame']==frame)]

                assert len(tpt_df) == 1, 'Should be only one timepoint in dataframe'

                # Apply a minimal cutoff to avoid averaging too small a number of frames.
                if len(t_wind_df) >= min_frames:

                    # Do the measurements for each factor
                    for factor in factor_list:

                        mean_str = factor + '_tmean'
                        ratio_str = factor + '_ratio'

                        # Mean value for factor across time window
                        tpt_df[mean_str] = np.nanmean(t_wind_df[factor]) #adds new col to df called 'area_tmean' for example

                        # Ratio
                        tpt_df[ratio_str] = tpt_df[factor] / tpt_df[mean_str]

                        # Keep a list of the factors in order to make DR methods easier to implement
                        new_factor_list.append(factor)
                        new_factor_list.append(mean_str)
                        new_factor_list.append(ratio_str)

                    df_list.append(tpt_df) # Append the row of new calculations to a list of dataframes

            # Increase the unique id given to each cell
            unique_id += 1

    # Assemble the df_list into a dataframe and reorder by index.
    df_out = pd.concat(df_list)
    df_out.sort_index(inplace=True)

    new_factor_list=np.unique(new_factor_list)



    return df_out, new_factor_list
