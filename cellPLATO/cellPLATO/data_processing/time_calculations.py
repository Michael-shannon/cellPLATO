# time_calculations.py

from initialization.config import *
from initialization.initialization import *

from data_processing.clustering import cluster_purity

import os
import numpy as np
import pandas as pd

def cluster_composition_timecourse(df):

    df_list = []

    for frame in df['frame'].unique():

        # Get dataframe at this timepoint
        tpt_sub_df = df[df['frame'] == frame]

        clust_sum_df = cluster_purity(tpt_sub_df)
        clust_sum_df['frame'] = frame

        df_list.append(clust_sum_df)

    df_out = pd.concat(df_list)
    df_out['Time (min)'] = df_out['frame'] * SAMPLING_INTERVAL
    df_out.reset_index(inplace=True)

    return df_out

def time_average(df):

    '''
    Needs a more descriptive name?
        average_across_time()?

    Function to generate a time-averaged dataframe,
    where the average value for each factor across all timepoints
    is calculated for each cell.

    Input:
        df: DataFrame [N * T * X]


    Returns:
        avg_df: DataFrame [N * X]
    '''

    time_avg_df = pd.DataFrame()
    unique_id = 0 # Create a unique cell id
    rep_list = df['Replicate_ID'].unique()

    for this_rep in rep_list:

        rep_df = df[df['Replicate_ID']==this_rep]
        cell_ids = rep_df['particle'].unique() # Particle ids only unique for replicate, not between.

        # For each cell, calculate the average value and add to new DataFrame
        for cid in cell_ids:

            cell_df = rep_df[rep_df['particle'] == cid]

            # A test to ensure there is only one replicate label included.
            assert len(cell_df['Rep_label'].unique()) == 1, 'check reps'

            avg_df = cell_df.mean() # Returns a series that is the mean value for each numerical column. Non-numerical columns are dropped.

            # Add back non-numeric data
            dropped_cols = list(set(cell_df.columns) - set(avg_df.index))

            for col in dropped_cols:

                assert len(cell_df[col].unique()) == 1, 'Invalid assumption: uniqueness of non-numerical column values'
                avg_df.loc[col] = cell_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)

            avg_df.loc['unique_id'] = unique_id # Add Unique cell ID for the analysis
            time_avg_df = time_avg_df.append(avg_df,ignore_index=True)
            unique_id += 1

    time_avg_df['frame'] = 'timeaverage' # Replace the meaningless average frame values with a string desciption

    return time_avg_df



def average_per_timepoint(df, t_window=None):

    '''
    For each timepoint, calculate the average across cells

    Note: this works for single timepoints or time windows, but
        doing these calculations at the level of the dataframe
        wont easily permit stdev and sem calculations

    Input:
        df: DataFrame [N * T * X]
        #poolreps: Boolean, default=False

    Returns:
        tpt_avg_df: DataFrame [T * X]

    '''

    tptavg_df = pd.DataFrame()

    frame_list = df['frame'].unique()
    cond_list = df['Condition'].unique()
    rep_list = df['Replicate_ID'].unique()

    '''
    Do we instead want to use FRAME_END?
    More user-controlled vs data-driven:
    frame_list = range(FRAME_END)
    '''

    for frame in frame_list:

        if t_window is not None:
            # get a subset of the dataframe across the range of frames
            frame_df = df[(df['frame']>=frame - t_window/2) &
                          (df['frame']<frame + t_window/2)]

        else:
            # Find the dataframe for a single frame
            frame_df = df[df['frame']==frame]

        # Separate by condition and **optionally** replicate
        for cond in cond_list:

            cond_df = frame_df[frame_df['Condition']==cond]

            for rep in rep_list:

                rep_df = cond_df[cond_df['Replicate_ID']==rep]

                if(len(rep_df) > MIN_CELLS_PER_TPT):

                    avg_df = rep_df.mean() # Returns a series that is the mean value for each numerical column. Non-numerical columns are dropped.

                    # Add back non-numeric data
                    dropped_cols = list(set(frame_df.columns) - set(avg_df.index))

                    for col in dropped_cols:

                        # Validate assumption that sub_df has only one rep/condition, then use this value in new frame
                        assert len(rep_df[col].unique()) == 1, 'Invalid assumption: uniqueness of non-numerical column values'
                        avg_df.loc[col] = rep_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)

                    if t_window is None: # assertion only works when no window is used.
                        assert avg_df.loc['frame'] == frame, 'Frame mismatch'

                    tptavg_df = tptavg_df.append(avg_df,ignore_index=True)
                else:
                    if(DEBUG):
                        print('Skipping: ',rep, ' N = ', len(rep_df))

    return tptavg_df
