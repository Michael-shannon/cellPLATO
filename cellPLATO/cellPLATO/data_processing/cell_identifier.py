#cell_identifier.py
# Functions for finding cells that meet a criteria, or random ones.
#test#

# from cellPLATO.cellPLATO.initialization.config_trackmate import *
from initialization.config import *
from initialization.initialization import *

import os
import numpy as np
import pandas as pd

def get_random_cell(df):

    # Select random row.
    i_row = np.random.randint(len(df))
    row = df.iloc[i_row]

    # Get sub_df for cell from random row
    cell_df = df[(df['Condition']==row['Condition']) &
                    (df['Replicate_ID']==row['Replicate_ID']) &
                    (df['particle']==row['particle'])]

    return cell_df

def get_cell_mean_variance(df,factor, sortby='mean'):

    '''
    Rank the cells in df with respect to their standard deviation of a given factor.
    Used to find example cells that show large changes in a specific factor over time.
    '''

    avg_list = []

    for rep in df['Replicate_ID'].unique():

        for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

            # Create a unique cell identifier
            rep_ind = list(df['Replicate_ID'].unique()).index(rep)

            cell_uniq_ident = str(rep_ind) + '_' + str(int(cell_id))

            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            avg_list.append((rep, cell_id, cell_uniq_ident, np.mean(cell_df[factor]),np.std(cell_df[factor])))

            df_inds = list(df.index[(df['Replicate_ID']==rep)
                           & (df['particle']==cell_id)])

            # Add unique ID back into the original dataframe
            df.at[df_inds,'uniq_id'] = cell_uniq_ident


    mean_std_df = pd.DataFrame(data=avg_list,columns=['rep', 'cell_id','cell_uniq_ident', 'mean','std'])

    if sortby=='mean':
        mean_std_df.sort_values(by='mean', ascending=False, inplace=True)

    elif sortby=='std':
        mean_std_df.sort_values(by='std', ascending=False, inplace=True)


    return mean_std_df


def get_cell_variance(df,factor):

    '''
    TEMP - TO DELETE.
    '''
    print('() is discontinued, use get_cell_mean_variance() instead.')


# Get that cell and confirm it has the same measured value.
def get_specific_cell(sum_df, full_df,nth):

    '''
    Having calculated the average and standard deviation for the factor of interest, find the specific cell from the main dataframe

    Input:
        sum_df: The dataframe containing summary measurements (ex: std)
        full_df: The full datafrme from which we want to extract an example cell
        nth: integer indicating which row of sum_df to extract the cell info.

    returns:
        cell_df: Section of full_df corresponding to the selected cell.
    '''

    this_rep = sum_df.iloc[nth]['rep']
    this_cell_id = sum_df.iloc[nth]['cell_id']
    this_std = sum_df.iloc[nth]['std']

    # Get sub_df for cell from random row
    cell_df = full_df[(full_df['Replicate_ID']==this_rep) &
                    (full_df['particle']==this_cell_id)]

    return cell_df




def get_cell_id(cell_df):

    '''
    For a given cell dataframe, return a string containing a unique identifier,
    accounting for the condition, replicate and cell number.
    '''

    assert len(np.unique(cell_df['particle'].values)) == 1, 'Should be only one cell in dataframe'
    cell_number = cell_df['particle'].values[0]

    rep_label = int(cell_df['Rep_label'].values[0])
    cond_label = cell_df['Cond_label'].values[0]
    cid_str = str(cond_label)+ '_' + str(rep_label)+ '_' + str(int(cell_number) )

    return cid_str







