#cleaning_Labeling.py

# from cellPLATO.cellPLATO.initialization.config_trackmate import *
from initialization.initialization import *
from initialization.config import *

import os
import numpy as np
import pandas as pd

import itertools

def clean_comb_df(df_in, deduplicate=True):

    '''
    A function with steps to clean up the comb_df
    to standardize the dataframe formatting for all the downstream processing steps

    '''

    df = df_in.copy()
    # Drop all-nan rows.
#     df.dropna(how='all', axis=1,inplace=True)
#     df.dropna(how='any', axis=0,inplace=True)

    if 'Replicate_ID' not in df.columns:
        print('No column Replicate_ID, renaming Experiment column')

        df.rename(columns = {'Experiment': 'Replicate_ID'}, inplace=True)

    df.dropna(subset = ['Condition', 'Replicate_ID'], inplace=True)

    # Create Rep_label column
    reps = df['Replicate_ID'].unique()
    allreps = df['Replicate_ID'].values

    rep_inds = np.empty([len(df)])

    for i, rep in enumerate(reps):
        rep_inds[np.where(allreps==rep)] = i

    df['Cond_label'] = df['Condition']
    df['Rep_label'] = rep_inds


    if 'level_0'  in df.columns:

        df.drop(columns=['level_0'], inplace=True)
        df.reset_index(inplace=True,drop=True)
        print('Dropped level_0 column.')


    if(deduplicate):

        #Prepare the combined dataframe for migration calculations
        #be ensuring there will be no overlap in columns

        overlap = list(set(df.columns).intersection(MIG_FACTORS))
        print('Overlap:', overlap)
        df.drop(columns=overlap, inplace=True)

        # Remove duplicate coloumns
        dedup_df = df.loc[:,~df.columns.duplicated()]
        df = dedup_df.copy()

    return df

def apply_unique_id(df):

    '''
    Add column to dataframe indicating a unique id for each cell, constructed as a concatenation of
    a numerical representation of the cells experimental replicate and the particle (cell) if.
    Of the form: XX_xx

    Additionally, adds column 'ntpts' to the dataframe, to make it easier to filter by track length.

    Input:
        df: DataFrame


    Returns:
        None. (Change is made directly to the passed dataframe.)

    '''

    for rep in df['Replicate_ID'].unique():

        for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

            # Create a unique cell identifier
            rep_ind = list(df['Replicate_ID'].unique()).index(rep)

            cell_uniq_ident = str(rep_ind) + '_' + str(int(cell_id))

            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            df_inds = list(df.index[(df['Replicate_ID']==rep)
                           & (df['particle']==cell_id)])
            ntpts = len(cell_df)
            # Add unique ID back into the original dataframe
            df.at[df_inds,'uniq_id'] = cell_uniq_ident
            df.at[df_inds,'ntpts'] = ntpts

def apply_unique_id_trackmate(df):

    '''
    Add column to dataframe indicating a unique id for each cell, constructed as a concatenation of
    a numerical representation of the cells experimental replicate and the particle (cell) if.
    Of the form: XX_xx

    Additionally, adds column 'ntpts' to the dataframe, to make it easier to filter by track length.

    Input:
        df: DataFrame


    Returns:
        None. (Change is made directly to the passed dataframe.)

    '''
    for filename in df['File_name'].unique():
         
        for cell_id in df[df['File_name'] == filename]['TRACK_ID'].unique():
            
            rep_ind = list(df['File_name'].unique()).index(filename)
            # print(f' The rep_ind is {rep_ind}')
            cell_uniq_ident = str(rep_ind) + '_' + str(int(cell_id))
            

            cell_df = df[(df['File_name']==filename) &
                            (df['TRACK_ID']==cell_id)]
            df_inds = list(df.index[(df['File_name']==filename)
                            & (df['TRACK_ID']==cell_id)])
            ntpts = len(cell_df)
            # Add unique ID back into the original dataframe
            df.at[df_inds,'uniq_id'] = cell_uniq_ident
            df.at[df_inds,'ntpts'] = ntpts




    # # in trackmate data, you can get a cell_df by minimally using the TRACK_ID, which is unique for each File_name

    # # for filename in df['File_name'].unique():
    # #     # extract a df
    # #     df_file = df[df['File_name'] == filename]
    # #     # then get the TRACK_IDs present in that df
    # #     for trackmate_id in df_file['TRACK_ID'].unique():

    # #         cell_df = df_file[df_file['TRACK_ID'] == trackmate_id]
    # #         # so you've got a cell_df, with the columns of trackmate

    # #         # now, create a unique cell ID
    # #         rep_ind = 


    #                     rep_ind = list(df['Replicate_ID'].unique()).index(rep)

    #         cell_uniq_ident = str(rep_ind) + '_' + str(int(cell_id))

    #         cell_df = df[(df['Replicate_ID']==rep) &
    #                     (df['Replicate_ID']==rep) &
    #                     (df['Replicate_ID']==rep) &
    #                         (df['particle']==cell_id)]

    #         df_inds = list(df.index[(df['Replicate_ID']==rep)
    #                     & (df['particle']==cell_id)])
            
    #         ntpts = len(cell_df)
    #         # Add unique ID back into the original dataframe
    #         df.at[df_inds,'uniq_id'] = cell_uniq_ident
    #         df.at[df_inds,'ntpts'] = ntpts      




    #         # then get the cell_ids present in that df
    #         for cell_id in df_file[df_file['TRACK_ID'] == trackmate_id]['particle'].unique():

    #             # Create a unique cell identifier
    #             trackmate_ind = list(df_file['TRACK_ID'].unique()).index(trackmate_id)

    #             cell_uniq_ident = str(trackmate_ind) + '_' + str(int(cell_id))

    #             cell_df = df_file[(df_file['TRACK_ID']==trackmate_id) &
    #                             (df_file['particle']==cell_id)]

    #             df_inds = list(df_file.index[(df_file['TRACK_ID']==trackmate_id)
    #                         & (df_file['particle']==cell_id)])
    #             ntpts = len(cell_df)
    #             # Add unique ID back into the original dataframe
    #             df.at[df_inds,'uniq_id'] = cell_uniq_ident
    #             df.at[df_inds,'ntpts'] = ntpts

    # for trackmate_id in df['Track_ID'].unique():
            
    #         for cell_id in df[df['Track_ID'] == trackmate_id]['particle'].unique():
    
    #             # Create a unique cell identifier
    #             trackmate_ind = list(df['Track_ID'].unique()).index(trackmate_id)
    
    #             cell_uniq_ident = str(trackmate_ind) + '_' + str(int(cell_id))
    
    #             cell_df = df[(df['Track_ID']==trackmate_id) &
    #                             (df['particle']==cell_id)]
    
    #             df_inds = list(df.index[(df['Track_ID']==trackmate_id)
    #                         & (df['particle']==cell_id)])
    #             ntpts = len(cell_df)
    #             # Add unique ID back into the original dataframe
    #             df.at[df_inds,'uniq_id'] = cell_uniq_ident
    #             df.at[df_inds,'ntpts'] = ntpts



    # for rep in df['Replicate_ID'].unique():

    #     for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

    #         # Create a unique cell identifier
    #         rep_ind = list(df['Replicate_ID'].unique()).index(rep)

    #         cell_uniq_ident = str(rep_ind) + '_' + str(int(cell_id))

    #         cell_df = df[(df['Replicate_ID']==rep) &
    #                      (df['Replicate_ID']==rep) &
    #                      (df['Replicate_ID']==rep) &
    #                         (df['particle']==cell_id)]

    #         df_inds = list(df.index[(df['Replicate_ID']==rep)
    #                        & (df['particle']==cell_id)])
            
    #         ntpts = len(cell_df)
    #         # Add unique ID back into the original dataframe
    #         df.at[df_inds,'uniq_id'] = cell_uniq_ident
    #         df.at[df_inds,'ntpts'] = ntpts            


def replace_labels_shortlabels(df):

                '''
                If shortlabels are used, Replace Condition labels with shortlabels.

                Should work on any dataframe, intended for adding shortlabels to the difference plots.
                '''

                assert USE_SHORTLABELS is True, 'This should only be used if USE_SHORTLABELS is True...'

                full_condition_list = list(df['Condition'])
                condition_shortlabels = []

                # Create a shortlabel per replicate
                rep_shortlabel_list = []

                for this_cond_label in full_condition_list:

                    this_cond_ind = CONDITIONS_TO_INCLUDE.index(this_cond_label)
                    this_shortlabel = CONDITION_SHORTLABELS[this_cond_ind]
                    condition_shortlabels.append(this_shortlabel)

                df['Condition'] = condition_shortlabels


def apply_filters(df, filter_cell=True, how = 'all', filter_dict=DATA_FILTERS):

    '''
    Apply the filters defines as FILTERS dictionary in config.py
    Apply in subsequent steps, and visualize the loss.

    Adds the 'included' column to the inputted dataframe

    Returns:
        Filtered dataframe


    '''

    print('Applying filters:')
    print(filter_dict)

    print('Beginning filtering ...')
    print(len(df.index), ' data points from ', len(df['uniq_id'].unique()), ' cells')

    df.to_csv(os.path.join(DATA_OUTPUT,'dr_df-prefilt.csv'))

    filt_counts=[]


    if(filter_cell is False):


        print('Applying data filters to individual timepoints:')
        print(filter_dict)
        print('...')

        for i,factor in enumerate(filter_dict.keys()):
            print(factor)
            print(filter_dict[factor][0], filter_dict[factor][1])

            '''Consider adding here the export csv summary step, to export along with plots'''
            filt_df = df[(df[factor] > filter_dict[factor][0]) &#]#)
                              (df[factor] < filter_dict[factor][1])]

            df.to_csv(os.path.join(DATA_OUTPUT,'filt_'+str(i)+'-'+factor+'.csv'))
            print(len(df.index), ' data points remaining.')
            assert len(df.index) > 0, 'Filtered out all the data.'
            filt_counts.append((factor, len(filt_df)))
    else:

        # Default filtering of entire cell.
        print('Applying filters to entire cell trajectory:')
        print(filter_dict)
        print('...')

        for cell_id in df['uniq_id'].unique():

            cell_df = df[df['uniq_id'] == cell_id]

            # make a list to hold the filter results per factor
            incl_list = []

            for i,factor in enumerate(filter_dict.keys()):

                if how == 'any':
                    included = cell_df[factor].between(filter_dict[factor][0],filter_dict[factor][1]).any()
                elif how == 'all':
                    included = cell_df[factor].between(filter_dict[factor][0],filter_dict[factor][1]).all()

                incl_list.append(included)
                filt_counts.append((factor, np.sum(included)))

            assert len(incl_list) == len(filter_dict.keys())

            # Get indices in the dataframe for this cell.
            df_inds = list(df.index[(df['uniq_id']==cell_id)])

            # Cell is only included if all of the list of criteria are met.
            if all(incl_list):

                # Add included flag if true
                df.at[df_inds,'included'] = True

            else:

                # Add unique ID back into the original dataframe
                df.at[df_inds,'included'] = False

    filt_df = df[df['included'] == True]

    print(' Finished filtering. Resulting dataframe contains:')
    print(len(filt_df.index), ' data points from ', len(filt_df['uniq_id'].unique()), ' cells')

    sum_counts = [(key, sum(num for _, num in value))
        for key, value in itertools.groupby(sorted(filt_counts), lambda x: x[0])]

    # Re-index the filtered dataframe, while keeping index of each row in the unfiltered dataframe.
    filt_df.reset_index(inplace=True)
    filt_df.rename(columns={'level_0': 'comb_df_row_ind'}, inplace=True)

    return filt_df, sum_counts



def factor_calibration(df, mixed_calibration=False):

    if mixed_calibration:
        print('Using mixed_calibration.')
        df_list = []

        # Make sure the lists of calibration factors are the correct length
        assert len(CONDITIONS_TO_INCLUDE) == len(MICRONS_PER_PIXEL_LIST), 'MICRONS_PER_PIXEL_LIST must be same sized list as CONDITIONS_TO_INCLUDE'
        assert len(CONDITIONS_TO_INCLUDE) == len(SAMPLING_INTERVAL_LIST),'SAMPLING_INTERVAL_LIST must be same sized list as CONDITIONS_TO_INCLUDE'

        for i, cond in enumerate(list(df['Condition'].unique())):

            microns_per_pixel = MICRONS_PER_PIXEL_LIST[i]
            sampling_interval = SAMPLING_INTERVAL_LIST[i]
            print(cond, microns_per_pixel,sampling_interval)

            sub_df = df[df['Condition'] == cond]

            for factor in FACTORS_TO_CONVERT:

                if(factor == 'area' or factor == 'filled_area' or factor == 'bbox_area'):
                    sub_df[factor] = sub_df[factor] * microns_per_pixel ** 2

                else:

                    sub_df[factor] = sub_df[factor] * microns_per_pixel

            # Special case for speed:

            ''' Be extra careful with speed
            May also need a correction relative to the base pixel calibration'''
            sub_df['speed'] = sub_df['speed'] * sampling_interval / SAMPLING_INTERVAL

            df_list.append(sub_df)

        df_out = pd.concat(df_list)


    else:

        df_out = df.copy()

        for factor in FACTORS_TO_CONVERT:

            if(factor == 'area' or factor == 'filled_area' or factor == 'bbox_area'):

                df_out[factor] = df_out[factor] * MICRONS_PER_PIXEL ** 2

            else:


                df_out[factor] = df_out[factor] * MICRONS_PER_PIXEL

    return df_out
