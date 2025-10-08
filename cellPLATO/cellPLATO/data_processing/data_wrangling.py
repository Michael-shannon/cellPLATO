#data_wrangling.py

from initialization.initialization import *
from initialization.config import *

import os
import numpy as np
import pandas as pd

import h5py

def format_for_superplots(df, metric, t, to_csv=False):

    '''
    Dataframe should contain the combination of all loaded datasets to be included in the superplots
    metric: a string relating to one of the dataframe column headers, telling which measurement to include in the superplots.

    t: timepoint for visualization

    '''
    # get sub dataframe at the selected timepoint.
    sub_df = df.loc[(df['frame'] == t)]

    # if(DEBUG):
    #     print(sub_df.head())

    if(USE_SHORTLABELS):
        # Create dataframe from the selected series within the original
        frame = { 'Replicate': sub_df['Rep_label'], 'Treatment': sub_df['Condition_shortlabel'], str(metric): sub_df[metric]  }
    else:
        # Create dataframe from the selected series within the original
        frame = { 'Replicate': sub_df['Rep_label'], 'Treatment': sub_df['Cond_label'], str(metric): sub_df[metric]  }
    output_df = pd.DataFrame(frame)

    assert len(df.index) > 0, 'Error with empty dataframe'

    if to_csv:
        output_df.to_csv(DATA_OUTPUT+'superplot_fmt_'+metric+'_t_'+str(t)+'.csv')

    return output_df

def get_data_matrix(df, dr_factors=DR_FACTORS): #can deprecate as it is just a one liner
    '''
    input dataframe (df):

    returns x: ndarray, matrix containing numerical values to be considered in
        dimensionality reduction methods.
    '''

    sub_df = df[dr_factors] # Filter original dataframe by select factors

    x = sub_df.values   # Matrix to be used in the dimensionality reduction

    return x


'''
Spacetime-cube related functions
previously in spacetimecube.py,
used only in blender_visualization_pipeline()
'''


def df2stc(df):#,exp_list):

    '''
    Input: DataFrame containing a data from a number of cells at multiple timepoints.
    Note: Should also work on combined dataframes.

    Returns: N*n*t Numpy Array, where:
            N = the unique cell index (not ID)
            D = the factor extracted from the dataframe (including unique ID)
            t = integer timepoint index,
                (can be converted to time with experimental parameters)

    '''

    # Open question whether these should be defined somewhere else,
    # or stored with the Object like data.n_cells, data.n_factors, etc.



    '''
    Note: Particle numbers are only unique to each experiment.
    Cannot assume otherwise.
    Maybe need to be sure that this function only run on separate experiments.
    OR that it splits them up from the beginning.
    i.e. assert len(df['Condition'].unique()) == 1
    or: if len(df['Condition'].unique()) > 1: Split them.

    '''



    '''
    For testing/development purposes, use only the first condition
    Eventually will loop through each condition, creating an array for each and returning the list of arrays
    (Assert that the length of the list of arrays is the same as the length of the list of conditions.)
    '''

    # conditions = exp_list['Condition']

    #
    # if(DEBUG):
    #
    #     display(df)
    #
    #     ax1 = df.plot.scatter(x='x',
    #                           y='y',
    #                           c='DarkBlue')


    # Take only the first condition from the list.
    # sub_df = df.loc[(df['Condition'] == exp_list.loc[0]['Condition'])] #conditions[0]

    # Override conditional selection above, use full DataFrame
    sub_df = df.copy()

    factor_list = list(sub_df.columns) # Otherwise is <class 'pandas.core.indexes.base.Index'>
    n_factors = len(factor_list)

    # Select the first row to know about the data types
    row = sub_df.iloc[0] # Select first row of data frame


    strings = row[row.apply(isinstance, args=(str,))]
    non_strings = row[~row.apply(isinstance, args=(str,))]
    n_num_cats = len(non_strings)# number of numerical catergoies

    #Get the list of headers for the numerical catergories
    headers = non_strings.index.values

    # if(DEBUG):
    #     display(row)
    #     display(non_strings)

    # Assertions to catch problematic data input
    assert  n_num_cats + len(strings) == n_factors, 'Mismatach between categories'
    assert len(headers) == n_num_cats, 'Number of headers doesnt match number of non-numerical categories'

    cells = np.sort(sub_df['particle'].unique())
    frames = np.sort(sub_df['frame'].unique())

    n_cells = len(cells)
    n_frames = len(frames)


    # Build a list of dataframes for each timepoint.
    df_list = []

    for t in frames:
        t_df = sub_df.loc[(sub_df['frame'] == t)]
        df_list.append(t_df)

    # Built the spacetime cubes with space for the non-string contents only.
    stc = np.empty([n_cells, n_num_cats, n_frames])

    for ind, row in sub_df.iterrows():

        # Split the row into strings and numbers (non-strings)
        '''
        The assumption above should be asserted.
        '''
        row_str = row[row.apply(isinstance, args=(str,))]
        row_data = row[~row.apply(isinstance, args=(str,))]

        frame = int(row['frame'])
        cell = int(row['particle']) - 1

        np_row = row_data.to_numpy(copy=True) # Get the data elements of row in numpy format

        '''
        using to_numpy allowed for strings, but the strings can't go into the array.
        Will need to convert them to np.nans or ignore text entries entirely
        '''

        # On the first pass, check that the number of factors is correct.
        if(ind == 0):
            assert np.shape(np_row)[0] == n_num_cats, ' # rows != n_factors'
            assert np.shape(np_row) ==  np.shape(stc[1,:,1]), ' # rows != shape of stc'

            '''
            If something changes in the labelling pattern from imageJ/Fiji or other
            upstream software, the asserts below will throw an error to let us know
            the arrays won't be indexed correctly.
            '''

            # Ensure frame is zero indexed.
            assert frame == 0, 'Frame not correctly zero-indexed for numpy.'
            assert cell == 0, ' Cell not correctly zero-indexed for numpy.'

        # Data transformation (recall 0 indexing of numpy array)
        # Recall spacetime-cube dimensions stc[n_cells, n_factors, n_frames]
        stc[cell,:,frame] = np_row


    assert len(df_list) == np.shape(stc)[2], 'df_list length doesnt match time dimension of array'

    return stc, list(headers), df_list     # Or a list of stc's


def verify_stc(stc):

    '''
    A testing fiinun to validate that the time-array is create as expected.
    Not currently implemented as not working properly:
    To Do:
        - pass stc, or replace from asserts.
        - Repair the ValueError:
            The truth value of an array with more than one element is ambiguous.
            Use a.any() or a.all()
    '''

    print('Verifying that time-array matches with corresponding dataframe for that time point.')

    for t in range(np.shape(stc)[2]):

        for n in range(np.shape(stc)[0] - 2):# -1 because of cell indexing

            this_df = df_list[t]
            sub_df = this_df.loc[(this_df['particle'] == n+1)] # +1 accounts for zero indexing of np array but not cell (particle) #.

            x_ind = int(headers.index('x'))
            y_ind = int(headers.index('y'))

            if (len(sub_df['x']) > 0):  # This avoids assert errors on empty series of the dataframe.

                #  assert statements to check that everything lines up correctly
                assert sub_df['x'].values == stc[n,x_ind,t], 'Error'
                assert sub_df['y'].values == stc[n,y_ind,t], 'Error'



    print("If this is the only text you see, it means it worked")

def condense_stc(stc,headers, zero_it=False, x_label='x', y_label='y'):

    '''
    Function to Condense the spacetime cube to a 2D + time output.

    Function to 'zero' all of the cell trajectories such that they all
    begin at the origin of the graph (0, 0).

    Importantly it also reduces the shape to only the x and y positions.

    Inputs:
        stc: spacetime cube (numpy array) where:
                    ax=0 : cell number
                    ax=1 : factor, measurement
                    ax=2 : timepoint
        headers: Column headers from original dataframe that are passed
                to columns of the ndarray
        zero_it: Boolean (optional), controls weather the zeroing operation is
                performed. Otherwise, allows this function to format for space-time cube visualization

        x_label, y_label: strings, indicate the name of the column headers to be used in the animation.
            Allows us to use the tSNE dimensions in the spacetime cube.
    Output:
        zerod_stc

    '''
    assert not ((x_label != 'x') and (zero_it == True)), 'Zeroing a non-spatial dimension is not supported.'

    n_cells = np.shape(stc)[0]
    n_frames = np.shape(stc)[2]
    zerod_stc = np.empty([n_cells, 2, n_frames]) # Creates a spacetime-cube, formatted like a spreadsheet, cells in rows, columns for X and Y, and t in Z

    x_ind = int(headers.index(x_label))
    y_ind = int(headers.index(y_label))

    # Convert all zero values of x and y position to NaN
    xpos_arr = stc[:,x_ind,:]
    ypos_arr = stc[:,y_ind,:]

    # Replace zero values with np.nan
    xpos_arr[xpos_arr == 0] = np.nan# or use np.nan
    ypos_arr[ypos_arr == 0] = np.nan# or use np.nan

    # Insert the corrected values back into the array
    stc[:,x_ind,:] = xpos_arr
    stc[:,y_ind,:] = ypos_arr

    for i in range(0,n_cells):

        # For each cell, find the first frame on which the cell appears
        # This will be the first non-NaN value
        # Solution using x position only
        non_nan_inds = np.argwhere(~np.isnan(stc[i,x_ind,:]))
        first_ind = non_nan_inds[0]


        for j in range(0,n_frames):
            zerod_stc[i,0,j] = stc[i,x_ind,j] - stc[i,x_ind,first_ind] * zero_it
            zerod_stc[i,1,j] = stc[i,y_ind,j] - stc[i,y_ind,first_ind] * zero_it


    return zerod_stc


def zero_stc(stc,headers, zero_it=True):

    print('Warning, this function will be replaced by condense_stc(). ')

    '''
    DELETE THIS FUNCTION ONLY WHEN SURE THAT ALL USES OF zero_stc have been replaced with condense_stc.
    '''

    n_cells = np.shape(stc)[0]
    n_frames = np.shape(stc)[2]
    zerod_stc = np.empty([n_cells, 2, n_frames]) # Creates a spacetime-cube, formatted like a spreadsheet, cells in rows, columns for X and Y, and t in Z

    x_ind = int(headers.index('x'))
    y_ind = int(headers.index('y'))

    # Convert all zero values of x and y position to NaN
    xpos_arr = stc[:,x_ind,:]
    ypos_arr = stc[:,y_ind,:]

    # Replace zero values with np.nan
    xpos_arr[xpos_arr == 0] = np.nan# or use np.nan
    ypos_arr[ypos_arr == 0] = np.nan# or use np.nan

    # Insert the corrected values back into the array
    stc[:,x_ind,:] = xpos_arr
    stc[:,y_ind,:] = ypos_arr

    for i in range(0,n_cells):

        # For each cell, find the first frame on which the cell appears
        # This will be the first non-NaN value
        # Solution using x position only
        non_nan_inds = np.argwhere(~np.isnan(stc[i,x_ind,:]))
        first_ind = non_nan_inds[0]


        for j in range(0,n_frames):
            zerod_stc[i,0,j] = stc[i,x_ind,j] - stc[i,x_ind,first_ind] * zero_it
            zerod_stc[i,1,j] = stc[i,y_ind,j] - stc[i,y_ind,first_ind] * zero_it


    return zerod_stc



def stc2df(stc_0d):

    '''
    Transform the origin-corrected ndarray to a format
    to be visualized in 3d with plotly.

    Input:
        stc0d: 'zeroed' ndarray (time-array, spacetime-cube)

    Output:
        out_df: DataFrame, transposed and reshaped such that
                origin-corrected cells are in rows, with columns:
                cell, X0, Y0, t (slice)
    '''

    n,m,t = stc_0d.shape

    # Transpose the array upsteam of the reshape
    transp_array = np.transpose(stc_0d,(0,2,1))
    out_arr = np.column_stack((np.repeat(np.arange(n),t),
                                transp_array.reshape(n*t,-1),
                                np.repeat(np.arange(t),n)))

    out_df = pd.DataFrame(out_arr,columns=['cell', 'X0', 'Y0', 't'])

    return out_df


def load_and_combine_surfaces(cp):
    """
    Walk cp.DATA_PATH/<Condition>/<Rep>/ for *_surfaces.csv, load each
    (skipping junk row1, using row2 as header), rename, and concatenate.
    Returns the combined DataFrame and also writes comb_df.csv to cp.OUTPUT_PATH.
    
    Updated to properly handle Experiment and Replicate_ID following cellPLATO conventions:
    - Experiment = replicate folder name (rep)
    - Replicate_ID = replicate folder name (rep) 
    This follows the pattern used in load_trackmate.py and data_io.py
    """
    save_dir = os.path.join(cp.OUTPUT_PATH, cp.DATASET_SHORTNAME, 'saved_data')
    os.makedirs(save_dir, exist_ok=True)

    combined = []
    cond2short = dict(zip(cp.CONDITIONS_TO_INCLUDE, cp.CONDITION_SHORTLABELS))
    
    # Create file_id mapping based on conditions
    file_id_map = {
        cond.replace("Condition_", ""): idx
        for idx, cond in enumerate(cp.CONDITIONS_TO_INCLUDE)
    }

    for cond in os.listdir(cp.DATA_PATH):
        if cond not in cp.CONDITIONS_TO_INCLUDE:
            continue
        cond_path = os.path.join(cp.DATA_PATH, cond)
        
        for rep in os.listdir(cond_path):
            rep_path = os.path.join(cond_path, rep)
            if not os.path.isdir(rep_path):
                continue

            for fn in os.listdir(rep_path):
                if not fn.lower().endswith('_surfaces.csv'):
                    continue

                path = os.path.join(rep_path, fn)
                # 1) skip row1, use row2 as header
                df = pd.read_csv(path, header=0)
                # 2) drop stray unnamed cols
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                print(f'The columns in the df natively are now: {df.columns.tolist()}')
                
                # 3) rename exactly the fields you want
                df = df.rename(columns={
                    'Time':        'frame',
                    # 'Time (s)':    'time',
                    'TrackID':     'particle',
                    'Position X':  'x_um',
                    'Position Y':  'y_um',
                    'Position Z':  'z_um'
                })
                
                # Convert frame to time in seconds using SAMPLING_INTERVAL
                df['time'] = df['frame'] * cp.SAMPLING_INTERVAL  # Convert minutes to seconds
                print(f'The columns in the df are now: {df.columns.tolist()}')

                # 4) make sure they exist
                needed = ['frame', 'time', 'particle', 'x_um', 'y_um', 'z_um']
                missing = set(needed) - set(df.columns)
                if missing:
                    raise ValueError(f"Missing columns {missing} in {path}")

                # 5) optionally convert types (or leave as float if preferred)
                # df['frame'] = df['frame'].astype(int)
                # df['particle'] = df['particle'].astype(int)
                df['time'] = df['time'].astype(float)

                # 6) annotate coordinates in pixels
                df['x_pix'] = df['x_um']
                df['y_pix'] = df['y_um']
                df['z_pix'] = df['z_um']

                # 7) annotate condition/experiment - CORRECTED VERSION
                df['Condition'] = cond
                # Use the replicate folder name (rep) as both Experiment and Replicate_ID
                # This follows the pattern used in load_trackmate.py and data_io.py
                df['Experiment'] = rep  # Use rep folder name, not derived from condition
                df['Replicate_ID'] = rep  # Use rep folder name consistently
                
                # file_id should map to condition, not experiment
                cond_key = cond.replace("Condition_", "")
                df['file_id'] = file_id_map.get(cond_key, 0)

                df['included'] = True
                df['Condition_shortlabel'] = cond2short.get(cond, cond)

                # 8) build unique ID and counts
                # Use file_id (condition-based) + particle for uniqueness across conditions
                df['uniq_id'] = df['file_id'].astype(str) + "_" + df['particle'].astype(str)
                df['ntpts'] = df.groupby('uniq_id')['uniq_id'].transform('size')

                combined.append(df)

    if not combined:
        raise RuntimeError("No '*_surfaces.csv' files found in DATA_PATH.")

    comb_df = pd.concat(combined, ignore_index=True)
    comb_df.columns = comb_df.columns.str.replace(' ', '_', regex=False) 

    # Create a mapping of unique Replicate_IDs to integers
    unique_reps = comb_df['Replicate_ID'].unique()
    rep_to_int_map = {rep: i for i, rep in enumerate(unique_reps)}
    comb_df['repID_int'] = comb_df['Replicate_ID'].map(rep_to_int_map)
    print(comb_df['repID_int'])
    # Add repID_int as a prefix to the existing uniq_id
    comb_df['uniq_id'] = comb_df['repID_int'].astype(str) + "_" + comb_df['uniq_id']

    comb_df.to_csv(os.path.join(save_dir, 'comb_df.csv'), index=False)
    return comb_df


def load_and_combine_tracks(cp):
    """
    Walk cp.DATA_PATH/<Condition>/<Rep>/ for *_tracks.csv, load each
    (skipping junk row1, using row2 as header), rename, annotate, and
    concatenate into a single DataFrame. Writes tavg_df.csv to cp.OUTPUT_PATH.
    
    Updated to properly handle Experiment and Replicate_ID following cellPLATO conventions:
    - Experiment = replicate folder name (rep)
    - Replicate_ID = replicate folder name (rep) 
    This follows the pattern used in load_trackmate.py and data_io.py
    """
    # prepare save directory
    save_dir = os.path.join(cp.OUTPUT_PATH, cp.DATASET_SHORTNAME, 'saved_data')
    os.makedirs(save_dir, exist_ok=True)

    combined = []
    # maps full condition names → short labels and file IDs
    cond2short = dict(zip(cp.CONDITIONS_TO_INCLUDE, cp.CONDITION_SHORTLABELS))
    
    # Create file_id mapping based on conditions
    file_id_map = {
        cond.replace("Condition_", ""): idx
        for idx, cond in enumerate(cp.CONDITIONS_TO_INCLUDE)
    }

    for cond in os.listdir(cp.DATA_PATH):
        if cond not in cp.CONDITIONS_TO_INCLUDE:
            continue
        cond_path = os.path.join(cp.DATA_PATH, cond)

        for rep in os.listdir(cond_path):
            rep_path = os.path.join(cond_path, rep)
            if not os.path.isdir(rep_path):
                continue

            for fn in os.listdir(rep_path):
                if not fn.lower().endswith('_tracks.csv'):
                    continue

                path = os.path.join(rep_path, fn)
                # 1) skip the junk first row, use row 2 as header
                df = pd.read_csv(path, header=0)
                # 2) drop any stray 'Unnamed:' columns
                df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]
                for col in df.columns:
                    print(col)
                # 3) rename 'ID' → 'particle'
                if 'ID' in df.columns:
                    df = df.rename(columns={'ID': 'particle'})
                else:
                    raise ValueError(f"No 'ID' column found in {path}")

                # 4) annotate metadata - CORRECTED VERSION
                df['Condition'] = cond
                # Use the replicate folder name (rep) as both Experiment and Replicate_ID
                # This follows the pattern used in load_trackmate.py and data_io.py
                df['Experiment'] = rep  # Use rep folder name, not derived from condition
                df['Replicate_ID'] = rep  # Use rep folder name consistently
                
                # file_id should map to condition, not experiment
                cond_key = cond.replace("Condition_", "")
                df['file_id'] = file_id_map.get(cond_key, 0)
                df['Condition_shortlabel'] = cond2short.get(cond, cond)

                # 5) build unique particle ID & count points per track
                # Use file_id (condition-based) + particle for uniqueness across conditions
                df['uniq_id'] = df['file_id'].astype(str) + "_" + df['particle'].astype(str)
                df['ntpts'] = df.groupby('uniq_id')['uniq_id'].transform('size')

                combined.append(df)

    if not combined:
        raise RuntimeError("No '*_tracks.csv' files found in DATA_PATH.")

    tavg_df = pd.concat(combined, ignore_index=True)
    # please replace the spaces in the column names with underscores
    tavg_df.columns = tavg_df.columns.str.replace(' ', '_', regex=False) 

    # Create a mapping of unique Replicate_IDs to integers
    unique_reps = tavg_df['Replicate_ID'].unique()
    rep_to_int_map = {rep: i for i, rep in enumerate(unique_reps)}
    tavg_df['repID_int'] = tavg_df['Replicate_ID'].map(rep_to_int_map)
    # Add repID_int as a prefix to the existing uniq_id
    # Note: This uses comb_df which should be passed in or generated separately
    # For now, using tavg_df's own uniq_id
    tavg_df['uniq_id'] = tavg_df['repID_int'].astype(str) + "_" + tavg_df['uniq_id']

    tavg_df.to_csv(os.path.join(save_dir, 'tavg_df.csv'), index=False)
    return tavg_df