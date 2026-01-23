#migration_calculations.py

from initialization.config import *
from initialization.initialization import *

import os
import numpy as np
import pandas as pd

from tqdm import tqdm

def cell_delta_calcs(cell_tarray, t_window=MIG_T_WIND, debug=False):
    '''
    Calculate delta x, delta y, delta z, and directionality in radians for a given cell through time.
    This function uses the same time windowing strategy as cell_calcs().
    
    Input:
        cell_tarray: [T * 4] NumPy array, where T is the number of frames over which this cell was tracked
                    [frame, x_um, y_um, index] (or [frame, x_um, y_um, z_um, index] for 3D data)
        t_window: int; width of the time window in # of frames
        debug: bool; if True, print debugging information
        
    Returns:
        cell_delta_calcs: list of calculations for each timepoint
    '''
    
    cell_calcs = []
    
    if cell_tarray.shape[0] > 0:
        # Check if we have z-coordinate data (should be 5 columns: frame, x, y, z, index)
        has_z = cell_tarray.shape[1] >= 5
        
        # Find the first and last frame in which this cell was tracked
        init_f = int(np.min(cell_tarray[:, 0]))
        final_f = int(np.max(cell_tarray[:, 0]))
        
        if debug:
            print(f"    Cell has {cell_tarray.shape[0]} timepoints")
            print(f"    Frame range: {init_f} to {final_f}")
            print(f"    Available frames: {sorted(cell_tarray[:, 0])}")
            print(f"    Required window size: {t_window}")
        
        # Enumerate across the range of frames (EXACT SAME AS ORIGINAL)
        for i, t in enumerate(range(init_f, final_f)):
            
            # Extract separate arrays for the timepoints and window of interest (EXACT SAME AS ORIGINAL)
            prev_frame_arr = np.squeeze(cell_tarray[np.where(cell_tarray[:, 0] == t-1)])
            this_frame_arr = np.squeeze(cell_tarray[np.where(cell_tarray[:, 0] == t)])
            
            # Extract time window array using same strategy as original function (EXACT SAME AS ORIGINAL)
            t_window_arr = np.squeeze(cell_tarray[np.where((cell_tarray[:, 0] >= t - t_window/2) &
                                                          (cell_tarray[:, 0] < t + t_window/2))])
            
            # Check if t_window_arr is empty (FIX FOR IndexError)
            if t_window_arr.size == 0:
                if debug:
                    print(f"      Frame {t}: Empty time window - skipping")
                    print(f"        Window range: {t - t_window/2} to {t + t_window/2}")
                    available_in_range = cell_tarray[:, 0][(cell_tarray[:, 0] >= t - t_window/2) & 
                                                          (cell_tarray[:, 0] < t + t_window/2)]
                    print(f"        Available frames in range: {sorted(available_in_range)}")
                continue  # Skip if time window is empty
            
            # Handle both 1D and 2D cases for t_window_arr (EXACT SAME AS ORIGINAL)
            if t_window_arr.ndim == 1:
                # If only one frame in window, t_window_arr is 1D
                init_frame_arr = t_window_arr
            else:
                # If multiple frames in window, t_window_arr is 2D
                init_frame_arr = t_window_arr[0, :]  # Use the first row of the window
            
            # Only process calculations for which we have the entire window (EXACT SAME AS ORIGINAL)
            window_size = t_window_arr.shape[0] if t_window_arr.ndim > 1 else 1
            if window_size == t_window:
                
                # Extract coordinates for delta calculations 
                # For deltas, we only need initial and final positions in the time window
                if has_z:
                    # 3D case: [frame, x_um, y_um, z_um, index]
                    x0, y0, z0 = init_frame_arr[1:4]
                    xf, yf, zf = this_frame_arr[1:4]
                else:
                    # 2D case: [frame, x_um, y_um, index]
                    x0, y0 = init_frame_arr[1:3]
                    xf, yf = this_frame_arr[1:3]
                    z0, zf = 0, 0  # Set z to 0 for 2D data
                
                # Use the index of the row of the subdf to insert value into original df
                ind = this_frame_arr[-1]  # Last column is always the index
                
                # Calculate delta values across the time window (windowed)
                delta_x = xf - x0
                delta_y = yf - y0
                delta_z = zf - z0
                
                # Calculate directionality in radians (windowed)
                # Angle from initial position to final position in the window
                directionality_radians = np.arctan2(delta_y, delta_x)
                
                # Calculate INSTANT delta values (adjacent frames) if prev_frame_arr exists
                if prev_frame_arr.size > 0 and prev_frame_arr.ndim > 0:
                    if has_z:
                        # 3D case for instant calculations
                        xi, yi, zi = prev_frame_arr[1:4]
                    else:
                        # 2D case for instant calculations
                        xi, yi = prev_frame_arr[1:3]
                        zi = 0
                    
                    # Instant deltas (frame-to-frame)
                    instant_delta_x = xf - xi
                    instant_delta_y = yf - yi
                    instant_delta_z = zf - zi
                    
                    # Instant directionality in radians
                    instant_directionality_radians = np.arctan2(instant_delta_y, instant_delta_x)
                else:
                    # No previous frame available, set instant values to NaN
                    instant_delta_x = np.nan
                    instant_delta_y = np.nan
                    instant_delta_z = np.nan
                    instant_directionality_radians = np.nan
                
                # Combine current calculations into a list for the current timepoint
                delta_calcs = [ind,
                              delta_x,
                              delta_y, 
                              delta_z,
                              directionality_radians,
                              instant_delta_x,
                              instant_delta_y,
                              instant_delta_z,
                              instant_directionality_radians]
                
                # Add the current timepoint calculations to the cell-specific list
                cell_calcs.append(delta_calcs)
    
    return cell_calcs


def cell_calcs(cell_tarray, t_window=MIG_T_WIND):#, calibrate):

    '''
    Cell migration calculations for a given cell through time.
    This function is passed a numpy array corresponding to the timecourse of a single cell,
    (from a single experimental replicate)


    Migration calcs accessory function that is optimized to use Numpy only, instead
    of pandas.

    Input:
        cell_tarray: [T * 4] NumPy array, where T is the number of frames over which this cell was tracker
                    [frame, x_um, y_um, index]:

        t_window = int; width of the time window in # of frames.

    Returns:
        cell_calcs: list;

    UPDATED: This version of the function calculates certain values across a time window.


    '''

    cell_calcs = []
    mig_calcs = []

    if(cell_tarray.shape[0] > 0):

        # Find the first and last frame in which this cell was tracked.
        init_f = int(np.min(cell_tarray[:,0]))
        final_f = int(np.max(cell_tarray[:,0]))

        # Enumerate across the range of frames
        for i, t in enumerate(range(init_f, final_f)): # Because we need a count and an index, for cases where cells arent included throughout
                
                # Adding actual window size
                # actual_window_size = min(t - init_f + 1, final_f - t, t_window) #trackmate

                # Extract separate arrays for the timepoints and window of interest
                prev_frame_arr = np.squeeze(cell_tarray[np.where(cell_tarray[:,0] == t-1)])
                this_frame_arr = np.squeeze(cell_tarray[np.where(cell_tarray[:,0] == t)])

                # if INPUT_FMT == 'trackmate':
                    
                #     #### trackmate
                #     # Extract the time window array considering the actual window size
                #     t_window_arr = np.squeeze(cell_tarray[np.where((cell_tarray[:,0] >= t - actual_window_size//2) &
                #                                                     (cell_tarray[:,0] < t + actual_window_size//2))])
                #     size_of_window = actual_window_size


                #     # Check if the t_window_arr is not empty
                #     if t_window_arr.size > 0 and t_window_arr.shape[0] == actual_window_size:
                #         # Access the first row of the window
                #         init_frame_arr = t_window_arr[0,:]

                #         # ... [rest of your calculations]
                #     else:
                #         # Handle the case where t_window_arr is empty
                #         # For example, you can continue to the next iteration of the loop
                #         continue



                # else:
                #     t_window_arr = np.squeeze(cell_tarray[np.where((cell_tarray[:,0] >= t - t_window/2) &
                #                                                     (cell_tarray[:,0] < t + t_window/2))])
                #     size_of_window = t_window
                t_window_arr = np.squeeze(cell_tarray[np.where((cell_tarray[:,0] >= t - t_window/2) &
                                                                (cell_tarray[:,0] < t + t_window/2))])
                # size_of_window = t_window # Redundant, equivalent to t_window

                #####
                # Handle both 1D and 2D cases for t_window_arr
                if t_window_arr.ndim == 1:
                    # If only one frame in window, t_window_arr is 1D
                    init_frame_arr = t_window_arr
                else:
                    # If multiple frames in window, t_window_arr is 2D
                    init_frame_arr = t_window_arr[0,:] # Use the first row of the window
                    end_frame_arr  = t_window_arr[-1,:] # Use the last row of the window

#                 segment_length = np.nan # default value


                # Only process calculations for which we have the entire window
                # Check if we have enough frames in the window
                window_size = t_window_arr.shape[0] if t_window_arr.ndim > 1 else 1
                if(window_size == t_window):

                    # Extract the critical coordinates for making mnigration calculations
                    x0, y0 = init_frame_arr[1:3]
                    xi, yi = prev_frame_arr[1:3]
                    xf, yf = this_frame_arr[1:3]
                    x1, y1 = end_frame_arr[1:3]

                    # Extract the xy-track across the window
                    if t_window_arr.ndim == 1:
                        # Single frame case - reshape to 2D for consistency
                        window_traj = t_window_arr[1:3].reshape(1, -1)
                    else:
                        # Multiple frames case
                        window_traj = t_window_arr[:,1:3]

                    # Use the index of the row of the subdf to insert value into original df
                    ind = this_frame_arr[3]

                    # Decide which one to keep
                    segment_length =  np.sqrt((xf-xi)**2 + (yf-yi)**2)
#                     dist =  np.sqrt((xf-xi)**2 + (yf-yi)**2) # Redundant, equivalent to segment_length

                    '''Decide which one to keep'''
                    euc_dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
#                     net_dist = np.sqrt((xf-x0)**2 + (yf-y0)**2) # Redundant, equivalent to euc_dist

                    speed = segment_length / SAMPLING_INTERVAL # Units will be in microns per unit of time of T_INC


                    # Efficient cumulative euclidean distance calculation:
                    diff = np.diff(window_traj, axis=0, prepend=window_traj[-1].reshape((1, -1)))
                    ss = np.power(diff, 2).sum(axis=1)
                    cumul_euc_dist = np.sqrt(ss).sum()
#
                    # Calculate the cumulative path length across the window
                    '''
                    Would be nice to replace with a more efficient implementation
                    as for cumulative euclidean above
                    '''


                    # Calculations to be made across the window

                    cumulative_dist_sqrd  = 0 # reset for each window
                    dist_list = []
                    turn_list = []
                    dist_from_start = []

                    for n in range(1,len(window_traj)):

                        x_, y_ = window_traj[n-1,:]
                        x__, y__ =  window_traj[n,:]
                        dist = np.sqrt((x__-x_)**2 + (y__-y_)**2)
                        dist_list.append(dist)
                        dist_from_start.append(np.sqrt((x__-x0)**2 + (y__-y0)**2))

                        # Global turn (relative to previous frame)
                        glob_turn = np.arctan((y__ - y_) / (x__ - x_)) # Equivalent to turn_angle_radians
                        turn_list.append(glob_turn)


                    if INPUT_FMT == 'trackmate':
                        actual_window_size = len(window_traj)
                        assert len(dist_list) == actual_window_size - 1, 'length of computed distances does not match actual window size'
                    else:

                        assert len(dist_list) == t_window-1, 'length of computed distances doesnt match time window'

                    # Summary measurements across the time window
                    cumulative_length = np.sum(dist_list)
                    max_dist = np.max(dist_from_start)

                    # Mean-squared displacement (MSD)
                    msd = np.sum(np.power(dist_list,2)) / t_window


                    cumulative_dist_sqrd = cumulative_dist_sqrd + segment_length**2

                    # Meandering index
#                     meandering_ind = net_dist / total_dist
                    meandering_ind = euc_dist / cumulative_length
                    # Outreach Ratio
#                     outreach_ratio = max_dist / total_dist
                    outreach_ratio = max_dist / cumulative_length

                    # Arrest coefficient - proportion of track cell is immobile (speed < x um)

                    arrest_coefficient = sum(dist < ARREST_THRESHOLD for dist in dist_list) / len(dist_list)


                    #
                    # Direction calculations
                    #

                    # Global turn for this frame
                    glob_turn = np.arctan((yf - y0) / (xf - x0))# change from yi and xi
                    glob_turn_deg = np.degrees(glob_turn) #
                    dir_autocorr = np.cos(turn_list[int(t_window/2)-1] -
                                          turn_list[int(t_window/2)-2])

                    '''
                    The directional autocorrelation is usually calculated between this and the previous frame
                    It would be more interesting as compared to the trajectory in the time window./
                    '''

                    # Orientation
                    axis_angle = np.arctan(yf / xf) # Temp
                    orientation = np.cos(2 * np.radians(axis_angle))

                    # Directedness
                    directedness = (xf - x0) / euc_dist

                    # Turned angle (Between two frames)
                    turn_angle_radians = np.arctan((yf - yi) / (xf - xi))
                    turn_angle = np.degrees(turn_angle_radians)

                    # Endpoint directionality ratio
                    endpoint_dir_ratio = euc_dist / cumulative_length


                    # Combine current calculations into a list for the current timepoint
                    mig_calcs = [ind,
                                 euc_dist,
                                 segment_length,
                                 cumulative_length,
                                 speed,
                                 orientation,
                                 directedness,
                                 turn_angle,
                                 endpoint_dir_ratio,
                                 # New ones added:
                                 dir_autocorr,
                                 outreach_ratio,
                                 msd,
                                 max_dist,
                                 glob_turn_deg,
                                 arrest_coefficient]

                    # Add the current timepoint calculations to the cell-sepecific list of calculations
                    cell_calcs.append(mig_calcs)


    return cell_calcs


def migration_calcs(df_in):#, calibrate=CALIBRATE_MIG):

    '''
    Re-implementation of the previous Usiigaci function to calculate cell
    migration measurements, for the dataframe instead of a numpy array.


    Function works in two steps:
        1. Calculate any frame-independent measures, i.e. that don't require
            comparing to a previous frame. These are applied to the entire sub_df
            associated with a given cell. (Orientation)
        2. Calculate frame-dependent measures, where the difference of a measurement
            is made with a previous frame. These must be done on a further segmented
            dataframe.

    Read from df_in, make changes to df_out.

    '''
    # calibrate = CALIBRATE_MIG # Previously an input argument, placed in function so cannot be changed.

    df_out = df_in.copy() # Make a copy so as not to modify the original input
    df_out.reset_index(inplace=True, drop=True)
    # df_out.drop(columns=['index'],inplace=True) # Dropped already in reset_index
    assert len(df_out.index.unique()) == len(df_out.index), 'Dataframe indexes not unique'

    # Determine if dataframe contains a single replicate or multiple
    # by seeing if the column Replicate_ID exists.
    if 'Replicate_ID' in df_in.columns.values:
        print('Processing migration calculations of pooled data')

    else:

        # Add Replicate_ID with arbitrary values to the dataframe
        print('Processing single experiment, adding arbitrary Replicate_ID = -1')
        df_out['Replicate_ID'] = -1
        df_out['Condition'] = 'unknown'


    calcs_list = [] # Initialize for the whole dataframe

    conditions = df_in['Condition'].unique()


    for cond in conditions:

        # cond_df = df_in[df_in['Condition'] == cond]
        cond_df = df_out[df_out['Condition'] == cond]

        # If a combined dataframe is provided, it will have duplicate particle(cell)
        # numbers, therefore we must treat them separately
        
        exp_reps = cond_df['Replicate_ID'].unique()

        '''
        NOTE:
            Important to use Replicate_ID (string of experiment name) instead of
            Rep_label (integer), as Rep_label is only unique per condition.
        '''
        print('Processing migration_calcs() for condition: ',cond)

        for exp_rep in exp_reps:

            print('Processing migration_calcs() for experiment: ',exp_rep)

            # Get subset of dataframe corresponding to this replicate
            exp_subdf = cond_df[cond_df['Replicate_ID'] == exp_rep]
            assert len(exp_subdf.index)==len(exp_subdf.index.unique()), 'exp_subdf indexes not unique'

            # Get the number of frames and cells in this selection
            n_frames = int(np.max(exp_subdf['frame']))
            # n_cells = int(np.max(exp_subdf['particle'])) # This
            n_cells = len(exp_subdf['particle'].unique())
            # for n in tqdm(range(n_cells)):

            # print('n_frames: ',n_frames )
            # print('n_cells: ',n_cells )
            if INPUT_FMT != 'trackmate':
                thing_to_iterate = 'particle'
            elif INPUT_FMT == 'trackmate':
                thing_to_iterate = 'uniq_id'

            for n in tqdm(exp_subdf[thing_to_iterate].unique()): #put in thing_to_iterate
                # For each cell, get another subset of the dataframe
                cell_subdf = exp_subdf[exp_subdf[thing_to_iterate] == n] # was 'particle', now thing_to_iterate
                assert len(cell_subdf.index)==len(cell_subdf.index.unique()), 'exp_subdf indexes not unique'

                tarray = cell_subdf[['frame', 'x_um', 'y_um']].to_numpy()#cell_subdf['frame', 'x', 'y']
                inds = cell_subdf.index.values

                assert tarray.shape[0] == len(inds), 'indexes doesnt match tarray shape'
                assert len(inds) == len(np.unique(inds)), 'indexes not unique'

                tarray = np.c_[tarray,inds] # Append index as 4th column to the array
                assert tarray.shape[1] == 4, ''
                mig_calcs = cell_calcs(tarray)#, calibrate)

                if len(mig_calcs) > 0:
                    calcs_list.append(mig_calcs)

    calcs_array = np.vstack(calcs_list) # Arrat from the list

    # Insert back into dataframe
    mig_calcs_df = pd.DataFrame(data=calcs_array[:,1:],    # values
                                index=calcs_array[:,0],    # 1st column as index
                                columns=['euclidean_dist',
                                     'segment_length',
                                     'cumulative_length',
                                     'speed',
                                     'orientedness',
                                     'directedness',
                                     'turn_angle',
                                     'endpoint_dir_ratio',
                                     'dir_autocorr',
                                     'outreach_ratio',
                                     'MSD',
                                     'max_dist',
                                     'glob_turn_deg',
                                     'arrest_coefficient'])

                                # The old ones from the previous version of cell_calcs, kept here just in case.
                                # columns=['euclidean_dist','segment_length','cumulative_length','speed',
                                #         'orientedness', 'directedness', 'turn_angle', 'endpoint_dir_ratio'])#, 'dir_autocorr'])

    assert len(mig_calcs_df.index.unique()) == len(np.unique(calcs_array[:,0])), 'Created dataframe indexes don match values from calcs_array'

    df_out = df_out.join(mig_calcs_df) # Add migration calcs to dataframr

    return df_out


def delta_calcs(df_in, debug_first_cell=False):
    '''
    Calculate delta x, delta y, delta z, and directionality in radians for all cells in the dataframe.
    Uses the same time windowing strategy as migration_calcs().
    
    Input:
        df_in: DataFrame containing cell tracking data
        debug_first_cell: if True, enables debug output for the first cell processed
        
    Returns:
        df_out: DataFrame with added delta calculations
    '''
    
    df_out = df_in.copy()  # Make a copy so as not to modify the original input
    df_out.reset_index(inplace=True, drop=True)
    assert len(df_out.index.unique()) == len(df_out.index), 'Dataframe indexes not unique'
    
    # Determine if dataframe contains a single replicate or multiple
    if 'Replicate_ID' in df_in.columns.values:
        print('Processing delta calculations of pooled data')
    else:
        # Add Replicate_ID with arbitrary values to the dataframe
        print('Processing single experiment, adding arbitrary Replicate_ID = -1')
        df_out['Replicate_ID'] = -1
        df_out['Condition'] = 'unknown'
    
    calcs_list = []  # Initialize for the whole dataframe
    conditions = df_in['Condition'].unique()
    
    for cond in conditions:
        cond_df = df_out[df_out['Condition'] == cond]
        exp_reps = cond_df['Replicate_ID'].unique()
        
        print('Processing delta_calcs() for condition: ', cond)
        
        for exp_rep in exp_reps:
            print('Processing delta_calcs() for experiment: ', exp_rep)
            
            # Get subset of dataframe corresponding to this replicate
            exp_subdf = cond_df[cond_df['Replicate_ID'] == exp_rep]
            assert len(exp_subdf.index) == len(exp_subdf.index.unique()), 'exp_subdf indexes not unique'
            
            # Get the number of frames and cells in this selection
            n_frames = int(np.max(exp_subdf['frame']))
            n_cells = len(exp_subdf['particle'].unique())
            
            if INPUT_FMT != 'trackmate':
                thing_to_iterate = 'particle'
            elif INPUT_FMT == 'trackmate':
                thing_to_iterate = 'uniq_id'
            
            first_cell_processed = False
            for n in tqdm(exp_subdf[thing_to_iterate].unique()):
                # For each cell, get another subset of the dataframe
                cell_subdf = exp_subdf[exp_subdf[thing_to_iterate] == n]
                assert len(cell_subdf.index) == len(cell_subdf.index.unique()), 'cell_subdf indexes not unique'
                
                # Check if we have z-coordinate data
                if 'z_um' in cell_subdf.columns:
                    tarray = cell_subdf[['frame', 'x_um', 'y_um', 'z_um']].to_numpy()
                else:
                    tarray = cell_subdf[['frame', 'x_um', 'y_um']].to_numpy()
                
                inds = cell_subdf.index.values
                
                assert tarray.shape[0] == len(inds), 'indexes doesnt match tarray shape'
                assert len(inds) == len(np.unique(inds)), 'indexes not unique'
                
                tarray = np.c_[tarray, inds]  # Append index as last column to the array
                
                # Enable debug for first cell if requested
                debug_this_cell = debug_first_cell and not first_cell_processed
                if debug_this_cell:
                    print(f"  DEBUG: Processing first cell (particle {n}):")
                    first_cell_processed = True
                
                delta_calcs_result = cell_delta_calcs(tarray, debug=debug_this_cell)
                
                if len(delta_calcs_result) > 0:
                    calcs_list.append(delta_calcs_result)
    
    if calcs_list:  # Only proceed if we have calculations
        calcs_array = np.vstack(calcs_list)  # Array from the list
        
        # Insert back into dataframe
        delta_calcs_df = pd.DataFrame(data=calcs_array[:, 1:],    # values
                                     index=calcs_array[:, 0],    # 1st column as index
                                     columns=['delta_x',
                                             'delta_y',
                                             'delta_z',
                                             'directionality_radians',
                                             'instant_delta_x',
                                             'instant_delta_y',
                                             'instant_delta_z',
                                             'instant_directionality_radians'])
        
        assert len(delta_calcs_df.index.unique()) == len(np.unique(calcs_array[:, 0])), 'Created dataframe indexes dont match values from calcs_array'
        
        df_out = df_out.join(delta_calcs_df)  # Add delta calcs to dataframe
    else:
        print("Warning: No delta calculations were computed. Check that your data has sufficient timepoints.")
    
    return df_out


def calculate_and_add_delta_factors(df, debug_first_cell=False):
    '''
    Convenience function to calculate and add delta factors to an existing dataframe.
    This function can be easily called from a notebook cell.
    
    Input:
        df: DataFrame containing cell tracking data
        debug_first_cell: if True, enables debug output for the first cell processed
        
    Returns:
        df_with_deltas: DataFrame with added delta calculations
        
    Example usage in notebook:
        from data_processing.migration_calculations import calculate_and_add_delta_factors
        df_with_deltas = calculate_and_add_delta_factors(your_dataframe)
    '''
    print("Calculating delta x, delta y, delta z, and directionality in radians...")
    print(f"Using time window of {MIG_T_WIND} frames")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_cols = ['frame', 'x_um', 'y_um']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        df_with_deltas = delta_calcs(df, debug_first_cell=debug_first_cell)
        print("Delta calculations completed!")
        
        # Check what we actually got
        windowed_columns = ['delta_x', 'delta_y', 'delta_z', 'directionality_radians']
        instant_columns = ['instant_delta_x', 'instant_delta_y', 'instant_delta_z', 'instant_directionality_radians']
        new_columns = windowed_columns + instant_columns
        for col in new_columns:
            if col in df_with_deltas.columns:
                non_null_count = df_with_deltas[col].notna().sum()
                total_count = len(df_with_deltas)
                print(f"  {col}: {non_null_count}/{total_count} non-null values")
                if non_null_count > 0:
                    print(f"    Range: {df_with_deltas[col].min():.3f} to {df_with_deltas[col].max():.3f}")
            else:
                print(f"  {col}: NOT FOUND")
        
        print("Added windowed columns: 'delta_x', 'delta_y', 'delta_z', 'directionality_radians'")
        print("Added instant columns: 'instant_delta_x', 'instant_delta_y', 'instant_delta_z', 'instant_directionality_radians'")
        return df_with_deltas
    except Exception as e:
        print(f"Error during delta calculations: {e}")
        print("This might be due to insufficient data or time window issues.")
        raise
