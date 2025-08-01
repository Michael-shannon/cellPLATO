# time_calculations.py

from initialization.initialization import *
from initialization.config import *

from data_processing.clustering import cluster_purity

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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

# def time_average(df):

#     '''
#     Needs a more descriptive name?
#         average_across_time()?

#     Function to generate a time-averaged dataframe,
#     where the average value for each factor across all timepoints
#     is calculated for each cell.

#     Input:
#         df: DataFrame [N * T * X]


#     Returns:
#         avg_df: DataFrame [N * X]
#     '''

#     time_avg_df = pd.DataFrame()
#     unique_id = 0 # Create a unique cell id
#     rep_list = df['Replicate_ID'].unique()


#     for this_rep in rep_list:

#         rep_df = df[df['Replicate_ID']==this_rep]
#         print(f'Replicate: {this_rep}')
#         cell_ids = rep_df['particle'].unique() # Particle ids only unique for replicate, not between.
#         print(f'cell_ids: {cell_ids} ')

#         # For each cell, calculate the average value and add to new DataFrame
#         for cid in cell_ids:

#             cell_df = rep_df[rep_df['particle'] == cid]

#             # A test to ensure there is only one replicate label included.
#             assert len(cell_df['Rep_label'].unique()) == 1, 'check reps'

#             avg_df = cell_df.mean() # Returns a series that is the mean value for each numerical column. Non-numerical columns are dropped.

#             # Add back non-numeric data
#             dropped_cols = list(set(cell_df.columns) - set(avg_df.index))

#             for col in dropped_cols:

#                 assert len(cell_df[col].unique()) == 1, 'Invalid assumption: uniqueness of non-numerical column values'
#                 avg_df.loc[col] = cell_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)

#             avg_df.loc['unique_id'] = unique_id # Add Unique cell ID for the analysis
#             time_avg_df = time_avg_df.append(avg_df,ignore_index=True)
#             unique_id += 1

#     time_avg_df['frame'] = 'timeaverage' # Replace the meaningless average frame values with a string desciption

#     return time_avg_df


def time_average(df):
    """
    Function to generate a time-averaged dataframe,
    where the average value for each factor across all timepoints
    is calculated for each unique `uniq_id`.

    Input:
        df: DataFrame with a `uniq_id` column

    Returns:
        time_avg_df: DataFrame with averaged values for each `uniq_id`
    """
    
    time_avg_df = pd.DataFrame()
    unique_ids = df['uniq_id'].unique()

    for uid in unique_ids:
        cell_df = df[df['uniq_id'] == uid]
        
        # Calculate the mean value for each numerical column
        avg_df = cell_df.mean()  # Returns a series
        
        # Add back non-numeric data (assuming they are consistent across the unique_id)
        non_numeric_cols = list(set(cell_df.columns) - set(avg_df.index))
        for col in non_numeric_cols:
            # Check if the column is indeed non-numeric
            if cell_df[col].dtype == 'object' or cell_df[col].dtype == 'category':
                # Make sure there's only one unique value for this column in the filtered dataframe
                assert len(cell_df[col].unique()) == 1, f"Non-unique values found in column {col} for uniq_id {uid}"
                avg_df.loc[col] = cell_df[col].values[0]

        avg_df.loc['uniq_id'] = uid  # Add the unique_id back to the dataframe
        time_avg_df = pd.concat([time_avg_df,pd.DataFrame([avg_df])], ignore_index=True)

    time_avg_df['frame'] = 'timeaverage'  # Replace the meaningless average frame values with a string description

    return time_avg_df


def time_average_trackmate(df):

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

    cell_ids = df['uniq_id'].unique() # Just use unique ids

    # For each cell, calculate the average value and add to new DataFrame
    for cid in tqdm(cell_ids, desc="Time-averaging cells"):

        cell_df = df[df['uniq_id'] == cid]

        # A test to ensure there is only one replicate label included.
        assert len(cell_df['Rep_label'].unique()) == 1, 'check reps'

        avg_df = cell_df.mean(numeric_only=True) # Returns a series that is the mean value for each numerical column. Non-numerical columns are dropped.

        # Add back non-numeric data
        dropped_cols = list(set(cell_df.columns) - set(avg_df.index))

        for col in dropped_cols:
            avg_df.loc[col] = cell_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)

        avg_df.loc['unique_id'] = unique_id # Add Unique cell ID for the analysis
        time_avg_df = pd.concat([time_avg_df, pd.DataFrame([avg_df])], ignore_index=True)
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

                    avg_df = rep_df.mean(numeric_only=True) # Returns a series that is the mean value for each numerical column. Non-numerical columns are dropped.

                    # Add back non-numeric data
                    dropped_cols = list(set(frame_df.columns) - set(avg_df.index))

                    for col in dropped_cols:

                        # Validate assumption that sub_df has only one rep/condition, then use this value in new frame
                        assert len(rep_df[col].unique()) == 1, 'Invalid assumption: uniqueness of non-numerical column values'
                        avg_df.loc[col] = rep_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)

                    if t_window is None: # assertion only works when no window is used.
                        assert avg_df.loc['frame'] == frame, 'Frame mismatch'

                    tptavg_df = pd.concat([tptavg_df,pd.DataFrame(avg_df)],ignore_index=True)
                else:
                    if(DEBUG):
                        print('Skipping: ',rep, ' N = ', len(rep_df))

    return tptavg_df


def analyze_time_window_settings(df, current_window_minutes=None, sampling_interval=None, verbose=True):
    """
    Analyze your data to suggest optimal time window settings.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your trajectory data
    current_window_minutes : float, optional
        Current window size in minutes (if None, uses config)
    sampling_interval : float, optional  
        Time between frames in minutes (if None, uses config)
    verbose : bool
        Whether to print detailed analysis
        
    Returns:
    --------
    dict : Analysis results and recommendations
    """
    import numpy as np
    
    # Get config values if not provided
    if current_window_minutes is None:
        try:
            from initialization.config import MigrationTimeWindow_minutes
            current_window_minutes = MigrationTimeWindow_minutes
        except:
            current_window_minutes = 5.0  # fallback default
            
    if sampling_interval is None:
        try:
            from initialization.config import SAMPLING_INTERVAL  
            sampling_interval = SAMPLING_INTERVAL
        except:
            sampling_interval = 1.0  # fallback default
    
    current_window_frames = round(current_window_minutes / sampling_interval)
    
    if verbose:
        print("=== TIME WINDOW ANALYSIS ===\n")
        print(f"⚙️  Current Settings:")
        print(f"   Window: {current_window_minutes} minutes ({current_window_frames} frames)")
        print(f"   Sampling: {sampling_interval:.2f} minutes/frame")
    
    # Analyze track lengths
    if 'TRACK_ID' in df.columns:
        track_id_col = 'TRACK_ID'
    elif 'uniq_id' in df.columns:
        track_id_col = 'uniq_id'  
    elif 'particle' in df.columns:
        track_id_col = 'particle'
    else:
        if verbose:
            print("❌ No track ID column found!")
        return None
    
    # Calculate track lengths
    track_lengths = df.groupby(track_id_col).size()
    
    if verbose:
        print(f"\n📊 Track Length Analysis:")
        print(f"   Total tracks: {len(track_lengths)}")
        print(f"   Track length range: {track_lengths.min()} to {track_lengths.max()} frames")
        print(f"   Mean track length: {track_lengths.mean():.1f} frames")
        print(f"   Median track length: {track_lengths.median():.1f} frames")
    
    # Calculate how many frames would have valid migration data
    frames_lost_per_track = current_window_frames - 1  # lose (w-1) frames per track
    valid_frames_per_track = track_lengths - frames_lost_per_track
    valid_frames_per_track = valid_frames_per_track.clip(lower=0)
    
    total_frames = track_lengths.sum()
    total_valid_frames = valid_frames_per_track.sum()
    data_loss_pct = (1 - total_valid_frames / total_frames) * 100
    
    if verbose:
        print(f"\n⚠️  Current Window Impact:")
        print(f"   Data loss due to windowing: {data_loss_pct:.1f}%")
        print(f"   Total frames: {total_frames}")
        print(f"   Frames with migration data: {total_valid_frames}")
        
        # Show tracks that would be completely lost
        completely_lost = (track_lengths < current_window_frames).sum()
        if completely_lost > 0:
            print(f"   Tracks with NO migration data: {completely_lost} ({100*completely_lost/len(track_lengths):.1f}%)")
    
    # Generate recommendations
    recommendations = {}
    
    # Conservative recommendation: window ≤ 25% of median track length
    conservative_frames = max(3, int(track_lengths.median() * 0.25))
    conservative_minutes = conservative_frames * sampling_interval
    
    # Balanced recommendation: window ≤ 33% of median track length  
    balanced_frames = max(3, int(track_lengths.median() * 0.33))
    balanced_minutes = balanced_frames * sampling_interval
    
    # Aggressive recommendation: window ≤ 50% of median track length
    aggressive_frames = max(3, int(track_lengths.median() * 0.50))
    aggressive_minutes = aggressive_frames * sampling_interval
    
    recommendations = {
        'conservative': {
            'minutes': conservative_minutes,
            'frames': conservative_frames,
            'data_loss_pct': _calculate_data_loss(track_lengths, conservative_frames),
            'description': 'Minimal data loss, smaller temporal context'
        },
        'balanced': {
            'minutes': balanced_minutes, 
            'frames': balanced_frames,
            'data_loss_pct': _calculate_data_loss(track_lengths, balanced_frames),
            'description': 'Good balance of data retention and temporal context'
        },
        'aggressive': {
            'minutes': aggressive_minutes,
            'frames': aggressive_frames, 
            'data_loss_pct': _calculate_data_loss(track_lengths, aggressive_frames),
            'description': 'More temporal context, higher data loss'
        }
    }
    
    if verbose:
        print(f"\n💡 Recommendations:")
        for name, rec in recommendations.items():
            print(f"   {name.title()}: {rec['minutes']:.1f} min ({rec['frames']} frames)")
            print(f"     → {rec['data_loss_pct']:.1f}% data loss - {rec['description']}")
            
        # Find the best match to current setting
        current_loss = data_loss_pct
        if current_loss < 15:
            status = "✅ Good"
        elif current_loss < 30:  
            status = "⚠️  Moderate" 
        else:
            status = "❌ High"
            
        print(f"\n🎯 Current setting status: {status} ({current_loss:.1f}% data loss)")
        
        if current_loss > 20:
            best_rec = min(recommendations.items(), key=lambda x: abs(x[1]['data_loss_pct'] - 15))
            print(f"   💡 Consider: {best_rec[1]['minutes']:.1f} minutes for ~{best_rec[1]['data_loss_pct']:.1f}% data loss")
    
    return {
        'current': {
            'minutes': current_window_minutes,
            'frames': current_window_frames,
            'data_loss_pct': data_loss_pct
        },
        'track_stats': {
            'n_tracks': len(track_lengths),
            'mean_length': track_lengths.mean(),
            'median_length': track_lengths.median(), 
            'min_length': track_lengths.min(),
            'max_length': track_lengths.max()
        },
        'recommendations': recommendations
    }


def _calculate_data_loss(track_lengths, window_frames):
    """Helper function to calculate data loss percentage for a given window size."""
    frames_lost_per_track = window_frames - 1
    valid_frames_per_track = track_lengths - frames_lost_per_track
    valid_frames_per_track = valid_frames_per_track.clip(lower=0)
    
    total_frames = track_lengths.sum()
    total_valid_frames = valid_frames_per_track.sum()
    
    return (1 - total_valid_frames / total_frames) * 100 if total_frames > 0 else 0


def suggest_time_window(df, target_data_loss=15.0, min_frames=3, max_frames=None):
    """
    Suggest optimal time window size for a target data loss percentage.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your trajectory data  
    target_data_loss : float
        Target percentage of data loss (default: 15%)
    min_frames : int
        Minimum allowed window size (default: 3)
    max_frames : int, optional
        Maximum allowed window size
        
    Returns:
    --------
    dict : Suggested window size and analysis
    """
    
    # Get track lengths
    if 'TRACK_ID' in df.columns:
        track_id_col = 'TRACK_ID'
    elif 'uniq_id' in df.columns:
        track_id_col = 'uniq_id'
    elif 'particle' in df.columns:
        track_id_col = 'particle'
    else:
        return None
        
    track_lengths = df.groupby(track_id_col).size()
    
    if max_frames is None:
        max_frames = int(track_lengths.median())
    
    # Test different window sizes
    best_window = min_frames
    best_loss = _calculate_data_loss(track_lengths, min_frames)
    
    for window_size in range(min_frames, max_frames + 1):
        loss = _calculate_data_loss(track_lengths, window_size)
        if abs(loss - target_data_loss) < abs(best_loss - target_data_loss):
            best_window = window_size
            best_loss = loss
    
    # Convert to minutes
    try:
        from initialization.config import SAMPLING_INTERVAL
        sampling_interval = SAMPLING_INTERVAL
    except:
        sampling_interval = 1.0
        
    best_minutes = best_window * sampling_interval
    
    return {
        'suggested_frames': best_window,
        'suggested_minutes': best_minutes,
        'actual_data_loss': best_loss,
        'target_data_loss': target_data_loss
    }


def analyze_individual_tracks_for_nans(df, track_id_col='uniq_id', migration_factors=None, 
                                      n_tracks_to_analyze=10, verbose=True):
    """
    Analyze individual tracks to understand where NaN values come from in migration calculations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your trajectory data
    track_id_col : str
        Column name for track IDs (default: 'uniq_id')
    migration_factors : list, optional
        List of migration factors to check for NaNs
    n_tracks_to_analyze : int
        Number of individual tracks to analyze in detail
    verbose : bool
        Whether to print detailed analysis
        
    Returns:
    --------
    dict : Analysis results
    """
    import numpy as np
    import pandas as pd
    
    if migration_factors is None:
        migration_factors = [
            'euclidean_dist', 'speed', 'cumulative_length', 'MSD',
            'turn_angle', 'directedness', 'arrest_coefficient'
        ]
    
    # Filter to factors that exist
    available_factors = [f for f in migration_factors if f in df.columns]
    
    if verbose:
        print("=== INDIVIDUAL TRACK NaN ANALYSIS ===\n")
        print(f"Analyzing migration factors: {available_factors}")
    
    # Get track information
    track_info = []
    track_ids = df[track_id_col].unique()
    
    for track_id in track_ids:
        track_data = df[df[track_id_col] == track_id].sort_values('frame')
        
        # Basic track info
        track_length = len(track_data)
        frame_range = track_data['frame'].max() - track_data['frame'].min() + 1
        frames = track_data['frame'].values
        has_gaps = len(set(range(int(frames.min()), int(frames.max()) + 1))) != len(frames)
        
        # Check for NaNs in migration factors
        nan_counts = {}
        for factor in available_factors:
            if factor in track_data.columns:
                nan_counts[factor] = track_data[factor].isna().sum()
        
        track_info.append({
            'track_id': track_id,
            'length': track_length,
            'frame_range': frame_range,
            'has_gaps': has_gaps,
            'frame_min': frames.min(),
            'frame_max': frames.max(),
            'frames': frames,
            'nan_counts': nan_counts,
            'total_nans': sum(nan_counts.values()) if nan_counts else 0
        })
    
    # Sort by total NaN count to find most problematic tracks
    track_info.sort(key=lambda x: x['total_nans'], reverse=True)
    
    if verbose:
        print(f"📊 Track Overview:")
        print(f"   Total tracks: {len(track_info)}")
        
        # Summary statistics
        lengths = [t['length'] for t in track_info]
        gaps = sum(1 for t in track_info if t['has_gaps'])
        
        print(f"   Track lengths: {min(lengths)} to {max(lengths)} frames")
        print(f"   Tracks with frame gaps: {gaps} ({100*gaps/len(track_info):.1f}%)")
        
        # NaN summary
        tracks_with_nans = sum(1 for t in track_info if t['total_nans'] > 0)
        print(f"   Tracks with migration NaNs: {tracks_with_nans} ({100*tracks_with_nans/len(track_info):.1f}%)")
    
    # Detailed analysis of specific tracks
    if verbose and n_tracks_to_analyze > 0:
        print(f"\n🔍 Detailed Analysis of {min(n_tracks_to_analyze, len(track_info))} Tracks:")
        
        for i, track in enumerate(track_info[:n_tracks_to_analyze]):
            print(f"\n   Track {track['track_id']}:")
            print(f"     Length: {track['length']} frames")
            print(f"     Frame range: {track['frame_min']:.0f} to {track['frame_max']:.0f}")
            print(f"     Has gaps: {track['has_gaps']}")
            
            # Show frame sequence for short tracks or with gaps
            if track['length'] <= 20 or track['has_gaps']:
                print(f"     Frame sequence: {track['frames'][:15].astype(int).tolist()}{'...' if len(track['frames']) > 15 else ''}")
            
            # Show NaN counts by factor
            if track['nan_counts']:
                print(f"     NaN counts by factor:")
                for factor, count in track['nan_counts'].items():
                    if count > 0:
                        pct = 100 * count / track['length']
                        print(f"       {factor}: {count}/{track['length']} ({pct:.1f}%)")
                        
                # Analyze WHERE the NaNs occur in this track
                track_data = df[df[track_id_col] == track['track_id']].sort_values('frame')
                example_factor = next((f for f, c in track['nan_counts'].items() if c > 0), None)
                
                if example_factor:
                    nan_positions = track_data[example_factor].isna()
                    nan_frame_indices = np.where(nan_positions)[0]
                    
                    if len(nan_frame_indices) > 0:
                        print(f"     NaN positions in track (first few):")
                        for idx in nan_frame_indices[:5]:
                            frame_num = track_data.iloc[idx]['frame']
                            relative_pos = idx / len(track_data)
                            position_desc = "start" if relative_pos < 0.33 else "end" if relative_pos > 0.67 else "middle"
                            print(f"       Frame {frame_num:.0f} (position {idx}/{len(track_data)}, {position_desc})")
    
    # Try to understand the windowing issue
    if verbose:
        print(f"\n🔧 Time Window Analysis:")
        
        try:
            from cellPLATO.initialization.config import MIG_T_WIND
            window_size = MIG_T_WIND
            half_window = window_size // 2
            
            print(f"   Window size: {window_size} frames (±{half_window})")
            
            # For a few tracks, simulate the windowing logic
            for track in track_info[:3]:
                track_data = df[df[track_id_col] == track['track_id']].sort_values('frame')
                frames = track_data['frame'].values
                
                print(f"\n   Track {track['track_id']} windowing simulation:")
                
                valid_windows = 0
                total_windows = 0
                
                for i, current_frame in enumerate(frames):
                    # Simulate the windowing logic from migration_calculations.py
                    window_start = current_frame - half_window
                    window_end = current_frame + half_window
                    
                    # Count frames in this window
                    frames_in_window = frames[(frames >= window_start) & (frames < window_end + 1)]
                    window_actual_size = len(frames_in_window)
                    
                    total_windows += 1
                    if window_actual_size == window_size:
                        valid_windows += 1
                
                valid_pct = 100 * valid_windows / total_windows if total_windows > 0 else 0
                print(f"     Valid windows: {valid_windows}/{total_windows} ({valid_pct:.1f}%)")
                
                if valid_pct < 50:
                    print(f"     ⚠️  Most windows are incomplete!")
                    
        except ImportError:
            print("   Could not import window size from config")
    
    return {
        'track_summary': {
            'total_tracks': len(track_info),
            'tracks_with_gaps': sum(1 for t in track_info if t['has_gaps']),
            'tracks_with_nans': sum(1 for t in track_info if t['total_nans'] > 0),
            'mean_track_length': np.mean([t['length'] for t in track_info]),
            'median_track_length': np.median([t['length'] for t in track_info])
        },
        'track_details': track_info[:n_tracks_to_analyze] if n_tracks_to_analyze > 0 else [],
        'problematic_tracks': [t for t in track_info if t['total_nans'] > 0 or t['has_gaps']]
    }


def check_migration_calculation_requirements(df, track_id_col='uniq_id', verbose=True):
    """
    Check if data meets requirements for migration calculations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your trajectory data
    track_id_col : str
        Column name for track IDs
    verbose : bool
        Whether to print detailed analysis
        
    Returns:
    --------
    dict : Requirements check results
    """
    import numpy as np
    
    if verbose:
        print("=== MIGRATION CALCULATION REQUIREMENTS CHECK ===\n")
    
    issues = []
    track_ids = df[track_id_col].unique()
    
    # Check each track
    problematic_tracks = []
    
    for track_id in track_ids:
        track_data = df[df[track_id_col] == track_id].sort_values('frame')
        
        # Check for required columns
        required_cols = ['frame', 'x_um', 'y_um']
        missing_cols = [col for col in required_cols if col not in track_data.columns]
        
        if missing_cols:
            problematic_tracks.append({
                'track_id': track_id,
                'issue': f'Missing columns: {missing_cols}'
            })
            continue
        
        # Check for frame gaps
        frames = track_data['frame'].values
        expected_frames = set(range(int(frames.min()), int(frames.max()) + 1))
        actual_frames = set(frames.astype(int))
        missing_frames = expected_frames - actual_frames
        
        if missing_frames:
            problematic_tracks.append({
                'track_id': track_id,
                'issue': f'Frame gaps: missing frames {sorted(list(missing_frames))[:5]}...'
            })
        
        # Check for duplicate frames
        if len(frames) != len(np.unique(frames)):
            duplicate_frames = [frame for frame in np.unique(frames) if (frames == frame).sum() > 1]
            problematic_tracks.append({
                'track_id': track_id,
                'issue': f'Duplicate frames: {duplicate_frames[:5]}...'
            })
        
        # Check for NaN coordinates
        if track_data[['x_um', 'y_um']].isna().any().any():
            problematic_tracks.append({
                'track_id': track_id,
                'issue': 'NaN coordinates'
            })
    
    if verbose:
        print(f"📊 Requirements Check Results:")
        print(f"   Total tracks checked: {len(track_ids)}")
        print(f"   Problematic tracks: {len(problematic_tracks)}")
        
        if problematic_tracks:
            print(f"\n⚠️  Issues Found:")
            issue_types = {}
            for track in problematic_tracks:
                issue_type = track['issue'].split(':')[0]
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            for issue_type, count in issue_types.items():
                print(f"   {issue_type}: {count} tracks")
            
            if verbose and len(problematic_tracks) <= 10:
                print(f"\n🔍 Detailed Issues:")
                for track in problematic_tracks[:10]:
                    print(f"   Track {track['track_id']}: {track['issue']}")
        else:
            print("   ✅ No major issues found!")
    
    return {
        'total_tracks': len(track_ids),
        'problematic_tracks': len(problematic_tracks),
        'issues': problematic_tracks
    }


def fill_frame_gaps_in_tracks(df, track_id_col='uniq_id', verbose=True):
    """
    Fill missing frames in tracks by interpolation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your trajectory data
    track_id_col : str
        Column name for track IDs
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    pandas.DataFrame : Data with filled frame gaps
    """
    import pandas as pd
    import numpy as np
    
    if verbose:
        print("🔧 Filling frame gaps by interpolation...")
    
    filled_tracks = []
    tracks_with_gaps = 0
    total_frames_added = 0
    
    for track_id in df[track_id_col].unique():
        track_data = df[df[track_id_col] == track_id].sort_values('frame')
        
        # Check if track has gaps
        frames = track_data['frame'].values
        min_frame = int(frames.min())
        max_frame = int(frames.max())
        expected_frames = set(range(min_frame, max_frame + 1))
        actual_frames = set(frames.astype(int))
        missing_frames = expected_frames - actual_frames
        
        if missing_frames:
            tracks_with_gaps += 1
            total_frames_added += len(missing_frames)
            
            # Create complete frame range
            complete_frames = range(min_frame, max_frame + 1)
            
            # Reindex to fill gaps
            track_data = track_data.set_index('frame')
            track_data = track_data.reindex(complete_frames)
            
            # Interpolate missing values for numeric columns
            numeric_cols = track_data.select_dtypes(include=[np.number]).columns
            track_data[numeric_cols] = track_data[numeric_cols].interpolate(method='linear')
            
            # Forward fill non-numeric columns
            non_numeric_cols = track_data.select_dtypes(exclude=[np.number]).columns
            track_data[non_numeric_cols] = track_data[non_numeric_cols].fillna(method='ffill')
            
            # Reset index and restore track_id
            track_data = track_data.reset_index()
            track_data[track_id_col] = track_id
        
        filled_tracks.append(track_data)
    
    result_df = pd.concat(filled_tracks, ignore_index=True)
    
    if verbose:
        print(f"   ✅ Filled gaps in {tracks_with_gaps} tracks")
        print(f"   ✅ Added {total_frames_added} interpolated frames")
        print(f"   ✅ Result: {len(df)} → {len(result_df)} frames")
    
    return result_df


def split_tracks_at_gaps(df, track_id_col='uniq_id', min_segment_length=8, verbose=True):
    """
    Split tracks at frame gaps to create continuous segments.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your trajectory data
    track_id_col : str
        Column name for track IDs
    min_segment_length : int
        Minimum length for track segments (shorter ones are discarded)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    pandas.DataFrame : Data with tracks split at gaps
    """
    import pandas as pd
    import numpy as np
    
    if verbose:
        print("✂️  Splitting tracks at frame gaps...")
    
    continuous_tracks = []
    new_track_id = 0
    original_tracks = 0
    final_tracks = 0
    discarded_segments = 0
    
    for track_id in df[track_id_col].unique():
        original_tracks += 1
        track_data = df[df[track_id_col] == track_id].sort_values('frame')
        frames = track_data['frame'].values
        
        # Find gap positions (where frame difference > 1)
        gaps = np.where(np.diff(frames) > 1)[0]
        
        if len(gaps) == 0:
            # No gaps, keep original track if long enough
            if len(track_data) >= min_segment_length:
                continuous_tracks.append(track_data)
                final_tracks += 1
            else:
                discarded_segments += 1
        else:
            # Split at gaps
            start_idx = 0
            for gap_idx in gaps:
                # Create segment from start_idx to gap_idx
                segment = track_data.iloc[start_idx:gap_idx+1].copy()
                if len(segment) >= min_segment_length:
                    segment[track_id_col] = f"{track_id}_seg{new_track_id}"
                    continuous_tracks.append(segment)
                    final_tracks += 1
                else:
                    discarded_segments += 1
                new_track_id += 1
                start_idx = gap_idx + 1
            
            # Add final segment
            if start_idx < len(track_data):
                segment = track_data.iloc[start_idx:].copy()
                if len(segment) >= min_segment_length:
                    segment[track_id_col] = f"{track_id}_seg{new_track_id}"
                    continuous_tracks.append(segment)
                    final_tracks += 1
                else:
                    discarded_segments += 1
                new_track_id += 1
    
    result_df = pd.concat(continuous_tracks, ignore_index=True) if continuous_tracks else pd.DataFrame()
    
    if verbose:
        print(f"   ✅ Original tracks: {original_tracks}")
        print(f"   ✅ Final tracks: {final_tracks}")
        print(f"   ✅ Discarded short segments: {discarded_segments}")
        print(f"   ✅ Result: {len(df)} → {len(result_df)} frames")
    
    return result_df


def fix_track_gaps(df, method='fill', track_id_col='uniq_id', min_segment_length=8, 
                   auto_detect=True, verbose=True):
    """
    Fix frame gaps in tracks using either filling or splitting approach.
    
    ⚠️  IMPORTANT: Always use 'uniq_id' as the track identifier column in cellPLATO.
    Other columns (like 'particle', 'TRACK_ID') may not correctly define unique tracks.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your trajectory data
    method : str
        Approach to use: 'fill', 'split', or 'auto'
        - 'fill': Interpolate missing frames
        - 'split': Split tracks at gaps into continuous segments  
        - 'auto': Automatically choose best method based on data
    track_id_col : str
        Column name for track IDs (default: 'uniq_id' - DO NOT CHANGE unless you know what you're doing)
    min_segment_length : int
        For 'split' method: minimum length for track segments
    auto_detect : bool
        Whether to first analyze and suggest best method
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    pandas.DataFrame : Fixed trajectory data
    """
    import pandas as pd
    import numpy as np
    
    # Validate track ID column
    if track_id_col != 'uniq_id':
        if verbose:
            print(f"⚠️  WARNING: Using '{track_id_col}' instead of 'uniq_id' for track identification!")
            print(f"⚠️  This may cause incorrect track handling. Use 'uniq_id' unless you're certain.")
    
    if track_id_col not in df.columns:
        raise ValueError(f"Track ID column '{track_id_col}' not found in dataframe. "
                        f"Available columns: {list(df.columns)}")
    
    if verbose:
        print("=== TRACK GAP FIXING ===")
        print(f"🔍 Using '{track_id_col}' column to identify tracks")
        print()
    
    # First, analyze the gap situation
    track_info = analyze_individual_tracks_for_nans(df, track_id_col=track_id_col, 
                                                   n_tracks_to_analyze=0, verbose=False)
    
    tracks_with_gaps = track_info['track_summary']['tracks_with_gaps']
    total_tracks = track_info['track_summary']['total_tracks']
    gap_percentage = 100 * tracks_with_gaps / total_tracks
    
    if verbose:
        print(f"📊 Gap Analysis:")
        print(f"   Tracks with gaps: {tracks_with_gaps}/{total_tracks} ({gap_percentage:.1f}%)")
    
    # Auto-select method if requested
    if method == 'auto':
        if gap_percentage < 20:
            method = 'fill'
            if verbose:
                print(f"   🤖 Auto-selected: 'fill' (few gaps)")
        elif gap_percentage > 60:
            method = 'split'  
            if verbose:
                print(f"   🤖 Auto-selected: 'split' (many gaps)")
        else:
            # Check average gap size
            avg_track_length = track_info['track_summary']['mean_track_length']
            if avg_track_length > 30:
                method = 'fill'
                if verbose:
                    print(f"   🤖 Auto-selected: 'fill' (long tracks)")
            else:
                method = 'split'
                if verbose:
                    print(f"   🤖 Auto-selected: 'split' (short tracks)")
    
    if verbose:
        print(f"\n🔧 Applying method: '{method}'")
        
        if method == 'fill':
            print("   📝 This will interpolate missing frames to create continuous tracks")
            print("   ✅ Preserves original track structure")
            print("   ⚠️  May introduce interpolation artifacts")
        elif method == 'split':
            print("   ✂️  This will split tracks at gaps into continuous segments")
            print("   ✅ No interpolation artifacts")
            print("   ⚠️  May create many short track segments")
    
    # Apply the chosen method
    if method == 'fill':
        result_df = fill_frame_gaps_in_tracks(df, track_id_col=track_id_col, verbose=verbose)
    elif method == 'split':
        result_df = split_tracks_at_gaps(df, track_id_col=track_id_col, 
                                       min_segment_length=min_segment_length, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'fill', 'split', or 'auto'")
    
    # Verify the fix
    if verbose and len(result_df) > 0:
        print(f"\n✅ Verification:")
        
        # Check a sample track
        sample_track = result_df[result_df[track_id_col] == result_df[track_id_col].iloc[0]].sort_values('frame')
        frames = sample_track['frame'].values
        has_gaps = len(set(range(int(frames.min()), int(frames.max()) + 1))) != len(frames)
        
        print(f"   Sample track has gaps: {has_gaps}")
        print(f"   Sample frame sequence: {frames[:10].astype(int).tolist()}{'...' if len(frames) > 10 else ''}")
        
        # Quick migration calculation test
        try:
            print(f"\n🧪 Quick test: Running migration calculations...")
            from data_processing.migration_calculations import migration_calcs
            
            # Test on a small subset
            test_df = result_df.head(1000).copy()
            test_result = migration_calcs(test_df)
            
            if 'euclidean_dist' in test_result.columns:
                nan_rate = test_result['euclidean_dist'].isna().sum() / len(test_result) * 100
                print(f"   Test NaN rate: {nan_rate:.1f}% (was ~35%)")
                
                if nan_rate < 15:
                    print("   ✅ Significant improvement!")
                elif nan_rate < 25:
                    print("   ⚠️  Some improvement, but gaps remain")
                else:
                    print("   ❌ Limited improvement - consider different approach")
            
        except Exception as e:
            print(f"   ⚠️  Could not run test migration calculation: {e}")
    
    if verbose:
        print(f"\n🎯 Result: {len(result_df)} frames ready for analysis")
        print("=" * 40)
    
    return result_df


def fix_gaps_and_run_pipeline(df, gap_method='auto', track_id_col='uniq_id', 
                              mixed=None, factors_to_timeaverage=None, verbose=True):
    """
    Convenience function to fix track gaps and run the full measurement pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your trajectory data
    gap_method : str
        Method to fix gaps: 'fill', 'split', or 'auto'
    track_id_col : str
        Column name for track IDs
    mixed : bool, optional
        Mixed scaling parameter for measurement pipeline
    factors_to_timeaverage : list, optional
        Factors for time averaging
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    tuple : (processed_df, new_factors)
    """
    if verbose:
        print("🚀 COMPLETE GAP FIX + MEASUREMENT PIPELINE\n")
    
    # Import config defaults if not provided
    if mixed is None:
        try:
            from initialization.config import MIXED_SCALING
            mixed = MIXED_SCALING
        except:
            mixed = False
            
    if factors_to_timeaverage is None:
        try:
            from initialization.config import ALL_FACTORS
            factors_to_timeaverage = ALL_FACTORS
        except:
            factors_to_timeaverage = []
    
    # Step 1: Fix gaps
    print("Step 1: Fixing track gaps...")
    df_fixed = fix_track_gaps(df, method=gap_method, track_id_col=track_id_col, verbose=verbose)
    
    if len(df_fixed) == 0:
        print("❌ No data remaining after gap fixing!")
        return df_fixed, []
    
    # Step 2: Run measurement pipeline
    print("\nStep 2: Running measurement pipeline...")
    try:
        from data_processing.pipelines import measurement_pipeline
        df_processed, new_factors = measurement_pipeline(df_fixed, mixed=mixed, 
                                                        factors_to_timeaverage=factors_to_timeaverage)
        
        if verbose:
            # Check final NaN rates
            migration_factors = ['euclidean_dist', 'speed', 'directedness', 'arrest_coefficient']
            available_factors = [f for f in migration_factors if f in df_processed.columns]
            
            if available_factors:
                print(f"\n📊 Final NaN Rates:")
                for factor in available_factors:
                    nan_rate = df_processed[factor].isna().sum() / len(df_processed) * 100
                    status = "✅" if nan_rate < 15 else "⚠️" if nan_rate < 30 else "❌"
                    print(f"   {status} {factor}: {nan_rate:.1f}%")
        
        return df_processed, new_factors
        
    except Exception as e:
        print(f"❌ Error in measurement pipeline: {e}")
        return df_fixed, []
