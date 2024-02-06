# load_trackmate.py
from initialization.initialization import *
from initialization.config import *
from data_processing.cleaning_formatting_filtering import *
from data_processing.migration_calculations import *
import re
import tqdm
import pandas as pd
import numpy as np

# Thanks to Guillaume Jacquemet for the following code structure. It does some trackmate loading followed by some cellPLATO formatting.

def populate_columns(df, filepath):
    # Extract the parts of the file path
    path_parts = os.path.normpath(filepath).split(os.sep)

    if len(path_parts) < 3:
        # if there are not enough parts in the path to extract folder and parent folder
        print(f"Error: Cannot extract parent folder and folder from the filepath: {filepath}")
        return df

    # Assuming that the file is located at least two levels deep in the directory structure
    folder_name = path_parts[-2]  # The folder name is the second last part of the path
    parent_folder_name = path_parts[-3]  # The parent folder name is the third last part of the path


    filename_without_extension = os.path.splitext(os.path.basename(filepath))[0]
    


    df['File_name'] = remove_suffix(filename_without_extension)
    df['Condition'] = parent_folder_name  # Populate 'Condition' with the parent folder name
    # df['experiment_nb'] = folder_name  # Populate 'Repeat' with the folder name
    df['Replicate_ID'] = parent_folder_name + folder_name  # Populate 'Repeat' with the folder name


    ###############
    return df


def load_and_populate(file_pattern, usecols=None, chunksize=100000, Folder_path = DATA_PATH, Results_Folder = SAVED_DATA_PATH):
    df_list = []
    pattern = re.compile(file_pattern)  # Compile the file pattern to a regex object
    files_to_process = []

    # First, list all the files we'll be processing
    for dirpath, dirnames, filenames in os.walk(Folder_path):
        # print(f"Dirpath is {dirpath}")
        # print(f"Dirnames is {dirnames}")
        # print(f"filenames is {filenames}")
        for filename in filenames:
            if pattern.match(filename):  # Check if the filename matches the file pattern
                filepath = os.path.join(dirpath, filename)
                files_to_process.append(filepath)

    # Metadata list
    metadata_list = []

    # Create a tqdm instance for progress tracking
    for filepath in tqdm.tqdm(files_to_process, desc="Processing Files"):
        # Get the expected number of rows in the file (subtracting header rows)
        expected_rows = sum(1 for row in open(filepath)) - 4

        # Get file size
        file_size = os.path.getsize(filepath)

        # Add to the metadata list
        metadata_list.append({
            'filename': os.path.basename(filepath),
            'expected_rows': expected_rows,
            'file_size': file_size
        })

        chunked_reader = pd.read_csv(filepath, skiprows=[1, 2, 3], usecols=usecols, chunksize=chunksize)

        for chunk in chunked_reader:
            processed_chunk = populate_columns(chunk, filepath)
            df_list.append(processed_chunk)

    if not df_list:  # if df_list is empty, return an empty DataFrame
        print(f"No files found with pattern: {file_pattern}")
        return pd.DataFrame()

    merged_df = pd.concat(df_list, ignore_index=True)
    # Verify the total rows in the merged dataframe matches the total expected rows from metadata
    total_expected_rows = sum(item['expected_rows'] for item in metadata_list)
    if len(merged_df) != total_expected_rows:
      print(f"Warning: Mismatch in total rows. Expected {total_expected_rows}, found {len(merged_df)} in the merged dataframe.")
    else:
      print(f"Success: The processed dataframe matches the metadata. Total rows: {len(merged_df)}")
    return merged_df, metadata_list



def sort_and_generate_repeat(merged_df):
    merged_df.sort_values(['Condition', 'experiment_nb'], inplace=True)
    merged_df = merged_df.groupby('Condition', group_keys=False).apply(generate_repeat)
    return merged_df

def generate_repeat(group):
    unique_experiment_nbs = sorted(group['experiment_nb'].unique())
    experiment_nb_to_repeat = {experiment_nb: i+1 for i, experiment_nb in enumerate(unique_experiment_nbs)}
    group['Repeat'] = group['experiment_nb'].map(experiment_nb_to_repeat)
    return group

def remove_suffix(filename):
    suffixes_to_remove = ["-tracks", "-spots"]
    for suffix in suffixes_to_remove:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            break
    return filename


def validate_tracks_df(df):
    """Validate the tracks dataframe for necessary columns and data types."""
    required_columns = ['TRACK_ID']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' missing in tracks dataframe.")
            return False

    # Additional data type checks or value ranges can be added here
    return True

def validate_spots_df(df):
    """Validate the spots dataframe for necessary columns and data types."""
    required_columns = ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'POSITION_T']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' missing in spots dataframe.")
            return False

    # Additional data type checks or value ranges can be added here
    return True

def check_unique_id_match(df1, df2):
    df1_ids = set(df1['Unique_ID'])
    df2_ids = set(df2['Unique_ID'])

    # Check if the IDs in the two dataframes match
    if df1_ids == df2_ids:
        print("The Unique_ID values in both dataframes match perfectly!")
    else:
        missing_in_df1 = df2_ids - df1_ids
        missing_in_df2 = df1_ids - df2_ids

        if missing_in_df1:
            print(f"There are {len(missing_in_df1)} Unique_ID values present in the second dataframe but missing in the first.")
            print("Examples of these IDs are:", list(missing_in_df1)[:5])

        if missing_in_df2:
            print(f"There are {len(missing_in_df2)} Unique_ID values present in the first dataframe but missing in the second.")
            print("Examples of these IDs are:", list(missing_in_df2)[:5])

#####

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    d = diff / np.sqrt(pooled_var)
    return d

def save_dataframe_with_progress(df, path, desc="Saving", chunk_size=50000):
    """Save a DataFrame with a progress bar."""

    # Estimating the number of chunks based on the provided chunk size
    num_chunks = int(len(df) / chunk_size) + 1

    # Create a tqdm instance for progress tracking
    with tqdm(total=len(df), unit="rows", desc=desc) as pbar:
        # Open the file for writing
        with open(path, "w") as f:
            # Write the header once at the beginning
            df.head(0).to_csv(f, index=False)

            for chunk in np.array_split(df, num_chunks):
                chunk.to_csv(f, mode="a", header=False, index=False)
                pbar.update(len(chunk))

def check_for_nans(df, df_name):
    """
    Checks the given DataFrame for NaN values and prints the count for each column containing NaNs.

    Args:
    df (pd.DataFrame): DataFrame to be checked for NaN values.
    df_name (str): The name of the DataFrame as a string, used for printing.
    """
    # Check if the DataFrame has any NaN values and print a warning if it does.
    nan_columns = df.columns[df.isna().any()].tolist()

    if nan_columns:
        for col in nan_columns:
            nan_count = df[col].isna().sum()
            print(f"Column '{col}' in {df_name} contains {nan_count} NaN values.")
    else:
        print(f"No NaN values found in {df_name}.")


def trackmate_to_cellPLATO(df):
        # This will become the function to make the comb_df

    input_df=df.copy()

    '''This part renames a lot of columns to match cellPLATO'''

    # rename LABEL to trackmate_label
    input_df = input_df.rename(columns={'LABEL':'trackmate_label'})
    # ID to particle
    input_df = input_df.rename(columns={'ID':'particle'})
    # change the data type of particle to float
    input_df['particle'] = input_df['particle'].astype(float)
    # rename POSITION_X to x, POSITION_Y to y, POSITION_Z to z, FRAME to t
    input_df = input_df.rename(columns={'POSITION_X':'x', 'POSITION_Y':'y', 'POSITION_Z':'z', 'FRAME':'frame'})

    # Convert the values in frame column to float
    input_df['frame'] = input_df['frame'].astype(float)

    '''This part makes the x_um, y_um, z_um columns just by replicating the existing ones'''
    # copy the x column to a new x_um column, and the y column to a new y_um column, and z to z_um
    input_df['x_um'] = input_df['x']
    input_df['y_um'] = input_df['y']
    input_df['z_um'] = input_df['z']
    # Same with the x_pix, y_pix, z_pix
    input_df['x_pix'] = input_df['x']
    input_df['y_pix'] = input_df['y']
    input_df['z_pix'] = input_df['z']

    '''This part makes the Rep_label column'''
    # Make a column of floats that corresponds to the 'Replicate_ID' column and call it 'Rep_label'
    # To do this, extract the 'Replicate_ID' columns from the merged_spots_df
    Replicate_ID = input_df['Replicate_ID']
    # Get the unique Replicate_IDs
    Replicate_ID_unique = np.unique(Replicate_ID)
    # Make a dictionary of the unique Replicate_IDs and a number (float) that corresponds to them
    Replicate_ID_dict = {}
    for i, ID in enumerate(Replicate_ID_unique):
        Replicate_ID_dict[ID] = i
    # Make a new column called 'Rep_label' and populate it with the float values from the dictionary
    input_df['Rep_label'] = input_df['Replicate_ID'].map(Replicate_ID_dict)
    # make those floats
    input_df['Rep_label'] = input_df['Rep_label'].astype(float)

    # Make a new column called 'Condition_shortlabel' which has the same value as 'Condition'
    input_df['Condition_shortlabel'] = input_df['Condition']

    ##########################

    # Then, add the ntpts and the uniq_id to the df

    apply_unique_id_trackmate(input_df)
    #sort by frame
    input_df = input_df.sort_values(by=['uniq_id', 'frame'])

    # display(input_df)

    # Then, do the cellPLATO migration calculations

    if DO_CP_METRICS_FOR_TRACKMATE:

        proto_comb_list = []
        proto_comb_df = pd.DataFrame()

    ########################################################

        for replicate in np.unique(input_df['Replicate_ID']):
            # extract the replicate
            replicate_df = input_df[input_df['Replicate_ID'] == replicate]
            # sort that df by uniq_id and frame
            replicate_df = replicate_df.sort_values(by=['uniq_id', 'frame'])


        #     print('For this replciate df, the replicated is ', replicate_df['Replicate_ID'].unique())
        #     print('And the rep_label is ', replicate_df['Rep_label'].unique())
        #     print('And the unique ID is ', replicate_df['uniq_id'].unique())
        #     print('And finally the file name is ', replicate_df['File_name'].unique())
        #     print('And the condition is ', replicate_df['Condition'].unique())

            # do the migration measurements
            mig_df = migration_calcs(replicate_df)

            mig_df.reset_index(inplace=True, drop=True)
        #     # add it to the proto_comb_df list 
            proto_comb_list.append(mig_df)

        proto_comb_df = pd.concat(proto_comb_list, ignore_index=True)

        # proto_comb_df =  pd.concat([proto_comb_df,mig_df])
        proto_comb_df.reset_index(inplace=True, drop=True)
    else:
        proto_comb_df = input_df



    #############
    return proto_comb_df
##### 

