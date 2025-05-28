#statistics.py

from initialization.initialization import *
from initialization.config import *

import os
import numpy as np
import pandas as pd

import scipy.stats as st
import scipy.stats as stats


def average_per_condition(df, avg_per_rep=False):

    '''

    Function to calculate average value for each metric in a dataframe, taking a time-averagede dataframe as input

    Input:
        df: time-averaged DataFrame [N * X]

    Returns:
        avg_df: DataFrame [N]
    '''


    assert df['frame'].unique()[0] == 'timeaverage', 'This function is intended for a time-averaged dataset.'

    avg_df = pd.DataFrame()
    std_df = pd.DataFrame()
    n_df = pd.DataFrame()
    cond_list = df['Condition'].unique()

    # Find the average value for each of the numerical columns

    for cond in cond_list:

        this_cond_df = df[df['Condition'] == cond]
        cond_avg_df = this_cond_df.mean()#skipna=True)
        cond_std_df = this_cond_df.std()#skipna=True)
        cond_n_df = this_cond_df.count()#skipna=True)

        # Additional nested level of processing if we want to calculate the average per replicate.
        if(avg_per_rep):

            rep_list = this_cond_df['Replicate_ID'].unique()

            for this_rep in rep_list:


                this_rep_df = this_cond_df[this_cond_df['Replicate_ID'] == this_rep]
                rep_avg_df = this_rep_df.mean()#skipna=True)
                rep_std_df = this_rep_df.std()
                rep_n_df = this_rep_df.count()

                 # Add back non-numeric data
                dropped_cols = list(set(this_rep_df.columns) - set(rep_avg_df.index))

                for col in dropped_cols:

                    assert len(this_rep_df[col].unique()) == 1, 'Invalid assumption: uniqueness of non-numerical column values'
                    rep_avg_df.loc[col] = this_rep_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)
                    rep_std_df.loc[col] = this_rep_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)
                    rep_n_df.loc[col] = this_rep_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)


                avg_df = pd.concat([avg_df,pd.DataFrame([rep_avg_df])],ignore_index=True)
                std_df = pd.concat([std_df,pd.DataFrame([rep_std_df])],ignore_index=True)
                n_df = pd.concat([n_df,pd.DataFrame([rep_n_df])],ignore_index=True)


        else:


             # Add back non-numeric data
            dropped_cols = list(set(this_cond_df.columns) - set(cond_avg_df.index))

            for col in dropped_cols:

                # Since we are averaging without considering replicates, we expect the list of Replicates_IDs to not be unique.
                if col != 'Replicate_ID' and col != 'Replicate_shortlabel' and col != 'File_name':
                    assert len(this_cond_df[col].unique()) == 1, f'Invalid assumption: uniqueness of non-numerical column values'
                    cond_avg_df.loc[col] = this_cond_df[col].values[0] # Get the non-numerical value from dataframe (assuming all equivalent)
                    cond_std_df.loc[col] = this_cond_df[col].values[0]
                    cond_n_df.loc[col] = this_cond_df[col].values[0]
                else:
                    cond_avg_df.loc[col] = 'NA' # Get the non-numerical value from dataframe (assuming all equivalent)
                    cond_std_df.loc[col] = 'NA'
                    cond_n_df.loc[col] = 'NA'

            avg_df = pd.concat([avg_df,pd.DataFrame([cond_avg_df])],ignore_index=True)
            std_df = pd.concat([std_df,pd.DataFrame([cond_std_df])],ignore_index=True)
            n_df = pd.concat([n_df,pd.DataFrame([cond_n_df])],ignore_index=True)


    avg_std_n = (avg_df, std_df, n_df)

    return avg_std_n


def generalized_stats(set1, set2, test=STAT_TEST):

    '''
    Function should work for any test between two datasets, so long as it returns two arguments
    the second of which is the P value.

    '''

    t, P = eval(test+'(set1, set2)')
#     print(t,P)

    return P

def stats_table(df, factor, grouping='Condition', test=STAT_TEST):

    '''
    Create a matrix of P-values for an exhaustive comparison of groupings.

    Inputs:
        df: pd.DataFrame
        factor: string, column in df.
        grouping: default: Condition, alternatively used with label
        test: Statistical test to use. Defaut STAT_TEST

    Returns:
        stat_table: pd.DataFrame
    '''

    print('Returning stats_table using test: ', test, ' for factor: ', factor)
    print('Note: for exploratory purposes only, no multiple comparison correction is being applied.')

    # Create a numpy array to hold the values, fill with NaNs
    n_cond = len((df['Condition'].unique()))
    stat_mat = np.empty([n_cond, n_cond])
    stat_mat[:] = np.NaN

    # Fill the table with the statistic of choice.
    for i, cond_i in enumerate(df['Condition'].unique()):
        for j, cond_j in enumerate(df['Condition'].unique()):

            if cond_i == cond_j:
                stat_mat[i,j] = np.NaN
            else:

                set1 = df[factor][df[grouping] == cond_i]
                set2 = df[factor][df[grouping] == cond_j]

                P = generalized_stats(set1, set2, test)

                stat_mat[i,j] = P

    # Turn the filled numpy array into a dataframe
    stat_table = pd.DataFrame(data=stat_mat,
                             index=df['Condition'].unique(),
                             columns=df['Condition'].unique())

    stat_table.to_csv(DATA_OUTPUT+factor+'_P_table.csv')

    return stat_table


# Bootstrapping function
def bootstrap_sample(df, n_samples=1000):

    measurements = df.values
    medians = []

    for i in range(n_samples):

        samples = np.random.choice(measurements, size = len(measurements))
        medians.append(np.median(samples))

    medians = np.asarray(medians)

    return medians




def bootstrap_sample_df(df,factor,ctl_label):

    '''
    Generate bootstrapped sample and return as dataframe, to be plotted with seaborn
    '''

    # Calculate the differences for each category and save them into dataframes for visualizing in Seaborn or Matplotlib
    bootstrap_diff_df = pd.DataFrame()

    # Get the control bootstrap
    ctl_bootstrap = bootstrap_sample(df[factor][df['Condition'] == ctl_label])

    for i in range(0,len(pd.unique(df['Condition']))):

        # Use the ctl_bootstrap if we're now on that condition, otherwise will create a new bootstrap sample that won't be the same.
        if(pd.unique(df['Condition'])[i] == ctl_label):
            bootstrap = ctl_bootstrap
        else:
            bootstrap = bootstrap_sample(df[factor][df['Condition'] == pd.unique(df['Condition'])[i]])

        difference = bootstrap - ctl_bootstrap
        this_cond =  pd.unique(df['Condition'])[i]
        this_diff_df = pd.DataFrame(data={'Difference':difference, 'Condition':this_cond})
        bootstrap_diff_df = pd.concat([bootstrap_diff_df, this_diff_df], ignore_index=True)

        
    # Calculate and print mean effect size for each condition
    mean_effect_sizes = bootstrap_diff_df.groupby('Condition')['Difference'].mean()
    print("Mean Effect Size for Each Condition Compared to Control:")
    for condition, mean_effect_size in mean_effect_sizes.items():
        print(f"The effect size for {condition} compared with control is: {mean_effect_size}")

    return bootstrap_diff_df

# Function to calculate median and mean for each condition per factor and save results to CSV
def calculate_median_mean_and_save(df, factors):
    for factor_name in factors:
        result_df = df.groupby('Condition_shortlabel')[factor_name].agg(['median', 'mean']).reset_index()
        output_file = f'{DATA_OUTPUT}/{factor_name}_median_mean_results.csv'
        result_df.to_csv(output_file, index=False)

# Function to perform statistical testing between two conditions for each factor and save results to CSV
def perform_statistical_testing_and_save(df, factors): # , output_folder,
    for factor_name in factors:
        conditions = df['Condition_shortlabel'].unique()
        condition1, condition2 = conditions[:2]  # Assuming only two conditions for simplicity
        
        data1 = df[df['Condition_shortlabel'] == condition1][factor_name]
        data2 = df[df['Condition_shortlabel'] == condition2][factor_name]
        
        # Perform Mann-Whitney U test for non-normal data
        stat_mw, p_value_mw = stats.mannwhitneyu(data1, data2)
        
        # Perform t-test for normal data (assuming normality for simplicity)
        stat_t, p_value_t = stats.ttest_ind(data1, data2)
        
        result_df = pd.DataFrame({
            'Factor': [factor_name],
            'Condition1': [condition1],
            'Condition2': [condition2],
            'Mann-Whitney U Statistic': [stat_mw],
            'Mann-Whitney U P-Value': [p_value_mw],
            't-test Statistic': [stat_t],
            't-test P-Value': [p_value_t]
        })
        
        output_file_mw = f'{DATA_OUTPUT}/{factor_name}_mannwhitneyu_results.csv'
        output_file_t = f'{DATA_OUTPUT}/{factor_name}_ttest_results.csv'
        # DATA_OUTPUT
        result_df.to_csv(output_file_mw, index=False)
        result_df.to_csv(output_file_t, index=False)
