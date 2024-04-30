from initialization.config import *
from initialization.initialization import *

from data_processing.cleaning_formatting_filtering import *
from data_processing.shape_calculations import *
from data_processing.statistics import *
from data_processing.time_calculations import *

from visualization.trajectory_visualization import *

import numpy as np
import pandas as pd
import os

import imageio
from IPython.display import Image

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def time_plot(df, factor='mean_intensity', savepath=PLOT_OUTPUT):

    '''
    Convert a dataframe with a specified parameter into a plot that shows
    that parameter changing over the course of the experiment.

    Input:
        df: DataFrame
        factor: String: column title for the factor to be visualized.

    Returns:
        fig_data: Plotly graph data
    '''


    n_frames = int(np.max(df['frame']))

    # Create a numpy array to contain the factor of interest, and the variance + N
    fac_array = np.empty([n_frames, 4])

    for t in range(0,n_frames):

        # sub_df method to extract frame of interest.
        sub_df = df.loc[df['frame'] == t]


        # Calculate mean, stdev and N
        sample_mean = np.nanmean(sub_df[factor])
        sample_std = np.nanstd(sub_df[factor])
        sample_n = len(sub_df[factor])# Count the non Nan values.

        '''
        ***
        Note: using length of the series works because NaN values have already been removed.
        Will have to see how this works when applied directly to a dataframe, instead of dr_df as previously...
        '''
        # Error bands displaying estimated standard error of the mean.
        fac_array[t,:] = [sample_mean,sample_mean+sample_std/np.sqrt(sample_n),sample_mean-sample_std/np.sqrt(sample_n),sample_n]


    # After filling the fac_array, plot it.
    t_vect = np.linspace(0,n_frames)

    # The plots with error bands are actually just 3 scatterplots
    fig_data=[
        go.Scatter(
            name=factor,
            x=t_vect,
            y=fac_array[:,0],
            mode='lines',
            line=dict(color='rgb(0,0,0)')

        ),
        go.Scatter(
            name=factor,
            x=t_vect,
            y=fac_array[:,1],
            mode='lines',
            marker=dict(color='#444'),
            showlegend=False
        ),
        go.Scatter(
            name=factor,
            x=t_vect,
            y=fac_array[:,2],
            mode='lines',
            marker=dict(color='#444'),
            showlegend=False,
            fillcolor='rgba(68,68,68,0.3)',
            fill='tonexty'
        )
    ]

    # It can also receive the info as a dict.
    fig_layout={
        'xaxis_title': 'Frame',
        'yaxis_title': factor,
        'title': factor
    }

#     fig = go.Figure(fig_data)

#     if STATIC_PLOTS:
#         fig.write_image(savepath+'timeplot_'+str(factor)+'.png')

#     if PLOTS_IN_BROWSER:
#         fig.show()

    return fig_data, fig_layout

def time_subplots(df, factors, savepath=PLOT_OUTPUT):

    '''
    Create a plotly subplot figure containing cellular measurements as as
    a function of time.

    Input:
        df: DataFrame
        factors: list of strings, column names t visualize for the dataframe
    '''

    fig_data_list = []
    # fig_layout_list = []

    for factor in factors:

        cur_fig_data, cur_fig_layout = time_plot(df, factor)
        fig_data_list.append(cur_fig_data)
        # fig_layout_list.append(cur_fig_layout)


    fig_subplots = make_subplots(
        rows=len(fig_data_list), cols=1,shared_xaxes=True,
        # You can define the subplot titles here:
        subplot_titles=(factors)

    )

    for i, subfig in enumerate(fig_data_list):

        # Because subfig here is actually a list of scatterplots (to draw the error bands)
        # We must loop through the entire number of entries at this item of the list,
        # Adding them to the same subplot.

        if(len(subfig) > 1):
            for sub_subfig in subfig:
                fig_subplots.add_trace(sub_subfig,row=i+1, col=1)
        else:

            fig_subplots.add_trace(subfig,row=i+1, col=1)#i

        # Update yaxes for each row individually
        fig_subplots.update_yaxes(title_text=factors[i],
                              row=i+1, # Makes sure the axis label on the botto
                              col=1)



    # Update properties for the shared x axis
    fig_subplots.update_xaxes(title_text="Frame # [Uncalibrated]",
                              row=len(fig_data_list), # Makes sure the axis label on the botto
                              col=1)

    # Update properties for the entire subplot figure
    var_height = 100 + len(factors) * 120 # Calculate figure height by # of plots
    fig_subplots.update_layout(showlegend=False, title_text='Timecourse analysis',
                                height=var_height)


    if PLOTS_IN_BROWSER:
        fig_subplots.show()
    if STATIC_PLOTS:
        fig_subplots.write_image(savepath+"time_plots_small_multiples.png")

    return fig_subplots

def timeavg_mean_error(df, n_frames, factor, t_window=None, err_metric='sem'):

    # Create a numpy array to contain the factor of interest, and the variance + N
    fac_array = np.empty([n_frames, 4])

    for t in range(0,n_frames):

        # If a time-window was passed to the function, average across the window
        if t_window is not None:
            # get a subset of the dataframe across the range of frames
            sub_df = df[(df['frame']>=t - t_window/2) &
                          (df['frame']<t + t_window/2)]

        else:
            # Find the dataframe for a single frame
            sub_df = df[df['frame']==t]

        # Calculate mean, stdev and N
        sample_mean = np.nanmean(sub_df[factor])
        sample_std = np.nanstd(sub_df[factor])
        sample_n = len(sub_df[factor])# Count the non Nan values.

        if sample_n > 1:

            sample_ci = np.nanpercentile(sub_df[factor], [2.5,97.5])
        else:
            sample_ci = [np.nan,np.nan]

        upper_err = np.nan
        lower_err = np.nan

        if err_metric=='std':
            upper_err = sample_mean+sample_std
            lower_err = sample_mean-sample_std

        elif err_metric=='percentile':
            upper_err = sample_ci[1]
            lower_err = sample_ci[0]

        if err_metric=='sem':
            upper_err = sample_mean+sample_std/np.sqrt(sample_n)
            lower_err = sample_mean-sample_std/np.sqrt(sample_n)

        # Error bands displaying estimated standard error of the mean.
        fac_array[t,:] = [sample_mean,upper_err,lower_err,sample_n]

    return fac_array


def timeplot_sample(fig_data, factor, fac_array, n_frames, color_rgb, label):#, y_range=None):

    t_vect = np.linspace(0,n_frames,n_frames) * SAMPLING_INTERVAL

    # The plots with error bands are actually just 3 scatterplots

    # First is the sample mean
    fig_data.append(go.Scatter(
        name=label, # Should depend on what's being plotted.
        x=t_vect,
        y=fac_array[:,0],
        mode='lines',
        line=dict(color='rgb'+str(color_rgb))))
        # line=dict(color=color_rgb) # MJS changed this 3-13-2024
    

    # Second and third plot the error bands.
    fig_data.append(go.Scatter(
        name=label,
        x=t_vect,
        y=fac_array[:,1],
        mode='lines',
        marker=dict(color='rgba' +str(color_rgb)[:-1]+',0.4)'),
        showlegend=False))

    fig_data.append(go.Scatter(
        name=label,
        x=t_vect,
        y=fac_array[:,2],
        mode='lines',
        marker=dict(color='rgba' +str(color_rgb)[:-1]+',0.4)'),
        showlegend=False,
        fillcolor='rgba' +str(color_rgb)[:-1]+',0.1)',
        fill='tonexty'))

    return fig_data



def time_superplot_old(df, factor,x_range=None, y_range=None,t_window=None, savepath=TIMEPLOT_DIR):

    '''
    Convert a dataframe with a specified parameter into a plot that shows
    that parameter changing over the course of the experiment.

    Input:
        df: DataFrame
        factor: String: column title for the factor to be visualized.

    Returns:
        fig_data: Plotly graph data
    '''
    cond_grouping = 'Condition'
    rep_grouping = 'Replicate_ID'

    if(USE_SHORTLABELS):
        cond_grouping = 'Condition_shortlabel'
        rep_grouping = 'Replicate_shortlabel'

    fig1_data = [] # Define as an empty list to add traces

    # Frames to be calculated ones for the whole set
    n_frames = int(np.max(df['frame']))

    # palette = 'tab10'
    colors = sns.color_palette(PALETTE, n_colors=len(df[cond_grouping].unique()))
    fig_data_list = []

    # Loop for each condition
    for i, cond in enumerate(df[cond_grouping].unique()):

        # Update n_frames to current. Makes subplots autoscale to num timepts of that trace
        sub_df = df[df[cond_grouping]==cond]
        n_frames = int(np.max(sub_df['frame']))

        fac_array = timeavg_mean_error(df[df[cond_grouping]==cond], n_frames,factor,t_window)

        fig1_data = timeplot_sample(fig1_data, factor, fac_array, n_frames,
                       color_rgb = colors[i],label=cond)#,y_range=y_range)

        subfig_data = []

        # Each replicate from within that condition
        for rep in df[df[cond_grouping]==cond][rep_grouping].unique():

            fac_array = timeavg_mean_error(df[(df[cond_grouping]==cond)&(df[rep_grouping]==rep)],
                                           n_frames,factor,t_window)

            # Generate a slightly different random hue for each replicate.
            rand_vect = np.random.normal(0, 0.10, 3) #(mu,sigma)
            this_color = np.clip(colors[i]+rand_vect, 0,1)
            this_color = np.around(this_color,decimals=4)

            subfig_data = timeplot_sample(subfig_data, factor, fac_array, n_frames,
                           color_rgb = str(tuple(this_color)), label=rep)#,y_range=y_range)

        fig_data_list.append(subfig_data) #New suplot for each condition


    fig_data_list.append(fig1_data)


    fig_subplots = make_subplots(
        rows=len(fig_data_list),
        cols=1,
#         shared_xaxes=True,
#         row_heights=[1,3],

        # You can define the subplot titles here:
#         subplot_titles=['Time-averaged per condition', 'Time-averaged per replicate']
        subplot_titles=list(df[cond_grouping].unique())+['Time-averaged per condition']

    )

    for i, subfig in enumerate(fig_data_list):

        for sub_subfig in subfig:
            fig_subplots.add_trace(sub_subfig,row=i+1, col=1)

    fig = go.Figure(fig_subplots)

    fig.update_layout(
            title_text=factor+ " as a function of time",
            autosize=False,
            width=1000,
            height=1500,
#             xaxis_title= 'Time (min)',
#             yaxis_title= factor,
                legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0,#-0.4,
                xanchor="left",
                x=1,
                traceorder='normal'
            ))

    # Specify axis labels
    fig['layout']['xaxis']['title']='Time (min)'
    fig['layout']['xaxis2']['title']='Time (min)'
    fig['layout']['xaxis3']['title']='Time (min)'
    # fig['layout']['xaxis4']['title']='Time (min)'
    # fig['layout']['xaxis5']['title']='Time (min)'
    fig['layout']['yaxis']['title']= factor
    fig['layout']['yaxis2']['title']= factor
    fig['layout']['yaxis3']['title']= factor
    # fig['layout']['yaxis4']['title']= factor
    # fig['layout']['yaxis5']['title']= factor

    if x_range is not None:

        # Ensure the subplots share the axis scale
        fig.update_xaxes(range=x_range, row=1, col=1)
        fig.update_xaxes(range=x_range, row=2, col=1)
        fig.update_xaxes(range=x_range, row=3, col=1)
        # fig.update_xaxes(range=x_range, row=4, col=1)
        # fig.update_xaxes(range=x_range, row=5, col=1)
    if y_range is not None:

        # Ensure the subplots share the axis scale
        fig.update_yaxes(range=y_range, row=1, col=1)
        fig.update_yaxes(range=y_range, row=2, col=1)
        fig.update_yaxes(range=y_range, row=3, col=1)
        # fig.update_yaxes(range=y_range, row=4, col=1)
        # fig.update_yaxes(range=y_range, row=5, col=1)

    if STATIC_PLOTS:
        fig.write_image(savepath+str(factor)+'_time_superplot.png')

    if PLOTS_IN_BROWSER:

        fig.show()

    return fig

### DEV this
def time_superplot(df, factor,x_range=None, y_range=None,t_window=None, savepath=TIMEPLOT_DIR):

    '''
    Convert a dataframe with a specified parameter into a plot that shows
    that parameter changing over the course of the experiment.

    Input:
        df: DataFrame
        factor: String: column title for the factor to be visualized.

    Returns:
        fig_data: Plotly graph data
    '''
    cond_grouping = 'Condition'
    rep_grouping = 'Replicate_ID'

    if(USE_SHORTLABELS):
        cond_grouping = 'Condition_shortlabel'
        rep_grouping = 'Replicate_shortlabel'

    fig1_data = [] # Define as an empty list to add traces

    # Frames to be calculated ones for the whole set
    n_frames = int(np.max(df['frame']))

    # palette = 'tab10'
    colors = sns.color_palette(PALETTE, n_colors=len(df[cond_grouping].unique()))
    fig_data_list = []

    # Loop for each condition
    for i, cond in enumerate(df[cond_grouping].unique()):

        # Update n_frames to current. Makes subplots autoscale to num timepts of that trace
        sub_df = df[df[cond_grouping]==cond]
        n_frames = int(np.max(sub_df['frame']))

        fac_array = timeavg_mean_error(df[df[cond_grouping]==cond], n_frames,factor,t_window)

        fig1_data = timeplot_sample(fig1_data, factor, fac_array, n_frames,
                       color_rgb = colors[i],label=cond)#,y_range=y_range)

        subfig_data = []

        # Each replicate from within that condition
        for rep in df[df[cond_grouping]==cond][rep_grouping].unique():

            fac_array = timeavg_mean_error(df[(df[cond_grouping]==cond)&(df[rep_grouping]==rep)],
                                           n_frames,factor,t_window)

            # Generate a slightly different random hue for each replicate.
            rand_vect = np.random.normal(0, 0.10, 3) #(mu,sigma)
            this_color = np.clip(colors[i]+rand_vect, 0,1)
            this_color = np.around(this_color,decimals=4)

            subfig_data = timeplot_sample(subfig_data, factor, fac_array, n_frames,
                           color_rgb = str(tuple(this_color)), label=rep)#,y_range=y_range)

        fig_data_list.append(subfig_data) #New suplot for each condition


    fig_data_list.append(fig1_data)


    fig_subplots = make_subplots(
        rows=len(fig_data_list),
        cols=1,
#         shared_xaxes=True,
#         row_heights=[1,3],

        # You can define the subplot titles here:
#         subplot_titles=['Time-averaged per condition', 'Time-averaged per replicate']
        subplot_titles=list(df[cond_grouping].unique())+['Time-averaged per condition']

    )

    for i, subfig in enumerate(fig_data_list):

        for sub_subfig in subfig:
            fig_subplots.add_trace(sub_subfig,row=i+1, col=1)

    fig = go.Figure(fig_subplots)

    fig.update_layout(
            title_text=factor+ " as a function of time",
            autosize=False,
            width=1000,
            height=1500,
#             xaxis_title= 'Time (min)',
#             yaxis_title= factor,
                legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0,#-0.4,
                xanchor="left",
                x=1,
                traceorder='normal'
            ))

    # Specify axis labels
    fig['layout']['xaxis']['title']='Time (min)'
    fig['layout']['xaxis2']['title']='Time (min)'
    fig['layout']['xaxis3']['title']='Time (min)'
    # fig['layout']['xaxis4']['title']='Time (min)'
    # fig['layout']['xaxis5']['title']='Time (min)'
    fig['layout']['yaxis']['title']= factor
    fig['layout']['yaxis2']['title']= factor
    fig['layout']['yaxis3']['title']= factor
    # fig['layout']['yaxis4']['title']= factor
    # fig['layout']['yaxis5']['title']= factor

    if x_range is not None:

        # Ensure the subplots share the axis scale
        fig.update_xaxes(range=x_range, row=1, col=1)
        fig.update_xaxes(range=x_range, row=2, col=1)
        fig.update_xaxes(range=x_range, row=3, col=1)
        # fig.update_xaxes(range=x_range, row=4, col=1)
        # fig.update_xaxes(range=x_range, row=5, col=1)
    if y_range is not None:

        # Ensure the subplots share the axis scale
        fig.update_yaxes(range=y_range, row=1, col=1)
        fig.update_yaxes(range=y_range, row=2, col=1)
        fig.update_yaxes(range=y_range, row=3, col=1)
        # fig.update_yaxes(range=y_range, row=4, col=1)
        # fig.update_yaxes(range=y_range, row=5, col=1)

    if STATIC_PLOTS:
        fig.write_image(savepath+str(factor)+'_time_superplot.png')

    if PLOTS_IN_BROWSER:

        fig.show()

    return fig


def timeplots_of_differences(df_in,factor='Value', ctl_label=CTL_LABEL,cust_txt='', save_path=TIMEPLOT_DIR, t_window=None):

    ''' This import until they are both in the same script'''

    assert ctl_label in df_in['Condition'].values, ctl_label + ' is not in the list of conditions'
    assert ctl_label != -1, 'Not yet adapted to compare between cluster groups, use plots_of_differences_plotly() instead'

    df = df_in.copy()

    cond_grouping = 'Condition'

    # Sort values according to custom order for drawing plots onto graph
    df[cond_grouping] = pd.Categorical(df[cond_grouping], CONDITIONS_TO_INCLUDE)
    df.sort_values(by=cond_grouping, inplace=True, ascending=True)

    # For each frame, do the bootstrapping calculations and add it to a list

    frames = df['frame'].unique()

    df_list = []
    for frame in frames:

        frame_df = df[df['frame'] == frame]

        # Get the bootstrapped sample as a dataframe
        bootstrap_diff_df = bootstrap_sample_df(frame_df,factor,ctl_label)

        # Add frame and append this sub_df to the list.
        bootstrap_diff_df['frame'] = frame
        df_list.append(bootstrap_diff_df)

    bootstrap_df = pd.concat(df_list)

    if(USE_SHORTLABELS):

        # Apply the shortlabels to the dataframe
        replace_labels_shortlabels(bootstrap_df)

    # Frames to be calculated ones for the whole set
    n_frames = int(np.max(bootstrap_df['frame']))
    fig1_data = []
    # palette = 'tab10'
    # colors = sns.color_palette(PALETTE, n_colors=len(bootstrap_df[cond_grouping].unique())) #removed in favour of the below

    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP)
    numcolors= len(bootstrap_df[cond_grouping].unique())
    for i in range(numcolors):
        colors.append(cmap(i))
    #Remove the last element of each item in the colors list to match the sns output but with the universally applicable cmap
    colors = [i[:-1] for i in colors]

    # Loop for each condition
    for i, cond in enumerate(bootstrap_df[cond_grouping].unique()):


        fac_array = timeavg_mean_error(bootstrap_df[bootstrap_df[cond_grouping]==cond], n_frames,
                                       factor='Difference', err_metric='percentile',t_window=t_window)


        fig1_data = timeplot_sample(fig1_data, factor, fac_array, n_frames,
                       color_rgb = colors[i],label=cond)#,y_range=y_range)

    fig = go.Figure(fig1_data)

    fig.update_layout(
        title_text=factor+ " difference as a function of time",
        autosize=True,
        width=1000,
        height=500,
        font_size=28, #MJS made this change 8-24-2022
#         width=1000,
#         height=1000,
            xaxis_title= 'Time (minutes)',
            yaxis_title= 'Difference',
            legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0,#-0.4,
            xanchor="left",
            x=1,
            traceorder='normal'
        ))

    if STATIC_PLOTS:
        fig.write_image(save_path+str(factor)+'_diff_timeplot.png')

    if PLOTS_IN_BROWSER:

        fig.show()

    return fig


#####

def timeplots_of_difference_DEV(df_in,factor='Value', ctl_label=CTL_LABEL,cust_txt='', save_path=TIMEPLOT_DIR, t_window=None):

    ''' This import until they are both in the same script'''

    assert ctl_label in df_in['Condition'].values, ctl_label + ' is not in the list of conditions'
    assert ctl_label != -1, 'Not yet adapted to compare between cluster groups, use plots_of_differences_plotly() instead'

    df = df_in.copy()

    cond_grouping = 'Condition'

    # Sort values according to custom order for drawing plots onto graph
    df[cond_grouping] = pd.Categorical(df[cond_grouping], CONDITIONS_TO_INCLUDE)
    df.sort_values(by=cond_grouping, inplace=True, ascending=True)

    # For each frame, do the bootstrapping calculations and add it to a list

    frames = df['frame'].unique()

    df_list = []
    for frame in frames:

        frame_df = df[df['frame'] == frame]

        # Get the bootstrapped sample as a dataframe
        bootstrap_diff_df = bootstrap_sample_df(frame_df,factor,ctl_label)

        # Add frame and append this sub_df to the list.
        bootstrap_diff_df['frame'] = frame
        df_list.append(bootstrap_diff_df)

    bootstrap_df = pd.concat(df_list)

    if(USE_SHORTLABELS):

        # Apply the shortlabels to the dataframe
        replace_labels_shortlabels(bootstrap_df)

    # Frames to be calculated ones for the whole set
    n_frames = int(np.max(bootstrap_df['frame']))
    fig1_data = []
    # palette = 'tab10'
    colors = sns.color_palette(PALETTE, n_colors=len(bootstrap_df[cond_grouping].unique()))

    # Loop for each condition
    for i, cond in enumerate(bootstrap_df[cond_grouping].unique()):


        fac_array = timeavg_mean_error(bootstrap_df[bootstrap_df[cond_grouping]==cond], n_frames,
                                       factor='Difference', err_metric='percentile',t_window=t_window)


        fig1_data = timeplot_sample(fig1_data, factor, fac_array, n_frames,
                       color_rgb = colors[i],label=cond)#,y_range=y_range)

    fig = go.Figure(fig1_data)

    fig.update_layout(
        title_text=factor+ " difference as a function of time",
        autosize=True,
        width=1000,
        height=500,
        font_size=28, #MJS made this change 8-24-2022
#         width=1000,
#         height=1000,
            xaxis_title= 'Time (minutes)',
            yaxis_title= 'Difference',
            legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0,#-0.4,
            xanchor="left",
            x=1,
            traceorder='normal'
        ))

    if STATIC_PLOTS:
        fig.write_image(save_path+str(factor)+'_diff_timeplot.png')

    if PLOTS_IN_BROWSER:

        fig.show()

    return fig

def multi_condition_timeplot(df, factor,t_window=None, savepath=TIMEPLOT_DIR):

    '''
    Convert a dataframe with a specified parameter into a plot that shows
    that parameter changing over the course of the experiment.

    Input:
        df: DataFrame
        factor: String: column title for the factor to be visualized.

    Returns:
        fig_data: Plotly graph data
    '''

    cond_grouping = 'Condition'
    rep_grouping = 'Replicate_ID'

    if(USE_SHORTLABELS):
        cond_grouping = 'Condition_shortlabel'
        rep_grouping = 'Replicate_shortlabel'

    fig1_data = [] # Define as an empty list to add traces

    # Frames to be calculated ones for the whole set
    n_frames = int(np.max(df['frame']))


    # palette = 'tab10'
    # colors = sns.color_palette(PALETTE, n_colors=len(df[cond_grouping].unique()))

    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP)
    numcolors= len(df[cond_grouping].unique())
    for i in range(numcolors):
        colors.append(cmap(i))
    #Remove the last element of each item in the colors list to match the sns output but with the universally applicable cmap
    colors = [i[:-1] for i in colors]


    fig_data_list = []

    # Loop for each condition
    for i, cond in enumerate(df[cond_grouping].unique()):


        fac_array = timeavg_mean_error(df[df[cond_grouping]==cond], n_frames,factor,t_window)

        fig1_data = timeplot_sample(fig1_data, factor, fac_array, n_frames,
                       color_rgb = colors[i],label=cond)

    fig_data_list.append(fig1_data)

    fig = go.Figure(fig1_data)


#     fig = go.Figure(fig_subplots)
# fig.update_layout(font_size=<VALUE>)
    fig.update_layout(
            title_text=factor+ " as a function of time",
            autosize=False,
            width=1000,
            height=500,
            font_size=28,

                legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0,#-0.4,
                xanchor="left",
                x=1,
                traceorder='normal'
            ))

    # Specify axis labels
    fig['layout']['xaxis']['title']='Time (min)'
#     fig['layout']['xaxis2']['title']='Time (min)'
    fig['layout']['yaxis']['title']= factor
#     fig['layout']['yaxis2']['title']= factor


    if STATIC_PLOTS:
        fig.write_image(savepath+str(factor)+'_timeplot_multi-condition.png')

    if PLOTS_IN_BROWSER:

        fig.show()

    return fig

def multi_condition_timeplot_DEV(df, factor,t_window=None, savepath=TIMEPLOT_DIR):

    '''
    Convert a dataframe with a specified parameter into a plot that shows
    that parameter changing over the course of the experiment.

    Input:
        df: DataFrame
        factor: String: column title for the factor to be visualized.

    Returns:
        fig_data: Plotly graph data
    '''

    cond_grouping = 'Condition'
    rep_grouping = 'Replicate_ID'

    if(USE_SHORTLABELS):
        cond_grouping = 'Condition_shortlabel'
        rep_grouping = 'Replicate_shortlabel'

    fig1_data = [] # Define as an empty list to add traces

    # Frames to be calculated ones for the whole set
    n_frames = int(np.max(df['frame']))


    # palette = 'tab10'
    colors = sns.color_palette(PALETTE, n_colors=len(df[cond_grouping].unique()))
    fig_data_list = []

    # Loop for each condition
    for i, cond in enumerate(df[cond_grouping].unique()):


        fac_array = timeavg_mean_error(df[df[cond_grouping]==cond], n_frames,factor,t_window)

        fig1_data = timeplot_sample(fig1_data, factor, fac_array, n_frames,
                       color_rgb = colors[i],label=cond)

    fig_data_list.append(fig1_data)

    fig = go.Figure(fig1_data)


#     fig = go.Figure(fig_subplots)
# fig.update_layout(font_size=<VALUE>)
    fig.update_layout(
            title_text=factor+ " as a function of time",
            autosize=False,
            width=1000,
            height=500,
            font_size=28,

                legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0,#-0.4,
                xanchor="left",
                x=1,
                traceorder='normal'
            ))

    # Specify axis labels
    fig['layout']['xaxis']['title']='Time (min)'
#     fig['layout']['xaxis2']['title']='Time (min)'
    fig['layout']['yaxis']['title']= factor
#     fig['layout']['yaxis2']['title']= factor


    if STATIC_PLOTS:
        fig.write_image(savepath+str(factor)+'_timeplot_multi-condition.png')

    if PLOTS_IN_BROWSER:

        fig.show()

    return fig



def animate_t_window(cell_df, dr_df, dr_method='tSNE', cid='test'):

    '''
    Create an animated gif to show the cell_df's measurements through physical and
    low dimensional space.
    Saves temporary files to TMP folder,
    '''

    frames = cell_df['frame'].values
    t_list = []
    start_frame = frames[0]
    end_frame = frames[-1]
    gif_frames = []
    filenames=[]

    # Load the contour list one time, and pass it to visualize_cell_t_window()
    contour_list = get_cell_contours(cell_df)

    # Populate the list of t values (proportions of track) associated for the frames
    for frame in frames:

        t = (frame - start_frame) / (end_frame - start_frame)
        t = float("{:.3f}".format(t))
        t_list.append(t)

    for t in t_list:

        # Export the figure and add its filename to a list
        identifier = str(cid) +'_'+ dr_method
        filename = identifier+'_t_'+str(t)+'.png'
        fig = visualize_cell_t_window(cell_df, dr_df,contour_list=contour_list,
                                        t=t, t_window=8, dr_method=dr_method)
        fig.savefig(TEMP_OUTPUT+filename, dpi=300)
        plt.clf()
        filenames.append(TEMP_OUTPUT+filename)

    # Build the gif
    animation_filename = ANIM_OUTPUT +str(cid) +'_'+ dr_method +'_anim.gif'
    with imageio.get_writer(animation_filename, mode='I',fps=5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)

    display(Image(data=open(animation_filename,'rb').read(), format='png'))




def cluster_timeplot(clust_sum_t_df):

    '''
    Create a line plot of the cluster purity through time.

    Input:
        clust_sum_t_df: DataFrame created by running the cluster_composition_timecourse() function

    '''

    # Define the custom colormap for the clusters
    cluster_colors = []
    cmap = cm.get_cmap(CLUSTER_CMAP, len(clust_sum_t_df['cluster_id'].unique()))

    for i in range(cmap.N):
        this_color = cmap([i][0])
        cluster_colors.append('rgba'+str(this_color[0:4]))   # Convert to a format that plotly can accept.

    fig = px.line(clust_sum_t_df, x="Time (min)", y=CONDITION_SHORTLABELS[0]+"_ncells_%", color='cluster_id',
                color_discrete_sequence=cluster_colors)

    return fig
