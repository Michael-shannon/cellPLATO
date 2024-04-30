from initialization.config import *
from initialization.initialization import *

from data_processing.data_wrangling import *


import numpy as np
import pandas as pd
import os

import scipy
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def superplots(df, factor, t=FRAME_END, save_path=SUPERPLOT_DIR):
    '''
    A function to implement the 'superplots' from Lord et al 2020,
    where eperimental replicates within pooled conditions are plotted such that they can be distinguished.



    df: a pandas DataFrame with the column headers:Replicate,Treatment,Speed
    factor:
    t:

    '''

    plt.clf()

    sns.set(rc={'figure.figsize':(8,6),#(11.7,8.27),
                "font.size":PLOT_TEXT_SIZE,
    #             "font.scale":2,
                "axes.titlesize":PLOT_TEXT_SIZE*1.2,
                "axes.labelsize":PLOT_TEXT_SIZE*1.2},
            style="white",
            font_scale=1
           )

    data = format_for_superplots(df, factor,t)

    # Sort the dataframe by custom category list to set draw order
    if(USE_SHORTLABELS): # Must instead sort by shortlabel list order
        # Sort the dataframe by custom category list to set draw order
        data['Treatment'] = pd.Categorical(data['Treatment'], CONDITION_SHORTLABELS)
    else:
         # Sort the dataframe by custom category list to set draw order
        data['Treatment'] = pd.Categorical(data['Treatment'], CONDITIONS_TO_INCLUDE)

    data.sort_values(by='Treatment', inplace=True, ascending=True)

    # Extract the actual treatment names
    treatment_list = pd.unique(data['Treatment'])

    ReplicateAverages = data.groupby(['Treatment','Replicate'], as_index=False).agg({factor: "mean"});
    ReplicateAvePivot = ReplicateAverages.pivot_table(columns='Treatment', values=factor, index="Replicate");

    fig = sns.swarmplot(x="Treatment", y=factor, hue="Replicate", data=data)
    ax = sns.swarmplot(x="Treatment", y=factor, hue="Replicate", size=15, edgecolor="k", linewidth=2, data=ReplicateAverages)
    ax.legend_.remove();

    # Set the axes limits custom (3 sigma)
    y_low = np.mean(data[factor]) - 3 * np.std(data[factor])

    # Avoid making lower limit below 0 if values don't extent that low.
    if y_low < 0 and np.min(data[factor] > 0):
        y_low = 0

    y_high = np.mean(data[factor]) + 3 * np.std(data[factor])

    filtered_df = data[(data[factor].values>y_low)  & (data[factor].values < y_high)]
    n_dropped = len(data[~data.isin(filtered_df)].dropna())
    ax.set(ylim=(y_low, y_high))
    print('Custom axis using 3 sigma rule, axis bounds not showing ' + str(n_dropped) + ' point(s): ')

    if STATIC_PLOTS and DRAW_SUPERPLOTS:

        plt.savefig(save_path + factor +'_superplots_sns_t_'+str(t)+'.png', format='png', dpi=600)


def superplots_plotly(df_in, factor, t=FRAME_END, grid=False, save_path=SUPERPLOT_DIR):

    '''
    A function to implement the 'superplots' from Lord et al 2020,
    where eperimental replicates within pooled conditions are plotted such that they can be distinguished.

    df_in: a pandas DataFrame with the column headers:Replicate,Treatment,Speed

    This plot started its life as a boxplot:
    https://plotly.com/python/reference/box/

    '''
    df = df_in.copy()

    # Get a colormap the length of unique replicates
    replicates = df['Replicate_ID'].unique()
    colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(replicates)))

    sp_df = format_for_superplots(df, factor,t)
    # print(sp_df)
    # print(sp_df['Replicate'].unique())
    # print('Check this is correct!')

    if(USE_SHORTLABELS): # Must instead sort by shortlabel list order
        # Sort the dataframe by custom category list to set draw order
        sp_df['Treatment'] = pd.Categorical(sp_df['Treatment'], CONDITION_SHORTLABELS)
    else:
         # Sort the dataframe by custom category list to set draw order
        sp_df['Treatment'] = pd.Categorical(sp_df['Treatment'], CONDITIONS_TO_INCLUDE)

    sp_df.sort_values(by='Treatment', inplace=True, ascending=True)
    # sp_df.reset_index(inplace=True, drop=True)


    # Extract the actual treatment names
    treatment_list = list(pd.unique(sp_df['Treatment']))
    # print(len(colors))
    # print(treatment_list)
    # assert len(colors) == len(treatment_list), 'Color range should equal the number of conditons (treatments)'

    fig = go.Figure()

    # For each condition,
    for treatment in treatment_list:

        stat_list = [] # List to contain the sumary statistic plotted on top.
        treat_subdf = sp_df.loc[sp_df['Treatment'] == treatment]
        rep_list = pd.unique(treat_subdf['Replicate'])
        n_reps = len(rep_list)
        # For each replicate
        for i, rep in enumerate(rep_list): # Using enumerate to keep track of the # of reps
            '''
            Note: This is needed to manually force the summary points to spread out along
            the x-dimension. Need to specify their position, based on the number of replicates.
            Important:
                    rep gives the id of the replicate, which determines the color. This may
                    or may not be shared with the other condition, depending on input data.
                    i is the index relative to len(rep_list), used to distinguish between replicates
                    of the same group.
            '''
            # Use seperate index to choose colors
            if(i < len(colors)):
                ci = i
            else:
                ci = i - len(colors)

            rel_pos = -0.5 + i / n_reps
            rep_subdf = treat_subdf.loc[treat_subdf['Replicate'] == rep]

            # Draw the swarm plots
            fig.add_trace(go.Box(y=rep_subdf[factor].values,#y0,
                                 name=treatment,#treatment_list[0],
                                 opacity=1,
                                 marker={
                                     'color':'rgb' + str(tuple(colors[ci,:])),# tuple(colors[ci,:])#rep]
                                 },
                                fillcolor='rgba(0,0,0,0)',
                                boxpoints='all',
                                jitter=0.8,
                                line={
                                    'width': 0
                                },
                                pointpos=0))

            # Save trace data to a list to draw summary stats on top.
            trace_data = go.Box(y=[np.mean(rep_subdf[factor].values)],#y0)],
                                 name=treatment,
                                 opacity=1,
                                 marker={
                                     'size':20,
                                     'color': 'rgb' + str(tuple(colors[ci,:])),#colors[ci,:],#rep],
                                     'line': {
                                         'color': 'black',
                                         'width': 2
                                     }
                                 },
                                fillcolor='rgba(0,0,0,0)',
                                boxpoints='all',
                                jitter=0,
                                line={
                                    'width': 0
                                },
                                pointpos=rel_pos)

            stat_list.append(trace_data)

        # After all replicates are drawn, THEN draw the summary stats fig_data
        for stat in stat_list:
             fig.add_trace(stat)

    fig.update_layout(showlegend=False,
                      plot_bgcolor = 'white',
                      yaxis_title=factor,
                      title_text="Superplots: "+factor,
                      font=dict(
                          #family="Courier New, monospace",
                          size=PLOT_TEXT_SIZE, #CHANGED BY MJS
                          # size=PLOT_TEXT_SIZE,
                          color="Black"))

    # Show the axis frame, and optionally the grid
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    if(grid):
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black')


    if STATIC_PLOTS and DRAW_SUPERPLOTS:

        fig.write_image(save_path + factor +'_superplots_plotly_t_'+str(t)+'.png')

    if PLOTS_IN_BROWSER:
        fig.show()

    # Superplots retuns the figure object, not to be added to subplot figure
    return fig# graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def superplots_plotly_grays(df_in, factor, t=FRAME_END, grid=False, save_path=SUPERPLOT_grays_DIR):

    '''
    A function to implement the 'superplots' from Lord et al 2020,
    where eperimental replicates within pooled conditions are plotted such that they can be distinguished.

    df_in: a pandas DataFrame with the column headers:Replicate,Treatment,Speed

    This plot started its life as a boxplot:
    https://plotly.com/python/reference/box/

    '''
    df = df_in.copy()

    # Get a colormap the length of unique replicates
    replicates = df['Replicate_ID'].unique()
    # colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(replicates)))
    # colors = np.asarray(sns.color_palette('Greys', n_colors=len(replicates)))
    colors = np.asarray(sns.color_palette('Greys', n_colors=6))
    timestorepeat_in=(len(replicates))/2
    timestorepeat = (np.ceil(timestorepeat_in)).astype(int)
    colors2=colors[2]
    colors3=colors[4]
    colors4=np.stack((colors2,colors3))
    colors5 = np.tile(colors4,(timestorepeat,1))
    colors=colors5

    sp_df = format_for_superplots(df, factor,t)

    if(USE_SHORTLABELS): # Must instead sort by shortlabel list order
        # Sort the dataframe by custom category list to set draw order
        sp_df['Treatment'] = pd.Categorical(sp_df['Treatment'], CONDITION_SHORTLABELS)
    else:
         # Sort the dataframe by custom category list to set draw order
        sp_df['Treatment'] = pd.Categorical(sp_df['Treatment'], CONDITIONS_TO_INCLUDE)

    sp_df.sort_values(by='Treatment', inplace=True, ascending=True)
    # sp_df.reset_index(inplace=True, drop=True)


    # Extract the actual treatment names
    treatment_list = list(pd.unique(sp_df['Treatment']))
    # print(len(colors))
    # print(treatment_list)
    # assert len(colors) == len(treatment_list), 'Color range should equal the number of conditons (treatments)'

    fig = go.Figure()

    # For each condition,
    for treatment in treatment_list:

        stat_list = [] # List to contain the sumary statistic plotted on top.
        treat_subdf = sp_df.loc[sp_df['Treatment'] == treatment]
        rep_list = pd.unique(treat_subdf['Replicate'])
        n_reps = len(rep_list)
        # For each replicate
        for i, rep in enumerate(rep_list): # Using enumerate to keep track of the # of reps
            '''
            Note: This is needed to manually force the summary points to spread out along
            the x-dimension. Need to specify their position, based on the number of replicates.
            Important:
                    rep gives the id of the replicate, which determines the color. This may
                    or may not be shared with the other condition, depending on input data.
                    i is the index relative to len(rep_list), used to distinguish between replicates
                    of the same group.
            '''
            # Use seperate index to choose colors
            if(i < len(colors)):
                ci = i
            else:
                ci = i - len(colors)

            rel_pos = -0.5 + i / n_reps
            rep_subdf = treat_subdf.loc[treat_subdf['Replicate'] == rep]

            # Draw the swarm plots
            fig.add_trace(go.Box(y=rep_subdf[factor].values,#y0,
                                 name=treatment,#treatment_list[0],
                                 opacity=1,
                                 marker={
                                     'color':'rgb' + str(tuple(colors[ci,:])),# tuple(colors[ci,:])#rep]
                                 },
                                fillcolor='rgba(0,0,0,0)',
                                boxpoints='all',
                                jitter=0.8,
                                line={
                                    'width': 0
                                },
                                pointpos=0))

            # Save trace data to a list to draw summary stats on top.
            trace_data = go.Box(y=[np.mean(rep_subdf[factor].values)],#y0)],
                                 name=treatment,
                                 opacity=1,
                                 marker={
                                     'size':20,
                                     'color': 'rgb' + str(tuple(colors[ci,:])),#colors[ci,:],#rep],
                                     'line': {
                                         'color': 'black',
                                         'width': 2
                                     }
                                 },
                                fillcolor='rgba(0,0,0,0)',
                                boxpoints='all',
                                jitter=0,
                                line={
                                    'width': 0
                                },
                                pointpos=rel_pos)

            stat_list.append(trace_data)

        # After all replicates are drawn, THEN draw the summary stats fig_data
        for stat in stat_list:
             fig.add_trace(stat)

    fig.update_layout(showlegend=False,
                      plot_bgcolor = 'white',
                      yaxis_title=factor,
                      title_text="Superplots: "+factor,
                      font=dict(
                          #family="Courier New, monospace",
                          size=PLOT_TEXT_SIZE, #CHANGED BY MJS
                          # size=PLOT_TEXT_SIZE,
                          color="Black"))

    # Show the axis frame, and optionally the grid
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    if(grid):
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='black')


    if STATIC_PLOTS and DRAW_SUPERPLOTS:

        fig.write_image(save_path + factor +'_superplots_plotly_t_'+str(t)+'.png')

    if PLOTS_IN_BROWSER:
        fig.show()

    # Superplots retuns the figure object, not to be added to subplot figure
    return fig# graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
