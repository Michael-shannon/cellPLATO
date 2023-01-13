from initialization.config import *
from initialization.initialization import *

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go


plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.),
    "figure.figsize":    (10,10),
    "font.size": 12
})

def visualize_filtering(df, filt_counts, plot_by='xy'):

    assert 'included' in df.columns, 'visualize_filtering() must be run on filtered dataframe'

    if plot_by == 'xy':
        x_name = 'x_um'
        y_name = 'y_um'
        color_by = 'rip_L'
        x_label='x position (microns)'
        y_label='y position (microns)'

    elif plot_by == 'pca':
        x_name = 'PC1'
        y_name = 'PC2'
        color_by = 'label'
        x_label='PC1'
        y_label='PC2'

    elif (plot_by == 'tsne' or plot_by == 'tSNE'):

        x_name = 'tSNE1'
        y_name = 'tSNE2'
        color_by = 'label'
        x_label='tSNE1'
        y_label='tSNE2'

    elif plot_by == 'umap':

        x_name = 'UMAP1'
        y_name = 'UMAP2'
        color_by = 'label'
        x_label = 'UMAP1'
        y_label = 'UMAP2'

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[20,10])

    df_filt = df[df['included'] == True]

    ax1.scatter(x=df[x_name], y=df[y_name], color='gray', s=0.5)
    ax1.scatter(x=df_filt[x_name], y=df_filt[y_name], c=df_filt[color_by], s=5) #
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    filt_cond = ['Pre-filtering']
    counts = [len(df['uniq_id'].unique())]

    for filt in filt_counts:
        filt_cond.append(filt[0])
        counts.append(filt[1])

    filt_cond.append('Post-filtering')
    counts.append(len(df_filt['uniq_id'].unique()))

    ax2.bar(filt_cond,counts)
    ax2.set_ylabel('Number of cells')

    return fig


def visualize_filt_loss():

    # Labels as names of exported dataframes
    labels = ['comb_df',
          'mig_df',
          'dr_df-prefilt',
          'dr_df_filt']

    # Add the programatically generated names for the filtered outputs
    # From the DATA_FILTERS dictionary
    for i,factor in enumerate(DATA_FILTERS.keys()):
        labels.append('filt_'+str(i)+'-'+factor)

    # Load each of the DataFrames into a list
    df_list = []
    for label in labels:
        df_list.append(pd.read_csv(DATA_OUTPUT + label+'.csv'))

    # Set up the subplot figure.
    fig = make_subplots(
        rows=2, cols=len(df_list),
    #     subplot_titles=(labels),
        specs=[[{} for _ in range(len(df_list))],
                [{'colspan': len(df_list)}, *[None for _ in range(len(df_list)-1)]]])

    count = []

    for i, df in enumerate(df_list): #enumerate here to get access to i
        label=labels[i]
        count.append(len(df.index))

        fig.add_trace(go.Scatter(x=df['x'],
                                 y=df['y'],
                                opacity=0.5),
                  row=1,
                  col=i+1)

    fig.add_trace(go.Scatter(x=labels, y=count),
                  row=2, col=1)

    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(tickangle=-90)
    fig.update_layout(showlegend=False)

    if STATIC_PLOTS:
        fig.write_image(PLOT_OUTPUT+'filter_loss.png')

    if PLOTS_IN_BROWSER:
        fig.show()
