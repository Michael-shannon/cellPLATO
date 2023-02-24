# comparative_visualization.py

from initialization.config import *
from initialization.initialization import *

from data_processing.data_wrangling import *
from data_processing.statistics import *

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



def scatter2dplotly_compare(comb_df, factors):

    '''
    2D scatter plot built with plotly graph objects, intended to visualize the
    results of the dimensionality reduction operations.
    This version is explicitly intended for comparing between conditions on the
    same scatter axis.


    Input:
        comb_df: DataFrame, contains combined data from multiple conditions, and/or replicates
        factors:
        color_by: Indicates what factor should be used to color the points.
                default='Condition'

    Returns:
        fig_data:
         Note: can be visualized normally by using:
             fig = go.Figure(fig_data)
             fig.show()

    '''

    # Extract the data to be used to color-code
    cmaps = ['Viridis', 'inferno']

    '''
    For each of the conditions to be plotted, assign them a colormap.
    Create trace_data for each,
    '''
    cond_list = comb_df['Condition'].unique()
    trace_list = [] # Keep traces in list to return, instead of fig object.
    for i, condition in enumerate(cond_list):

        sub_df = comb_df.loc[comb_df['Condition'] == condition]

        x = sub_df[factors[0]]
        y = sub_df[factors[1]]

        trace_data = go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=sub_df['frame'],                # set color to an array/list of desired values
                        colorscale=cmaps[i],   # choose a colorscale
                        opacity=0.5))

        trace_list.append(trace_data)

    # After all replicates are drawn, THEN draw the summary stats fig_data
    # fig.update_layout(showlegend=False,
    #              yaxis_title=factor)

    # Define fig layout as dict, to return and apply in the pipeline
    fig_layout={
        'xaxis_title': factors[0],
        'yaxis_title': factors[1],
        'showlegend': False,
        'title': 'Low-dimension scatterplot'
    }

    # Create the Plotly figure.
    scatter_comp = go.Figure()
    for trace in dr_data:
         scatter_comp.add_trace(trace)
    scatter_comp.update_layout(layout)

    if STATIC_PLOTS:
        scatter_comp.write_image(PLOT_OUTPUT+str(factors)+"_compartive_scatter_plotly.png")

    if PLOTS_IN_BROWSER:
        scatter_comp.show()


    return trace_list, fig_layout #fig_data




def scatter3dplotly_compare(comb_df, factors):

    '''
    3D scatter plot built with plotly graph objects, intended to visualize the
    results of the dimensionality reduction operations.
    This version is explicitly intended for comparing between conditions on the
    same scatter axis.



    Input:
        comb_df: DataFrame, contains combined data from multiple conditions, and/or replicates
        factors:
        color_by: Indicates what factor should be used to color the points.
                default='Condition'

    Returns:

        trace_list:
        fig_layout:
        OR
        fig_data:
         Note: can be visualized normally by using:
             fig = go.Figure(fig_data)
             fig.show()

    '''

    # Extract the data to be used to color-code
    cmaps = ['Viridis', 'inferno']

    '''
    For each of the conditions to be plotted, assign them a colormap.
    Create trace_data for each,
    '''
    cond_list = comb_df['Condition'].unique()
    trace_list = [] # Keep traces in list to return, instead of fig object.
    for i, condition in enumerate(cond_list):

        sub_df = comb_df.loc[comb_df['Condition'] == condition]

        x = sub_df[factors[0]]
        y = sub_df[factors[1]]
        z = sub_df[factors[2]]

        trace_data = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=sub_df['frame'],                # set color to an array/list of desired values
                        colorscale=cmaps[i],   # choose a colorscale
                        opacity=1))

        trace_list.append(trace_data)


    # Define fig layout as dict, to return and apply in the pipeline
    fig_layout={
        'xaxis_title': factors[0],
        'yaxis_title': factors[1],
        'showlegend': False,
        'title': 'Low-dimension scatterplot'
    }

    # Create the Plotly figure.
    scatter_comp = go.Figure()
    for trace in dr_data:
         scatter_comp.add_trace(trace)
    scatter_comp.update_layout(layout)

    if STATIC_PLOTS:
        scatter_comp.write_image(PLOT_OUTPUT+str(factors)+"_compartive_scatter_plotly.png")

    if PLOTS_IN_BROWSER:
        scatter_comp.show()

    return trace_list, fig_layout #fig_data



def plotly_marginal_scatter(df, pair, save_path=MARGSCAT_DIR):

    '''
    Create a plotly express scatterplot comparing multiple conditions, for a user-provided
    pair of factors.

    input:
        df: DataFrame
        pair: list of factors to compare.
    '''

    assert len(pair) == 2, 'Marginal scatter requires 2 factors as input'

    fig = px.scatter(df, x=pair[0], y=pair[1], color="Condition",
                     marginal_x="violin", marginal_y="violin",
                      title="Comparative marginal scatter: "+str(pair))

    if STATIC_PLOTS:
        fig.write_image(save_path+'marginal_scatter_'+str(pair)+'.png')

    if PLOTS_IN_BROWSER:
        fig.show()



def marginal_xy(df, pair, plot_type='scatter', renderer='plotly', save_path=MARGSCAT_DIR, bounds=None,supp_label=''):

    '''
    Create a plotly express scatterplot comparing multiple conditions, for a user-provided
    pair of factors.

    input:
        df: DataFrame
        pair: list of factors to compare.
        plot_type: scatter, contour, hex
        renderer: plotly or seaborn

    '''

    assert len(pair) == 2, 'Marginal scatter requires 2 factors as input'

    cond_grouping = 'Condition'
#     rep_grouping = 'Replicate_ID'

    if(USE_SHORTLABELS):
        cond_grouping = 'Condition_shortlabel'
#         rep_grouping = 'Replicate_shortlabel'

    # Unpack th ebounds if they exist
    if bounds is not None:
        x_min, x_max,y_min, y_max = bounds

    if renderer == 'plotly':

        if plot_type == 'scatter':
            fig = px.scatter(df, x=pair[0], y=pair[1], color=cond_grouping,
                 marginal_x="violin", marginal_y="violin",
                  title='marginal_xy_'+plot_type+ '_'+str(pair))

        elif plot_type == 'contour':

            fig = px.density_contour(df, x=pair[0], y=pair[1], color=cond_grouping,
                         marginal_x="violin", marginal_y="violin",
                          title='marginal_xy_'+plot_type+ '_'+str(pair))

        elif plot_type == 'hex':

            print('No hexbin plot type in plotly.')
            fig = go.Figure()

        if STATIC_PLOTS:
            fig.write_image(save_path+'marginal_xy_plotly_'+plot_type+ '_'+str(pair)+'_'+supp_label+'.png')

        if PLOTS_IN_BROWSER:
            fig.show()

    elif renderer == 'seaborn':

        if plot_type == 'scatter':

            # If only one condition:
            if len(df[cond_grouping].unique())==1:

                fig =  sns.jointplot(data=df, x=pair[0], y=pair[1], color='black',
                          joint_kws={'s': 1}, alpha=0.5)
            else:
                fig =  sns.jointplot(data=df, x=pair[0], y=pair[1], hue = df[cond_grouping],
                      joint_kws={'s': 1}, alpha=0.5)
                plt.legend(loc='best')

        elif plot_type == 'contour':

            # Remove the legend if there's only one condition in the provided dataset.
            if len(df[cond_grouping].unique())==1:

                fig =  sns.jointplot(data=df, x=pair[0], y=pair[1], color = 'black', kind="kde", palette='magma')
                plt.suptitle(supp_label, y=1.05, fontsize = 16)

            else:
                fig =  sns.jointplot(data=df, x=pair[0], y=pair[1], hue = df[cond_grouping],kind="kde", palette=PALETTE)
                plt.suptitle(supp_label, y=1.05, fontsize = 16)
                sns.color_palette(PALETTE, as_cmap=True)

        elif plot_type == 'hex':
            print('No multi-condition hexplot available, consider making small multiples.')
            fig =  sns.jointplot(data=df, x=pair[0], y=pair[1],kind="hex",palette='magma')
            plt.suptitle(supp_label, y=1.05, fontsize = 16)
            sns.color_palette("magma", as_cmap=True)

        if bounds is not None:
            fig.ax_marg_x.set_xlim(x_min, x_max)
            fig.ax_marg_y.set_ylim(y_min, y_max)


        if STATIC_PLOTS:

            fig.savefig(save_path+'marginal_xy_sns_'+plot_type+ '_'+str(pair)+'_'+supp_label+'.png', dpi=300)#plt.



def comparative_bar(df_tup, x, y, title='',  height=400, to_plot='avg',error='SEM', save_path=BAR_DIR): #color='Condition'
    # print('THIS IS THE INPUT DF')
    # display(df_tup)
    # print(df_tup.columns)

    widthmultiplier = len(df_tup)
    print("widthmultiplier: ", widthmultiplier)

    '''
    Simple bar plot conveneince function that allows plotting of color-coded conditions either on a per-conditions
    or per-replicate basis.
    Eventually to be replaced by a bar plot that includes measure of variance to be plotted as error bars

    Inputs:
        df: DataFrame to be plotted
        x: Grouping, 'Condition' 'Replicate_ID'
        y: factor to be visualized
        title: str, additional label for the saved plot.
        color: factor to color by, default: 'Condition'
        height: plot height, default: 400px
        to_plot: str, what to plot: 'avg' or 'n'
        error: Measure of variance for error bars, str: SEM or STD
    '''
    # This part extracts an sns colormap for use in plotly express ###

    pal = sns.color_palette(CONDITION_CMAP) #extracts a colormap from the seaborn stuff.
    cmap=pal.as_hex()[:] #outputs that as a hexmap which is compatible with plotlyexpress below

    # Split up the input tuple:(avg, std, n)
    df = df_tup[0]
    std_df = df_tup[1]
    n_df = df_tup[2]

    if(USE_SHORTLABELS):

        # df = add_shortlabels(df)
        grouping = 'Condition_shortlabel'

        # Sort the dataframe by custom category list to set draw order
        df[grouping] = pd.Categorical(df[grouping], CONDITION_SHORTLABELS)

        # Also replace the x-labels on the plot and legend.
        if(x=='Condition'):
            x = 'Condition_shortlabel'
        elif(x=='Replicate_ID'):
            x = 'Replicate_shortlabel'

    else:

        grouping = 'Condition'

        # Sort the dataframe by custom category list to set draw order
        df[grouping] = pd.Categorical(df[grouping], CONDITIONS_TO_INCLUDE)

    color = grouping


    df.sort_values(by=grouping, inplace=True, ascending=True)

    if error == 'SEM':
        y_error = std_df[y] / np.sqrt(n_df[y]) #Estimate of SEM (std / sqare root of n)

    elif error == 'STD':
        y_error = std_df[y] # Stadard deviation

    if(to_plot == 'avg'):
        # Plot the means between groups for this factor, between conditions and between replicates.
        fig = px.bar(df, x=x, y=y, color=color, height=height,
                     # color_discrete_sequence=eval(PX_COLORS),#cmap
                     color_discrete_sequence=cmap,
                    error_y = y_error)

    elif(to_plot == 'n'):

        fig = px.bar(n_df, x=x, y=y, color=color, height=height,
                     # color_discrete_sequence=eval(PX_COLORS),
                     color_discrete_sequence=cmap,
                     labels = dict(y="Number of cells"))
    
    widthofplot = 220* widthmultiplier
    #change the font size of the axis labels
    fig.update_layout(showlegend=False,
                    # plot_bgcolor = 'white',  
                    autosize=False,
                    width = widthofplot,
                    height = 650,                  
                    font=dict(
                        #family="Courier New, monospace",
                        size=PLOT_TEXT_SIZE,
                        color="Black"))  
    # fig.update_xaxes(tickangle=90)
    # Remove the x axis label
    fig.update_xaxes(title_text='', tickangle=45)

 
    # change the font size of the y and x axis tick labels

        


    if STATIC_PLOTS:
        fig.write_image(save_path+y+'_'+to_plot+'_'+title + '.png')

    if PLOTS_IN_BROWSER:
        fig.show()

    return fig

def comparative_SNS_bar(df, save_path=BAR_SNS_DIR):
    import seaborn as sns
    whattoplot=ALL_FACTORS
    CLUSTER_CMAP = 'tab20'
    CONDITION_CMAP = 'dark'

    colors = np.asarray(sns.color_palette('Greys', n_colors=6))
    timestorepeat_in=(len(df['Condition'].unique()))/2
    timestorepeat = (np.ceil(timestorepeat_in)).astype(int)
    colors2=colors[2]
    colors3=colors[4]
    colors4=np.stack((colors2,colors3))
    colors5 = np.tile(colors4,(timestorepeat,1))
    colors=colors5

    import seaborn as sns
    sns.set_theme(style="ticks")
    # sns.set_palette(CONDITION_CMAP)

    x_lab = whattoplot
    plottitle = ""

    for factor in np.arange(len(whattoplot)):
        # f, ax = plt.subplots(1, 1, figsize=(10, 10)) #sharex=True
        f, ax = plt.subplots() #sharex=True
        sns.barplot(ax=ax, x="Condition_shortlabel", y=whattoplot[factor], data=df, palette=colors,capsize=.2, dodge=False) #ci=85, # estimator=np.mean,
        # sns.catplot(ax=ax, x="Condition_shortlabel", y=whattoplot[factor], data=df, palette=colors, kind="boxen") #errorbar=('ci', 95)
        sns.stripplot(ax=ax, x="Condition_shortlabel", y=whattoplot[factor], data=df, size=5, color=".1",alpha = 0.6, linewidth=0, jitter=0.2)

        ax.xaxis.grid(True)
        ax.set(xlabel="")
        ax.set_ylabel(whattoplot[factor], fontsize=PLOT_TEXT_SIZE)
        ax.set_title("", fontsize=PLOT_TEXT_SIZE)
        # ax.tick_params(axis='both', labelsize=36)
        ax.tick_params(axis='y', labelsize=PLOT_TEXT_SIZE)
        ax.tick_params(axis='x', labelsize=PLOT_TEXT_SIZE)
        # f.tight_layout()
        plt.setp(ax.patches, linewidth=3, edgecolor='k')
    # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
        f.savefig(save_path+str(whattoplot[factor])+'_gray_barplot.png', dpi=300)#plt.
    if PLOTS_IN_BROWSER:
        f.show()

    return
