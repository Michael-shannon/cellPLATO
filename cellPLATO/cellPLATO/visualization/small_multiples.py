#small_multiples.py
from initialization.config import *
from initialization.initialization import *

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def hist_sm(df, factor='mean_intensity'):

    '''
    Plotly graph object histogram,
    minimalist: intended for small multiples
    '''


    fig_data = go.Histogram(x=df[factor])

    return fig_data

def scatter_sm(df, dep_var='frame',factor='mean_intensity'):

    '''
    Plotly graph object scatter plot,
    minimalist: intended for small multiples

    Input:
        df Datafame
        dep_var: the variabe to be used on the x axis

    '''

    fig_data = go.Scatter(x=df[dep_var], y=df[factor],
                        mode='markers',
                        marker=dict(
                            size=3,
                        ))

    return fig_data


def small_multiples(df, factors, plot_type='hist', savepath=PLOT_OUTPUT):

    '''
    Create a plotly subplot figure containing cellular measurements
    Based off of the timeplots function.

    Input:
        df: DataFrame
        factors: list of strings, column names t visualize for the dataframe
        plot_type: Type of plot to draw:
            'scatter'
            'hist'
    '''

    fig_data_list = []

    # Compute the grid of subplots
    n_cols = 3
    n_rows = int(np.floor(len(factors) / n_cols)) + 1

    ys = np.linspace(1,n_rows, n_rows)
    xs = np.linspace(1,n_cols, n_cols)

    xx,yy = np.meshgrid(xs, ys) # reversed
    grid_x = np.reshape(xx, -1)
    grid_y = np.reshape(yy, -1)

    for factor in factors:

        # Get fig data for plot type of choice
        if(plot_type=='hist'):
            cur_fig_data = hist_sm(df, factor)

        elif(plot_type=='scatter'):
            cur_fig_data = scatter_sm(df, factor)

        fig_data_list.append(cur_fig_data)

    fig_subplots = make_subplots(
        rows=n_rows, cols=n_cols,shared_yaxes=True,
        # You can define the subplot titles here:
        subplot_titles=(factors)
    )

    for i, subfig in enumerate(fig_data_list):

        this_row = int(grid_y[i])
        this_col = int(grid_x[i])

        fig_subplots.add_trace(subfig,row=this_row, col=this_col)#i

        # Update yaxes for each row individually
        fig_subplots.update_yaxes(title_text=factors[i],
                              row=this_row, # Makes sure the axis label on the botto
                              col=this_col)


    # Update properties for the entire subplot figure
    var_height = 100 + n_rows * 120 # Calculate figure height by # of plots
    fig_subplots.update_layout(showlegend=False, title_text='Small-multiples: '+plot_type,
                                height=var_height)

    if STATIC_PLOTS:
        fig_subplots.write_image(savepath+"small_multiples_"+plot_type+".png")

    if PLOTS_IN_BROWSER:
        fig_subplots.show()

    return fig_subplots


def fig2subplot(fig_list,savepath=PLOT_OUTPUT):

    '''
    Combine the fig_data objects together into a multi-panel figure,
    that can be returned as a single object to the html render template.

    Input:
        fig_list: list of fig_data

    Output:
        fig: Plotly go Figure() with subplots in fig_list

    '''

    # Still need to figure out how to get this format correctly
    # By appending the type dictionaries to a list.
    # In its current form I need to know in advance the number and types
    specs=[[{"type": "scatter3d"},
            {"type": "scatter3d"}]]

    fig_subplots = make_subplots(
        rows=1, cols=len(fig_list),
        specs=specs,
        subplot_titles=('Cell migration trajectories',
                        'tSNE visualization of cell shape')
    )

    # Change the defaultview to rembind of 2D 'hairball' plots
    camera = dict(
        up=dict(x=1, y=0, z=1),
        # center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.5)
        )

    for i, subfig in enumerate(fig_list):

        fig_subplots.add_trace(subfig,
                  row=1, col=i+1)


    fig_subplots.update_layout(showlegend=False, title_text='3D plots',
                                scene_camera=camera)

    if STATIC_PLOTS:
        fig_subplots.write_image(savepath+'3D_subplots.png')

    return fig_subplots
