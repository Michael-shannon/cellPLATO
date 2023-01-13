#scatterplots.py
from initialization.config import *
from initialization.initialization import *

import seaborn as sns
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def seaborn_jointplot(df, pair, savepath=PLOT_OUTPUT):

    sns.jointplot(data=df,
          x=pair[0],
          y=pair[1],
          kind="reg")

    if STATIC_PLOTS:

        plt.savefig(savepath + '\jointplot'+str(pair)+'.png', format='png', dpi=600)


def scatter2dplotly(df, factors, savepath=PLOT_OUTPUT):

    '''
    3D scatte rplot built woth plotly graph objects, intended to visualize the
    results of the dimensionality reduction operations.

    Note: return the graph data, instead of a figure object.
    This way, multiple graph_data objects can be combined more easily into a
    mult-panel subplot figure.

    Input:
        df
        factors

    Returns:
        fig_data:
         Note: can be visualized normally by using:
             fig = go.Figure(fig_data)
             fig.show()

    '''

    df.sort_values(by=['particle', 'frame'], inplace=True)

    x = df[factors[0]]
    y = df[factors[1]]

    #     fig = go.Figure( # Keep this bit in case we revert to retirng the entire fig
    fig_data=go.Scatter(
                x=x,
                y=y,

                mode='markers',
                marker=dict(
                    size=8,
                    color=df['particle'],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8),
                )

    fig = go.Figure(fig_data)

    if STATIC_PLOTS:
        fig.write_image(savepath+'scatter_2d'+str(factors)+'.png')

    if PLOTS_IN_BROWSER:
        fig.show()

    return fig_data

def scatter3dplotly(df, factors, savepath=PLOT_OUTPUT):

    '''
    3D scatte rplot built woth plotly graph objects, intended to visualize the
    results of the dimensionality reduction operations.

    Note: return the graph data, instead of a figure object.
    This way, multiple graph_data objects can be combined more easily into a
    mult-panel subplot figure.

    Input:
        df
        factors

    Returns:
        fig_data:
         Note: can be visualized normally by using:
             fig = go.Figure(fig_data)
             fig.show()

    '''

    df.sort_values(by=['particle', 'frame'], inplace=True)

    x = df[factors[0]]
    y = df[factors[1]]
    z = df[factors[2]]

    #     fig = go.Figure( # Keep this bit in case we revert to retirng the entire fig
    fig_data=go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['particle'],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8),
                )

    fig = go.Figure(fig_data)

    if STATIC_PLOTS:
        fig.write_image(savepath+'scatter_3d'+str(factors)+'.png')

    if PLOTS_IN_BROWSER:
        fig.show()

    return fig_data



def hairball_2D(zerod_stc, savepath=PLOT_OUTPUT):

    xmin = np.nanmin(zerod_stc[:,0,:])
    xmax = np.nanmax(zerod_stc[:,0,:])
    ymin = np.nanmin(zerod_stc[:,1,:])
    ymax = np.nanmax(zerod_stc[:,1,:])

    # Do we prefer defining these globally instead of extracting from the time array?
    n_cells = np.shape(zerod_stc)[0]
    n_frames = np.shape(zerod_stc)[2]

    x = zerod_stc[:,0,:]
    y = zerod_stc[:,1,:]
    fig = plt.figure(figsize = (10,10),facecolor='w')
    ax = fig.add_subplot(111)

    segs = np.zeros((n_cells, n_frames, 2), float)
    segs[:, :, 0] = x
    segs[:, :, 1] = y


    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    ax.set_xlabel('X position ($\mu$m)[uncalibrated]', fontsize =18) #
    ax.set_ylabel('Y position ($\mu$m)[uncalibrated]',fontsize =18)

    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    line_segments = LineCollection(segs,colors=colors, cmap=plt.get_cmap('jet'))

    ax.add_collection(line_segments)
    ax.set_title('Cell migration trajectories',fontsize =22)


    if STATIC_PLOTS:

        plt.savefig(savepath + '2d_hairball.png', format='png', dpi=600)
        print('Using plt.savefig for 2d_hairball, need fig.savefig instead??')

def hairball_3d(zerod_stc, anim=False, savepath = PLOT_OUTPUT):

    '''
    Input:
        zerod_stc

    Output:
        None, draws figure to screen.
    '''

    n_cells = np.shape(zerod_stc)[0]
    n_frames = np.shape(zerod_stc)[2]

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')

    for n in range(0,n_cells):

        x = zerod_stc[n,0,:]
        y = zerod_stc[n,1,:]
        t = np.linspace(0,1,n_frames)#n_frames*T_INC,n_frames)

        ax.plot(x, y, t)#, c=)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)


    if(anim):

        frames = []

        for angle in range(0, 360):

            ax.view_init(30, angle)

            fig.canvas.draw()

            # Convert to numpy array, and append to list
            np_fig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(np_fig)
            imageio.mimsave(savepath + '/3d_hairball.gif', frames)

    else:

        angle = 60
        ax.view_init(30, angle)

    # Draw the figure
    fig.canvas.draw()


def hairball_3d_plotly(stc0d,colorscale='hsv',savepath=PLOT_OUTPUT):

    '''
    Input:
        np.ndarray: Origin-corrected 3d numpy array
        colorscale (optional) : ensures same scale used
                                for points and lines

    Returns:
        fig_data:
         Note: can be visualized normally by using:
             fig = go.Figure(fig_data)
             fig.show()

    '''

    out_df = stc2df(stc0d)

    # fig = go.Figure(
    fig_data=go.Scatter3d(
        x=out_df['X0'], y=out_df['Y0'], z=out_df['t'],
        marker=dict(
            size=3,
            color=out_df['cell'],
            colorscale=colorscale,
        ),
        line=dict(
            color=out_df['cell'],
            colorscale=colorscale,
            width=2
        )
    )

    # fig.show()
    fig = go.Figure(fig_data)

    if STATIC_PLOTS:
        fig.write_image(savepath+'hairball_3d_plotly.png')

    if PLOTS_IN_BROWSER:
        fig.show()


    return fig_data
