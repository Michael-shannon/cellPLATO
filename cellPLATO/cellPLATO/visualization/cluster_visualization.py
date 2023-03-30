#cluster_visualization.py

from initialization.config import *
from initialization.initialization import *

from visualization.low_dimension_visualization import colormap_pcs

import numpy as np
import pandas as pd
import os

import plotly.graph_objs as go


# matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.),
    "figure.figsize":    (10,10),
    "font.size": 12,
    "agg.path.chunksize": 20000
})

import sklearn.preprocessing
import sklearn.pipeline
import scipy.spatial

# DBscan functon for parameter sweeping Eps
def cluster_vis_sns(df_in,eps, cluster_by,min_samples=10):

    '''
    Visualize labels from DBScan in a scatterplot.

    Intended to work with positional or dimension-reduced input data.

    Input:
        df_in: dataframe containing only the columns to be clustered and plotted
        eps: value that influences the clustering.
        cluster_by: str: 'pos', 'tsne', 'pca'
    '''

    if cluster_by == 'pos':
        x_name = 'x'
        y_name = 'y'
    elif cluster_by == 'pca':
        x_name = 'PC1'
        y_name = 'PC2'
    elif cluster_by == 'tsne':
        x_name = 'tSNE1'
        y_name = 'tSNE2'


    #DBScan
    X = StandardScaler().fit_transform(df_in.values)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = ['label'])
    dr_label_df = pd.concat([df_in,lab_df], axis=1)

    g = sns.jointplot(data=dr_label_df[dr_label_df['label']!=-1],x=x_name, y=y_name, hue="label",
                      kind='scatter',
                   palette=palette,
                      joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
    if STATIC_PLOTS:
        plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)

    if PLOTS_IN_BROWSER:
        plt.show()

    plt.clf() # Because it will be run in a loop.

    return plt


def calculate_hull(
        X,
        scale=1.0,
        padding="scale",
        n_interpolate=100,
        interpolation="quadratic_periodic",
        return_hull_points=False):

    """
    Calculates a "smooth" hull around given points in `X`.
    The different settings have different drawbacks but the given defaults work reasonably well.
    Parameters
    ----------
    X : np.ndarray
        2d-array with 2 columns and `n` rows
    scale : float, optional
        padding strength, by default 1.1
    padding : str, optional
        padding mode, by default "scale"
    n_interpolate : int, optional
        number of interpolation points, by default 100
    interpolation : str or callable(ix,iy,x), optional
        interpolation mode, by default "quadratic_periodic"

    From: https://stackoverflow.com/questions/17553035/draw-a-smooth-polygon-around-data-points-in-a-scatter-plot-in-matplotlib

    """

    if padding == "scale":

        # scaling based padding
        scaler = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(with_std=False),
            sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1)))
        points_scaled = scaler.fit_transform(X) * scale
        hull_scaled = scipy.spatial.ConvexHull(points_scaled, incremental=True)
        hull_points_scaled = points_scaled[hull_scaled.vertices]
        hull_points = scaler.inverse_transform(hull_points_scaled)
        hull_points = np.concatenate([hull_points, hull_points[:1]])

    elif padding == "extend" or isinstance(padding, (float, int)):
        # extension based padding
        # TODO: remove?
        if padding == "extend":
            add = (scale - 1) * np.max([
                X[:,0].max() - X[:,0].min(),
                X[:,1].max() - X[:,1].min()])
        else:
            add = padding
        points_added = np.concatenate([
            X + [0,add],
            X - [0,add],
            X + [add, 0],
            X - [add, 0]])
        hull = scipy.spatial.ConvexHull(points_added)
        hull_points = points_added[hull.vertices]
        hull_points = np.concatenate([hull_points, hull_points[:1]])
    else:
        raise ValueError(f"Unknown padding mode: {padding}")

    # number of interpolated points
    nt = np.linspace(0, 1, n_interpolate)

    x, y = hull_points[:,0], hull_points[:,1]

    # ensures the same spacing of points between all hull points
    t = np.zeros(x.shape)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t /= t[-1]

    # interpolation types
    if interpolation is None or interpolation == "linear":
        x2 = scipy.interpolate.interp1d(t, x, kind="linear")(nt)
        y2 = scipy.interpolate.interp1d(t, y, kind="linear")(nt)
    elif interpolation == "quadratic":
        x2 = scipy.interpolate.interp1d(t, x, kind="quadratic")(nt)
        y2 = scipy.interpolate.interp1d(t, y, kind="quadratic")(nt)

    elif interpolation == "quadratic_periodic":
        x2 = scipy.interpolate.splev(nt, scipy.interpolate.splrep(t, x, per=True, k=4))
        y2 = scipy.interpolate.splev(nt, scipy.interpolate.splrep(t, y, per=True, k=4))

    elif interpolation == "cubic":
        x2 = scipy.interpolate.CubicSpline(t, x, bc_type="periodic")(nt)
        y2 = scipy.interpolate.CubicSpline(t, y, bc_type="periodic")(nt)
    else:
        x2 = interpolation(t, x, nt)
        y2 = interpolation(t, y, nt)

    X_hull = np.concatenate([x2.reshape(-1,1), y2.reshape(-1,1)], axis=1)
    if return_hull_points:
        return X_hull, hull_points
    else:
        return X_hull





def draw_cluster_hulls(df_in, cluster_by=CLUSTER_BY, min_pts=5, color_by='cluster',cluster_label='label',ax=None,draw_pts=False,save_path=CLUST_DIR, legend=False):

    df = df_in.copy()

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'

    elif cluster_by == 'pca':
        x_name = 'PC1'
        y_name = 'PC2'

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):

        x_name = 'tSNE1'
        y_name = 'tSNE2'

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):

        x_name = 'UMAP1'
        y_name = 'UMAP2'

    labels = list(set(df[cluster_label].unique()))

    # conditions = df['Cond_label'].unique()
    # colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(conditions)))
    # print(conditions, colors)

    catcol = 'Cond_label'
    categories = np.unique(df[catcol])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))
    df["Color"] = df[catcol].apply(lambda x: colordict[x])

    # Define a list of cluster colors to be consistent across the module

    cluster_colors = []
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))


    # If no axis is supplied, then createa simple fig, ax and default to drawing the points.
    if ax is None:

        fig, ax = plt.subplots()
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

        all_pts = df[[x_name, y_name]].values
        draw_pts = True
        ax.scatter(x=all_pts[:,0], y=all_pts[:,1], s=0.1, c='grey')

    if(color_by=='condition' and draw_pts):
        # Draw the scatter points for the current cluster
        scatter = ax.scatter(x=df[x_name], y=df[y_name], s=0.3, c=df['Color'],cmap=CONDITION_CMAP)
        # scatter = ax.scatter(x=df[x_name], y=df[y_name], s=0.3, c=df['Condition_shortlabel'].values,cmap=CONDITION_CMAP)

        if(DEBUG):
            print(list(df['Condition_shortlabel'].unique()))
            print(scatter.legend_elements())
            print(list(df['Color'].unique()))
            print(categories)

        ''' Note: for some reason we cant get the conditions labels correctly on the legend.'''
        legend1 = ax.legend(*scatter.legend_elements(),labels=list(df['Condition_shortlabel'].unique()),
                            loc="upper right", title="Condition")


        ax.add_artist(legend1)

    elif(color_by=='PCs' and draw_pts):
        pc_colors = colormap_pcs(df, cmap='rgb')
        # pc_colors = np.asarray(df[['PC1','PC2','PC3']])
        # scaler = MinMaxScaler()
        # scaler.fit(pc_colors)
        # pc_colors= scaler.transform(pc_colors)

        # Color code the scatter points by their respective PC 1-3 values
        ax.scatter(x=df[x_name], y=df[y_name], s=0.5, c=pc_colors)

    for i_lab,curr_label in enumerate(labels[:-2]):


        curr_lab_df = df[df[cluster_label] == curr_label]
        curr_lab_pts = curr_lab_df[[x_name, y_name]].values

        if curr_lab_pts.shape[0] > min_pts:

            x=curr_lab_pts[:,0]
            y=curr_lab_pts[:,1]

            if(color_by=='cluster' and draw_pts):
                '''
                Having this color_by block within the cluster loop makes sure that the
                points within the cluster have the same default color order as the cluster hulls
                drawn below in the same loop.

                Ideally it would be possible to apply a custom colormap to the clusters, apply them to
                the scatter plots, and use them in other plots to make the link.
                '''

                # Draw the scatter points for the current cluster
                ax.scatter(x=x, y=y, s=0.3, color=cluster_colors[i_lab])

            # Catch cases where we can't draw the hull because the interpolation fails.
            try:
                X_hull  = calculate_hull(
                    curr_lab_pts,
                    scale=1.0,
                    padding="scale",
                    n_interpolate=100,
                    interpolation="quadratic_periodic")

                ax.plot(X_hull[:,0], X_hull[:,1],c=cluster_colors[i_lab], label=curr_label)
                if(legend):
                    ax.legend()
            except:
                print('Failed to draw cluster, failed to draw cluster: ',curr_label, ' with shape: ', curr_lab_pts.shape)


    if STATIC_PLOTS:
        plt.savefig(save_path+'clusterhull_scatter_'+cluster_by+'.png')
    if PLOTS_IN_BROWSER:
        plt.show()


    return ax

def plot_3D_scatter_deprecated(df, x, y, z, colorby, ticks=False, identifier='', dotsize = 3, alpha=0.2, markerscale=5): #new matplotlib version of scatter plot for umap 1-26-2023
    import matplotlib.pyplot as plt
    from numpy.random import random
    from mpl_toolkits.mplot3d import Axes3D

    font_size = 24
    # df = lab_dr_df

    fig = plt.figure(figsize=(15, 15))

    if colorby == 'label':
        coloredby = 'label'
        colors = cm.Dark2(np.linspace(0, 1, len(df['label'].unique())))  
        df = df.sort_values(by=['label'])
          
    elif colorby == 'condition':
        coloredby = 'Condition_shortlabel'
        colors = cm.Dark2(np.linspace(0, 1, len(df['Condition_shortlabel'].unique())))

    ax = plt.subplot(111, projection='3d')
     
    for colorselector in range(len(colors)): #you have to plot each label separately to get the legend to work
        if colorby == 'label':
            ax.scatter(df[df['label'] == df['label'].unique()[colorselector]][x], df[df['label'] == df['label'].unique()[colorselector]][y],
             df[df['label'] == df['label'].unique()[colorselector]][z], '.', color=colors[colorselector], label = df['label'].unique()[colorselector], s=dotsize, alpha = alpha)
        elif colorby == 'condition':
            ax.scatter(df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][x], df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][y],
             df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][z], '.', color=colors[colorselector], label = df['Condition_shortlabel'].unique()[colorselector], s=dotsize, alpha = alpha)

    leg=plt.legend(loc='upper left', numpoints=1, ncol=1, fontsize=font_size, bbox_to_anchor=(1.05, 1.0), markerscale=markerscale)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    
    # plt.tight_layout()
    
    # Set the axis labels with or without ticks
    if ticks == False:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel(x, fontsize=font_size, linespacing=3.2) # gives a new line to make space for the axis label
        ax.set_ylabel(y, fontsize=font_size, linespacing=3.2)
        ax.set_zlabel(z, fontsize=font_size, linespacing=3.2)
    elif ticks == True:
        # Set the axis labels
        ax.set_xlabel('\n ' + x, fontsize=font_size, linespacing=3.2) # gives a new line to make space for the axis label
        ax.set_ylabel('\n ' + y, fontsize=font_size, linespacing=3.2)
        ax.set_zlabel('\n ' + z, fontsize=font_size, linespacing=3.2)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    # set the axis limits to tight
    ax.set_xlim3d(np.min(df[x]), np.max(df[x]))
    ax.set_ylim3d(np.min(df[y]), np.max(df[y]))
    ax.set_zlim3d(np.min(df[z]), np.max(df[z]))

    plt.show()

    fig.savefig(CLUST_DISAMBIG_DIR+identifier+'3D_scatter.png', dpi=300, bbox_inches='tight')

    return ax    

def plot_3D_scatter(df, x, y, z, colorby, ticks=False, identifier='', dotsize = 3, alpha=0.2, markerscale=5): #new matplotlib version of scatter plot for umap 1-26-2023
    import matplotlib.pyplot as plt
    from numpy.random import random
    from mpl_toolkits.mplot3d import Axes3D

    font_size = 24
    # df = lab_dr_df

    fig = plt.figure(figsize=(15, 15))

    # colors=[]
    # cmap = cm.get_cmap(CONDITION_CMAP, len(df_in['Condition_shortlabel'].unique()))
    # for i in range(cmap.N):
    #     colors.append(cmap(i))

    if colorby == 'label':
        coloredby = 'label'
        # colors = cm.Dark2(np.linspace(0, 1, len(df['label'].unique())))  
        cmap = cm.get_cmap(CLUSTER_CMAP, len(df['label'].unique()))
        colors=[]
        for i in range(cmap.N):
            colors.append(cmap(i))
        df = df.sort_values(by=['label'])
          
    elif colorby == 'condition':
        coloredby = 'Condition_shortlabel'
        # colors = cm.Dark2(np.linspace(0, 1, len(df['Condition_shortlabel'].unique())))
        # cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique())) # deleted...
        # colors=[] # deleted...
        # for i in range(cmap.N): # deleted
        #     colors.append(cmap(i)) # Deleted...

        colors=[]
        if CONDITION_CMAP != 'Dark24':
            cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
            for i in range(cmap.N):
                colors.append(cmap(i))
        else:
            colors = plotlytomatplotlibcolors()
            colors=colors[:len(df['Condition_shortlabel'].unique())]    



    ax = plt.subplot(111, projection='3d')
     
    for colorselector in range(len(colors)): #you have to plot each label separately to get the legend to work
        if colorby == 'label':
            ax.scatter(df[df['label'] == df['label'].unique()[colorselector]][x], df[df['label'] == df['label'].unique()[colorselector]][y],
             df[df['label'] == df['label'].unique()[colorselector]][z], '.', color=colors[colorselector], label = df['label'].unique()[colorselector], s=dotsize, alpha = alpha)
        elif colorby == 'condition':
            ax.scatter(df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][x], df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][y],
             df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][z], '.', color=colors[colorselector], label = df['Condition_shortlabel'].unique()[colorselector], s=dotsize, alpha = alpha)

    leg=plt.legend(loc='upper left', numpoints=1, ncol=1, fontsize=font_size, bbox_to_anchor=(1.05, 1.0), markerscale=markerscale)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    
    # plt.tight_layout()
    
    # Set the axis labels with or without ticks
    if ticks == False:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel(x, fontsize=font_size, linespacing=3.2) # gives a new line to make space for the axis label
        ax.set_ylabel(y, fontsize=font_size, linespacing=3.2)
        ax.set_zlabel(z, fontsize=font_size, linespacing=3.2)
    elif ticks == True:
        # Set the axis labels
        ax.set_xlabel('\n ' + x, fontsize=font_size, linespacing=3.2) # gives a new line to make space for the axis label
        ax.set_ylabel('\n ' + y, fontsize=font_size, linespacing=3.2)
        ax.set_zlabel('\n ' + z, fontsize=font_size, linespacing=3.2)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    # set the axis limits to tight
    ax.set_xlim3d(np.min(df[x]), np.max(df[x]))
    ax.set_ylim3d(np.min(df[y]), np.max(df[y]))
    ax.set_zlim3d(np.min(df[z]), np.max(df[z]))

    plt.show()

    fig.savefig(CLUST_DISAMBIG_DIR+identifier+'3D_scatter.png', dpi=300, bbox_inches='tight')

    return ax    

def plot_3D_UMAP(df, colorby = 'label', symbolby = 'Condition_shortlabel', what = ''):

    import plotly.io as pio
    import seaborn as sns
    import plotly.express as px


    if colorby == 'Condition_shortlabel':

        if CONDITION_CMAP != 'Dark24':
            pal = sns.color_palette(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
            # pal = sns.color_palette(CONDITION_CMAP) #extracts a colormap from the seaborn stuff.
            cmap=pal.as_hex()[:] #outputs that as a hexmap which is compatible with plotlyexpress below
        else:
            cmap = px.colors.qualitative.Dark24
            # colors=colors[:len(df_in['Condition_shortlabel'].unique())]
            # cmap=cmap[:len(df['Condition_shortlabel'].unique())] 
    else:
        if CLUSTER_CMAP != 'Dark24':
            pal = sns.color_palette(CLUSTER_CMAP, len(df['Condition_shortlabel'].unique()))
            cmap=pal.as_hex()[:] #outputs that as a hexmap which is compatible with plotlyexpress below
        else: 
            cmap = px.colors.qualitative.Dark24
    print(cmap)
    if 'label' in df.columns:
        df['label'] = pd.Categorical(df.label)


    # import plotly.express as px
    # df = px.data.iris()
    fig = px.scatter_3d(df, x='UMAP1', y='UMAP2', z='UMAP3',
                  color=colorby, #Condition_shortlabel
                        symbol=symbolby,
                        size_max=1, opacity=0.7,width=1800, height=1200, color_discrete_sequence=cmap) # 1) Increase fontsize of labels

    fig.update_traces(marker_size = 2)#2
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),legend_font_size=24)
    # fig.update_layout(legend=dict(title_font_family="Times New Roman", font=dict(size= 20)) )

    fig.update_layout(font=dict(family="Courier New, monospace",size=24)) #Courier New, monospace
    fig.update_layout(legend= {'itemsizing': 'constant'})

    # "Arial", "Balto", "Courier New", "Droid Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas One", "Old Standard TT", "Open Sans", "Overpass", "PT Sans Narrow", "Raleway", "Times New Roman".

    pio.write_image(fig, CLUST_DISAMBIG_DIR + what + ' UMAP_Clusters.png',scale=1, width=1800, height=1200)
    fig.show()
    return

def interactive_umap_plot_choosecondition(df, condition):
    # filter dataframe for the chosen condition
    df_condition = df[df['Condition_shortlabel'] == condition]

    # create trace for all data points in grey
    trace_all = go.Scatter3d(
        x=df['UMAP1'],
        y=df['UMAP2'],
        z=df['UMAP3'],
        mode='markers',
        marker=dict(
            size=2,
            color='grey',
            opacity=0.2
        ),
        name='Other Conditions'
    )

    # create trace for chosen condition in a different color
    trace_condition = go.Scatter3d(
        x=df_condition['UMAP1'],
        y=df_condition['UMAP2'],
        z=df_condition['UMAP3'],
        mode='markers',
        marker=dict(
            size=2,
            color='#ff7f0e',
            opacity=1
        ),
        name=f'Condition: {condition}'
    )

    # create data list to pass to plotly figure
    data = [trace_all, trace_condition]

    # create layout for plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='UMAP1'),
            yaxis=dict(title='UMAP2'),
            zaxis=dict(title='UMAP3')
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # create figure object
    fig = go.Figure(data=data, layout=layout)
    #change figure size to 800x800
    fig.update_layout(
        width=1800,
        height=1200,
        autosize=False,
        margin=dict(l=20, r=20, t=20, b=20,),
        legend_font_size=24,
        legend= {'itemsizing': 'constant'},
        font=dict(family="Courier New, monospace",size=24),
    )

    # show plot
    fig.show()



def plot_plasticity_changes(df, identifier='\_allcells', miny=None, maxy=None):

    # f, axes = plt.subplots(1, 3, figsize=(15, 5)) #sharex=True
    # f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=True) #sharex=True
    f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=False) #sharex=True

    whattoplot=['label','twind_n_changes', 'twind_n_labels']

    CLUSTER_CMAP = 'tab20'
    # CONDITION_CMAP = 'dark'

    time = df['frame']
    # SAMPLING_INTERVAL=10/60 #This shouldn't be hardcoded!
    timeminutes=time*SAMPLING_INTERVAL

    # dfnumericals = df.select_dtypes('number')

    # extracted_col = df["Condition_shortlabel"]

    # df=dfnumericals.join(extracted_col)

    ##
    if miny != None or maxy != None:
        minimumy=miny
        maximumy1=maxy
        maximumy2=maxy
        maximumy3=maxy
    else:
        minimumy=0
        maximumy1=np.nanmax(df[whattoplot[0]])
        maximumy2=np.nanmax(df[whattoplot[1]])
        maximumy3=np.nanmax(df[whattoplot[2]])

    ##
    import seaborn as sns
    sns.set_theme(style="ticks")
    sns.set_palette(CONDITION_CMAP)

    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
    for i in range(cmap.N):
        colors.append(cmap(i))

    # display(df)
    df=df.dropna(how='any')
    # display(df)
    # Plot the responses for different events and regions
    sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
                 hue="Condition_shortlabel",
                 data=df, palette=colors)

    sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
                 hue="Condition_shortlabel",
                 data=df, palette=colors)

    sns.lineplot(ax=axes[2], x=timeminutes, y=whattoplot[2], #n_labels #n_changes #label
                 hue="Condition_shortlabel",
                 data=df, palette=colors)

    timewindowmins = MIG_T_WIND*SAMPLING_INTERVAL
    text1 = "Cluster ID"
    text2 = "Cluster switches (per " + str(timewindowmins) + " min)"
    text3 = "Unseen cluster switches (per " + str(timewindowmins) + " min)"

    x_lab = "Distinct Behaviors"
    plottitle = ""

    the_yticks = np.arange(0, len(df[whattoplot[0]].unique()), 1)
    the_yticks = [int(x) for x in the_yticks]
    axes[0].set_yticks(the_yticks) # set new tick positions
    axes[0].margins(y=0) # set tight margins
    the_yticks = np.arange(0, len(df[whattoplot[1]].unique()), 1)
    the_yticks = [int(x) for x in the_yticks]
    axes[1].set_yticks(the_yticks) # set new tick positions
    axes[1].margins(y=0) # set tight margins
    the_yticks = np.arange(0, len(df[whattoplot[2]].unique()), 1)
    the_yticks = [int(x) for x in the_yticks]
    axes[2].set_yticks(the_yticks) # set new tick positions
    axes[2].margins(y=0) # set tight margins

    # the_xticks = np.arange(0, len(timeminutes), 1)
    # the_xticks = [int(x) for x in the_xticks]
    # axes[0].set_xticks(the_xticks) # set new tick positions
    # axes[0].margins(x=0) # set tight margins
    # axes[1].set_xticks(the_xticks) # set new tick positions
    # axes[1].margins(x=0) # set tight margins
    # axes[2].set_xticks(the_xticks) # set new tick positions
    # axes[2].margins(x=0) # set tight margins


    # Tweak the visual presentation
    axes[0].xaxis.grid(True)
    axes[1].xaxis.grid(True)
    axes[2].xaxis.grid(True)

    # axes[0].set_ylabel(whattoplot[0], fontsize=36)
    # axes[1].set_ylabel(whattoplot[1], fontsize=36)
    # axes[2].set_ylabel(whattoplot[2], fontsize=36)
    axes[0].set_ylabel(text1, fontsize=36)
    axes[1].set_ylabel(text2, fontsize=36)
    axes[2].set_ylabel(text3, fontsize=36)

    axes[0].set_title("", fontsize=36)
    axes[1].set_title("", fontsize=36)
    axes[2].set_title("", fontsize=36)

    axes[0].set_xlabel("Time (min)", fontsize=36)
    axes[1].set_xlabel("Time (min)", fontsize=36)
    axes[2].set_xlabel("Time (min)", fontsize=36)

    # axes[0].set_ylim(0, np.nanmax(df[whattoplot[0]]))
    # axes[1].set_ylim(0, np.nanmax(df[whattoplot[1]]))
    # axes[2].set_ylim(0, np.nanmax(df[whattoplot[2]]))
    axes[0].set_ylim(0, maximumy1)
    axes[1].set_ylim(0, maximumy2)
    axes[2].set_ylim(0, maximumy3)


    

    # ax.set_ylabel(y_lab, fontsize=36)
    axes[0].tick_params(axis='both', labelsize=36)
    axes[1].tick_params(axis='both', labelsize=36)
    axes[2].tick_params(axis='both', labelsize=36)

    axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
    axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
    axes[2].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)

    f.tight_layout()
    # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
    f.savefig(CLUST_DISAMBIG_DIR+identifier+'_plasticity_cluster_changes_over_time.png', dpi=300)#plt.

    return


def plot_UMAP_subplots_coloredbymetrics(df_in, x= 'UMAP1', y= 'UMAP2', z = 'UMAP3', n_cols = 5, ticks=False, metrics = ALL_FACTORS, scalingmethod='log2minmax', identifier='', colormap='viridis'): #new matplotlib version of scatter plot for umap 1-26-2023
    import matplotlib.pyplot as plt
    from numpy.random import random
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PowerTransformer
    from matplotlib.gridspec import GridSpec

    font_size = 20
   
    # Scale the data
    sub_set = df_in[metrics]
    Z = sub_set.values

    if scalingmethod == 'minmax':
        x_=MinMaxScaler().fit_transform(Z)
        scaled_df = pd.DataFrame(data=x_, columns = metrics) 
        df_out = pd.concat([scaled_df,df_in[[x,y,z]]], axis=1)
        # fresh_df = pd.concat([df,lab_list_tpt_df['tavg_label']], axis=1)
        
    elif scalingmethod == 'log2minmax':
        
        negative_FACTORS = []
        positive_FACTORS = []
        for factor in metrics:
            if np.min(df_in[factor]) < 0:
                print('factor ' + factor + ' has quite a few negative values')
                negative_FACTORS.append(factor)
                    
            else:
                print('factor ' + factor + ' has no negative values')
                positive_FACTORS.append(factor)
            
            
        pos_df = df_in[positive_FACTORS]
        pos_x = pos_df.values
        neg_df = df_in[negative_FACTORS]
        neg_x = neg_df.values

        if len(neg_x[0]) == 0: #This controls for an edge case in which there are no negative factors - must be implemented in the other transforms as well (pipelines and clustering)
            print('No negative factors at all!')
            neg_x_ = neg_x
        else:
            neg_x_ = MinMaxScaler().fit_transform(neg_x) 

        pos_x_constant = pos_x + 0.000001
        pos_x_log = np.log2(pos_x + pos_x_constant)
        pos_x_ = MinMaxScaler().fit_transform(pos_x_log)
        x_ = np.concatenate((pos_x_, neg_x_), axis=1)
        newcols=positive_FACTORS + negative_FACTORS

        scaled_df = pd.DataFrame(x_, columns = newcols)
        df_out = pd.concat([scaled_df,df_in[[x,y,z]]], axis=1)


    elif scalingmethod == 'powertransformer':    
        
        pt = PowerTransformer(method='yeo-johnson')
        x_ = pt.fit_transform(Z)
        scaled_df = pd.DataFrame(data=x_, columns = metrics) 
        df_out = pd.concat([scaled_df,df_in[[x,y,z]]], axis=1)

    # do not forget constrained_layout=True to have some space between axes
    fig = plt.figure(constrained_layout=True, figsize=(20, 20))
    # Define the grid of subplots - you can change the number of columns and this changes the rows accordingly
    n_metrics = len(metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    print('Length of metrics is ', n_metrics)

    gs = GridSpec(n_rows, n_cols, figure=fig)

    df_out = df_out.reset_index(drop=True)
    # define the axes in a for loop according to the grid
    for number, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[number], projection='3d') # Initializes each plot
        g = ax.scatter(df_out[x], df_out[y], df_out[z], '.', c=df_out[metric], cmap=colormap, s=0.5, alpha = 0.5) # Makes each plot
        metric_name = metric.replace('_', ' ') #removes the underscore from the metric name
        ax.set_title(metric_name, fontsize=font_size)

    fig.colorbar(g, shrink=0.5)
    fig.savefig(CLUST_DISAMBIG_DIR+identifier+'UMAP_subplots.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return     

def plot_plasticity_countplots_deprecated(df, identifier='\_allcells'):

    import matplotlib.pyplot as plt
    import seaborn as sns

    # f, axes = plt.subplots(1, 2, figsize=(30, 15), sharey=True)
    f, axes = plt.subplots(1, 2, figsize=(30, 8), sharey=False)
    # f, axes = plt.subplots(2, 1, figsize=(15, 30), sharey=False)
    # ax.set_xscale("log")

    # whattoplot=['n_changes', 'n_labels'] #'twind_n_changes', 'twind_n_labels'
    whattoplot=['twind_n_changes', 'twind_n_labels']
    # CLUSTER_CMAP = 'tab20'
    # CONDITION_CMAP = 'dark'

    sns.set_theme(style="ticks")
    sns.set_palette(CONDITION_CMAP)


    x_lab = whattoplot
    plottitle = ""
    # Plot the orbital period with horizontal boxes
    sns.countplot(x=whattoplot[0], hue="Condition_shortlabel", data=df, ax=axes[0], orient='h')
    sns.countplot(x=whattoplot[1], hue="Condition_shortlabel", data=df, ax=axes[1], orient='h')
    ######
    # sns.boxplot(ax=axes[0], x=whattoplot[0], y="Condition", data=df,
    #             whis=[0, 100], width=.6)#palette="vlag"
    # # Add in points to show each observation
    # sns.stripplot(ax=axes[0], x=whattoplot[0], y="Condition", data=df,
    #               size=5, color=".3", linewidth=0, jitter=True)
    #
    # sns.boxplot(ax=axes[1], x=whattoplot[1], y="Condition", data=df,
    #             whis=[0, 100], width=.6)#palette="vlag"
    # # Add in points to show each observation
    # sns.stripplot(ax=axes[1], x=whattoplot[1], y="Condition", data=df,
    #               size=5, color=".3", linewidth=0, jitter=True) #color=".3",
    #########
    timewindowmins = MIG_T_WIND*SAMPLING_INTERVAL
    text1 = "Distinct changes per " + str(int(timewindowmins)) + " min time window"
    text2 = "New cluster changes per " + str(int(timewindowmins)) + " min time window"

    axes[0].xaxis.grid(True)
    axes[1].xaxis.grid(True)
    axes[0].set_ylabel("Frequency",fontsize=36)
    axes[1].set_ylabel("Frequency",fontsize=36)
    # axes[0].set_xlabel(whattoplot[0], fontsize=36)
    # axes[1].set_xlabel(whattoplot[1], fontsize=36)
    axes[0].set_xlabel(text1, fontsize=36)
    axes[1].set_xlabel(text2, fontsize=36)
    axes[0].set_title("", fontsize=36)
    axes[1].set_title("", fontsize=36)
    axes[0].tick_params(axis='both', labelsize=36)
    axes[1].tick_params(axis='both', labelsize=36)

    axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper right',fontsize=36,markerscale=20,fancybox=True)
    axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper right',fontsize=36,markerscale=20,fancybox=True)

    f.tight_layout()
    # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
    f.savefig(CLUST_DISAMBIG_DIR+identifier+'plasticity_sum_of_cluster_changes.png', dpi=300)#plt.
    return


##################################
##################################


def plot_plasticity_countplots(df, identifier='\_allcells'):

    import matplotlib.pyplot as plt
    import seaborn as sns

    font_size = 36
    i=0
    labelchanges = df['twind_n_changes'].unique() #IMportantly computed for the whole set.
    label_counts_df = pd.DataFrame(columns=['Condition', 'Replicate_ID', 'twind_n_changes', 'count', 'percent'])
    cond_list = df['Condition_shortlabel'].unique()

    for cond in cond_list:
            this_cond_df = df[df['Condition_shortlabel'] == cond]
            totalforthiscondition=len(this_cond_df.index)
            for labelchange in labelchanges:
                 # Keep this dataframe being made for when we want to look at distributions
                this_lab_df = this_cond_df[this_cond_df['twind_n_changes'] == labelchange]                
                fraction_in_label = len(this_lab_df.index)/totalforthiscondition
                percent_in_label = fraction_in_label*100
                label_counts_df.loc[i] = [cond, 'NA', labelchange, len(this_lab_df.index), percent_in_label]
                i+=1
    label_counts_df.dropna(inplace=True)
    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
    for i in range(cmap.N):
        colors.append(cmap(i))

    sub_df = df[['Condition_shortlabel', 'twind_n_changes', 'twind_n_labels']]    
    
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle("Cluster switching", fontsize=font_size)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1= sns.histplot(data=sub_df, x='twind_n_changes', hue='Condition_shortlabel',
     multiple='dodge', palette=colors, shrink=.8, stat='percent', common_norm=False, 
     discrete=True, binwidth=1,
      element='bars', legend=True)
    the_xticks = np.arange(0, len(df['twind_n_changes'].unique())-1, 1)
    xticks = the_xticks
    ax1.set_xticks(xticks) # set new tick positions
    # ax1.tick_params(axis='x', rotation=30) # set tick rotation
    ax1.margins(x=0) # set tight margins

    ax2 = fig.add_subplot(1, 2, 2)
    ax2= sns.histplot(data=sub_df, x='twind_n_labels', hue='Condition_shortlabel',
        multiple='dodge', palette=colors, shrink=.8, stat='percent', common_norm=False, 
        discrete=True, binwidth=1,
        element='bars', legend=True)
    the_xticks = np.arange(0, len(df['twind_n_labels'].unique())-1, 1)
    xticks = the_xticks
    ax2.set_xticks(xticks) # set new tick positions
    # ax1.tick_params(axis='x', rotation=30) # set tick rotation
    ax2.margins(x=0) # set tight margins    

    timewindowmins = MIG_T_WIND*SAMPLING_INTERVAL

    text1 = "Cluster switches (per " + str(int(timewindowmins)) + " min)"
    text2 = "Unseen cluster switches (per " + str(int(timewindowmins)) + " min)"

    ax1.xaxis.grid(True)
    ax2.xaxis.grid(True)
    ax1.set_ylabel("Percent cells",fontsize=font_size)
    ax2.set_ylabel("Percent cells",fontsize=font_size)
    ax1.set_xlabel(text1, fontsize=font_size)
    ax2.set_xlabel(text2, fontsize=font_size)
    ax1.set_title("", fontsize=font_size)
    ax1.tick_params(axis='both', labelsize=font_size)
    ax2.tick_params(axis='both', labelsize=font_size)
    # f.tight_layout()
    # # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
    fig.savefig(CLUST_DISAMBIG_DIR+identifier+'plasticity_sum_of_cluster_changes.png', dpi=300)#plt.
    return label_counts_df

##################################
##################################


def purity_plots(lab_dr_df, clust_sum_df,traj_clust_df,trajclust_sum_df,cluster_by=CLUSTER_BY, save_path=CLUST_DIR ):


    if(cluster_by == 'tsne' or cluster_by == 'tSNE'):

        x_name = 'tSNE1'
        y_name = 'tSNE2'


    elif cluster_by == 'umap':

        x_name = 'UMAP1'
        y_name = 'UMAP2'




    # Create a Subplot figure that shows the effect of clustering between conditions

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=[20,20])

    fig.suptitle("Low dimensional cluster analysis and purity", fontsize='x-large')

    #
    # First subplot: Scatter of lowD space with cluster outlines
    #

    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y_name)
    ax1.set_title('Low-dimensional scatter with cluster outlines', fontsize=18)
    # ax1.scatter(x=lab_dr_df['tSNE1'], y=lab_dr_df['tSNE2'], c='grey', s=0.1)
    draw_cluster_hulls(lab_dr_df,cluster_by=cluster_by, color_by='condition',ax=ax1, draw_pts=True,legend=True)

    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP, len(lab_dr_df['Condition_shortlabel'].unique()))
    for i in range(cmap.N):
        colors.append(cmap(i))

    # Define a custom colormap for the clusters (To match what's done in draw_cluster_hulls())
    cluster_colors = []
    labels = list(set(lab_dr_df['label'].unique()))
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))



    #
    # Second subplot: stacked bar plots of cluster purity
    #
    ax2.set_title('Low-dimension cluster purity', fontsize=18)

    for i, cond in enumerate(lab_dr_df['Condition_shortlabel'].unique()):
        '''
        Assumes that the conditions are already in the correct order in the dataframe.
        '''
        if(i==0):
            ax2.bar(clust_sum_df['cluster_id'], clust_sum_df[cond+'_ncells_%'], label=cond,color=colors[i])
            prev_bottom = clust_sum_df[cond+'_ncells_%']
        else:
            ax2.bar(clust_sum_df['cluster_id'], clust_sum_df[cond+'_ncells_%'], bottom=prev_bottom, label=cond,color=colors[i])

    ax2.set_xticks(clust_sum_df['cluster_id'])
    ax2.set_ylabel('% of cells per cluster')
    ax2.legend()


    for ticklabel, tickcolor in zip(ax2.get_xticklabels(), cluster_colors):
        ticklabel.set_color(tickcolor)





    #
    # Thirds subplot: Trajectories through lowD space
    #

    # Define a custom colormap for the trajectory clusters (To match what's done in draw_cluster_hulls())
    cluster_colors = []
    labels = list(set(traj_clust_df['traj_id'].unique()))
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))

    ax3.set_title('Cell trajectory clustering through low-dimensional space', fontsize=18)
    ax3.scatter(x=traj_clust_df['UMAP1'], y=traj_clust_df['UMAP2'], s=0.5,alpha=0.5,
            c='gray')
    for i, traj in enumerate(labels[:-2]): # Same as when drawing contours
        traj_sub_df = traj_clust_df[traj_clust_df['traj_id'] == traj]
#         display(traj_sub_df)
        # Draw lines and scatters individually for each label
        ax3.plot(traj_sub_df[x_name], traj_sub_df[y_name], alpha=1, c=cluster_colors[i],linewidth=0.1)
        ax3.scatter(x=traj_sub_df[x_name], y=traj_sub_df[y_name], s=0.8,alpha=0.5,
            color=cluster_colors[i])

    '''Cannot colormap lines explicitly with matplotlib.
        If you need to the lines colormapped, they would have to be iterated and drawn with discrete colors.
    '''
#     ax3.plot(traj_clust_df['UMAP1'], traj_clust_df['UMAP2'], alpha=0.5, c='gray',linewidth=0.1)




    ax3.set_xlabel(x_name)
    ax3.set_ylabel(y_name)
    # Alt: Colormap by cluster, using custom map. This will include the unlabeled ones
#     ax3.scatter(x=traj_clust_df['UMAP1'], y=traj_clust_df['UMAP2'], s=1,alpha=0.9,
#                 c=traj_clust_df['traj_id'], cmap= CLUSTER_CMAP)

    #
    # ALT: Using datashader for 3rd plot.
    #
    # traj_clust_df = trajectory_clustering_pipeline(std_dr_df, traj_factor='umap', dist_metric='hausdorff', filename_out='std_dr_df_traj_')
    # datashader_lines(traj_clust_df, 'UMAP1', 'UMAP2',color_by='traj_id', categorical=True)
    # ax3.set_facecolor('#000000')

    # cvs = ds.Canvas()#,x_range=x_range, y_range=y_range)
    # artist = dsshow(cvs.line(traj_clust_df,'UMAP1', 'UMAP2', agg=ds.count_cat('Cat')),ax=ax3)

    # Datashader for 3rd panel (points only working, not lines)
    # ax3.set_facecolor('#000000')
    # traj_clust_df['Cat'] = traj_clust_df['Condition'].astype('category')
    # artist = dsshow(
    #     traj_clust_df,
    #     ds.Point('UMAP1', 'UMAP2'),
    #     ds.count_cat('Cat'),
    # #     norm='log',
    #     ax=ax3
    # )
    #


    draw_cluster_hulls(traj_clust_df,cluster_by=cluster_by, cluster_label='traj_id', ax=ax3, color_by='cluster',legend=True)

    #
    # Fourth subplot: Trajectories cluster purity
    #

    ax4.set_title('Cell trajectory cluster purity', fontsize=18)

    for i, cond in enumerate(lab_dr_df['Condition_shortlabel'].unique()):
        '''
        Assumes that the conditions are already in the correct order in the dataframe.
        '''
        if(i==0):
            ax4.bar(trajclust_sum_df['cluster_id'], trajclust_sum_df[cond+'_ncells_%'], label=cond,color=colors[i])
            prev_bottom = trajclust_sum_df[cond+'_ncells_%']
        else:
            ax4.bar(trajclust_sum_df['cluster_id'], trajclust_sum_df[cond+'_ncells_%'], bottom=prev_bottom, label=cond,color=colors[i])

    ax4.set_xticks(trajclust_sum_df['cluster_id'])
    ax4.set_ylabel('Number of cells per trajectory cluster')
    ax4.legend()

    for ticklabel, tickcolor in zip(ax4.get_xticklabels(), cluster_colors):
        ticklabel.set_color(tickcolor)

    if STATIC_PLOTS:
        plt.savefig(save_path+'purity_plots'+cluster_by+'.png')
    if PLOTS_IN_BROWSER:
        plt.show()



    return fig


######################################################

def draw_cluster_hulls_dev(df_in, cluster_by='umap', min_pts=5, color_by='cluster',cluster_label='label',ax=None,draw_pts=False,save_path=CLUST_DIR, legend=False):

    df = df_in.copy()

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'

    elif cluster_by == 'pca':
        x_name = 'PC1'
        y_name = 'PC2'

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):

        x_name = 'tSNE1'
        y_name = 'tSNE2'

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):

        x_name = 'UMAP1'
        y_name = 'UMAP2'
        z_name = 'UMAP3'

    labels = list(set(df[cluster_label].unique()))

    # conditions = df['Cond_label'].unique()
    # colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(conditions)))
    # print(conditions, colors)

    catcol = 'Cond_label'
    categories = np.unique(df[catcol])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))
    df["Color"] = df[catcol].apply(lambda x: colordict[x])

    # Define a list of cluster colors to be consistent across the module

    cluster_colors = []
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))


    # # If no axis is supplied, then createa simple fig, ax and default to drawing the points.
    # if ax is None:

    #     fig, ax = plt.subplots()
    #     ax.set_xlabel(x_name)
    #     ax.set_ylabel(y_name)

    #     all_pts = df[[x_name, y_name]].values
    #     draw_pts = True
    #     ax.scatter(x=all_pts[:,0], y=all_pts[:,1], s=0.1, c='grey')

    # if(color_by=='condition' and draw_pts):
    #     # Draw the scatter points for the current cluster
    #     # ax = plt.subplot(111, projection='3d')
    #     scatter = ax.scatter(x=df[x_name], y=df[y_name], s=0.3, c=df['Color'],cmap=CONDITION_CMAP)
    #     # scatter = ax.scatter(x=df[x_name], y=df[y_name], s=0.3, c=df['Condition_shortlabel'].values,cmap=CONDITION_CMAP)

    #     if(DEBUG):
    #         print(list(df['Condition_shortlabel'].unique()))
    #         print(scatter.legend_elements())
    #         print(list(df['Color'].unique()))
    #         print(categories)

    #     ''' Note: for some reason we cant get the conditions labels correctly on the legend.'''
    #     legend1 = ax.legend(*scatter.legend_elements(),labels=list(df['Condition_shortlabel'].unique()),
    #                         loc="upper right", title="Condition")


    #     ax.add_artist(legend1)

    # elif(color_by=='PCs' and draw_pts):
    #     pc_colors = colormap_pcs(df, cmap='rgb')
    #     # pc_colors = np.asarray(df[['PC1','PC2','PC3']])
    #     # scaler = MinMaxScaler()
    #     # scaler.fit(pc_colors)
    #     # pc_colors= scaler.transform(pc_colors)

    #     # Color code the scatter points by their respective PC 1-3 values
    #     ax.scatter(x=df[x_name], y=df[y_name], s=0.5, c=pc_colors)

    for i_lab,curr_label in enumerate(labels[:-2]):


        curr_lab_df = df[df[cluster_label] == curr_label]
        curr_lab_pts = curr_lab_df[[x_name, y_name, z_name]].values

        if curr_lab_pts.shape[0] > min_pts:

            x=curr_lab_pts[:,0]
            y=curr_lab_pts[:,1]
            z=curr_lab_pts[:,2]

            if(color_by=='cluster' and draw_pts):
                '''
                Having this color_by block within the cluster loop makes sure that the
                points within the cluster have the same default color order as the cluster hulls
                drawn below in the same loop.

                Ideally it would be possible to apply a custom colormap to the clusters, apply them to
                the scatter plots, and use them in other plots to make the link.
                '''

                # Draw the scatter points for the current cluster
                # ax = plt.subplot(111, projection='3d')
                ax.scatter(x=x, y=y,z=z, s=0.3, color=cluster_colors[i_lab], )

            # Catch cases where we can't draw the hull because the interpolation fails.
            try:
                X_hull  = calculate_hull(
                    curr_lab_pts,
                    scale=1.0,
                    padding="scale",
                    n_interpolate=100,
                    interpolation="quadratic_periodic")

                ax.plot(X_hull[:,0], X_hull[:,1],c=cluster_colors[i_lab], label=curr_label)
                if(legend):
                    ax.legend()
            except:
                print('Failed to draw cluster, failed to draw cluster: ',curr_label, ' with shape: ', curr_lab_pts.shape)


    if STATIC_PLOTS:
        plt.savefig(save_path+'clusterhull_scatter_'+cluster_by+'.png')
    if PLOTS_IN_BROWSER:
        plt.show()


    return ax

def plotlytomatplotlibcolors():
    import plotly.express as px
    import matplotlib.colors
    color_discrete_sequence=px.colors.qualitative.Dark24
    freshcolors=[]
    for i in range(len(color_discrete_sequence)):
        freshcolortriplet=matplotlib.colors.to_rgb(color_discrete_sequence[i])
        freshcolors.append(freshcolortriplet)
    return freshcolors

def purity_plots_dev(df, clust_sum_df, cluster_by=CLUSTER_BY, save_path=CLUST_DIR ):


    if(cluster_by == 'tsne' or cluster_by == 'tSNE'):

        x_name = 'tSNE1'
        y_name = 'tSNE2'


    elif cluster_by == 'umap':

        x_name = 'UMAP1'
        y_name = 'UMAP2'
        z_name = 'UMAP3'
    fontsize = 24
    # Create a Subplot figure that shows the effect of clustering between conditions

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Low dimensional cluster analysis and purity", fontsize=fontsize)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # df=lab_dr_df
    x = 'UMAP1'
    y = 'UMAP2'
    z = 'UMAP3'
    dotsize = 0.1
    alpha = 0.5
    markerscale = 5
    # colors = cm.Dark2(np.linspace(0, 1, len(df['Condition_shortlabel'].unique())))
   

    colors=[]
    if CONDITION_CMAP != 'Dark24':

        cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
        for i in range(cmap.N):
            colors.append(cmap(i))
    else:
        colors = plotlytomatplotlibcolors()
        colors=colors[:len(df['Condition_shortlabel'].unique())]

    ######

    # if CONDITION_CMAP == 'Dark24':
    #     colors = plotlytomatplotlibcolors()
    #     colors=colors[:len(df['Condition_shortlabel'].unique())]


    
    # ######
    # coop = 'Dark24'
    # import plotly.express as px
    # import matplotlib.colors
    # color_discrete_sequence=px.colors.qualitative.Dark24
    # # replace the final bit of px.colors.qualitative.Dark24 with the name of the plotly colormap you want to use which is defined as coop above


    # color_discrete_sequence=color_discrete_sequence[:len(df['Condition_shortlabel'].unique())]
    # freshcolors=[]
    # for i in range(len(color_discrete_sequence)):
    #     freshcolortriplet=matplotlib.colors.to_rgb(color_discrete_sequence[i])
    #     freshcolors.append(freshcolortriplet)
    # # freshcolors=matplotlib.colors.to_rgb(color_discrete_sequence)
    # print('These are the fresh colors:')
    # print(freshcolors)
    # colors=freshcolors

 


    # ax1.set_title('Low-dimensional scatter with cluster outlines', fontsize=fontsize)
    # ax1.scatter(x=lab_dr_df['tSNE1'], y=lab_dr_df['tSNE2'], c='grey', s=0.1)
    # ax1.scatter(x=lab_dr_df[x], y=lab_dr_df[y], z=lab_dr_df[z] , c='grey', s=0.1)
    for colorselector in range(len(colors)):
        ax1.scatter(df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][x],
                     df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][y],
                     df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][z],
                       '.', color=colors[colorselector], label = df['Condition_shortlabel'].unique()[colorselector],s=dotsize, alpha = alpha)
        
    # ax1.set_xlabel('\n ' + x, fontsize=fontsize, linespacing=3.2) # gives a new line to make space for the axis label
    # ax1.set_ylabel('\n ' + y, fontsize=fontsize, linespacing=3.2)
    # ax1.set_zlabel('\n ' + z, fontsize=fontsize, linespacing=3.2)
    # ax1.tick_params(axis='both', which='major', labelsize=fontsize)    

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xlabel(x, fontsize=fontsize, linespacing=3.2) # gives a new line to make space for the axis label
    ax1.set_ylabel(y, fontsize=fontsize, linespacing=3.2)
    ax1.set_zlabel(z, fontsize=fontsize, linespacing=3.2)
                  

    # draw_cluster_hulls_dev(lab_dr_df,cluster_by='UMAP', color_by='cluster',ax=ax1, draw_pts=True,legend=True) #This won't work right now, because the plots are 3D and the hulls are 2D

    # Define a custom colormap for the clusters (To match what's done in draw_cluster_hulls())
    cluster_colors = []
    labels = list(set(df['label'].unique()))
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))

    # print("Cluster colors are: ", cluster_colors)  
    # Drop the first element of the cluster_colors list, which is the color for the 'unclustered' points
    cluster_colors = cluster_colors[1:]
    # print("Cluster colors without the unclustered portion are: ", cluster_colors)  

    #
    # Second subplot: stacked bar plots of cluster purity
    #
    ax2 = fig.add_subplot(1, 2, 2)

    # ax2.set_title('Low-dimension cluster purity', fontsize=fontsize)

    for i, cond in enumerate(df['Condition_shortlabel'].unique()):
        '''
        Assumes that the conditions are already in the correct order in the dataframe.
        '''
        print(df['Condition_shortlabel'].unique())

        if(i==0):
            ax2.bar(clust_sum_df['cluster_id'], clust_sum_df[cond+'_ncells_%'], label=cond,color=colors[i])
            prev_bottom = clust_sum_df[cond+'_ncells_%']

        else:
            ax2.bar(clust_sum_df['cluster_id'], clust_sum_df[cond+'_ncells_%'], bottom=prev_bottom, label=cond,color=colors[i])
            prev_bottom = clust_sum_df[cond+'_ncells_%'] + prev_bottom
            

    ax2.set_xticks(clust_sum_df['cluster_id'])
    ax2.set_ylabel('% of cells per cluster', fontsize = fontsize)
    ax2.set_xlabel('Cluster ID', fontsize = fontsize)

    # leg=plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=fontsize, bbox_to_anchor=(1.05, 1.0), markerscale=markerscale)
    # for lh in leg.legendHandles: 
    #     lh.set_alpha(1)
    ax2.legend(loc='upper right', numpoints=1, ncol=1, fontsize=fontsize, bbox_to_anchor=(1.6, 1.00), markerscale=markerscale)


    for ticklabel, tickcolor in zip(ax2.get_xticklabels(), cluster_colors):
        ticklabel.set_color(tickcolor)
    # change the xticklabel fontsize to fontsize variable

    ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    # plt.savefig(save_path+'purity_plots'+cluster_by+'.png')

    fig.savefig(CLUST_DISAMBIG_DIR+'PurityPlot.png', dpi=300, bbox_inches='tight')

    # if STATIC_PLOTS:
    #     plt.savefig(save_path+'purity_plots'+cluster_by+'.png')
    # if PLOTS_IN_BROWSER:
    #     plt.show()



    return fig


def plot_UMAP_subplots_coloredbymetricsorconditions(df_in, x= 'UMAP1', y= 'UMAP2', z = 'UMAP3', n_cols = 5, ticks=False, metrics = ALL_FACTORS, scalingmethod='choice', identifier='',
                                                    colormap='viridis', coloredbycondition = False, samplethedf = True): #new matplotlib version of scatter plot for umap 1-26-2023
    import matplotlib.pyplot as plt
    from numpy.random import random
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PowerTransformer
    from matplotlib.gridspec import GridSpec

    font_size = 20
    savedir = CLUST_DISAMBIG_DIR
   
    # Scale the data
    sub_set = df_in[metrics]
    Z = sub_set.values

    # colors=[] # replaced with new one that allows dark24 as well as all the others
    # cmap = cm.get_cmap(CONDITION_CMAP, len(df_in['Condition_shortlabel'].unique())) # replaced with new one that allows dark24 as well as all the others
    # for i in range(cmap.N): # replaced with new one that allows dark24 as well as all the others
    #     colors.append(cmap(i)) # replaced with new one that allows dark24 as well as all the others

    colors=[]
    if CONDITION_CMAP != 'Dark24':
        cmap = cm.get_cmap(CONDITION_CMAP, len(df_in['Condition_shortlabel'].unique()))
        for i in range(cmap.N):
            colors.append(cmap(i))
    else:
        colors = plotlytomatplotlibcolors()
        colors=colors[:len(df_in['Condition_shortlabel'].unique())]

    ############################   

    if coloredbycondition == True:

        print('Coloring each UMAP by each condition')
        conds = df_in['Condition_shortlabel'].unique().tolist()
        print(conds)
        n_conds = len(conds)
        n_rows = int(np.ceil(n_conds / n_cols))
        print('Number of conditions/plots is ', n_conds)
        print('Number of columns is ', n_cols)
        print('Number of rows is ', n_rows)
        
        figsizerows = n_rows*4
        figsizeconds= n_cols*4

        fig = plt.figure(constrained_layout=True, figsize=(figsizeconds,figsizerows))
        # Define the grid of subplots - you can change the number of columns and this changes the rows accordingly
        


        gs = GridSpec(n_rows, n_cols, figure=fig)

        df_out = df_in[['UMAP1', 'UMAP2', 'UMAP3', 'Condition_shortlabel']]
        # display(df_out)
        df_out = df_out.reset_index(drop=True)
        fraction_to_sample = 0.1
        
        main_df_valuecount = len(df_out)
        sample_of_cells = fraction_to_sample*main_df_valuecount
        sample_of_cells = int(sample_of_cells)
        print('Main df has ', main_df_valuecount, ' cells. Sampling ', sample_of_cells, ' cells.')
        main_df_sampled = df_out.sample(sample_of_cells)

        # Loop through each condition and make a 3D scatter plot of the UMAPs in grey, and then overlay the condition of interest in orange

        for number, cond in enumerate(conds):

            ax = fig.add_subplot(gs[number], projection='3d') # Initializes each plot
            chosencolor=colors[number]
            chosencolor=np.array(chosencolor).reshape(1, -1)
            sub_df = df_out[df_out['Condition_shortlabel'] == cond]

            if samplethedf == True:
                # Sample the df with only 10 % of the cells
                

                sub_df_valuecount = len(sub_df)
                sample_of_cells = fraction_to_sample*sub_df_valuecount
                sample_of_cells = int(sample_of_cells)
                print(cond, ' has ', sub_df_valuecount, ' cells. Sampling ', sample_of_cells, ' cells.')
                sub_df = sub_df.sample(sample_of_cells)

                ax.scatter(main_df_sampled[x], main_df_sampled[y], main_df_sampled[z], '.', c='gainsboro',  cmap=colormap, s=0.1, alpha = 0.02) # Makes each plot # c=df_out[cond], cmap=colormap,

            else:
                print('Not sampling the df')
                ax.scatter(df_out[x], df_out[y], df_out[z], '.', c='gainsboro',  cmap=colormap, s=0.1, alpha = 0.02) # Makes each plot # c=df_out[cond], cmap=colormap,
            ax.scatter(sub_df[x], sub_df[y], sub_df[z], '.', c=chosencolor, cmap=colormap, s=0.1, alpha = 0.9) # , edgecolors='Black' # Makes each plot # c=df_out[cond], cmap=colormap,
            # ax.plot_trisurf(sub_df[x], sub_df[y], sub_df[z],  cmap='white', edgecolor='Black')
            # ax.plot_trisurf(Xs-Xs.mean(), Ys-Ys.mean(), Zs, cmap=cm.jet, linewidth=0)
            # ax.plot_trisurf(sub_df[x]-sub_df[x].mean(), sub_df[y]-sub_df[y].mean(), sub_df[z], linewidth=0, shade=False, facecolor=None,  edgecolor='Black', antialiased=False)
            ax.set_title(cond, fontsize=font_size)

        # Set the axis labels with or without ticks
            if ticks == False:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])

            elif ticks == True:
                # Set the axis labels

                ax.tick_params(axis='both', which='major', labelsize=font_size)
            

    else:
        print('Coloring each UMAP by normalized metric values')

        # do not forget constrained_layout=True to have some space between axes
                
        n_metrics = len(metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        print('Length of metrics is ', n_metrics)    
        figsizerows = n_rows*4
        figsizeconds= n_cols*4    
        fig = plt.figure(constrained_layout=True, figsize=(figsizeconds, figsizerows))
        # Define the grid of subplots - you can change the number of columns and this changes the rows accordingly
        gs = GridSpec(n_rows, n_cols, figure=fig)

        if scalingmethod == 'minmax':
            x_=MinMaxScaler().fit_transform(Z)
            scaled_df = pd.DataFrame(data=x_, columns = metrics) 
            df_out = pd.concat([scaled_df,df_in[[x,y,z]]], axis=1)
            # fresh_df = pd.concat([df,lab_list_tpt_df['tavg_label']], axis=1)
            
        elif scalingmethod == 'log2minmax':
            
            negative_FACTORS = []
            positive_FACTORS = []
            for factor in metrics:
                if np.min(df_in[factor]) < 0:
                    print('factor ' + factor + ' has quite a few negative values')
                    negative_FACTORS.append(factor)
                        
                else:
                    print('factor ' + factor + ' has no negative values')
                    positive_FACTORS.append(factor)
                
                
            pos_df = df_in[positive_FACTORS]
            pos_x = pos_df.values
            neg_df = df_in[negative_FACTORS]
            neg_x = neg_df.values

            if len(neg_x[0]) == 0: #This controls for an edge case in which there are no negative factors - must be implemented in the other transforms as well (pipelines and clustering)
                print('No negative factors at all!')
                neg_x_ = neg_x
            else:
                neg_x_ = MinMaxScaler().fit_transform(neg_x) 

            pos_x_constant = pos_x + 0.000001
            pos_x_log = np.log2(pos_x + pos_x_constant)
            pos_x_ = MinMaxScaler().fit_transform(pos_x_log)
            x_ = np.concatenate((pos_x_, neg_x_), axis=1)
            newcols=positive_FACTORS + negative_FACTORS

            scaled_df = pd.DataFrame(x_, columns = newcols)
            df_out = pd.concat([scaled_df,df_in[[x,y,z]]], axis=1)

        elif scalingmethod == 'choice': 
            print('Factors to be scaled using log2 and then minmax:')
            FactorsNOTtotransform = ['arrest_coefficient', 'rip_L', 'rip_p', 'rip_K', 'eccentricity', 'orientation', 'directedness', 'turn_angle', 'dir_autocorr', 'glob_turn_deg']
            FactorsNottotransform_actual=[]
            FactorsToTransform_actual=[]
            for factor in metrics:
                if factor in FactorsNOTtotransform:
                    print('Factor: ' + factor + ' will not be transformed')
                    FactorsNottotransform_actual.append(factor)
                else:
                    print('Factor: ' + factor + ' will be transformed')
                    FactorsToTransform_actual.append(factor)
            trans_df = df_in[FactorsToTransform_actual]
            trans_x=trans_df.values
            nontrans_df = df_in[FactorsNottotransform_actual]
            nontrans_x=nontrans_df.values
            trans_x_constant=trans_x + 0.000001
            # trans_x_log = np.log2(trans_x + trans_x_constant) # Wait, this might be a mistake. You have already added the constant to the data. Here you are adding the data plus constant to the data without constant...
            trans_x_log = np.log2(trans_x_constant) # This is what it should be.
            trans_x_=MinMaxScaler().fit_transform(trans_x_log)
            nontrans_x_=MinMaxScaler().fit_transform(nontrans_x)
            x_=np.concatenate((trans_x_, nontrans_x_), axis=1)
            newcols=FactorsToTransform_actual + FactorsNottotransform_actual
            scaled_df_here = pd.DataFrame(x_, columns = newcols)
            scaled_df_here.hist(column=newcols, bins = 160, figsize=(20, 10),color = "black", ec="black")
            plt.tight_layout()
            plt.title('Transformed data')
            # plt.show()
            plt.savefig(savedir+ str(scalingmethod) +'.png')
            df_out = pd.concat([scaled_df_here,df_in[[x,y,z]]], axis=1)

        elif scalingmethod == 'powertransformer':    
            
            pt = PowerTransformer(method='yeo-johnson')
            x_ = pt.fit_transform(Z)
            scaled_df = pd.DataFrame(data=x_, columns = metrics) 
            df_out = pd.concat([scaled_df,df_in[[x,y,z]]], axis=1)

    ############################    

    if coloredbycondition == True:

        fig.savefig(CLUST_DISAMBIG_DIR+identifier+'UMAP_subplots_coloredbyconditions.png', dpi=300, bbox_inches='tight')


    else:

        df_out = df_out.reset_index(drop=True)
    # define the axes in a for loop according to the grid
        for number, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[number], projection='3d') # Initializes each plot
            g = ax.scatter(df_out[x], df_out[y], df_out[z], '.', c=df_out[metric], cmap=colormap, s=0.5, alpha = 0.5) # Makes each plot
            metric_name = metric.replace('_', ' ') #removes the underscore from the metric name
            ax.set_title(metric_name, fontsize=font_size)
            if ticks == False:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            elif ticks == True:
                ax.tick_params(axis='both', which='major', labelsize=font_size)
        fig.colorbar(g, shrink=0.5)
        fig.savefig(CLUST_DISAMBIG_DIR+identifier+'UMAP_subplots.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return     
