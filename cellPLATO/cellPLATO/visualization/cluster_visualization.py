#cluster_visualization.py

from initialization.initialization import *
from initialization.config import *

from visualization.low_dimension_visualization import colormap_pcs

import numpy as np
import pandas as pd
import os

import plotly.graph_objs as go
import plotly.express as px


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
        cmap = cm.get_cmap(CLUSTER_CMAP) 
                           
        numcolors=len(df['label'].unique())
        colors=[]
        for i in range(numcolors):
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
            # cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique())) #changedforbelow
            # for i in range(cmap.N):
            #     colors.append(cmap(i))
            cmap = cm.get_cmap(CONDITION_CMAP) 
            numcolors=len(df['Condition_shortlabel'].unique())
            for i in range(numcolors):
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

def plot_3D_scatter_dev(df, x, y, z, colorby, ticks=False, identifier='', dotsize = 3, alpha=0.2, markerscale=5): #new matplotlib version of scatter plot for umap 1-26-2023
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
        cmap = cm.get_cmap(CLUSTER_CMAP) 
                           
        numcolors=len(df['label'].unique())
        colors=[]
        for i in range(numcolors):
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
            # cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique())) #changedforbelow
            # for i in range(cmap.N):
            #     colors.append(cmap(i))
            cmap = cm.get_cmap(CONDITION_CMAP) 
            numcolors=len(df['Condition_shortlabel'].unique())
            for i in range(numcolors):
                colors.append(cmap(i)) 
        else:
            colors = plotlytomatplotlibcolors()
            colors=colors[:len(df['Condition_shortlabel'].unique())]   

    elif colorby == 'frame':
        cmap = cm.get_cmap('inferno') 
                           
        numcolors=max(df['frame'])
        numcolors=int(numcolors)
        colors=[]
        for i in range(numcolors):
            colors.append(cmap(i))
        # print(colors)
        # df = df.sort_values(by=['frame'])         

    elif colorby == 'uniq_id':
        cmap = cm.get_cmap('inferno') 
                           
        numcolors=len(df['uniq_id'].unique())
        numcolors=int(numcolors)
        colors=[]
        for i in range(numcolors):
            colors.append(cmap(i))
        df = df.sort_values(by=['uniq_id'])             

 



    ax = plt.subplot(111, projection='3d')
     
    for colorselector in range(len(colors)): #you have to plot each label separately to get the legend to work
        if colorby == 'label':
            ax.scatter(df[df['label'] == df['label'].unique()[colorselector]][x], df[df['label'] == df['label'].unique()[colorselector]][y],
             df[df['label'] == df['label'].unique()[colorselector]][z], '.', color=colors[colorselector], label = df['label'].unique()[colorselector], s=dotsize, alpha = alpha)
        elif colorby == 'condition':
            ax.scatter(df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][x], df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][y],
             df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][z], '.', color=colors[colorselector], label = df['Condition_shortlabel'].unique()[colorselector], s=dotsize, alpha = alpha)
    if colorby == 'frame':
        # make a scatter plot of x, y and z, colored by frame number
        ax.scatter(df[x], df[y], df[z], c=df['frame'], cmap='cividis', s=dotsize, alpha = alpha) #label = df['Condition_shortlabel'].unique()    
    if colorby == 'uniq_id':
        # make a scatter plot of x, y and z, colored by frame number
        ax.scatter(df[df['uniq_id'] == df['uniq_id'].unique()[colorselector]][x], df[df['uniq_id'] == df['uniq_id'].unique()[colorselector]][y],
             df[df['uniq_id'] == df['uniq_id'].unique()[colorselector]][z], '.', color=colors[colorselector],  s=dotsize, alpha = alpha) #label = df['uniq_id'].unique()[colorselector],
        # ax.scatter(df[x], df[y], df[z], c=df['uniq_id'], cmap='inferno', s=dotsize, alpha = alpha) #label = df['Condition_shortlabel'].unique()    

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

def interactive_plot_3D_UMAP(df, colorby = 'label', symbolby = 'Condition_shortlabel', what = ''):

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

def interactive_plot_3D_UMAP_chosen_condition(df, chosen_condition, colorby='Condition_shortlabel', symbolby='Condition_shortlabel', 
                                              opacity_grey=0.5, marker_size_all=2, what=''):

    import plotly.express as px
    import plotly.io as pio
    import seaborn as sns

    # Generate the initial colormap
    if CONDITION_CMAP != 'Dark24':
        pal = sns.color_palette(CONDITION_CMAP, len(df[colorby].unique()))
        cmap = pal.as_hex()
    else:
        cmap = px.colors.qualitative.Dark24

    # Overwrite colors to make all conditions except the chosen one grey
    chosen_color = cmap[df[colorby].unique().tolist().index(chosen_condition)]
    cmap = [chosen_color if cond == chosen_condition else '#808080' for cond in df[colorby].unique()]

    fig = px.scatter_3d(df, x='UMAP1', y='UMAP2', z='UMAP3',
                        color=colorby,
                        symbol=symbolby,
                        opacity=opacity_grey,  # Adjust opacity for grey points
                        # size_max=1,
                        #   opacity=0.7,
                            width=1800, height=1200, color_discrete_sequence=cmap)

    fig.update_traces(marker_size=marker_size_all)  # Adjust marker size for all points
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend_font_size=24)
    fig.update_layout(font=dict(family="Courier New, monospace", size=24))
    fig.update_layout(legend={'itemsizing': 'constant'})

    pio.write_image(fig, CLUST_DISAMBIG_DIR + what + ' UMAP_Clusters.png', scale=1, width=1800, height=1200)
    fig.show()
    return

def plot_plasticity_changes(df, identifier='\_allcells', miny=None, maxy=None, t_window_multiplier = T_WINDOW_MULTIPLIER): #spidey

    # Check for required columns and provide informative error messages
    required_columns = ['label', 'twind_n_changes', 'twind_n_labels', 'frame', 'Condition_shortlabel']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print(f"Original dataframe shape: {df.shape}")
    print(f"NaN counts per column:")
    for col in required_columns:
        nan_count = df[col].isna().sum()
        print(f"  {col}: {nan_count} NaN values")
    
    # f, axes = plt.subplots(1, 3, figsize=(15, 5)) #sharex=True
    # f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=True) #sharex=True
    f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=False) #sharex=True

    whattoplot=['label','twind_n_changes', 'twind_n_labels']

    # CLUSTER_CMAP = 'tab20'
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
        # Use safe max calculations with fallback values
        try:
            maximumy1=np.nanmax(df[whattoplot[0]]) if not df[whattoplot[0]].isna().all() else 1
            maximumy2=np.nanmax(df[whattoplot[1]]) if not df[whattoplot[1]].isna().all() else 1
            maximumy3=np.nanmax(df[whattoplot[2]]) if not df[whattoplot[2]].isna().all() else 1
        except Exception as e:
            print(f"Warning: Error calculating max values: {e}")
            maximumy1 = maximumy2 = maximumy3 = 1

    ##
    import seaborn as sns
    sns.set_theme(style="ticks")
    # sns.set_palette(CONDITION_CMAP) #removed
    # colors=[]
    # cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
    # for i in range(cmap.N):
    #     colors.append(cmap(i))

    # Check if we have valid condition data
    unique_conditions = df['Condition_shortlabel'].dropna().unique()
    if len(unique_conditions) == 0:
        print("ERROR: No valid conditions found in 'Condition_shortlabel' column")
        return None
    
    print(f"Found {len(unique_conditions)} unique conditions: {list(unique_conditions)}")

    colors=[]
    if CONDITION_CMAP != 'Dark24':

        cmap = cm.get_cmap(CONDITION_CMAP)
        numcolors= len(unique_conditions)
        sns.set_palette(CONDITION_CMAP)
        for i in range(numcolors):
            colors.append(cmap(i))
    else:
        colors = plotlytomatplotlibcolors()
        colors=colors[:len(unique_conditions)]



    # display(df)
    df_before_dropna = df.copy()
    
    # Instead of dropping all rows with any NaN, be more selective
    # Only drop rows where the columns we're actually plotting have NaN values
    plotting_columns = whattoplot + ['Condition_shortlabel']
    df = df.dropna(subset=plotting_columns, how='any')
    
    print(f"Dataframe shape after removing NaNs in plotting columns: {df.shape}")
    print(f"Removed {len(df_before_dropna) - len(df)} rows due to NaN values in plotting columns")
    
    if len(df) == 0:
        print("ERROR: No data remaining after removing NaN values!")
        print("NaN summary for plotting columns:")
        for col in plotting_columns:
            if col in df_before_dropna.columns:
                nan_count = df_before_dropna[col].isna().sum()
                print(f"  {col}: {nan_count}/{len(df_before_dropna)} are NaN")
        return None
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

    timewindowmins = (MIG_T_WIND * t_window_multiplier)*SAMPLING_INTERVAL #Here is just a superficial matching of the time window specified in the config for plotting purposes
    # The actual calculation is done in count_cluster_changes_with_tavg function
    timewindowmins = round(timewindowmins, 1)

    print('Time window mins: ', timewindowmins)
    text1 = "Cluster ID"
    text2 = "Cluster switches / " + str(timewindowmins) + " min"
    text3 = "New clusters / " + str(timewindowmins) + " min"

    x_lab = "Distinct Behaviors"
    plottitle = ""

    # get the max value of the whattoplot[1] column of df
    max1=np.nanmax(df[whattoplot[1]])
    max2=np.nanmax(df[whattoplot[2]])

    # ensure tick frequencies are at least 1 to prevent division by zero
    tickfrequency1 = max(int(max1/5), 1)
    tickfrequency2 = max(int(max2/5), 1)

    the_yticks = np.arange(0, len(df[whattoplot[0]].unique()), 1)
    the_yticks = [int(x) for x in the_yticks]
    axes[0].set_yticks(the_yticks) # set new tick positions


    axes[0].margins(y=0) # set tight margins
    the_yticks = np.arange(0, len(df[whattoplot[1]].unique()), tickfrequency1)
    the_yticks = [int(x) for x in the_yticks]


    axes[1].set_yticks(the_yticks) # set new tick positions
    axes[1].margins(y=0) # set tight margins
    the_yticks = np.arange(0, len(df[whattoplot[2]].unique()), tickfrequency2)
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

####################### DEVELOPMENTAL PLOTS #############################

# def plot_cumulative_plasticity_changes(df, identifier='\_allcells', miny=None, maxy=None, t_window_multiplier = T_WINDOW_MULTIPLIER, plotallcells = False): #spidey

#     # f, axes = plt.subplots(1, 3, figsize=(15, 5)) #sharex=True
#     # f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=True) #sharex=True
#     f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False) #sharex=True

#     # whattoplot=['cum_n_changes','cum_n_labels', 'twind_n_labels']
#     whattoplot=['cum_n_changes','cum_n_labels',]

#     # CLUSTER_CMAP = 'tab20'
#     # CONDITION_CMAP = 'dark'

#     time = df['frame']
#     # SAMPLING_INTERVAL=10/60 #This shouldn't be hardcoded!
#     timeminutes=time*SAMPLING_INTERVAL

#     # dfnumericals = df.select_dtypes('number')

#     # extracted_col = df["Condition_shortlabel"]

#     # df=dfnumericals.join(extracted_col)

#     ##
#     if miny != None or maxy != None:
#         minimumy=miny
#         maximumy1=maxy
#         maximumy2=maxy
#         # maximumy3=maxy
#     else:
#         minimumy=0
#         maximumy1=np.nanmax(df[whattoplot[0]])
#         maximumy2=np.nanmax(df[whattoplot[1]])
#         # maximumy3=np.nanmax(df[whattoplot[2]])

#     ##
#     import seaborn as sns
#     sns.set_theme(style="ticks")
#     # sns.set_palette(CONDITION_CMAP) #removed
#     # colors=[]
#     # cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
#     # for i in range(cmap.N):
#     #     colors.append(cmap(i))

#     colors=[]
#     if CONDITION_CMAP != 'Dark24':

#         cmap = cm.get_cmap(CONDITION_CMAP)
#         numcolors= len(df['Condition_shortlabel'].unique())
#         sns.set_palette(CONDITION_CMAP)
#         for i in range(numcolors):
#             colors.append(cmap(i))
#     else:
#         colors = plotlytomatplotlibcolors()
#         colors=colors[:len(df['Condition_shortlabel'].unique())]



#     # display(df)
#     df=df.dropna(how='any')
#     # display(df)

#     if plotallcells == False:
            
#         # Plot the responses for different events and regions
#         sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors)

#         sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors)
#     else:
#         sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",  units = 'particle', estimator = None, #style = 'particle', dashes = False, 
#                     data=df, palette=colors)


#         # w = axes[0].properties()
#         # print("Display all Properties\n")
#         # for i in w:
#         #     print(i, ":", w[i])
#         for line, name in zip(axes[0].lines, df['particle'].unique()): #df['particle'].unique()
#             # print(line)
#             # print(name)
#             y = line.get_ydata()[-1]
#             # t= line.get_label()
#             # h = line.get_label()[-1]
#             # print(y)
#             # print(t)
#             # print(h)
#             axes[0].annotate(name, xy=(1,y), xytext=(6,0), color=line.get_color(), 
#             xycoords = axes[0].get_yaxis_transform(), textcoords="offset points",
#             size=14, va="center")

#         sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors, lw=4)
    
#         sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",  units = 'particle', estimator = None, #style = 'particle', dashes = False,
#                     data=df, palette=colors)

#         for line, name in zip(axes[1].lines, df['particle'].unique()):
#             y = line.get_ydata()[-1]
#             axes[1].annotate(name, xy=(1,y), xytext=(6,0), color=line.get_color(), 
#             xycoords = axes[1].get_yaxis_transform(), textcoords="offset points",
#             size=14, va="center")

#         sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors, lw=4)            

#     # sns.lineplot(ax=axes[2], x=timeminutes, y=whattoplot[2], #n_labels #n_changes #label
#     #              hue="Condition_shortlabel",
#     #              data=df, palette=colors)

#     # timewindowmins = (MIG_T_WIND * t_window_multiplier)*SAMPLING_INTERVAL #Here is just a superficial matching of the time window specified in the config for plotting purposes
#     # # The actual calculation is done in count_cluster_changes_with_tavg function
#     timewindowmins = MIG_T_WIND*SAMPLING_INTERVAL #Here is just a superficial matching of the time window specified in the config for plotting purposes
#     # The actual calculation is done in count_cluster_changes_with_tavg function

#     print('Time window mins: ', timewindowmins)
#     text1 = "Cumulative cluster switches"
#     text2 = "Cumulative new clusters"
#     # text3 = "New clusters / " + str(timewindowmins) + " min"

#     x_lab = "Distinct Behaviors"
#     plottitle = ""

#     # get the max value of the whattoplot[1] column of df
#     max1=np.nanmax(df[whattoplot[0]])
#     max2=np.nanmax(df[whattoplot[1]])
#     tickfrequency1 = int(max1/5)
#     tickfrequency2 = int(max2/5)

#     the_yticks = np.arange(0, len(df[whattoplot[0]].unique()), tickfrequency1)
#     the_yticks = [int(x) for x in the_yticks]
#     axes[0].set_yticks(the_yticks) # set new tick positions
#     axes[0].margins(y=0) # set tight margins
#     the_yticks = np.arange(0, len(df[whattoplot[1]].unique()),tickfrequency2 ) #tickfrequency1)
#     the_yticks = [int(x) for x in the_yticks]


#     axes[1].set_yticks(the_yticks) # set new tick positions
#     axes[1].margins(y=0) # set tight margins

#     # the_yticks = np.arange(0, len(df[whattoplot[2]].unique()), tickfrequency2)
#     # the_yticks = [int(x) for x in the_yticks]
#     # axes[2].set_yticks(the_yticks) # set new tick positions
#     # axes[2].margins(y=0) # set tight margins

#     # the_xticks = np.arange(0, len(timeminutes), 1)
#     # the_xticks = [int(x) for x in the_xticks]
#     # axes[0].set_xticks(the_xticks) # set new tick positions
#     # axes[0].margins(x=0) # set tight margins
#     # axes[1].set_xticks(the_xticks) # set new tick positions
#     # axes[1].margins(x=0) # set tight margins
#     # axes[2].set_xticks(the_xticks) # set new tick positions
#     # axes[2].margins(x=0) # set tight margins


#     # Tweak the visual presentation
#     axes[0].xaxis.grid(True)
#     axes[1].xaxis.grid(True)
#     # axes[2].xaxis.grid(True)

#     # axes[0].set_ylabel(whattoplot[0], fontsize=36)
#     # axes[1].set_ylabel(whattoplot[1], fontsize=36)
#     # axes[2].set_ylabel(whattoplot[2], fontsize=36)
#     axes[0].set_ylabel(text1, fontsize=36)
#     axes[1].set_ylabel(text2, fontsize=36)
#     # axes[2].set_ylabel(text3, fontsize=36)

#     axes[0].set_title("", fontsize=36)
#     axes[1].set_title("", fontsize=36)
#     # axes[2].set_title("", fontsize=36)

#     axes[0].set_xlabel("Time (min)", fontsize=36)
#     axes[1].set_xlabel("Time (min)", fontsize=36)
#     # axes[2].set_xlabel("Time (min)", fontsize=36)

#     # axes[0].set_ylim(0, np.nanmax(df[whattoplot[0]]))
#     # axes[1].set_ylim(0, np.nanmax(df[whattoplot[1]]))
#     # axes[2].set_ylim(0, np.nanmax(df[whattoplot[2]]))
#     axes[0].set_ylim(0, max1+1)
#     axes[1].set_ylim(0, max2+2)
#     # axes[2].set_ylim(0, maximumy3)


    

#     # ax.set_ylabel(y_lab, fontsize=36)
#     axes[0].tick_params(axis='both', labelsize=36)
#     axes[1].tick_params(axis='both', labelsize=36)
#     # axes[2].tick_params(axis='both', labelsize=36)

#     axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
#     axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
#     # axes[2].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)

#     f.tight_layout()
#     # fig.write_image(CLUST_DISAMBIG_DIR+'\cluster_label_counts.png')
#     f.savefig(CLUST_DISAMBIG_DIR+identifier+'_cumulative_plasticity_cluster_changes_over_time.png', dpi=300)#plt.

#     return

############# Developmental plots end ######################

############# DEV PLOTS 2 ######################

# def plot_cumulative_plasticity_changes_test(df, identifier='\_allcells', miny=None, maxy=None, t_window_multiplier = T_WINDOW_MULTIPLIER, plotallcells = False): #spidey

#     f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False) #sharex=True

#     whattoplot=['cum_n_changes','cum_n_labels',]

#     time = df['frame']

#     timeminutes=time*SAMPLING_INTERVAL

#     ##
#     if miny != None or maxy != None:
#         minimumy=miny
#         maximumy1=maxy
#         maximumy2=maxy

#     else:
#         minimumy=0
#         maximumy1=np.nanmax(df[whattoplot[0]])
#         maximumy2=np.nanmax(df[whattoplot[1]])


#     ##
#     import seaborn as sns
#     sns.set_theme(style="ticks")

#     colors=[]
#     if CONDITION_CMAP != 'Dark24':

#         cmap = cm.get_cmap(CONDITION_CMAP)
#         numcolors= len(df['Condition_shortlabel'].unique())
#         sns.set_palette(CONDITION_CMAP)
#         for i in range(numcolors):
#             colors.append(cmap(i))
#     else:
#         colors = plotlytomatplotlibcolors()
#         colors=colors[:len(df['Condition_shortlabel'].unique())]

#     df=df.dropna(how='any')


#     if plotallcells == False:
            
#         # Plot the responses for different events and regions
#         sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors)

#         sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors)
#     else:
#         sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",  units = 'particle', estimator = None, #style = 'particle', dashes = False, 
#                     data=df, palette=colors)

#         # sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
#         #             hue="Condition_shortlabel",
#         #             data=df, palette=colors, lw=4)
    
#         sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",  units = 'particle', estimator = None, #style = 'particle', dashes = False,
#                     data=df, palette=colors)

#         # sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
#         #             hue="Condition_shortlabel",
#         #             data=df, palette=colors, lw=4)         
#         # 
#         # Annotate the end of each line with 'uniq_ID' values


#     # # Annotate the end of each line with 'uniq_ID' values outside the main plot
#     # for ax in axes:
#     #     lines = ax.get_lines()
#     #     for line in lines:
#     #         x_data = line.get_xdata()
#     #         y_data = line.get_ydata()
#     #         if len(x_data) > 0 and len(y_data) > 0:
#     #             uniq_ID = int(y_data[-1])  # Assuming 'uniq_ID' is an integer, adjust accordingly if it's not
#     #             ax.annotate(f'Uniq_ID: {uniq_ID}', xy=(x_data[-1], y_data[-1]), xytext=(5, 5), textcoords='offset points', fontsize=14, ha='left', va='bottom', zorder=5)
#     #             ax.plot([x_data[-1], x_data[-1]], [y_data[-1], y_data[-1] + 0.2], linestyle='dotted', color='gray', transform=ax.transData, zorder=4)  # Dotted line from the end of the line to the text position        

#     # Annotate the end of each line with 'uniq_ID' values outside the main plot
#     # for ax in axes:
#     #     lines = ax.get_lines()
#     #     for line in lines:
#     #         x_data = line.get_xdata()
#     #         y_data = line.get_ydata()
#     #         if len(x_data) > 0 and len(y_data) > 0:
#     #             uniq_ID = int(y_data[-1])  # Assuming 'uniq_ID' is an integer, adjust accordingly if it's not
#     #             ax.annotate(f'Uniq_ID: {uniq_ID}', xy=(1.02, y_data[-1]), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center')

#     # Annotate the end of each line with 'uniq_ID' values outside the main plot

#     ############# THIS WORKS ###############################
#     # for n, ax in enumerate(axes):

#     #     max_x = np.max(timeminutes)
#     #     print(max_x)

#     #     lines = ax.get_lines()
#     #     # alldata=lines.get_data()
#     #     for line in lines:
#     #         x_data = line.get_xdata()
#     #         y_data = line.get_ydata()
#     #         if len(x_data) > 0 and len(y_data) > 0:
#     #             # uniq_ID = int(y_data[-1])  # Assuming 'uniq_ID' is an integer, adjust accordingly if it's not
#     #             uniq_id = df['uniq_id'].iloc[n]  # Get the 'uniq_id' value for the corresponding line
#     #             ax.annotate(f'Uniq_ID: {uniq_ID}', xy=(max_x + 10, y_data[-1]), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center')

#     ############# THIS WORKS ###############################

#     # # Annotate the end of each line with 'uniq_id' values outside the main plot
#     # for n, ax in enumerate(axes):
#     #     max_x = np.max(timeminutes)
#     #     lines = ax.get_lines()
#     #     for line in lines:
#     #         x_data = line.get_xdata()
#     #         y_data = line.get_ydata()
#     #         if len(x_data) > 0 and len(y_data) > 0:
#     #             uniq_id = df['uniq_id'].iloc[n]  # Get the 'uniq_id' value for the corresponding line
#     #             ax.annotate(f'Uniq_ID: {uniq_id}', xy=(max_x + 10, y_data[-1]), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center')    



#     # # Annotate the end of each line with 'uniq_id' values outside the main plot
#     # for ax in axes:
#     #     lines = ax.get_lines()
#     #     for line in lines:
#     #         x_data = line.get_xdata()
#     #         y_data = line.get_ydata()
#     #         if len(x_data) > 0 and len(y_data) > 0:
#     #             # line_index = lines.index(line)  # Get the index of the line in the lines list
#     #             # print('This is the line index')
#     #             # print(line_index)
#     #             # uniq_id = df['uniq_id'].iloc[line_index]  # Get the 'uniq_id' value using the line index
#     #             # print('this is the uniq_id')
#     #             # print(uniq_id)
#     #             print(x_data[-1])
#     #             print(y_data[-1])

#     #             ax.annotate(f'Uniq_ID: {uniq_id}', xy=(x_data[-1], y_data[-1]), xytext=(5, 5), textcoords='offset points', fontsize=14, ha='left', va='bottom', zorder=5)
#     #             ax.plot([x_data[-1], x_data[-1]], [y_data[-1], y_data[-1] + 0.2], linestyle='dotted', color='gray', transform=ax.transData, zorder=4)  # Dotted line from the end of the line to the text position        

#     # Annotate the end of each line with 'uniq_id' values outside the main plot
#     for n, ax in enumerate(axes):
#         lines = ax.get_lines()
#         max_x = np.max(timeminutes)
#         for line in lines:
#             x_data = line.get_xdata()
#             y_data = line.get_ydata()
#             if len(x_data) > 0 and len(y_data) > 0:
#                 x_final = x_data[-1]  # Final entry in x_data
#                 y_final = y_data[-1]  # Final entry in y_data
                
#                 x_finalframe = x_final/SAMPLING_INTERVAL
#                 # print('The x final frame non integer is: ', x_finalframe)
#                 # x_finalframe = int(x_finalframe)
#                 print('The x final (minutes) is: ', x_final)
#                 print('The x final frame (frames) is: ', x_finalframe)
#                 # # Find the corresponding 'uniq_id' using the final x_data and y_data
#                 # uniq_id = df.loc[(df['frame'] == x_finalframes) & (df[whattoplot[n]] == y_final), 'uniq_id'].iloc[0]
#                 # uniq_id = df.loc[(df['frame'] == x_finalframe) & (df[whattoplot[n]] == y_final), 'uniq_id'].iloc[0] #original line
#                 print('This is the y_final')
#                 print(y_final)
#                 print('This is the df that contains the y_final')
#                 print(df.loc[(df[whattoplot[n]] == y_final)])

#                 #####
#                 # uniq_id_row = df.loc[(df['frame'] == x_finalframe) & (df[whattoplot[n]] == y_final), 'uniq_id']


#                 uniq_id_row = df.loc[(df[whattoplot[n]] == y_final) , 'uniq_id']
#                 # Option 1 - translate all of the df_frames into minutes first
#                 # Or test without this and see whether it works.

#                 # nicely display the uniq_id_row
#                 print('This is the uniq_id_row')
#                 print(uniq_id_row)
#                 # Get the number of elements in the uniq_id_row
#                 uniq_id_row_length = len(uniq_id_row)
#                 print('This is the length of the uniq_id_row')
#                 print(uniq_id_row_length)
#                 # Separate out each element of the uniq_id_row and print them
#                 print('These are the elements of the uniq_id_row')
#                 for i in range(uniq_id_row_length):
#                     print(uniq_id_row.iloc[i])
#                 uniq_id=uniq_id_row
#                 # uniq_id = uniq_id_row.iloc[0]
#                 # print(uniq_id)
#                 #####

#                 # uniq_id = df.loc[(df[whattoplot[n]] == y_final), 'uniq_id'].iloc[0]
#                 ax.annotate(f'Cell ID: {uniq_id}', xy=(max_x+85, y_final), xytext=(5, 5), textcoords='offset points', fontsize=14, ha='left', va='bottom', zorder=5)
#                 ax.plot([x_final, max_x+85], [y_final, y_final], linestyle='dotted', color='gray', transform=ax.transData, zorder=4)  # Dotted line from the end of the line to the text position 




#     timewindowmins = MIG_T_WIND*SAMPLING_INTERVAL #Here is just a superficial matching of the time window specified in the config for plotting purposes
#     # The actual calculation is done in count_cluster_changes_with_tavg function

#     print('Time window mins: ', timewindowmins)
#     text1 = "Cumulative cluster switches"
#     text2 = "Cumulative new clusters"
#     # text3 = "New clusters / " + str(timewindowmins) + " min"

#     x_lab = "Distinct Behaviors"
#     plottitle = ""

#     # get the max value of the whattoplot[1] column of df
#     max1=np.nanmax(df[whattoplot[0]])
#     max2=np.nanmax(df[whattoplot[1]])
#     tickfrequency1 = int(max1/5)
#     tickfrequency2 = int(max2/5)

#     the_yticks = np.arange(0, len(df[whattoplot[0]].unique()), tickfrequency1)
#     the_yticks = [int(x) for x in the_yticks]
#     axes[0].set_yticks(the_yticks) # set new tick positions
#     axes[0].margins(y=0) # set tight margins
#     the_yticks = np.arange(0, len(df[whattoplot[1]].unique()),tickfrequency2 ) #tickfrequency1)
#     the_yticks = [int(x) for x in the_yticks]


#     axes[1].set_yticks(the_yticks) # set new tick positions
#     axes[1].margins(y=0) # set tight margins

#     # Tweak the visual presentation
#     axes[0].xaxis.grid(True)
#     axes[1].xaxis.grid(True)

#     axes[0].set_ylabel(text1, fontsize=36)
#     axes[1].set_ylabel(text2, fontsize=36)

#     axes[0].set_title("", fontsize=36)
#     axes[1].set_title("", fontsize=36)

#     axes[0].set_xlabel("Time (min)", fontsize=36)
#     axes[1].set_xlabel("Time (min)", fontsize=36)

#     axes[0].set_ylim(0, max1+1)
#     axes[1].set_ylim(0, max2+2)
  
#     axes[0].tick_params(axis='both', labelsize=36)
#     axes[1].tick_params(axis='both', labelsize=36)

#     axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
#     axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)

#     f.tight_layout()
#     f.savefig(CLUST_DISAMBIG_DIR+identifier+'_cumulative_plasticity_cluster_changes_over_time.png', dpi=300)#plt.

#     return


# def plot_cumulative_plasticity_changes_main(df, identifier='\_allcells', miny=None, maxy=None, t_window_multiplier = T_WINDOW_MULTIPLIER, plotametric=None, plotallcells = False): #spidey
    
#     if plotallcells:
#         f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False)
#     else:
#         f, axes = plt.subplots(2, 1, figsize=(15, 20), sharex=False)
#     # f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False) #sharex=True

#     if plotametric != None:
#         whattoplot=[plotametric, plotametric]
#     else:
#         whattoplot=['cum_n_changes','cum_n_labels',]

#     time = df['frame']

#     timeminutes=time*SAMPLING_INTERVAL

#     ##
#     if miny != None or maxy != None:
#         minimumy=miny
#         maximumy1=maxy
#         maximumy2=maxy

#     else:
#         minimumy=0
#         maximumy1=np.nanmax(df[whattoplot[0]])
#         maximumy2=np.nanmax(df[whattoplot[1]])

#     ##
#     import seaborn as sns
#     sns.set_theme(style="ticks")

#     colors=[]
#     if CONDITION_CMAP != 'Dark24':

#         cmap = cm.get_cmap(CONDITION_CMAP)
#         numcolors= len(df['Condition_shortlabel'].unique())
#         sns.set_palette(CONDITION_CMAP)
#         for i in range(numcolors):
#             colors.append(cmap(i))
#     else:
#         colors = plotlytomatplotlibcolors()
#         colors=colors[:len(df['Condition_shortlabel'].unique())]

#     df=df.dropna(how='any')

#     conditionsindf = df['Condition_shortlabel'].unique()


#     if plotallcells == False:
            
#         # Plot the responses for different events and regions
#         sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors)

#         sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors)
#     else:
#         # Create a dictionary to map condition_shortlabel to colors
#         condition_colors = {condition: color for condition, color in zip(df['Condition_shortlabel'].unique(), colors)}
#         # Make timeminuteslist and get the last element for the annotations
#         timeminuteslist = timeminutes.tolist()
#         # x_final = timeminuteslist[-1] # Final entry in x_data
#         max_x = np.max(timeminutes)
#         # Create a set to keep track of plotted condition_shortlabels
#         plotted_labels = set()
#         # This will be used to offset the x position of the annotations
#         # x_offsets = np.arange(20, len(df['uniq_id'].unique()), 1)
#         x_offsets = [20, 100, 180, 260, 340, 420, 500] * int(len(df['uniq_id'].unique()) / 2 + 1) #used for placing labels in readable positions

#         y_offsets = [0, -40, -80, -120] * int(len(df['uniq_id'].unique()) / 2 + 1) #used for placing labels in readable positions

#         for offset, uniq_id in enumerate(df['uniq_id'].unique()):
#             df_uniq = df[df['uniq_id'] == uniq_id]
#             condition_label = df_uniq['Condition_shortlabel'].iloc[0]  # Get the condition_shortlabel for this 'uniq_id'
#             line_color = condition_colors[condition_label]  # Get the corresponding color for the condition_shortlabel
#             if condition_label not in plotted_labels:
#                 # Plot the line with a label for the legend only if it hasn't been plotted before
#                 singleline = sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], data=df_uniq, color=line_color, label=condition_label)
#                 singleline2 = sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], data=df_uniq, color=line_color, label=condition_label)
#                 plotted_labels.add(condition_label)  # Add the condition_shortlabel to the set of plotted labels
#             else:
#                 # If the label has already been plotted, don't add it to the legend again
#                 singleline = sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], data=df_uniq, color=line_color)
#                 singleline2 = sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], data=df_uniq, color=line_color) #, label=condition_label

#             x_final_frames = df_uniq['frame'].iloc[-1]  # Final entry in x_data
#             x_final = x_final_frames*SAMPLING_INTERVAL

#             y_final = df_uniq[whattoplot[0]].iloc[-1]  # Final entry in y_data
#             y_final1 = df_uniq[whattoplot[1]].iloc[-1]  # Final entry in y_data
            
#             uniq_id_label = f'Cell_ID: {uniq_id}'
#             axes[0].annotate(uniq_id_label, xy=(max_x + x_offsets[offset], y_final), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center', arrowprops=dict(arrowstyle="<-", color='black'))
#             axes[0].plot([x_final, max_x + x_offsets[offset]], [y_final, y_final], linestyle='dotted', color='gray', transform=axes[0].transData, zorder=4)

#             axes[1].annotate(uniq_id_label, xy=(max_x + x_offsets[offset], y_final1 ), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center',arrowprops=dict(arrowstyle="<-", color='black'))
#             axes[1].plot([x_final, max_x + x_offsets[offset]], [y_final1, y_final1], linestyle='dotted', color='gray', transform=axes[1].transData, zorder=4)            
        
#     timewindowmins = (MIG_T_WIND*t_window_multiplier) * SAMPLING_INTERVAL #Here is just a superficial matching of the time window specified in the config for plotting purposes
#     # The actual calculation is done in count_cluster_changes_with_tavg function
#     timewindowmins = round(timewindowmins, 1)

#     print('Time window mins: ', timewindowmins)
#     text1 = "Cumulative cluster switches"
#     text2 = "Cumulative new clusters"
#     # text3 = "New clusters / " + str(timewindowmins) + " min"

#     x_lab = "Distinct Behaviors"
#     plottitle = ""

#     # get the max value of the whattoplot[1] column of df
#     max1=np.nanmax(df[whattoplot[0]])
#     max2=np.nanmax(df[whattoplot[1]])
#     tickfrequency1 = int(max1/5)
#     tickfrequency2 = int(max2/5)

#     the_yticks = np.arange(0, len(df[whattoplot[0]].unique()), tickfrequency1)
#     the_yticks = [int(x) for x in the_yticks]
#     axes[0].set_yticks(the_yticks) # set new tick positions
#     axes[0].margins(y=0) # set tight margins
#     the_yticks = np.arange(0, len(df[whattoplot[1]].unique()),tickfrequency2 ) #tickfrequency1)
#     the_yticks = [int(x) for x in the_yticks]


#     axes[1].set_yticks(the_yticks) # set new tick positions
#     axes[1].margins(y=0) # set tight margins

#     # Tweak the visual presentation
#     axes[0].xaxis.grid(True)
#     axes[1].xaxis.grid(True)

#     axes[0].set_ylabel(text1, fontsize=36)
#     axes[1].set_ylabel(text2, fontsize=36)

#     axes[0].set_title("", fontsize=36)
#     axes[1].set_title("", fontsize=36)

#     axes[0].set_xlabel("Time (min)", fontsize=36)
#     axes[1].set_xlabel("Time (min)", fontsize=36)

#     axes[0].set_ylim(0, max1+1)
#     axes[1].set_ylim(0, max2+2)
  
#     axes[0].tick_params(axis='both', labelsize=36)
#     axes[1].tick_params(axis='both', labelsize=36)

#     axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
#     axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)

#     f.tight_layout()
#     f.savefig(CLUST_DISAMBIG_DIR+identifier+'_cumulative_plasticity_cluster_changes_over_time.png', dpi=300)#plt.

#     return

def plot_cumulative_plasticity_changes_main(df, identifier='\_allcells', miny=None, maxy=None, t_window_multiplier = T_WINDOW_MULTIPLIER, plotametric=None, plotallcells = False): #spidey
    
    if plotallcells:
        f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False)
    else:
        f, axes = plt.subplots(2, 1, figsize=(15, 20), sharex=False)
    # f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False) #sharex=True

    if plotametric != None:
        whattoplot=[plotametric, plotametric]
    else:
        whattoplot=['cum_n_changes','cum_n_labels',]

    time = df['frame']

    timeminutes=time*SAMPLING_INTERVAL

    ##
    if miny != None or maxy != None:
        minimumy=miny
        maximumy1=maxy
        maximumy2=maxy

    else:
        minimumy=0
        maximumy1=np.nanmax(df[whattoplot[0]])
        maximumy2=np.nanmax(df[whattoplot[1]])

    ##
    import seaborn as sns
    sns.set_theme(style="ticks")

    colors=[]
    if CONDITION_CMAP != 'Dark24':

        cmap = cm.get_cmap(CONDITION_CMAP)
        numcolors= len(df['Condition_shortlabel'].unique())
        sns.set_palette(CONDITION_CMAP)
        for i in range(numcolors):
            colors.append(cmap(i))
    else:
        colors = plotlytomatplotlibcolors()
        colors=colors[:len(df['Condition_shortlabel'].unique())]

    df=df.dropna(how='any')

    conditionsindf = df['Condition_shortlabel'].unique()

    if plotallcells == False:
            
        # Plot the responses for different events and regions
        sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0],
                    hue="Condition_shortlabel",
                    data=df, palette=colors)

        sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1],
                    hue="Condition_shortlabel",
                    data=df, palette=colors)
    else:
        # Create a dictionary to map condition_shortlabel to colors
        condition_colors = {condition: color for condition, color in zip(df['Condition_shortlabel'].unique(), colors)}
        # Make timeminuteslist and get the last element for the annotations
        timeminuteslist = timeminutes.tolist()
        # x_final = timeminuteslist[-1] # Final entry in x_data
        max_x = np.max(timeminutes)
        # Create a set to keep track of plotted condition_shortlabels
        plotted_labels = set()
        # This will be used to offset the x position of the annotations
        x_offsets = [20, 100, 180, 260, 340, 420, 500] * int(len(df['uniq_id'].unique()) / 2 + 1)
        y_offsets = [0, -40, -80, -120] * int(len(df['uniq_id'].unique()) / 2 + 1)

        for offset, uniq_id in enumerate(df['uniq_id'].unique()):
            df_uniq = df[df['uniq_id'] == uniq_id]
            condition_label = df_uniq['Condition_shortlabel'].iloc[0]  # Get the condition_shortlabel for this 'uniq_id'
            line_color = condition_colors[condition_label]  # Get the corresponding color for the condition_shortlabel
            if condition_label not in plotted_labels:
                # Plot the line with a label for the legend only if it hasn't been plotted before
                singleline = sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], data=df_uniq, color=line_color, label=condition_label)
                singleline2 = sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], data=df_uniq, color=line_color, label=condition_label)
                plotted_labels.add(condition_label)  # Add the condition_shortlabel to the set of plotted labels
            else:
                # If the label has already been plotted, don't add it to the legend again
                singleline = sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], data=df_uniq, color=line_color)
                singleline2 = sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], data=df_uniq, color=line_color)

            x_final_frames = df_uniq['frame'].iloc[-1]  # Final entry in x_data
            x_final = x_final_frames*SAMPLING_INTERVAL

            y_final = df_uniq[whattoplot[0]].iloc[-1]  # Final entry in y_data
            y_final1 = df_uniq[whattoplot[1]].iloc[-1]  # Final entry in y_data
            
            uniq_id_label = f'Cell_ID: {uniq_id}'
            axes[0].annotate(uniq_id_label, xy=(max_x + x_offsets[offset], y_final), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center', arrowprops=dict(arrowstyle="<-", color='black'))
            axes[0].plot([x_final, max_x + x_offsets[offset]], [y_final, y_final], linestyle='dotted', color='gray', transform=axes[0].transData, zorder=4)

            axes[1].annotate(uniq_id_label, xy=(max_x + x_offsets[offset], y_final1 ), xytext=(10, 0), textcoords='offset points', fontsize=14, ha='left', va='center', arrowprops=dict(arrowstyle="<-", color='black'))
            axes[1].plot([x_final, max_x + x_offsets[offset]], [y_final1, y_final1], linestyle='dotted', color='gray', transform=axes[1].transData, zorder=4)            
        
    timewindowmins = (MIG_T_WIND*t_window_multiplier) * SAMPLING_INTERVAL
    timewindowmins = round(timewindowmins, 1)

    print('Time window mins: ', timewindowmins)
    text1 = "Cumulative cluster switches"
    text2 = "Cumulative new clusters"

    x_lab = "Distinct Behaviors"
    plottitle = ""

    # get the max value of the whattoplot columns of df
    max1 = np.nanmax(df[whattoplot[0]])
    max2 = np.nanmax(df[whattoplot[1]])
    # ensure tick frequencies are at least 1
    tickfrequency1 = max(int(max1/5), 1)
    tickfrequency2 = max(int(max2/5), 1)

    the_yticks1 = np.arange(0, len(df[whattoplot[0]].unique()), tickfrequency1)
    the_yticks1 = [int(x) for x in the_yticks1]
    axes[0].set_yticks(the_yticks1)
    axes[0].margins(y=0)

    the_yticks2 = np.arange(0, len(df[whattoplot[1]].unique()), tickfrequency2)
    the_yticks2 = [int(x) for x in the_yticks2]
    axes[1].set_yticks(the_yticks2)
    axes[1].margins(y=0)

    # Tweak the visual presentation
    axes[0].xaxis.grid(True)
    axes[1].xaxis.grid(True)

    axes[0].set_ylabel(text1, fontsize=36)
    axes[1].set_ylabel(text2, fontsize=36)

    axes[0].set_title("", fontsize=36)
    axes[1].set_title("", fontsize=36)

    axes[0].set_xlabel("Time (min)", fontsize=36)
    axes[1].set_xlabel("Time (min)", fontsize=36)

    axes[0].set_ylim(0, max1+1)
    axes[1].set_ylim(0, max2+2)
  
    axes[0].tick_params(axis='both', labelsize=36)
    axes[1].tick_params(axis='both', labelsize=36)

    axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left', fontsize=36, markerscale=20, fancybox=True)
    axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left', fontsize=36, markerscale=20, fancybox=True)

    f.tight_layout()
    f.savefig(CLUST_DISAMBIG_DIR + identifier + '_cumulative_plasticity_cluster_changes_over_time.png', dpi=300)

    return


# def plot_cumulative_plasticity_changes_multiples(df, identifier='\_allcells', miny=None, maxy=None, t_window_multiplier = T_WINDOW_MULTIPLIER, plotallcells = False): #spidey

#     if plotallcells:

#         # find the number of unique uniq_ids
#         num_uniq_ids = len(df['uniq_id'].unique())
#         # find the number of conditions
#         num_conditions = len(df['Condition_shortlabel'].unique())
#         # Use that to make an array of subplots for each condition 
#         f, axes = plt.subplots(num_uniq_ids, num_conditions, figsize=(5*num_uniq_ids, 5*num_conditions), sharex=False) #sharex=True
#         f2, axes2 = plt.subplots(num_uniq_ids, num_conditions, figsize=(5*num_uniq_ids, 5*num_conditions), sharex=False) #sharex=True

#     else:
#         f, axes = plt.subplots(2, 1, figsize=(15, 20), sharex=False)
#     # f, axes = plt.subplots(2, 1, figsize=(25, 20), sharex=False) #sharex=True

#     whattoplot=['cum_n_changes','cum_n_labels',]

#     time = df['frame']

#     timeminutes=time*SAMPLING_INTERVAL

#     ##
#     import seaborn as sns
#     sns.set_theme(style="ticks")

#     colors=[]
#     if CONDITION_CMAP != 'Dark24':

#         cmap = cm.get_cmap(CONDITION_CMAP)
#         numcolors= len(df['Condition_shortlabel'].unique())
#         sns.set_palette(CONDITION_CMAP)
#         for i in range(numcolors):
#             colors.append(cmap(i))
#     else:
#         colors = plotlytomatplotlibcolors()
#         colors=colors[:len(df['Condition_shortlabel'].unique())]

#     df=df.dropna(how='any')

#     conditionsindf = df['Condition_shortlabel'].unique()


#     if plotallcells == False:
            
#         # Plot the responses for different events and regions
#         sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors)

#         sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
#                     hue="Condition_shortlabel",
#                     data=df, palette=colors)
#     else:
#         # Create a dictionary to map condition_shortlabel to colors
#         condition_colors = {condition: color for condition, color in zip(df['Condition_shortlabel'].unique(), colors)}
#         # Make timeminuteslist and get the last element for the annotations
#         timeminuteslist = timeminutes.tolist()
#         # x_final = timeminuteslist[-1] # Final entry in x_data
#         max_x = np.max(timeminutes)
#         # Create a set to keep track of plotted condition_shortlabels
#         plotted_labels = set()
#         # This will be used to offset the x position of the annotations
#         # x_offsets = np.arange(20, len(df['uniq_id'].unique()), 1)
#         # x_offsets = [20, 100, 180, 260, 340, 420, 500] * int(len(df['uniq_id'].unique()) / 2 + 1) #used for placing labels in readable positions



#         for offset, uniq_id in enumerate(df['uniq_id'].unique()):
#             df_uniq = df[df['uniq_id'] == uniq_id]
#             condition_label = df_uniq['Condition_shortlabel'].iloc[0]  # Get the condition_shortlabel for this 'uniq_id'
#             line_color = condition_colors[condition_label]  # Get the corresponding color for the condition_shortlabel
#             if condition_label not in plotted_labels:
#                 # Plot the line with a label for the legend only if it hasn't been plotted before
#                 f = sns.lineplot(ax=axes[0, offset], x=timeminutes, y=whattoplot[0], data=df_uniq, color=line_color, label=condition_label)
#                 f2 = sns.lineplot(ax=axes[0,offset], x=timeminutes, y=whattoplot[1], data=df_uniq, color=line_color, label=condition_label)
#                 plotted_labels.add(condition_label)  # Add the condition_shortlabel to the set of plotted labels
#             else:
#                 # If the label has already been plotted, don't add it to the legend again
#                 singleline = sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], data=df_uniq, color=line_color)
#                 singleline2 = sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], data=df_uniq, color=line_color) #, label=condition_label

#             x_final_frames = df_uniq['frame'].iloc[-1]  # Final entry in x_data
#             x_final = x_final_frames*SAMPLING_INTERVAL

#             y_final = df_uniq[whattoplot[0]].iloc[-1]  # Final entry in y_data
#             y_final1 = df_uniq[whattoplot[1]].iloc[-1]  # Final entry in y_data
            
#             uniq_id_label = f'Cell_ID: {uniq_id}'
    
        
#     timewindowmins = (MIG_T_WIND*t_window_multiplier) * SAMPLING_INTERVAL #Here is just a superficial matching of the time window specified in the config for plotting purposes
#     # The actual calculation is done in count_cluster_changes_with_tavg function
#     timewindowmins = round(timewindowmins, 1)

#     print('Time window mins: ', timewindowmins)
#     text1 = "Cumulative cluster switches"
#     text2 = "Cumulative new clusters"
#     # text3 = "New clusters / " + str(timewindowmins) + " min"

#     x_lab = "Distinct Behaviors"
#     plottitle = ""

#     # get the max value of the whattoplot[1] column of df
#     max1=np.nanmax(df[whattoplot[0]])
#     max2=np.nanmax(df[whattoplot[1]])
#     tickfrequency1 = int(max1/5)
#     tickfrequency2 = int(max2/5)

#     the_yticks = np.arange(0, len(df[whattoplot[0]].unique()), tickfrequency1)
#     the_yticks = [int(x) for x in the_yticks]
#     axes[0].set_yticks(the_yticks) # set new tick positions
#     axes[0].margins(y=0) # set tight margins
#     the_yticks = np.arange(0, len(df[whattoplot[1]].unique()),tickfrequency2 ) #tickfrequency1)
#     the_yticks = [int(x) for x in the_yticks]


#     axes[1].set_yticks(the_yticks) # set new tick positions
#     axes[1].margins(y=0) # set tight margins

#     # Tweak the visual presentation
#     axes[0].xaxis.grid(True)
#     axes[1].xaxis.grid(True)

#     axes[0].set_ylabel(text1, fontsize=36)
#     axes[1].set_ylabel(text2, fontsize=36)

#     axes[0].set_title("", fontsize=36)
#     axes[1].set_title("", fontsize=36)

#     axes[0].set_xlabel("Time (min)", fontsize=36)
#     axes[1].set_xlabel("Time (min)", fontsize=36)

#     axes[0].set_ylim(0, max1+1)
#     axes[1].set_ylim(0, max2+2)
  
#     axes[0].tick_params(axis='both', labelsize=36)
#     axes[1].tick_params(axis='both', labelsize=36)

#     axes[0].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)
#     axes[1].legend(title='', bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=36,markerscale=20,fancybox=True)

#     f.tight_layout()
#     f.savefig(CLUST_DISAMBIG_DIR+identifier+'_cumulative_plasticity_cluster_changes_over_time.png', dpi=300)#plt.

#     return

############# DEV PLOTS 2 END ######################


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
    # colors=[]
    # cmap = cm.get_cmap(CONDITION_CMAP, len(df['Condition_shortlabel'].unique()))
    # for i in range(cmap.N):
    #     colors.append(cmap(i))

    colors=[]
    if CONDITION_CMAP != 'Dark24':
        cmap = cm.get_cmap(CONDITION_CMAP, )
        numcolors=len(df['Condition_shortlabel'].unique())
        # sns.set_palette(CONDITION_CMAP)
        for i in range(numcolors):
            colors.append(cmap(i))
    else:
        colors = plotlytomatplotlibcolors()
        colors=colors[:len(df['Condition_shortlabel'].unique())]    

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

def purityplot_percentcluspercondition(df, df2, cluster_by=CLUSTER_BY, save_path=CLUST_DIR, cluster_label='label',dotsize = 0.1 ):

    fontsize = 24
    # Create a Subplot figure that shows the effect of clustering between conditions
    fig = plt.figure(figsize=(25, 10))
    fig.suptitle("How much each condition occupies each cluster", fontsize=fontsize)

    alpha = 0.5
    markerscale = 5
    markerscale2 = 100
    colors=[]
    if CONDITION_CMAP != 'Dark24':
        cmap = cm.get_cmap(CONDITION_CMAP)
        numcolors= len(df['Condition_shortlabel'].unique())
        for i in range(numcolors):
            colors.append(cmap(i))
    else:
        colors = plotlytomatplotlibcolors()
        colors=colors[:len(df['Condition_shortlabel'].unique())]    

    if cluster_label == 'label': 

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

        x = 'UMAP1'
        y = 'UMAP2'
        z = 'UMAP3'

    # PLOT NUMBER 1: UMAP COLORED BY CONDITION

        for colorselector in range(len(colors)):
            ax1.scatter(df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][x],
                            df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][y],
                            df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][z],
                            '.', color=colors[colorselector], label = df['Condition_shortlabel'].unique()[colorselector],s=dotsize, alpha = alpha)
            
        leg=ax1.legend(loc='upper left', numpoints=1, ncol=1, fontsize=fontsize, bbox_to_anchor=(1.05, 1.0), markerscale=markerscale2)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.set_xlabel(x, fontsize=fontsize, linespacing=3.2) # gives a new line to make space for the axis label
        ax1.set_ylabel(y, fontsize=fontsize, linespacing=3.2)
        ax1.set_zlabel(z, fontsize=fontsize, linespacing=3.2)

    elif cluster_label == 'trajectory_id':

        ax1 = fig.add_subplot(1, 2, 1)
        x = 'UMAP_traj_1'
        y = 'UMAP_traj_2'

        for colorselector in range(len(colors)):
            ax1.scatter(df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][x],
                            df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][y],
                            s=dotsize, color=colors[colorselector], marker='o', label = df['Condition_shortlabel'].unique()[colorselector],alpha = alpha)
            
            # x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None,
            
        leg=ax1.legend(loc='upper left', numpoints=1, ncol=1, fontsize=fontsize, bbox_to_anchor=(1.05, 1.0), markerscale=markerscale)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)        
        
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        # ax1.set_zticklabels([])
        ax1.set_xlabel(x, fontsize=fontsize, linespacing=3.2) # gives a new line to make space for the axis label
        ax1.set_ylabel(y, fontsize=fontsize, linespacing=3.2)
        # ax1.set_zlabel(z, fontsize=fontsize, linespacing=3.2)

    ##### Making the second axis: Stacked bar plot of cluster purity

    ax2 = fig.add_subplot(1, 2, 2)

    #defining a colormap for the clusters
    cluster_colors = []
    labels = list(set(df[cluster_label].unique())) #changed
    # print('first you make this one')
    # print(labels)
    if -1 in labels:
        labels.remove(-1)
        # add -1 to the start of the list
        labels.insert(0, -1)
    print(labels)
    cmap = cm.get_cmap(CLUSTER_CMAP) #Changes made here
    numlabelcolors= len(labels)
    for i in range(numlabelcolors):
        cluster_colors.append(cmap(i))

    print(f'These are the cluster colors: {cluster_colors}')

    # cluster_colors = []
    # cmap = cm.get_cmap(CLUSTER_CMAP) 
    # numcolors=len(df['trajectory_id'].unique())
    # for i in range(numcolors):
    #     cluster_colors.append(cmap(i))    
    # set x axis as 'Condition' column
    x = df2['Condition']
    # set the number of ClusterID columns to plot
    num_clusters = len(labels)
    # create empty list for bottom values of each bar
    bottoms = [0] * len(df2)
    # create stacked bar plot

    ###################################################
# Create stacked bar plot and add text labels
    for i, clus in enumerate(labels):
        # Set y values as the ith ClusterID column
        col_name = 'Percent_condition_pts_in_ClusterID_' + str(clus)
        
        # Check if column exists, if not, create it with zeros
        if col_name not in df2.columns:
            y = pd.Series([0.0] * len(df2), index=df2.index)
        else:
            y = df2[col_name].fillna(0.0)  # Fill any remaining NaN values with 0.0

        # Skip bars with 0 percentage
        if all(y == 0):
            continue
        # Create a bar plot for the ith ClusterID column
        # if 'color' in df.columns:
        #     bars = ax2.bar(x, y, bottom=bottoms, label='Cluster ID ' + str(clus), color=df['color'])
        # else:
        #     bars = ax2.bar(x, y, bottom=bottoms, label='Cluster ID ' + str(clus), color=cluster_colors[i])
        bars = ax2.bar(x, y, bottom=bottoms, label='Cluster ID ' + str(clus), color=cluster_colors[i])
        
        # Calculate the positions for text labels
        label_x = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        label_y = [bottom + height / 2 for bottom, height in zip(bottoms, y)]
        
        # Add text labels with percentages
        for j, txt in enumerate(y):
            if txt != 0:
                ax2.text(label_x[j], label_y[j], f'{txt:.2f}%', ha='center', va='center', fontsize=fontsize)
        
        # Update the bottom values of each bar to include the current y values
        bottoms += y

    #rotate the xticklabels 90 degrees
    ax2.set_xticklabels(x, rotation=90)    
    ax2.set_ylabel('Percent condition per cluster', fontsize = fontsize)
    ax2.set_xlabel('', fontsize = fontsize)
    ax2.legend(loc='upper right', numpoints=1, ncol=1, fontsize=fontsize, bbox_to_anchor=(1.6, 1.00), markerscale=markerscale)

    # Set the colors of the x tick labels to match with the condition CMAP for easy reference
    for ticklabel, tickcolor in zip(ax2.get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)

    ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    # Increase the spacing between the subplots so that ax1 legend is not cut off
    fig.subplots_adjust(wspace=0.8)

    # show the plot
    plt.show()
    if cluster_label == 'trajectory_id': 
        fig.savefig(TRAJECTORY_DISAMBIG_DIR+'PurityPlotConditionsinClusters.png', dpi=300, bbox_inches='tight')
    elif cluster_label == 'label':
        fig.savefig(CLUST_DISAMBIG_DIR+'PurityPlotClustersinConditions.png', dpi=300, bbox_inches='tight')
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

  
        elif scalingmethod == 'standard':
            # apply StandardScaler to all metrics, then re-attach UMAP columns
            from sklearn.preprocessing import StandardScaler
            x_std = StandardScaler().fit_transform(Z) 
            scaled_df = pd.DataFrame(data=x_std, columns=metrics)
            df_out = pd.concat([scaled_df, df_in[[x, y, z]]], axis=1)


            
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


def plot_small_multiples(dataframe, column_name):
    # Calculate the 'timeminutes' variable
    dataframe['timeminutes'] = dataframe['frame'] * SAMPLING_INTERVAL

    # Get unique 'Condition_shortlabel' values from the DataFrame
    unique_conditions = dataframe['Condition_shortlabel'].unique()

    FONT_SIZE = 12

    # Find the global minimum and maximum values for the 'y' axis
    global_y_min = dataframe[column_name].min()
    global_y_max = dataframe[column_name].max()

    global_x_min = dataframe['timeminutes'].min()
    global_x_max = dataframe['timeminutes'].max()

    # Create a mapping of 'Condition_shortlabel' to colors
    color_map = plt.get_cmap(CONDITION_CMAP)
    # CONDITION_CMAP
    condition_colors = {condition: color_map(i) for i, condition in enumerate(unique_conditions)}

    # Create a dictionary to store separate sets of subplots for each condition
    condition_subplots = {}

    # Find the maximum number of rows and columns across both sets of subplots
    max_rows = max_cols = 0
    for condition in unique_conditions:
        condition_df = dataframe[dataframe['Condition_shortlabel'] == condition]
        num_plots = len(condition_df['uniq_id'].unique())
        num_cols = int(num_plots ** 0.5)
        num_rows = (num_plots + num_cols - 1) // num_cols
        max_rows = max(max_rows, num_rows)
        max_cols = max(max_cols, num_cols)

    # Calculate the common figure size based on the maximum number of rows and columns
    figsize = (max_cols * 5, max_rows * 2)  # Adjust the width (5) to control the aspect ratio

    # Create subplots for each condition
    for condition in unique_conditions:
        condition_df = dataframe[dataframe['Condition_shortlabel'] == condition]
        num_plots = len(condition_df['uniq_id'].unique())
        num_cols = int(num_plots ** 0.5)
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=False)

        # Set the ylabel based on the column_name
        ylabel = 'Cluster ID' if column_name == 'label' else column_name

        # Flatten the axes array if it's a single row or column grid
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Adjust layout and show the plots for each condition
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.5, w_pad=1.5)   
        
        # Set the font size for all subplots on all axes in all figures to be larger
        plt.rcParams.update({'font.size': FONT_SIZE})     
            # Set the x tick font size to be larger
        plt.xticks(fontsize=FONT_SIZE)
        # and y tick font size to be larger
        plt.yticks(fontsize=FONT_SIZE)
        # and the y label font size to be larger
        plt.ylabel(ylabel, fontsize=FONT_SIZE)
        # and the x axis label font size to be larger
        plt.xlabel('Time (minutes)', fontsize=FONT_SIZE)

        # fig.savefig(CLUST_DISAMBIG_DIR+ylabel+'smallmultiplesovertime.png', dpi=300, bbox_inches='tight')

        for i, uniq_id in enumerate(condition_df['uniq_id'].unique()):
            ax = axes[i // num_cols, i % num_cols]
            df_subset = condition_df[condition_df['uniq_id'] == uniq_id]
            ax.plot(df_subset['timeminutes'], df_subset[column_name], label=f'ID: {uniq_id}', color=condition_colors[condition])
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(ylabel)

            # Set y-axis limits for each subplot to be consistent with global min and max
            ax.set_ylim(global_y_min-0.5, global_y_max+0.5)
            ax.set_xlim(global_x_min-0.5, global_x_max+0.5)

            # Show all integer tick values on the y-axis if the column_name is 'label'
            if column_name == 'label':
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                

            ax.legend()

            # ax.grid(True)
            # f.savefig(CLUST_DISAMBIG_DIR+ylabel+'smallmultiplesovertime.png', dpi=300, bbox_inches='tight')

        # Remove any empty subplots
        if num_plots < num_rows * num_cols:
            for i in range(num_plots, num_rows * num_cols):
                axes.flat[i].set_visible(False)

        # Add the title for each set of subplots based on the 'Condition_shortlabel'
        fig.suptitle(f'Condition: {condition}')

        # Add the subplots for this condition to the dictionary
        condition_subplots[condition] = (fig, axes)

        # filename = os.path.join(output_directory, f'plot_{condition}.png')
        # fig.savefig(filename)
        fig.savefig(CLUST_DISAMBIG_DIR+ylabel+f'smallmultiplesovertime_{condition}.png', dpi=300, bbox_inches='tight')

    # Add the legend for each condition using the color_map
    fig, axes = plt.subplots(1, 1, figsize=(12, 1))
    for i, condition in enumerate(unique_conditions):
        axes.plot([], [], label=f'Condition: {condition}', color=color_map(i))
    axes.set_axis_off()
    axes.legend(ncol=len(unique_conditions))

    # Set the font size for all subplots on all axes in all figures to be larger
    plt.rcParams.update({'font.size': FONT_SIZE})     
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylabel(ylabel, fontsize=FONT_SIZE)
    plt.xlabel('Time (minutes)', fontsize=FONT_SIZE)
    fig.savefig(CLUST_DISAMBIG_DIR+ylabel+'smallmultiplesovertime_legend.png', dpi=300, bbox_inches='tight')

    plt.show()
    return

import pandas as pd
import random
import matplotlib.pyplot as plt

############################################################

def diagnose_plasticity_data(df):
    """
    Diagnostic function to help identify issues with plasticity plotting data.
    Call this before plot_plasticity_changes to debug issues.
    """
    print("=== PLASTICITY DATA DIAGNOSIS ===")
    print(f"Dataframe shape: {df.shape}")
    print(f"Dataframe columns: {list(df.columns)}")
    
    required_columns = ['label', 'twind_n_changes', 'twind_n_labels', 'frame', 'Condition_shortlabel']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n❌ MISSING REQUIRED COLUMNS: {missing_columns}")
        return False
    else:
        print(f"\n✅ All required columns present: {required_columns}")
    
    print(f"\n=== NaN ANALYSIS ===")
    for col in required_columns:
        nan_count = df[col].isna().sum()
        total_count = len(df)
        percentage = (nan_count / total_count) * 100 if total_count > 0 else 0
        print(f"{col}: {nan_count}/{total_count} ({percentage:.1f}%) are NaN")
    
    print(f"\n=== DATA TYPE ANALYSIS ===")
    for col in required_columns:
        print(f"{col}: {df[col].dtype}")
    
    print(f"\n=== UNIQUE VALUE COUNTS ===")
    for col in required_columns:
        if col in df.columns:
            unique_count = df[col].nunique(dropna=False)
            print(f"{col}: {unique_count} unique values")
            if col == 'Condition_shortlabel':
                print(f"  Conditions: {list(df[col].dropna().unique())}")
    
    # Check what happens when we remove NaN values
    plotting_columns = ['label', 'twind_n_changes', 'twind_n_labels', 'Condition_shortlabel']
    df_clean = df.dropna(subset=plotting_columns, how='any')
    print(f"\n=== DATA AFTER REMOVING NaN ===")
    print(f"Rows remaining: {len(df_clean)}/{len(df)} ({len(df_clean)/len(df)*100:.1f}%)")
    
    if len(df_clean) == 0:
        print("❌ NO DATA REMAINS after removing NaN values!")
        print("This is likely the cause of your plotting error.")
        return False
    else:
        print("✅ Data remains after cleaning")
        return True

###############################################################


