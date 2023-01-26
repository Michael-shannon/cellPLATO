#cluster_visualization.py

from initialization.config import *
from initialization.initialization import *

from visualization.low_dimension_visualization import colormap_pcs

import numpy as np
import pandas as pd
import os


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

def plot_3D_scatter(df, x, y, z, colorby, ticks=False, identifier=''): #new matplotlib version of scatter plot for umap 1-26-2023
    import matplotlib.pyplot as plt
    from numpy.random import random
    from mpl_toolkits.mplot3d import Axes3D

    font_size = 24
    # df = lab_dr_df

    fig = plt.figure(figsize=(15, 15))

    if colorby == 'label':
        coloredby = 'label'
        colors = cm.tab20(np.linspace(0, 1, len(df['label'].unique())))  
        df = df.sort_values(by=['label'])
       
    elif colorby == 'condition':
        coloredby = 'Condition_shortlabel'
        colors = cm.rainbow(np.linspace(0, 1, len(df['Condition_shortlabel'].unique())))

    ax = plt.subplot(111, projection='3d')
     
    for colorselector in range(len(colors)): #you have to plot each label separately to get the legend to work
        if colorby == 'label':
            ax.scatter(df[df['label'] == df['label'].unique()[colorselector]][x], df[df['label'] == df['label'].unique()[colorselector]][y],
             df[df['label'] == df['label'].unique()[colorselector]][z], '.', color=colors[colorselector], label = df['label'].unique()[colorselector], s=3)
        elif colorby == 'condition':
            ax.scatter(df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][x], df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][y],
             df[df['Condition_shortlabel'] == df['Condition_shortlabel'].unique()[colorselector]][z], '.', color=colors[colorselector], label = df['Condition_shortlabel'].unique()[colorselector], s=3)

    plt.legend(loc='upper left', numpoints=1, ncol=1, fontsize=font_size, bbox_to_anchor=(1.05, 1.0), markerscale=5)
    
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

    CLUSTER_CMAP = 'tab20'
    CONDITION_CMAP = 'dark'

    # CLUST_DISAMBIG_DIR = 'D:/Michael_Shannon/CELLPLATO_MASTER/OCTOBERTESTING_/ThreeConditions_Go_Stopping_Stopped_FinalFig1_OUTPUT/ThreeConditions_Go_Stopping_Stopped_12-14-2022/2022-12-15_10-23-41-828185/plots/Clustering/Cluster_Disambiguation'

    # df = lab_dr_df #lab_tavg_dr_df #lab_dr_df

    pal = sns.color_palette(CONDITION_CMAP) #extracts a colormap from the seaborn stuff.
    cmap=pal.as_hex()[:] #outputs that as a hexmap which is compatible with plotlyexpress below

    if 'label' in df.columns:
        df['label'] = pd.Categorical(df.label)


    import plotly.express as px
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

def plot_plasticity_changes(df, identifier='\_allcells', miny=None, maxy=None):

    # f, axes = plt.subplots(1, 3, figsize=(15, 5)) #sharex=True
    # f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=True) #sharex=True
    f, axes = plt.subplots(3, 1, figsize=(15, 30), sharex=False) #sharex=True

    whattoplot=['label','twind_n_changes', 'twind_n_labels']

    CLUSTER_CMAP = 'tab20'
    CONDITION_CMAP = 'dark'

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
    # display(df)
    df=df.dropna(how='any')
    # display(df)
    # Plot the responses for different events and regions
    sns.lineplot(ax=axes[0], x=timeminutes, y=whattoplot[0], #n_labels #n_changes #label
                 hue="Condition_shortlabel",
                 data=df)

    sns.lineplot(ax=axes[1], x=timeminutes, y=whattoplot[1], #n_labels #n_changes #label
                 hue="Condition_shortlabel",
                 data=df)

    sns.lineplot(ax=axes[2], x=timeminutes, y=whattoplot[2], #n_labels #n_changes #label
                 hue="Condition_shortlabel",
                 data=df)

    timewindowmins = MIG_T_WIND*SAMPLING_INTERVAL
    text1 = "Cluster ID per frame"
    text2 = "Distinct changes per " + str(timewindowmins) + " min time window"
    text3 = "New cluster changes per " + str(timewindowmins) + " min time window"

    x_lab = "Distinct Behaviors"
    plottitle = ""
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

def plot_plasticity_countplots(df, identifier='\_allcells'):

    import matplotlib.pyplot as plt
    import seaborn as sns

    # f, axes = plt.subplots(1, 2, figsize=(30, 15), sharey=True)
    f, axes = plt.subplots(1, 2, figsize=(30, 8), sharey=False)
    # f, axes = plt.subplots(2, 1, figsize=(15, 30), sharey=False)
    # ax.set_xscale("log")

    # whattoplot=['n_changes', 'n_labels'] #'twind_n_changes', 'twind_n_labels'
    whattoplot=['twind_n_changes', 'twind_n_labels']
    CLUSTER_CMAP = 'tab20'
    CONDITION_CMAP = 'dark'

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
