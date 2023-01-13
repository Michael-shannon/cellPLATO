#low_dimension_visualization.py

from initialization.config import *
from initialization.initialization import *

from data_processing.clustering import hdbscan_clustering
from data_processing.dimensionality_reduction import *
from data_processing.shape_calculations import *


import numpy as np
import pandas as pd
import os
import imageio

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# matplotlib imports
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.),
    "figure.figsize":    (10,10),
    "font.size": 12
})


from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.),
    "figure.figsize":    (10,10),
    "font.size": 12
})

# Datashader imports
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler

import math
import ternary

def correlation_matrix(df):

    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number, cmap='viridis')
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()

    return f

def pca_factor_vis(df, pca_tuple, dr_factors=DR_FACTORS):

    pca_df, components, expl = pca_tuple#do_pca(df[dr_factors])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(15, 15))

    ax1.imshow(components)
    ax1.set_yticklabels(dr_factors)
    ax1.set_yticks(np.arange(len(dr_factors)))
    ax1.set_xticklabels(range(1,len(components)))
    ax1.set_xticks(range(0,len(components[0])))
    ax1.title.set_text('Principal compnents')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Factor')

    ax2.imshow(components*expl)
    ax2.set_yticklabels(dr_factors)
    ax2.set_yticks(np.arange(len(dr_factors)))
    ax2.set_xticklabels(range(1,len(components)))
    ax2.set_xticks(range(0,len(components[0])))
    ax2.title.set_text('Componenets * variance explained')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Factor')

    ax3.imshow(np.expand_dims(np.sum(components*expl,axis=1), axis=1))
    ax3.set_yticklabels(dr_factors)
    ax3.set_yticks(np.arange(len(dr_factors)))
    ax3.set_xticks(range(0))
    ax3.set_ylabel('Factor')
    ax3.title.set_text('Sum variance contribution per factor')

    ax4.plot(expl)
    ax4.title.set_text('Variance accounted for')
    ax4.set_xlabel('Principal Component')
    factor_variance = np.sum(components*expl,axis=1)


    if STATIC_PLOTS:

        plt.savefig(DR_DIR + '\pca_variance.png', format='png', dpi=600)

    return fig

def pca_factor_matrix(df,pca_tuple, dr_factors=DR_FACTORS, ax=None):

    # If no axis is supplied, then createa simple fig, ax and default to drawing the points.
    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')

    x = df[dr_factors].values

    pca_df, components, expl = pca_tuple#do_pca(x)
    dr_df = pd.concat([df,pca_df], axis=1)

    # Make the matrix plot
    # fig, ax = plt.subplots(1, 1,figsize=(15, 20))
    # fig.patch.set_facecolor('white')
    ax.imshow(components)
    ax.set_yticklabels(dr_factors)
    ax.set_yticks(np.arange(len(dr_factors)))
    ax.set_xticklabels(range(1,len(components)))
    ax.set_xticks(range(0,len(components[0])))
    ax.title.set_text('Principal components')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Factor')

    # if ax is None:

    return ax, dr_df

def colormap_pcs(dr_df, cmap = 'rgb'):

    pcs = np.asarray(dr_df[['PC1','PC2','PC3']])
    scaler = MinMaxScaler()
    scaler.fit(pcs)
    pc_colors = scaler.transform(pcs)

    if cmap != 'rgb':

        if cmap == 'cmy':

            pc_colors = rgb2cmy(pc_colors)

    pc_colors = np.clip(pc_colors, 0, 1)

    return pc_colors


def rgb2cmy(rgb_arr):

    '''
    Allows recoloring the 3-factor rgb array into cyan, magenta, yellow
    '''

    x = rgb_arr[:,0]
    y = rgb_arr[:,1]
    z = rgb_arr[:,2]

    w = 255
    x_color = x * w #/ float(scale)
    y_color = y * w #/ float(scale)
    z_color = z * w #/ float(scale)

    r = np.abs(w - y_color) / w
    g = np.abs(w - x_color) / w
    b = np.abs(w - z_color) / w

    color_arr = np.c_[r,g,b]
    print(color_arr.shape)


    return color_arr


def datashader_lines(df_in, x,y,color_by='Condition', output_res=500, aspect=1,categorical=False, export=False, identifier = ''):

    df = df_in.copy()

    # Need to add conditions as category datatype to use multi-color datashader
    width = output_res
    height = int(output_res / aspect)

    cvs = ds.Canvas(plot_width=width, plot_height=height)#,x_range=x_range, y_range=y_range)

    if categorical:

        df['Cat'] = df[color_by].astype('category')
        # Multicolor categorical
        agg = cvs.line(df,  x, y, agg=ds.count_cat('Cat'))
        img = tf.set_background(tf.shade(agg, how='eq_hist'),"black")

    else:

        agg = cvs.line(df, x, y, agg=ds.count())
        img = tf.set_background(tf.shade(agg, cmap=cm.inferno, how='linear'),"black")

    if STATIC_PLOTS:

        # plt.savefig(CLUST_DIR+label+'.png', dpi=300)
        figname = CLUST_DIR+identifier+'_datashaderlines.png'
        export_image(img, figname, background="black")

    return img


def spatial_img_coloc(df_in, xy='tSNE',thresh=2,n_bins=50):

    '''
    Visualize dimensionally reduced space as a histogram and perfrom image
    collocalization between the images.

    TO DO: Update this to work for the inputted conditions...
    By Default it assumes the first is the control and the second is for comparison.
    '''


    if(xy == 'tsne' or xy == 'tSNE'):

            x_lab = 'tSNE1'
            y_lab = 'tSNE2'

    elif xy == 'PCA':

            x_lab = 'PC1'
            y_lab = 'PC2'

    elif (xy == 'umap' or xy == 'UMAP'):

            x_lab = 'UMAP1'
            y_lab = 'UMAP2'

    df = df_in.copy()

    # Get the list of conditions included in the dataframe. By default show the first two.
    cond_list = df['Condition_shortlabel'].unique()
    print(cond_list)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,  figsize=(10,10))

    ax1.title.set_text('Combined set')
    # ax2.title.set_text(CTL_LABEL)
    # ax3.title.set_text(CONDITIONS_TO_INCLUDE[1])
    # ax2.title.set_text(CONDITION_SHORTLABELS[0])
    # ax3.title.set_text(CONDITION_SHORTLABELS[1])
    ax2.title.set_text(cond_list[0])
    ax3.title.set_text(cond_list[1])
    ax4.title.set_text('Colocalization')


    # xy_range = [[-60, 60], [-40, 40]]
    xy_range = [[np.min(df[x_lab]), np.max(df[x_lab])], [np.min(df[y_lab]), np.max(df[y_lab])]]

    H, xedges, yedges = np.histogram2d(df[x_lab], df[y_lab], bins=n_bins, range=xy_range, normed=None, weights=None, density=None)
    H = H.T
    ax1.imshow(H)
    # ctl_df = df[df['Condition']==CTL_LABEL]
    ctl_df = df[df['Condition_shortlabel']==cond_list[0]]#CONDITION_SHORTLABELS[0]]
    H_ctl, xedges, yedges = np.histogram2d(ctl_df[x_lab], ctl_df[y_lab], bins=n_bins, range=xy_range, normed=None, weights=None, density=None)
    H_ctl = H_ctl.T
    ax2.imshow(H_ctl)

    comp_df = df[df['Condition_shortlabel']==cond_list[1]]#CONDITION_SHORTLABELS[1]]
    H_comp, xedges, yedges = np.histogram2d(comp_df[x_lab], comp_df[y_lab], bins=n_bins, range=xy_range, normed=None, weights=None, density=None)
    H_comp = H_comp.T
    ax3.imshow(H_comp)

    # Image Colocalization

    # Inds that will be max value
    thresh_1 = thresh
    thresh_2 = thresh
    inds = (H_comp > thresh_1) & (H_ctl > thresh_2)

    # Convert inds to white

    # im=H_ctl
    im = np.zeros(H_ctl.shape)
    im[inds] = 1000.0 # An arbitrarily high intensity value so you'll effectively only see this in the plot

    ax4.imshow(im)

    # Invert axes to be consistent with the scatter plots
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()

    return fig



def dr_contour_matrix(df_in,n_grid_pts=10, dr_method='tSNE', t_window=None):

    '''

    n_grid_pts
    '''

    df = df_in.copy()

    if(dr_method == 'tsne' or dr_method == 'tSNE'):

            x_lab = 'tSNE1'
            y_lab = 'tSNE2'

    elif dr_method == 'PCA':

            x_lab = 'PC1'
            y_lab = 'PC2'

    elif dr_method == 'umap':

            x_lab = 'UMAP1'
            y_lab = 'UMAP2'


    # Make the figure
    fig, ax = plt.subplots(1, 1, figsize=(10,10))#(ax1, ax2), (ax3, ax4)

    # ax.scatter(x=df[x_lab],y=df[y_lab],c='gray', alpha=0.1, s=1)
    pc_colors = colormap_pcs(df, cmap='rgb') # cmap='cmy'
    ax.scatter(x=df[x_lab],y=df[y_lab], alpha=0.5, s=1, c=pc_colors)


    # Create a meshgrid covering the area of DR space
    x_bounds = [np.min(df[x_lab]), np.max(df[x_lab])]
    y_bounds = [np.min(df[y_lab]), np.max(df[y_lab])]

    xs = np.linspace(x_bounds[0], x_bounds[1], n_grid_pts)
    ys = np.linspace(y_bounds[0], y_bounds[1], n_grid_pts)

    xx, yy = np.meshgrid(xs,ys, indexing='ij')

    # Lists to store the shapes
    df_list = []
    traj_list = []

    for i in range(n_grid_pts):

        for j in range(n_grid_pts):

            grid_x = xx[i,j]
            grid_y = yy[i,j]

            plt.scatter(x=grid_x,y=grid_y,c='black', alpha=0.3, s=2)

            # Find the closest cell to this.
            dr_arr = df[[x_lab,y_lab]].values

            # Calculate the distance between grid points and DR points
            distances = np.sqrt((dr_arr[:,0] - grid_x)**2 + (dr_arr[:,1] - grid_y)**2)

            # Sort, but keep indices
            dist_inds = np.argsort(distances)
            row_ind = dist_inds[0] # The first is the closest point

            # Gee the sub dataframe of this cell
            row_df = df.loc[row_ind].to_frame().transpose()

            df_list.append(row_df)

#             if distances[row_ind] < 5:

            plt.scatter(x=row_df[x_lab],y=row_df[y_lab],c='red', alpha=0.3, s=5)

    # Get a dataframe containing the cells that fall closest to the grid points
    grid_cell_df = pd.concat(df_list)
    grid_cell_df.sort_index(inplace=True)


    # For each of these cells, extract their track.
    for i,row in grid_cell_df.iterrows():

        this_rep = row['Replicate_ID']
        this_cell_id = row['particle']
        frame = row['frame']

        # Get sub_df for cell from row
        cell_df = df[(df['Replicate_ID']==this_rep) &
                        (df['particle']==this_cell_id)]


        if t_window is not None:

            # get a subset of the dataframe across the range of frames
            cell_df = cell_df[(cell_df['frame']>=frame - t_window/2) &
                          (cell_df['frame']<frame + t_window/2)]


        traj_list.append(cell_df[['x_pix','y_pix']])


    contour_list = get_cell_contours(grid_cell_df)

    assert len(contour_list) == len(traj_list), 'trajec and contour lists not same length'

    # Colormap the contours
    PALETTE = 'flare'
    colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(contour_list)))
    cm_data = np.asarray(colors)

    for i,contour in enumerate(contour_list):

            rgb=cm_data[i,:]
            this_colour = rgb#'red' # Eventually calculate color along colormap
            contour_arr = np.asarray(contour).T
            traj_arr = np.asarray(traj_list[i])

            x = grid_cell_df['x_pix'].values[i]# - window / 2
            y = grid_cell_df['y_pix'].values[i]# - window / 2

            # Need to cell contours to be centered on the cells position within the image
            x_dr = grid_cell_df[x_lab].values[i] - x# - window / 2
            y_dr = grid_cell_df[y_lab].values[i] - y# - window / 2

            # Cell contour relative to tSNE positions
            if not np.isnan(np.sum(contour_arr)):
                plt.plot(x_dr+contour_arr[:,0],y_dr+contour_arr[:,1],'-o',markersize=1,c=this_colour)

                # Draw also the contour of the cell in this space.
                plt.plot(x_dr+traj_arr[:,0],y_dr+traj_arr[:,1],'-o',markersize=1,c=this_colour)

    if STATIC_PLOTS:

        plt.savefig(DR_DIR + dr_method + '_contour_matrix.png', format='png', dpi=600)




'''
Ternary plot functions for PCA COLORMAPPING
'''

def color_point(x, y, z, scale, cmap='rgb'):

    if(cmap == 'rgb'):
        # Pure RGB
        r = x / float(scale)
        g = y / float(scale)
        b = z / float(scale)

    elif(cmap == 'cmy'):
        w = 255
        x_color = x * w / float(scale)
        y_color = y * w / float(scale)
        z_color = z * w / float(scale)
        r = math.fabs(w - y_color) / w
        g = math.fabs(w - x_color) / w
        b = math.fabs(w - z_color) / w

    return (r, g, b, 1.)

def generate_heatmap_data(scale=5, cmap = 'rgb'):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point(i, j, k, scale, cmap)
    return d

def dimension_reduction_subplots(dr_df,pca_tuple, cmap = 'rgb'):

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=[20,20])

    # Get the pca matrix of the subset of the dr_df
    pca_factor_matrix(dr_df,pca_tuple, DR_FACTORS, ax=ax1)

    # tax = ternary.TernaryAxesSubplot(ax=ax2)

    scale = 40
    figure, tax = ternary.figure(scale=scale, ax=ax2)
    data = generate_heatmap_data(scale, cmap=cmap)
    tax.heatmap(data, style="hexagonal", use_rgba=True,colorbar=False)
    tax.boundary()

    # Set Axis labels and Title
    fontsize = 18
    offset = 0#0.14
    # tax.set_title("RGBA Heatmap\n", fontsize=20)
    tax.right_corner_label("PC1", fontsize=fontsize)
    tax.top_corner_label("PC2", fontsize=fontsize)
    tax.left_corner_label("PC3", fontsize=fontsize)
    tax.get_axes().axis('off')

    pc_colors = colormap_pcs(dr_df, cmap=cmap) # cmap='cmy'

    ax3.set_title('tSNE embedding', fontsize=18)
    ax3.scatter(x=dr_df['tSNE1'],y=dr_df['tSNE2'], s=2, c=pc_colors)
    ax3.set_xlabel('tSNE1')
    ax3.set_ylabel('tSNE2')

    ax4.set_title('UMAP embedding', fontsize=18)
    ax4.scatter(x=dr_df['UMAP1'],y=dr_df['UMAP2'], s=2, c=pc_colors)
    ax4.set_xlabel('UMAP1')
    ax4.set_ylabel('UMAP2')



    if STATIC_PLOTS:

        plt.savefig(DR_DIR + '\dr_subplots.png', format='png', dpi=600)




    return fig
