#panel_apps.py

from initialization.config import *
from initialization.initialization import *

import numpy as np
import pandas as pd
import os

# matplotlib imports
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.),
    "figure.figsize":    (10,10),
    "font.size": 12
})


import panel as pn
import param

import plotly.graph_objects as go
import plotly.express as px
pn.extension('plotly')

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as skTSNE
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# For tests with openTSNE
from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from openTSNE.callbacks import ErrorLogger

from data_processing.data_io import *
from data_processing.clustering import *
from data_processing.pipelines import *
from data_processing.cleaning_formatting_filtering import *
from visualization.cluster_visualization import *
from visualization.filter_visualization import *
from visualization.timecourse_visualization import *
from visualization.trajectory_visualization import *
# Previously we just imported *everything* from the module:
# from comparative_visualization import *
# from spacetimecube import *
# from data_visualization import *
# from config import *
# from data_processing import apply_filters, get_data_matrix, do_tsne, migration_calcs, format_for_superplots
# from combine_compare import load_data, populate_experiment_list, combine_dataframes, csv_summary
# from tsne_embedding import do_open_tsne



marginals = ['rug', 'violin', 'histogram']
# Comparative PLots (Superplots, PLots of differences)
plots = ['Superplot', 'Plot of Differences']

EXP_LIST = populate_experiment_list(fmt=INPUT_FMT, save=False)
COND_LIST =list(EXP_LIST['Condition'].unique())# ['Condition1, Condition2, Condition3']#
COND_LIST.append('all')
REP_LIST= list(EXP_LIST['Experiment'].unique())#[]#list(dr_df['Replicate_ID'].unique())
REP_LIST.append('pooled')

inputs = NUM_FACTORS  + ['x', 'y'] # Need to maintain the ability to troubleshoot using position.

'''
Note:
This could all be cleaned up by defining functions to use directly within the class
Rather then having them sit in the script between Classes.
'''

print('COND_LIST: ',COND_LIST)
print('REP_LIST: ',REP_LIST)


class MetricsExplorer(param.Parameterized):

    factors = DR_FACTORS + ['x_um', 'y_um']
    fac1 = param.Selector(factors, default='speed')
    fac2 = param.Selector(factors, default='area')
    frame = param.Integer(default=10, bounds=(1, FRAME_END))
    plot_by = param.Selector(['xy', 'pca', 'tSNE', 'umap'], default='xy')



    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @param.depends('fac1','fac2', 'frame', 'plot_by')

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)


    def get_plot(self):

        df = self.df
        fig = plot_cell_metrics(df, self.frame, mig_display_factors=[self.fac1], shape_display_factors=[self.fac2])

        return fig


class LowDExplorer(param.Parameterized):

    plot_by = param.Selector(['tSNE', 'umap'], default='umap')
    cell_i = param.Integer(default=20, bounds=(1, 100))
    contour_scale = param.Number(1.0,bounds=(0.2,5))

    def __init__(self, df, mean_df, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.mean_df = mean_df

    @param.depends('plot_by', 'cell_i', 'contour_scale')

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)

    def get_plot(self):

        df = self.df
        mean_df = self.mean_df

        this_cell_df = get_specific_cell(mean_df, df,self.cell_i)

        # get this cells id
        this_cell_id = get_cell_id(this_cell_df)

        fig = plot_cell_trajectories(this_cell_df, df, dr_method=self.plot_by,contour_scale=self.contour_scale)   # Recall contour_scale doing nothing

        return fig


class FilterExplorer(param.Parameterized):

    factors = DR_FACTORS + ['x_um', 'y_um']
    filt1 = param.Selector(factors, default='speed')
    filt1_range = param.Range(default=(0,50.0), bounds=(0, 50))

    filt2 = param.Selector(factors, default='area')
    filt2_range = param.Range(default=(0,100.0), bounds=(0, 100))
    ntpts_range = param.Range(default=(0, 100.0), bounds = (0, 100.0))

    how = param.Selector(['any', 'all'], default='any')
    plot_by = param.Selector(['xy', 'pca', 'tSNE', 'umap'], default='xy')



    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @param.depends('filt1', 'filt1_range',
                   'filt2','filt2_range','ntpts_range', 'how', 'plot_by')#, 'fraction_to_display')

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)


    def get_plot(self):

        df = self.df


        # User-defined filters in dict {factor:(min, max)}
        data_filters = {
        #   "speed": (10, 100),
          self.filt1: self.filt1_range,
          self.filt2: self.filt1_range,
          # "area": (0, 10000), # Warning: range will change if self-normalized
          "ntpts": self.ntpts_range#(0,1000)
        }

        # Returns a filtered dataframe, while also adding included column to comb_df
        filt_df, filt_counts = apply_filters(df,filter_dict=data_filters)

        fig = visualize_filtering(df, filt_counts,plot_by=self.plot_by)


        # fig = time_superplot(df, factor, t_window)



        return fig








class ScatterExplorer(param.Parameterized):


    factors = DR_FACTORS + ['x_um', 'y_um']
    X_variable = param.Selector(factors, default='area')
    Y_variable = param.Selector(factors, default='speed')
    frame = param.Integer(default=FRAME_END, bounds=(1, FRAME_END))

    filt1 = param.Selector(factors, default='speed')
    filt1_thresh = param.Number(0.0,bounds=(-100.0,200.0))
    filt2 = param.Selector(factors, default='area')
    filt2_range = param.Range(default=(-100,400.0), bounds=(-100, 400))
    filt3 = param.Selector(factors)
    filt3_range = param.Range(default=(-100.0, 100.0), bounds = (-100.0, 100.0))

    fraction_to_display = param.Number(1.0,bounds=(0.01,1.0))

    def __init__(self, df_in, **kwargs):
        super().__init__(**kwargs)
        self.df_in = df_in

    @param.depends('X_variable', 'Y_variable', 'frame', 'filt1', 'filt1_thresh',
                   'filt2','filt2_range', 'filt3','filt3_range', 'fraction_to_display')
    def plot(self):
        return self.get_plot()#self.df_in, self.X_variable, self.Y_variable, self.frame)#, self.filt, self.filt2)

    def panel(self):
        return pn.Row(self.param, self.plot)

    def get_df(self, df):
        self.bounds = (np.min(df[self.X_variable]-2),
                  np.max(df[self.X_variable]+2),
                  np.min(df[self.Y_variable]-2),
                  np.max(df[self.Y_variable]+2))
        df = df[df['frame']==self.frame]

        # Begin with hard-coded variables, eventually choose from dropdown menu.
        df = df[df[self.filt1] > self.filt1_thresh]
        df = df[(df[self.filt2] > self.filt2_range[0] ) & (df[self.filt2]  < self.filt2_range[1])]
        df = df[(df[self.filt3] > self.filt3_range[0] ) & (df[self.filt3]  < self.filt3_range[1])]
        return df

    def get_plot(self):#, df_in, X_variable, Y_variable, frame):

        df = self.get_df(self.df_in)

#         fig = px.scatter(df, x=self.X_variable, y=self.Y_variable, color="Condition",
#                          marginal_x="violin", marginal_y="violin",
#                          title="Comparative marginal scatter",
#                          range_x = [self.bounds[0], self.bounds[1]],
#                          range_y = [self.bounds[2], self.bounds[3]])


        # Need to map condition labels to numerical values to apply colormap
        palette = PALETTE#'tab10'
        colors = sns.color_palette(palette, n_colors=len(df['Condition'].unique()))
#         colors[i]

        '''
        Currently, we're anle to get improvement through numpy random choice OR scattergl. However, scattergl version doesn't have
        marginal histograms working well (not multiple colors per condition, bins instead of violins or KDE.)
        Ideally both solutions will be included, just need to add more histogram traces (per condition)
        '''

#         '''
#         Using scattergl
#         '''

#         fig = go.Figure(data=go.Scattergl(
#             x = df[self.X_variable],
#             y = df[self.Y_variable],
#             mode='markers',
# #             marker=dict(
# #                 color=df["Condition"].values,
# #                 colorscale='Viridis',
# #                 line_width=1
# #             )
#         ))

#         fig.update_xaxes(range=(self.bounds[0], self.bounds[1]))
#         fig.update_yaxes(range=(self.bounds[2], self.bounds[3]))

#         '''
#         This works to add marginal histograms, but will need to have the different conditions overlaid.
#         '''

#         trace2 = fig.add_histogram(x=df[self.X_variable], name=self.X_variable, marker=dict(color='#1f77b4', opacity=0.7),
#                               yaxis='y2',nbinsx=20
#                              )
#         trace3 = fig.add_histogram(y=df[self.Y_variable], name=self.X_variable, marker=dict(color='#1f77b4', opacity=0.7),
#                               xaxis='x2',nbinsy=20
#                              )

#         fig.layout = dict(xaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
#                           yaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
#                           showlegend=False,
#                           margin=dict(t=50),
#                           hovermode='closest',
#                           bargap=0,
#                           xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
#                           yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
#                           height=600,
#                          )




#         '''Older marginal plots'''

#         fig = px.scatter(df, x=self.X_variable, y=self.Y_variable, color="Condition",
#                          marginal_x="violin", marginal_y="violin",
#                          title="Comparative marginal scatter",
#                          range_x = [self.bounds[0], self.bounds[1]],
#                          range_y = [self.bounds[2], self.bounds[3]])


        '''Older marginal plots, using numpy random.choice()'''

        fig = px.scatter(df.sample(frac=self.fraction_to_display), x=self.X_variable, y=self.Y_variable, color="Condition_shortlabel",
                         marginal_x="violin", marginal_y="violin",
                         title="Comparative marginal scatter",
                         range_x = [self.bounds[0], self.bounds[1]],
                         range_y = [self.bounds[2], self.bounds[3]])



        return fig
#
# class JointExplorer(param.Parameterized):
#
#     def __init__(self, df_in,**kwargs):
#         super().__init__(**kwargs)
#         self.df_in = df_in
#
#     X_variable = param.Selector(inputs, default='tSNE1')
#     Y_variable = param.Selector(inputs, default='tSNE2')
#     marginals = param.Selector(marginals, default='histogram')
#     frame = param.Integer(default=1, bounds=(1, FRAME_END))
#     cond = param.Selector(COND_LIST, default=COND_LIST[0])
#     rep = param.Selector(REP_LIST, default='pooled')
# #     rep_i = param.Integer(default=0, bounds=(0,10))
#
#     @param.depends('X_variable', 'Y_variable', 'frame', 'marginals', 'cond', 'rep') # , 'rep_i'
#     def plot(self):
#         return get_jointplot(self.df_in, self.X_variable, self.Y_variable, self.marginals, self.frame, self.cond, self.rep) # , self.rep_i)#
#
#     def panel(self):
#         return pn.Row(self.param, self.plot)
#
#

###########################
# Jointplot
##########################

def get_sub_df(df,X_var, Y_var, frame, cond, rep):#='pooled'):#cond, rep_i

    bounds = (np.min(df[X_var]), np.max(df[X_var]), np.min(df[Y_var]), np.max(df[Y_var]))
    df = df[df['frame']==frame]# & (df['Condition'] == cond)]
    if cond != 'all':
        df = df[df['Condition']==cond]

    if rep != 'pooled':
        df = df[df['Replicate_ID'] == rep]
    return df, bounds

def get_jointplot(df_in,X_variable, Y_variable, marginals, frame, cond, rep): # , rep_i

    df, bounds = get_sub_df(df_in, X_variable, Y_variable,frame, cond, rep) # , rep_i)#
    fig = px.scatter(df, x=X_variable, y=Y_variable, color="Condition",
                            marginal_x=marginals, marginal_y=marginals,
                            title="Exploratory jointplot: " + rep, trendline="ols",
                            range_x = [bounds[0], bounds[1]],
                            range_y = [bounds[2], bounds[3]])
    # fig = sns.jointplot(data=df, x=X_variable, y=Y_variable, hue="Condition", kind="kde")


    return fig

class JointExplorer(param.Parameterized):

    def __init__(self, df_in,**kwargs):
        super().__init__(**kwargs)
        self.df_in = df_in

    X_variable = param.Selector(inputs, default='tSNE1')
    Y_variable = param.Selector(inputs, default='tSNE2')
    marginals = param.Selector(marginals, default='histogram')
    frame = param.Integer(default=10, bounds=(1, FRAME_END))
    cond = param.Selector(COND_LIST, default='all')
    rep = param.Selector(REP_LIST, default='pooled')

    @param.depends('X_variable', 'Y_variable', 'frame', 'marginals', 'cond', 'rep') # , 'rep_i'
    def plot(self):
        return get_jointplot(self.df_in, self.X_variable, self.Y_variable, self.marginals, self.frame, self.cond, self.rep) # , self.rep_i)#

    def panel(self):
        return pn.Row(self.param, self.plot)




def get_comp_plot(df_in,plot,factor, frame, ctl_label):

    if(plot=='Superplot'):

        fig =  superplots_plotly(df_in, factor,t=frame)# sp_dr_df

    elif(plot=='Plot of Differences'):
        test_df = df_in[df_in['frame'] == frame] # Choose a frame to visualize
        fig = plots_of_differences_plotly(test_df, factor=factor, ctl_label=ctl_label)

    return fig

#
# class ComparativeExplorer(param.Parameterized):
#
#     plot_type = param.Selector(plots)
#     factor = param.Selector(inputs, default='tSNE1')
#     frame = param.Integer(default=1, bounds=(1, FRAME_END))
#
#     def __init__(self, df_in, ctl_label,**kwargs):
#         super().__init__(**kwargs)
#         self.df_in = df_in
#         self.ctl_label = ctl_label
#
#     @param.depends('plot_type', 'factor', 'frame')
#     def plot(self):
#         return get_comp_plot(self.df_in, self.plot_type, self.factor, self.frame, self.ctl_label)
#
#     def panel(self):
#         return pn.Row(self.param, self.plot)

def get_comp_plot(df_in,plot,factor, frame, ctl_label):

    bounds = (np.min(df_in[factor]), np.max(df_in[factor]))

    if(plot=='Superplot'):

        fig =  superplots_plotly(df_in, factor,t=frame)# sp_dr_df
        fig.update_yaxes(range=bounds)

    elif(plot=='Plot of Differences'):
        test_df = df_in[df_in['frame'] == frame] # Choose a frame to visualize
        fig = plots_of_differences_plotly(test_df, factor=factor, ctl_label=ctl_label)
        fig.update_xaxes(range=bounds)

    return fig


class ComparativeExplorer(param.Parameterized):

    plot_type = param.Selector(plots)
    factor = param.Selector(inputs, default='area')
    frame = param.Integer(default=1, bounds=(1, FRAME_END))

    def __init__(self, df_in, ctl_label,**kwargs):
        super().__init__(**kwargs)
        self.df_in = df_in
        self.ctl_label = ctl_label

    @param.depends('plot_type', 'factor', 'frame')
    def plot(self):
        return get_comp_plot(self.df_in, self.plot_type, self.factor, self.frame, self.ctl_label)

    def panel(self):
        return pn.Row(self.param, self.plot)




class TsneKmeans(param.Parameterized):

    perplexity = param.Number(30.0,bounds=(0.0,200.0))
    early_exaggeration = param.Number(12.0,bounds=(0.0,50.0))
    n_iter = param.Integer(default=250, bounds=(25, 1000))
    learning_rate = param.Number(200.0,bounds=(10.0,1000.0))
    random_state = param.Integer(default=11, bounds=(0, 20))
    n_clusters = param.Integer(default=5, bounds=(1, 20))
    cond = param.Selector(COND_LIST, default=COND_LIST[0])
    rep = param.Selector(REP_LIST, default=REP_LIST[0])

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @param.depends('perplexity', 'early_exaggeration', 'n_iter',
                   'learning_rate','random_state', 'n_clusters', 'cond', 'rep')

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)

    def do_tsne(self, df):#x):

        # Matrix from DataFrame
        if self.cond != 'all':
            df = df[df['Condition']==self.cond]

        if self.rep != 'pooled':
            df = df[df['Replicate_ID'] == self.rep]

        sub_df = df[DR_FACTORS] # Filter original dataframe by select factors
        x = sub_df.values   # Matrix to be used in the dimensionality reduction

        n_components = 2
        n_iter_wo_prog = 300
        init = 'pca' # 'random'
        verbose = 0
        n_jobs = -1

        tsne = skTSNE(n_components=n_components,
                      early_exaggeration=self.early_exaggeration,
                      learning_rate=self.learning_rate,
                      verbose=verbose,
                      perplexity=self.perplexity,
                      n_iter=self.n_iter,
                      n_jobs = n_jobs,
                      random_state=self.random_state)

        tsne_results = tsne.fit_transform(x)

        return tsne_results

    def get_plot(self):

        tsne_x = self.do_tsne(self.df)#x)

        kmeans = KMeans(n_clusters=self.n_clusters)
        est = kmeans.fit(tsne_x)
        labels = est.labels_

        fig = Figure()
        FigureCanvas(fig) # not needed in mpl >= 3.1
        ax = fig.add_subplot()
        ax.scatter(tsne_x[:,0], tsne_x[:,1], s=1, c=labels,alpha=0.5)
        return fig







class HdbscanExplorer(param.Parameterized):

    perplexity = param.Number(TSNE_PERP,bounds=(0.0,200.0))
    early_exaggeration = param.Number(12.0,bounds=(0.0,50.0))
    n_iter = param.Integer(default=250, bounds=(25, 1000))
    learning_rate = param.Number(200.0,bounds=(10.0,1000.0))
    random_state = param.Integer(default=11, bounds=(0, 20))

    umap_nn = param.Integer(default=15, bounds=(1, 50))
    umap_min_dist = param.Number(UMAP_MIN_DIST,bounds=(0.1,1.0))

    # rep = param.Selector(REP_LIST, default=REP_LIST[0])
    factor = param.Selector(DR_FACTORS, default='area')
    min_cluster_size = param.Integer(default=20, bounds=(1, 100))
    cluster_by = param.Selector(['xy','PCs', 'tSNE', 'UMAP'], default='tSNE')
    color_by = param.Selector(['condition', 'PCs', 'cluster'])



    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @param.depends('perplexity', 'early_exaggeration', 'n_iter',
                   'learning_rate','random_state','factor',
                   'min_cluster_size', 'color_by','factor', 'cluster_by')

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)


    def get_plot(self):


        if self.cluster_by == 'xy':
            x_name = 'x'
            y_name = 'y'
        elif (self.cluster_by == 'pca' or self.cluster_by == 'PCA' or self.cluster_by == 'PCs'):
            x_name = 'PC1'
            y_name = 'PC2'
        elif (self.cluster_by == 'tsne' or self.cluster_by == 'tSNE'):
            x_name = 'tSNE1'
            y_name = 'tSNE2'

        elif (self.cluster_by == 'umap' or self.cluster_by == 'UMAP'):
            x_name = 'UMAP1'
            y_name = 'UMAP2'






        # Make the subplot figures with matplotlib
        fig, axes = plt.subplots(1,2, figsize=(10, 5))
        FigureCanvas(fig) # not needed in mpl >= 3.1

        # Subplots using seaborn
        palette = PALETTE#'tab10'

        dr_df = dr_pipeline(self.df, dr_factors=DR_FACTORS, dr_input='factors', tsne_perp=self.perplexity,umap_nn=self.umap_nn,min_dist=self.umap_min_dist)

        # dr_df = dr_pipeline(self.df, dr_factors=DR_FACTORS)

        lab_dr_df = hdbscan_clustering(dr_df,min_cluster_size=self.min_cluster_size,cluster_by=self.cluster_by, plot=False)
        # sns.scatterplot(ax=axes[0], data=dr_df, x=x_name, y=y_name, palette=palette, hue='label', legend=False)

        draw_cluster_hulls(lab_dr_df,cluster_by=self.cluster_by, color_by=self.color_by,legend=False,ax=axes[0],draw_pts=True,save_path=CLUST_PARAMS_DIR+'condition')



        # Subplots using seaborn

        palette = PALETTE#'tab10'
        sns.scatterplot(ax=axes[0], data=lab_dr_df, x=x_name, y=y_name, palette=palette, hue='label', legend=False)

        # Get a colormap the length of the list of labels
        labels=lab_dr_df['label'].unique()
        labels=labels[~np.isnan(labels)]
        cmap = np.asarray(sns.color_palette(palette, n_colors=len(labels)))

        axes[1].set_title(self.factor)

        for i, val in enumerate(labels):

            this_color = cmap[i,:]
            sns.kdeplot(lab_dr_df[lab_dr_df['label']==val][self.factor], shade=True, color=this_color,ax=axes[1], legend=False)


        return fig





class TsneDbscanExplorer(param.Parameterized):

    perplexity = param.Number(30.0,bounds=(0.0,200.0))
    early_exaggeration = param.Number(12.0,bounds=(0.0,50.0))
    n_iter = param.Integer(default=250, bounds=(25, 1000))
    learning_rate = param.Number(200.0,bounds=(10.0,1000.0))
    random_state = param.Integer(default=11, bounds=(0, 20))

    eps = param.Number(0.15,bounds=(0.05,1.0))
    min_samples = param.Integer(default=10, bounds=(1, 50))

    cond = param.Selector(COND_LIST, default=COND_LIST[0])
    rep = param.Selector(REP_LIST, default=REP_LIST[0])
    factor = param.Selector(DR_FACTORS, default='area')


    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @param.depends('perplexity', 'early_exaggeration', 'n_iter',
                   'learning_rate','random_state', 'eps','min_samples', 'cond', 'rep', 'factor')

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)

    def do_tsne(self, df):#x):

        # Matrix from DataFrame
        if self.cond != 'all':
            df = df[df['Condition']==self.cond]

        if self.rep != 'pooled':
            df = df[df['Replicate_ID'] == self.rep]

        sub_df = df[DR_FACTORS] # Filter original dataframe by select factors
        x = sub_df.values   # Matrix to be used in the dimensionality reduction

        x_ = StandardScaler().fit_transform(x)

        n_components = 2
        n_iter_wo_prog = 300
        init = 'pca' # 'random'
        verbose = 0
        n_jobs = -1

        tsne = skTSNE(n_components=n_components,
                      early_exaggeration=self.early_exaggeration,
                      learning_rate=self.learning_rate,
                      verbose=verbose,
                      perplexity=self.perplexity,
                      n_iter=self.n_iter,
                      n_jobs = n_jobs,
                      random_state=self.random_state)

        tsne_results = tsne.fit_transform(x_)

        return df, tsne_results

    def get_plot(self):

        sub_df, tsne_x = self.do_tsne(self.df)#x)

        X = StandardScaler().fit_transform(tsne_x)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Assemble a dataframe from the results
        tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])
        lab_df = pd.DataFrame(data = labels, columns = ['label'])
        dr_df = pd.concat([sub_df,tsne_df,lab_df], axis=1)


        # Make the subplot figures with matplotlib
        fig, axes = plt.subplots(1,2, figsize=(10, 5))
        FigureCanvas(fig) # not needed in mpl >= 3.1


        # Subplots using seaborn

        palette = PALETTE#'tab10'
        sns.scatterplot(ax=axes[0], data=dr_df, x='tSNE1', y='tSNE2', palette=palette, hue='label', legend=False)

        # Get a colormap the length of the list of labels
        labels=dr_df['label'].unique()
        labels=labels[~np.isnan(labels)]
        cmap = np.asarray(sns.color_palette(palette, n_colors=len(labels)))

        axes[1].set_title(self.factor)

        for i, val in enumerate(labels):

            this_color = cmap[i,:]
            sns.kdeplot(dr_df[dr_df['label']==val][self.factor], shade=True, color=this_color,ax=axes[1], legend=False)


        return fig

class ReEmbeddingExplorer(param.Parameterized):

    '''
    Panel applet that allows users to test the consistency of tSNE embeddings
    from different subsets of the analysed data.

    Each time the dropdown menus or sliders on the GUI are updated,
    the dataframe is re-embedded into the saved tSNE embedding.
    Allowing users to observe the effect of visualizing
    individual experimental replicates.

    Users can also modify the value of eps and the minimum number of samples
    inputs for the DBSCAN clustering

    The goal is to observe whether there is any consistency of the tSNE clustered
    groups between experimental replicates

    Input:
        df: dataframe
        perplexity: int or float

    Returns: Dynamic applet.

    '''

    # tsne values for consistency
    early_exaggeration = 12
    n_iter =296
    learning_rate = 200
    random_state = 11

    eps = param.Number(0.15,bounds=(0.05,0.2))
    min_samples = param.Integer(default=10, bounds=(1, 50))

    rep = param.Selector(REP_LIST, default=REP_LIST[0])
    factor = param.Selector(DR_FACTORS, default='area')


    def __init__(self, df,perplexity, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.perplexity = perplexity
        self.do_tsne(self.df)

    @param.depends('eps','min_samples', 'rep', 'factor')#'cond',

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)

    def do_tsne(self, df):#x):

        if self.rep != 'pooled':
            df = df[df['Replicate_ID'] == self.rep]

        sub_df = df[DR_FACTORS] # Filter original dataframe by select factors
        x = sub_df.values   # Matrix to be used in the dimensionality reduction

        n_components = 2
        n_iter_wo_prog = 300
        init = 'pca' # 'random'
        verbose = 0
        n_jobs = -1

        tsne = TSNE(
            perplexity=self.perplexity,
            n_jobs=-1,
            random_state=TSNE_R_S,
        )

        x_ = StandardScaler().fit_transform(x)

        #Save temp embedding
        embedded_x = tsne.fit(x_)
        embedding_array = np.asarray(embedded_x) # as array for easy visualization
        np.save(DATA_OUTPUT+EMBEDDING_FILENAME,embedding_array)
        np.save(DATA_OUTPUT+TRAINX_FILENAME,x)

    def embed_tsne(self, df):#x):

        if self.rep != 'pooled':
            df = df[df['Replicate_ID'] == self.rep]

        sub_df = df[DR_FACTORS] # Filter original dataframe by select factors
        x = sub_df.values   # Matrix to be used in the dimensionality reduction

        # Load the embedding as a numpy array
        loaded_embedd_arr = np.load(DATA_OUTPUT+EMBEDDING_FILENAME)
        loaded_x_train = np.load(DATA_OUTPUT+TRAINX_FILENAME)

        # Build the affinities needed for the embedding.
        affinities = affinity.PerplexityBasedNN(
            loaded_x_train,
            perplexity=self.perplexity,
            n_jobs=-1,
            random_state=TSNE_R_S,
        )

        # Reconstruct the tSNEEmbedding.
        saved_embedding = TSNEEmbedding(
            loaded_embedd_arr,
            affinities,
        )
        x_ = StandardScaler().fit_transform(x)
        # Apply the saved embedding to the existing current testing data x.
        embedded_x = saved_embedding.transform(x_)
        embedding_array = np.asarray(embedded_x) # as array for easy visualization

        return df, embedding_array#tsne_results


    def get_plot(self):

        # Re-embed tsne
        sub_df, tsne_x = self.embed_tsne(self.df)#x)

        X = StandardScaler().fit_transform(tsne_x)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Assemble a dataframe from the results
        tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])
        lab_df = pd.DataFrame(data = labels, columns = ['label'])
        dr_df = pd.concat([sub_df,tsne_df,lab_df], axis=1)

        # Make the subplot figures with matplotlib
        fig, axes = plt.subplots(1,2, figsize=(10, 5))
        FigureCanvas(fig) # not needed in mpl >= 3.1

        # Subplots using seaborn
        palette = PALETTE#'tab10'
        sns.scatterplot(ax=axes[0], data=dr_df, x='tSNE1', y='tSNE2', palette=palette, hue='label', legend=False)

        # Get a colormap the length of the list of labels
        labels=dr_df['label'].unique()
        labels=labels[~np.isnan(labels)]
        cmap = np.asarray(sns.color_palette(palette, n_colors=len(labels)))

        axes[1].set_title(self.factor)


        for i, val in enumerate(labels):

            this_color = cmap[i,:]
            sns.kdeplot(dr_df[dr_df['label']==val][self.factor], shade=True, color=this_color,ax=axes[1], legend=False)

        return fig

class SupertimePlotExplorer(param.Parameterized):

    factor = param.Selector(NUM_FACTORS, default='area')
    t_window = param.Integer(default=1, bounds=(1, 50))

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df

    @param.depends('factor', 't_window')

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)


    def get_plot(self):

        df = self.df
        factor = self.factor
        t_window = self.t_window
        fig = time_superplot(df, factor, t_window)
        return fig





class ClusterExplorer(param.Parameterized):

    '''

    Input:
        df: dataframe

    Returns: Dynamic applet.

    '''

    # eps_x100 = param.Number(15,bounds=(1,200))#)0.15,bounds=(0.05,0.2))
    # min_samples = param.Integer(default=10, bounds=(1, 50))

    # rep = param.Selector(REP_LIST, default=REP_LIST[0])
    factor = param.Selector(DR_FACTORS, default='area')
    min_cluster_size = param.Integer(default=20, bounds=(1, 100))
    cluster_by = param.Selector(['xy','PCs', 'tSNE', 'UMAP'], default='tSNE')
    color_by = param.Selector(['condition', 'PCs', 'cluster'])

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df


    # @param.depends('eps_x100','min_samples', 'rep', 'factor', 'cluster_by')
    @param.depends('min_cluster_size', 'color_by','factor', 'cluster_by')

    def plot(self):
        return self.get_plot()

    def panel(self):
        return pn.Row(self.param, self.plot)


    def get_plot(self):
        x_name,y_name = ' ', ' '

        if self.cluster_by == 'xy':
            x_name = 'x'
            y_name = 'y'
        elif (self.cluster_by == 'pca' or self.cluster_by == 'PCA' or self.cluster_by == 'PCs'):
            x_name = 'PC1'
            y_name = 'PC2'
        elif (self.cluster_by == 'tsne' or self.cluster_by == 'tSNE'):
            x_name = 'tSNE1'
            y_name = 'tSNE2'

        elif (self.cluster_by == 'umap' or self.cluster_by == 'UMAP'):
            x_name = 'UMAP1'
            y_name = 'UMAP2'

        # sub_set = self.df[[x_name, y_name]]
        # X = StandardScaler().fit_transform(sub_set)
        # db = DBSCAN(eps=self.eps_x100/100, min_samples=self.min_samples).fit(X)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_

        # Assemble a dataframe from the results
        # lab_df = pd.DataFrame(data = labels, columns = ['label'])
        # dr_df = pd.concat([self.df,lab_df], axis=1)

        # Make the subplot figures with matplotlib
        fig, axes = plt.subplots(1,2, figsize=(10, 5))
        FigureCanvas(fig) # not needed in mpl >= 3.1

        # Subplots using seaborn
        palette = PALETTE#'tab10'

        dr_df = hdbscan_clustering(self.df,min_cluster_size=self.min_cluster_size,cluster_by=self.cluster_by, plot=False)
        # sns.scatterplot(ax=axes[0], data=dr_df, x=x_name, y=y_name, palette=palette, hue='label', legend=False)

        draw_cluster_hulls(dr_df,cluster_by=self.cluster_by, color_by=self.color_by,legend=False,ax=axes[0],draw_pts=True,save_path=CLUST_PARAMS_DIR+'condition')

        # Get a colormap the length of the list of labels
        labels=dr_df['label'].unique()
        labels=labels[~np.isnan(labels)]
        cmap = np.asarray(sns.color_palette(palette, n_colors=len(labels)))

        axes[1].set_title(self.factor)


        for i, val in enumerate(labels):

            this_color = cmap[i,:]
            sns.kdeplot(dr_df[dr_df['label']==val][self.factor], shade=True, color=this_color,ax=axes[1], legend=False)

        return fig
