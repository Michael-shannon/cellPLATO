from initialization.config import *
from initialization.initialization import *

from data_processing.statistics import *

import numpy as np
import pandas as pd
import os

import scipy
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib # spidercilantro

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib import cm


def plots_of_differences_plotly(df_in,factor='Value', ctl_label=CTL_LABEL,plot_type=DIFF_PLOT_TYPE,cust_txt='', save_path=DIFFPLOT_DIR):

    '''
    A function to create the plots of differences plots with bootstrapping CIs on sample data.
    Based on the method and code from Joachim Goedhart doi: https://doi.org/10.1101/578575
        https://www.biorxiv.org/content/10.1101/578575v1.full.pdf+html
        and related coe in R:
        https://github.com/JoachimGoedhart/PlotsOfDifferences/blob/master/app.R

    **Adapted to allow comparisons between tSNE-derived subgroups**

    '''
    df = df_in.copy()

    if ctl_label == -1:
        grouping = 'label'
        assert len(df[grouping].unique()) < 30, str(len(df[grouping].unique()))+ ' groups will be difficult to display, try optimizing the clustering.'


    else:

        if(USE_SHORTLABELS):

            # df = add_shortlabels(df)
            grouping = 'Condition_shortlabel'

            # Sort the dataframe by custom category list to set draw order
            df[grouping] = pd.Categorical(df[grouping], CONDITION_SHORTLABELS)
            this_cond_ind = CONDITIONS_TO_INCLUDE.index(ctl_label)
            ctl_label = CONDITION_SHORTLABELS[this_cond_ind]

        else:

            grouping = 'Condition'

            # Sort the dataframe by custom category list to set draw order
            df[grouping] = pd.Categorical(df[grouping], CONDITIONS_TO_INCLUDE)


    df.sort_values(by=grouping, inplace=True, ascending=True)


    # Set up the figure and some plotting parameters
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    # Resize points based on number of samples to reduce overplotting.
    if(len(df) > 1000):
        pt_size = 1
    else:
        pt_size = 3

    # Get a colormap the length of unique condition (or whatever they're being grouped by)
    colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(pd.unique(df[grouping]))))


    # Get a bootstrapped sample for the control condition
    ctl_bootstrap = bootstrap_sample(df[factor][df[grouping] == ctl_label])

    # Store the calculated CIs in a list of shapes to add to the plot using shape in update layout.
    shape_list = []

    for i in range(0,len(pd.unique(df[grouping]))):

        if (plot_type=='swarm'):

            # Plot the points
            fig.add_trace(go.Violin(
                                    # Select the subset of the dataframe we need by chaining another [] as condition to plot the susbset of the dataframe.
                                    x=df[factor][df[grouping] == pd.unique(df[grouping])[i]],
                                    y=df[grouping][df[grouping] == pd.unique(df[grouping])[i]],
                                    line={
                                        'width': 0
                                    },
                                    points="all",
                                    pointpos=0,
                                    marker={
                                         'color': 'rgb' + str(tuple(colors[i,:])),#colors[i],#'black' #diff_df['Value'].values
                                         'size': pt_size
                                         #'color': colors[np.where(cond_list == diff_df['Condition'])[0]]
                                    },
                                    orientation="h",
                                    jitter=1,
                                    fillcolor='rgba(0,0,0,0)',
                                    width= 0.75, # Width of the violin, will influence extent of jitter
                                   ),

                                    row=1, col=1)

        elif (plot_type=='violin'):

            # Plot the points
            fig.add_trace(go.Violin(
                                    # Select the subset of the dataframe we need by chaining another [] as condition to plot the susbset of the dataframe.
                                    x=df[factor][df[grouping] == pd.unique(df[grouping])[i]],
                                    y=df[grouping][df[grouping] == pd.unique(df[grouping])[i]],
                                    line={
                                        'width': 1
                                    },
                                    pointpos=0,
                                    marker={
                                         'color': 'rgb' + str(tuple(colors[i,:])),#colors[i],#'black' #diff_df['Value'].values
                                         'size': pt_size

                                    },
                                    orientation="h",
                                    side='positive',
                                    meanline_visible=True,
                                    points=False,
                                    width= 0.75, # Width of the violin, will influence extent of jitter
                                   ),

                                    row=1, col=1)

        elif (plot_type=='box'):

            # Plot the points
            fig.add_trace(go.Box(
                                # Select the subset of the dataframe we need by chaining another [] as condition to plot the susbset of the dataframe.
                                x=df[factor][df[grouping] == pd.unique(df[grouping])[i]],
                                y=df[grouping][df[grouping] == pd.unique(df[grouping])[i]],
                                line={
                                    'width': 1
                                },
#                                 pointpos=0,
                                marker={
                                     'color': 'rgb' + str(tuple(colors[i,:])),#colors[i],#'black' #diff_df['Value'].values
                                     'size': pt_size

                                },
                                orientation="h",
                                boxpoints='all',
                               ),

                                row=1, col=1)


        # Use the ctl_bootstrap if we're now on that condition, otherwise will create a new bootstrap sample that won't be the same.
        if(pd.unique(df[grouping])[i] == ctl_label):
            bootstrap = ctl_bootstrap
        else:
            bootstrap = bootstrap_sample(df[factor][df[grouping] == pd.unique(df[grouping])[i]])

        # Calculate difference between this condition and the control bootstrap sample
        difference = bootstrap - ctl_bootstrap

        # 2-sample Randomization test (computationally intensive for large Ns)
#         result = randtest(set1, set2, num_permutations=1)#-1)
#         print('Randomization  P: ',result.p_value)

        if pd.unique(df[grouping])[i] != ctl_label:

            # Statistical hypothesis testing
            set1 = df[factor][df[grouping] == ctl_label]
            set2 = df[factor][df[grouping] == pd.unique(df[grouping])[i]]

            ts, tP = st.ttest_ind(set1, set2)
#             F, aP = stats.f_oneway(set1, set2)
#             print('P = ', tP, ' (Using t-test without multiple comparison correction)')

          # Properties for right axis (Difference plots)
            cond_label = pd.unique(df[grouping])[i]

            # Draw P-values to the plot
#             fig.add_annotation(x=1, y=3,
#                 text=str(cond_label) + ' vs ' + str(ctl_label) + ': P = ' + str(round(tP,3)),
#                 showarrow=False,
#                 yshift=i*20, row=1, col=2)

            # Print p-value to the console
            print(str(cond_label) + ' vs ' + str(ctl_label) + ': P = ' + str(round(tP,3)))
            print('P = ', tP, ' (Using t-test without multiple comparison correction)')
            print("------")

        # Calculate the confidence interval
        sample = df[factor][df[grouping] == pd.unique(df[grouping])[i]]
        raw_ci = st.t.interval(0.95, len(sample)-1, loc=np.mean(sample), scale=st.sem(sample))
        diff_ci = np.percentile(difference, [2.5,97.5])

        # Create the difference plots in the second subplot
        fig.add_trace(go.Violin(
                                # Select the subset of the dataframe we need by chaining another [] as condition to plot the susbset of the dataframe.
                                x=difference,
                                y=df[grouping][df[grouping] == pd.unique(df[grouping])[i]],

                                side="positive",
                                orientation="h",
                                points=False,
                                line_color = 'rgb' + str(tuple(colors[i,:])),#colors[i], #'black'
                                width= 1#0.85,
                                #bandwidth = 0.5 # Controls the kde
                               ),
                                row=1, col=2)

        # Add CI to shape list WITHIN the loop through conditions

        # Add raw CI to the list of dicts
        shape_list.append(dict(type="line", xref="x1", yref="y1",
                                 x0=raw_ci[0], y0=i, x1=raw_ci[1], y1=i, line_width=6))
        # Add diff CI to the list of dicts
        shape_list.append(dict(type="line", xref="x2", yref="y2",
                                 x0=diff_ci[0], y0=i, x1=diff_ci[1], y1=i, line_width=6))

    # Add zero line to the difference plots
    shape_list.append(dict(type="line", xref="x2", yref="y2",x0=0, y0=-1, x1=0, y1=3, line_width=2)) # Thick line at x=0 for the difference plot

    fig.update_layout(height=500, width=800,showlegend=False,
                      title_text="Plots of Differences: "+factor,
                      shapes=shape_list, # Accepts a list of dicts.
                      plot_bgcolor = 'white',
                      font=dict(
                         #family="Courier New, monospace",
                          size=PLOT_TEXT_SIZE,
                          color="Black"))

    # Properties for left axis (categorical scatter)
    fig.update_xaxes(title_text=factor, row=1, col=1,
                    showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(row=1, col=1,showline=True,
                     linewidth=2, linecolor='black')

    # Properties for right axis (Difference plots)
    fig.update_xaxes(title_text="Difference", row=1, col=2,
                    showline=True, linewidth=2, linecolor='black')

    fig.update_yaxes(row=1, col=2,showline=True,
                     linewidth=2, linecolor='black')

    # ANOVA on the full dataset.
    #if len(pd.unique(df[grouping])) > 2:

    #    set1 = df[factor][df[grouping] == pd.unique(df[grouping])[0]]
    #    set2 = df[factor][df[grouping] == pd.unique(df[grouping])[1]]
    #    set3= df[factor][df[grouping] == pd.unique(df[grouping])[2]]

    #    if(len(pd.unique(df[grouping])) > 3):

    #        print('Warning: only set up t calculate anova for 3 conditions')
    #    else:
    #        F, aP = st.f_oneway(set1,set2,set3)

            # fig.add_annotation(x=2, y=3.5,
            #     text="Anova F =  "+ str(round(F,3))+", "+"P = " + str(round(aP,3)),
            #     showarrow=False,
            #     yshift=30, row=1, col=2)

    if STATIC_PLOTS:

        fig.write_image(save_path+cust_txt+factor+'_plots_of_differences_plotly.png')

    if PLOTS_IN_BROWSER:
        fig.show()

    return fig



def plots_of_differences_sns_saved(df_in,factor='Value', ctl_label=CTL_LABEL,cust_txt='', save_path=DIFFPLOT_DIR):

    '''
    A function to create the plots of differences plots with bootstrapping CIs on sample data.
    Based on the method and code from Joachim Goedhart doi: https://doi.org/10.1101/578575
        https://www.biorxiv.org/content/10.1101/578575v1.full.pdf+html
        and related coe in R:
        https://github.com/JoachimGoedhart/PlotsOfDifferences/blob/master/app.R

    **NOT Adapted to allow comparisons between tSNE-derived subgroups**

    '''


    # import matplotlib.pyplot as plt
    # import seaborn as sns
    plt.rcParams.update({'font.size': PLOT_TEXT_SIZE})
    plt.clf()

    assert ctl_label in df_in['Condition'].values, ctl_label + ' is not in the list of conditions'
    assert ctl_label != -1, 'Not yet adapted to compare between cluster groups, use plots_of_differences_plotly() instead'

    df = df_in.copy()

    # Sort values according to custom order for drawing plots onto graph
    df['Condition'] = pd.Categorical(df['Condition'], CONDITIONS_TO_INCLUDE)
    df.sort_values(by='Condition', inplace=True, ascending=True)

    # Use Matplotlib to create subplot and set some properties
    fig_width = 11 # Inches
    aspect = 2

    fig, axes = plt.subplots(1, 2, figsize=(fig_width,fig_width/aspect))
#     plt.rcParams['savefig.facecolor'] = 'w'
    fig.suptitle('Plots of Differences: '+ factor)

    # Resize points based on number of samples to reduce overplotting.
    if(len(df) > 1000):
        pt_size = 1
    else:
        pt_size = 10

    # Get the bootstrapped sample as a dataframe
    bootstrap_diff_df = bootstrap_sample_df(df,factor,ctl_label)


    # colors=[]
    # cmap = cm.get_cmap(CONDITION_CMAP, len(lab_dr_df['Condition_shortlabel'].unique()))
    # for i in range(cmap.N):
    #     colors.append(cmap(i))

    #
    # Left subplot: horizontal scatter
    #

    #sns.swarmplot(ax=axes[0], x=factor, y="Condition",size=2, data=df)#, ax=g.ax) # Built with sns.swarmplot (no ci arg.)
    sns.stripplot(ax=axes[0], x=factor, y="Condition",size=pt_size,jitter=0.25, data=df)

    # Draw confidence intervalswith point plot onto scatter plot
    sns.pointplot(ax=axes[0], x=factor, y="Condition", data=df,color='black', ci = 95, join=False, errwidth=10.0) #calamansi kind="swarm",

    #
    # Right subplot: differences
    #

    sns.violinplot(ax=axes[1], x="Difference", y="Condition",inner='box', data=bootstrap_diff_df, split=True, linewidth=2)
    axes[1].axvline(0, ls='-', color='black')
    axes[1].set(ylabel=None)
    axes[1].set(yticklabels=[])

    # Invert both y axis to be consistent with original plots of difference
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()


    fig.savefig(save_path+cust_txt+factor+'_plots_of_differences_sns.png', dpi=300)#plt.

    # fig.savefig(PLOT_OUTPUT+cust_txt+factor+'_plots_of_differences_sns.png', dpi=300)#plt.

    return fig


################################################################################

################################################################################

################################################################################


def plots_of_differences_sns(df_in,factor='Value', ctl_label=CTL_LABEL,cust_txt='', save_path=DIFFPLOT_DIR):

    '''
    A function to create the plots of differences plots with bootstrapping CIs on sample data.
    Based on the method and code from Joachim Goedhart doi: https://doi.org/10.1101/578575
        https://www.biorxiv.org/content/10.1101/578575v1.full.pdf+html
        and related coe in R:
        https://github.com/JoachimGoedhart/PlotsOfDifferences/blob/master/app.R

    **NOT Adapted to allow comparisons between tSNE-derived subgroups**

    '''


    # import matplotlib.pyplot as plt
    # import seaborn as sns
    plt.rcParams.update({'font.size': PLOT_TEXT_SIZE})
    plt.clf()

    assert ctl_label in df_in['Condition'].values, ctl_label + ' is not in the list of conditions'
    assert ctl_label != -1, 'Not yet adapted to compare between cluster groups, use plots_of_differences_plotly() instead'

    df = df_in.copy()

    # Sort values according to custom order for drawing plots onto graph
    df['Condition'] = pd.Categorical(df['Condition'], CONDITIONS_TO_INCLUDE)
    df.sort_values(by='Condition', inplace=True, ascending=True)

    # Use Matplotlib to create subplot and set some properties
    fig_width = 15#11 # Inches
    aspect = 2

    fig, axes = plt.subplots(1, 2, figsize=(fig_width,fig_width/aspect))
#     plt.rcParams['savefig.facecolor'] = 'w'
    # Remove the underscore from factor
    factor_ = factor.replace('_',' ')
    if factor == 'rip_L': # SPIDERMAN #
        rip_r_microns= RIP_R*MICRONS_PER_PIXEL
        factor_ = factor + '(' + str(rip_r_microns) + ')' # Add the value of the rip to the label
        factor_ = factor_.replace('_',' ')
    fig.suptitle('Plots of data and effect size: '+ factor_, fontsize= PLOT_TEXT_SIZE)

    numberoftotalpoints = len(df)

    if(numberoftotalpoints > 1000 and numberoftotalpoints < 5000):
        pt_size = 3
        alphavalue = 0.9
    elif(numberoftotalpoints > 5000 and numberoftotalpoints < 10000):
        pt_size = 4
        alphavalue = 0.7
    elif(numberoftotalpoints > 10000 and numberoftotalpoints < 20000):
        pt_size=3
        alphavalue = 0.6
    elif(numberoftotalpoints > 10 and numberoftotalpoints < 500):
        pt_size = 10
        alphavalue = 0.8
    
    else:
        pt_size = 2
        alphavalue = 0.3

    
    print('There are ' + str(len(df)) + ' points, so using point size: ' + str(pt_size))
    # Get the bootstrapped sample as a dataframe
    bootstrap_diff_df = bootstrap_sample_df(df,factor,ctl_label)


    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP)
    numcolors = len(df['Condition_shortlabel'].unique())
    for i in range(numcolors):
        colors.append(cmap(i))

    #### VERSION 2 #####
    gax=axes[0]
    sns.violinplot(ax=axes[0], x=factor, y="Condition",palette=colors, inner='box', data=df, split=False, linewidth=2)
    sns.boxplot(ax=axes[0], x=factor, y="Condition_shortlabel", data=df, medianprops={'color': 'k', 'ls': '-', 'lw': 2},whiskerprops={'visible': False},
                showbox=False, notch=True, showcaps=False, showfliers=False,
                flierprops={"marker": "x", "markeredgecolor": "black","markersize": "3","fillstyle": "none" },)
    for collection in gax.collections:
        if isinstance(collection, matplotlib.collections.PolyCollection):
            collection.set_edgecolor('none')
            collection.set_facecolor('none')
            collection.set_zorder(10)
        
    plt.setp(gax.lines, zorder=100)
    sns.stripplot(ax=axes[0], x=factor, y="Condition_shortlabel",size=pt_size,jitter=0.25, palette=colors, data=df, alpha=alphavalue)   

    ######### VERSION 2 ENDS #####

    #
    # Right subplot: differences
    #

    sns.violinplot(ax=axes[1], x="Difference", y="Condition", palette=colors, inner='box', data=bootstrap_diff_df, split=True, linewidth=2)
    axes[1].axvline(0, ls='-', color='black')
    axes[1].set(ylabel=None)
    axes[1].set(yticklabels=[])
    # axes[0].set(yticklabels=[])
    axes[0].set(ylabel=None)

    axes[1].set(xlabel='Effect size')
    axes[0].set(xlabel=factor_) # SPIDERMAN #

    # change the font size of the x label on axes[1]
    axes[1].xaxis.label.set_size(PLOT_TEXT_SIZE)
    axes[0].xaxis.label.set_size(PLOT_TEXT_SIZE)

    # set the font size of the y ticks
    axes[1].tick_params(axis='y', labelsize=PLOT_TEXT_SIZE)
    axes[0].tick_params(axis='y', labelsize=PLOT_TEXT_SIZE)
    axes[1].tick_params(axis='x', labelsize=PLOT_TEXT_SIZE)
    axes[0].tick_params(axis='x', labelsize=PLOT_TEXT_SIZE)

    # Invert both y axis to be consistent with original plots of difference
    axes[0].invert_yaxis()
    axes[1].invert_yaxis() #scallywag

    # set the font size for the xticks

    

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path+cust_txt+factor+'_plots_of_differences_sns.png', dpi=300)#plt.
    # make it tight
    

    # fig.savefig(PLOT_OUTPUT+cust_txt+factor+'_plots_of_differences_sns.png', dpi=300)#plt.

    return fig



def plots_of_differences_sns_donors(df_in,factor='Value', ctl_label=CTL_LABEL,cust_txt='', save_path=DIFFPLOT_DIR):

    '''
    A function to create the plots of differences plots with bootstrapping CIs on sample data.
    Based on the method and code from Joachim Goedhart doi: https://doi.org/10.1101/578575
        https://www.biorxiv.org/content/10.1101/578575v1.full.pdf+html
        and related coe in R:
        https://github.com/JoachimGoedhart/PlotsOfDifferences/blob/master/app.R

    **NOT Adapted to allow comparisons between tSNE-derived subgroups**

    '''


    # import matplotlib.pyplot as plt
    # import seaborn as sns
    plt.rcParams.update({'font.size': PLOT_TEXT_SIZE})
    plt.clf()

    assert ctl_label in df_in['Condition'].values, ctl_label + ' is not in the list of conditions'
    assert ctl_label != -1, 'Not yet adapted to compare between cluster groups, use plots_of_differences_plotly() instead'

    df = df_in.copy()

    # Sort values according to custom order for drawing plots onto graph
    df['Condition'] = pd.Categorical(df['Condition'], CONDITIONS_TO_INCLUDE)
    df.sort_values(by='Condition', inplace=True, ascending=True)

    # Use Matplotlib to create subplot and set some properties
    fig_width = 15#11 # Inches
    aspect = 2

    fig, axes = plt.subplots(1, 2, figsize=(fig_width,fig_width/aspect))
#     plt.rcParams['savefig.facecolor'] = 'w'
    # Remove the underscore from factor
    factor_ = factor.replace('_',' ')
    if factor == 'rip_L': # SPIDERMAN #
        rip_r_microns= RIP_R*MICRONS_PER_PIXEL
        factor_ = factor + '(' + str(rip_r_microns) + ')' # Add the value of the rip to the label
        factor_ = factor_.replace('_',' ')
    fig.suptitle('Plots of data and effect size: '+ factor_, fontsize= PLOT_TEXT_SIZE)

    numberoftotalpoints = len(df)

    if(numberoftotalpoints > 1000 and numberoftotalpoints < 5000):
        pt_size = 3
        alphavalue = 0.9
    elif(numberoftotalpoints > 5000 and numberoftotalpoints < 10000):
        pt_size = 4
        alphavalue = 0.7
    elif(numberoftotalpoints > 10000 and numberoftotalpoints < 20000):
        pt_size=3
        alphavalue = 0.6
    elif(numberoftotalpoints > 10 and numberoftotalpoints < 500):
        pt_size = 10
        alphavalue = 0.8
    
    else:
        pt_size = 2
        alphavalue = 0.3

    
    print('There are ' + str(len(df)) + ' points, so using point size: ' + str(pt_size))
    # Get the bootstrapped sample as a dataframe
    bootstrap_diff_df = bootstrap_sample_df(df,factor,ctl_label)


    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP)
    numcolors = len(df['Condition_shortlabel'].unique())
    for i in range(numcolors):
        colors.append(cmap(i))

    #### VERSION 2 #####
    gax=axes[0]
    sns.violinplot(ax=axes[0], x=factor, y="Condition",palette=colors, inner='box', data=df, split=False, linewidth=2)
    sns.boxplot(ax=axes[0], x=factor, y="Condition_shortlabel", data=df, medianprops={'color': 'k', 'ls': '-', 'lw': 2},whiskerprops={'visible': False},
                showbox=False, notch=True, showcaps=False, showfliers=False,
                flierprops={"marker": "x", "markeredgecolor": "black","markersize": "3","fillstyle": "none" },)
    for collection in gax.collections:
        if isinstance(collection, matplotlib.collections.PolyCollection):
            collection.set_edgecolor('none')
            collection.set_facecolor('none')
            collection.set_zorder(10)
        
    plt.setp(gax.lines, zorder=100)
    sns.stripplot(ax=axes[0], x=factor, y="Condition_shortlabel",size=pt_size,jitter=0.25, palette=colors, data=df, alpha=alphavalue)   

    ######### VERSION 2 ENDS #####

    #
    # Right subplot: differences
    #

    sns.violinplot(ax=axes[1], x="Difference", y="Condition", palette=colors, inner='box', data=bootstrap_diff_df, split=True, linewidth=2)
    axes[1].axvline(0, ls='-', color='black')
    axes[1].set(ylabel=None)
    axes[1].set(yticklabels=[])
    # axes[0].set(yticklabels=[])
    axes[0].set(ylabel=None)

    axes[1].set(xlabel='Effect size')
    axes[0].set(xlabel=factor_) # SPIDERMAN #

    # change the font size of the x label on axes[1]
    axes[1].xaxis.label.set_size(PLOT_TEXT_SIZE)
    axes[0].xaxis.label.set_size(PLOT_TEXT_SIZE)

    # set the font size of the y ticks
    axes[1].tick_params(axis='y', labelsize=PLOT_TEXT_SIZE)
    axes[0].tick_params(axis='y', labelsize=PLOT_TEXT_SIZE)
    axes[1].tick_params(axis='x', labelsize=PLOT_TEXT_SIZE)
    axes[0].tick_params(axis='x', labelsize=PLOT_TEXT_SIZE)

    # Invert both y axis to be consistent with original plots of difference
    axes[0].invert_yaxis()
    axes[1].invert_yaxis() #scallywag

    # set the font size for the xticks

    

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path+cust_txt+factor+'_plots_of_differences_sns.png', dpi=300)#plt.
    # make it tight
    

    # fig.savefig(PLOT_OUTPUT+cust_txt+factor+'_plots_of_differences_sns.png', dpi=300)#plt.

    return fig
