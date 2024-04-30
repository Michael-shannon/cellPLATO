from initialization.config import *
from initialization.initialization import *

from data_processing.shape_calculations import *

from visualization.cluster_visualization import *
from visualization.low_dimension_visualization import colormap_pcs

import numpy as np
import pandas as pd
import os
import imageio
import plotly

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.),
    "figure.figsize":    (10,10),
    "font.size": 12
})



def plot_cell_metrics_timepoint_deprecated(cell_df, i_step, XYRange,boxoff, top_dictionary, Cluster_ID, mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS):
    '''
    For a selected cell, at a selected step of its trajectory, make a plot
    '''
    # Using the same sample cell trajectory, print out some measurements
    fig, (ax) = plt.subplots(1,1, figsize=(0.08*XYRange,0.08*XYRange))
    # print(top_dictionary)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Arial'] + plt.rcParams['font.serif'] #Times New Roman
    plt.rcParams['mathtext.default'] = 'regular'
    ax.title.set_text('Cell contour: xy space')
    ax.plot(cell_df['x_pix'],cell_df['y_pix'],'-o',markersize=3,c='black')

    ################
    xminimum=cell_df['x_pix'].min()
    xmaximum=cell_df['x_pix'].max()
    xmid = np.median([xmaximum, xminimum])
    xmin=xmid-XYRange/2
    xmax=xmid+XYRange/2

    yminimum=cell_df['y_pix'].min()
    ymaximum=cell_df['y_pix'].max()
    ymid = np.median([ymaximum, yminimum])
    ymin=ymid-XYRange/2
    ymax=ymid+XYRange/2

    #######################

    contour_list = get_cell_contours(cell_df)
    # Draw all contours faintly as image BG
    for i,contour in enumerate(contour_list):

        rgb = np.random.rand(3,)
        this_colour = rgb#'red' # Eventually calculate color along colormap
        contour_arr = np.asarray(contour).T


        if not np.isnan(np.sum(contour_arr)): # If the contour is not nan

            x = cell_df['x_pix'].values[i]# - window / 2
            y = cell_df['y_pix'].values[i]# - window / 2
            # Cell contour relative to x,y positions
            '''Want to double check that the x,y positions not mirrored from the contour function'''
            ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c='gray', alpha=0.2)
    # Draw this contour, highlighted.
    contour = contour_list[i_step]
    contour_arr = np.asarray(contour).T
    x = cell_df['x_pix'].values[i_step]# - window / 2
    y = cell_df['y_pix'].values[i_step]# - window / 2

    # Cell contour relative to x,y positions
    '''Want to double check that the x,y positions not mirrored from the contour function'''
    if not np.isnan(np.sum(contour_arr)):
        ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c='tab:orange',linewidth=5, alpha=1)

        # Current segment
        if i_step > 0:
            x_seg = cell_df['x_pix'].values[i_step-1:i_step+1]# - window / 2
            y_seg = cell_df['y_pix'].values[i_step-1:i_step+1]# - window / 2
            # ax.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c='red', alpha=1)
            # ax.plot(x_seg*2,y_seg*2,'-o',markersize=10,c='tab:blue', linewidth=4)
            ax.plot(x_seg,y_seg,'-o',markersize=10,c='tab:blue', linewidth=4)
    text_x = xmid
    text_y = ymid
    # SPIDER #
    # p=cell_df.iloc[i_step]
    # ClusterID = cell_df['label'].iloc[i_step]
    # print('TRYING TO MAKE THIS A CLUSTER ID TO USE ON THE DICTIONARY >>>>>>>>' )
    # display(ClusterID)
    # # ClusterID=p['label']
    # print(row['label']) #here, the row label is something different
    # ClusterID = row['label']
    display_factors = top_dictionary[Cluster_ID]
    # You want the list of factors in top_dictionary that relates to the cluster in question.
    # IN this case, the cluster in question is the one at iloc[i_step]

    for n, fac in enumerate(display_factors): #Positioning is currently relative to data. Can it be relative to plot?
        facwithoutunderscores = fac.replace('_',' ')
        text_str = facwithoutunderscores +': '+ format(cell_df.iloc[i_step][fac], '.1f')
        ax.text(text_x + 0.5*XYRange, text_y - (0.3*XYRange) + (0.08*XYRange) + (n*(0.08*XYRange)), 
                text_str, #These weird numbers were worked out manually
                color='k', fontsize=PLOT_TEXT_SIZE, fontdict = None) #spidermoose )
        
        ################################
        #Original line#
        
        # ax.text(text_x + 0.6*XYRange, text_y - 47 + (0.08*XYRange) + (n*(0.08*XYRange)), facwithoutunderscores +': '+ format(cell_df.iloc[i_step][fac], '.1f'), #These weird numbers were worked out manually
        #         color='k', fontsize=30, fontdict = None) #spidermoose )

        ################################


    # for n, fac in enumerate(shape_display_factors):
    #
    #     ax.text(text_x + 0.6*XYRange,text_y -  (n*(0.08*XYRange)), fac +': '+ format(cell_df.iloc[i_step][fac], '.1f'),
    #             color='tab:orange', fontsize=30,size = 36, fontdict = None)

    # General plot improvements
    plottitle="Cluster ID: " + str(cell_df['label'].iloc[i_step]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[i_step] + ")"
    # plottitle="Cluster ID: " + str(Cluster_ID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[i_step] + ")"
    ax.set_title(plottitle, fontname="Arial",fontsize=PLOT_TEXT_SIZE) #TEMPORARILY COMMENTED OUT
    ax.set_xlabel('x (px)', fontname="Arial",fontsize=PLOT_TEXT_SIZE)
    ax.set_ylabel("y (px)", fontname="Arial",fontsize=PLOT_TEXT_SIZE)
    ax.tick_params(axis='both', labelsize=PLOT_TEXT_SIZE)
    ax.set_aspect('equal')
    ax.set_adjustable("datalim")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if boxoff:
        ax.axis('off')

    # plt.autoscale()
    # sns.despine(left=True)

    # ax.set_yticklabels(['eNK','eNK+CytoD'])
    return fig #delete?


def plot_cell_metrics_tavg(cell_df, XYRange,boxoff, row, top_dictionary, mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS): #removed the i_step
    '''
    For a selected cell, at a selected step of its trajectory, make a plot
    '''
    # Using the same sample cell trajectory, print out some measurements
    fig, (ax) = plt.subplots(1,1, figsize = (0.08*XYRange,0.08*XYRange))

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Arial'] + plt.rcParams['font.serif'] #Times New Roman
    plt.rcParams['mathtext.default'] = 'regular'

    ax.title.set_text('Cell contour: xy space')
    ax.plot(cell_df['x_pix'],cell_df['y_pix'],'-o',markersize=3,c='black')#black

    ################
    xminimum=cell_df['x_pix'].min()
    xmaximum=cell_df['x_pix'].max()
    xmid = np.median([xmaximum, xminimum])
    xmin=xmid-XYRange/2
    xmax=xmid+XYRange/2

    yminimum=cell_df['y_pix'].min()
    ymaximum=cell_df['y_pix'].max()
    ymid = np.median([ymaximum, yminimum])
    ymin=ymid-XYRange/2
    ymax=ymid+XYRange/2

    #######################

    contour_list = get_cell_contours(cell_df)
    # Draw all contours faintly as image BG
    for i,contour in enumerate(contour_list):

        rgb = np.random.rand(3,)
        this_colour = rgb#'red' # Eventually calculate color along colormap
        contour_arr = np.asarray(contour).T


        if not np.isnan(np.sum(contour_arr)): # If the contour is not nan

            x = cell_df['x_pix'].values[i]# - window / 2
            y = cell_df['y_pix'].values[i]# - window / 2
            # Cell contour relative to x,y positions
            '''Want to double check that the x,y positions not mirrored from the contour function'''
            ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c='gray', alpha=0.2)

            # for transparency in np.arange(len(contour_arr))

    # Draw this contour, highlighted.
    # contour = contour_list[i_step]
    # contour_arr = np.asarray(contour).T
    # x = cell_df['x_pix'].values[i_step]# - window / 2
    # y = cell_df['y_pix'].values[i_step]# - window / 2

    # Cell contour relative to x,y positions
    '''Want to double check that the x,y positions not mirrored from the contour function'''
    if not np.isnan(np.sum(contour_arr)):
        ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c='gray',linewidth=5, alpha=0.2)

        # Current segment
        # if i_step > 0:
        #     x_seg = cell_df['x_pix'].values[i_step-1:i_step+1]# - window / 2
        #     y_seg = cell_df['y_pix'].values[i_step-1:i_step+1]# - window / 2
        #     # ax.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c='red', alpha=1)
        #     # ax.plot(x_seg*2,y_seg*2,'-o',markersize=10,c='tab:blue', linewidth=4)
        #     ax.plot(x_seg,y_seg,'-o',markersize=10,c='tab:blue', linewidth=4)

    text_x = xmid
    text_y = ymid
    print('ATTENTION, this is text y:')
    print(text_y)
    # You would have to import the 'row' you got from the tavg_exemplar_df here
    # Here, you can do something with top_dictionary
    # display(row)
    print(row['label'])
    ClusterID = row['label']
    display_factors = top_dictionary[ClusterID]
    for n, fac in enumerate(display_factors): #Positioning is currently relative to data. Can it be relative to plot?

        # ax.text(text_x + 0.6*XYRange,text_y + (0.08*XYRange) + (n*(0.08*XYRange)), fac +': '+ format(row.loc[fac], '.1f'), #These weird numbers were worked out manually
        #         color='tab:blue', fontsize=30,size = 36, fontdict = None) ORIGINAL
        
        thisnumber= row.loc[fac]
        print(thisnumber)
        print(type(thisnumber))
        thisnumberstring = str(thisnumber)
        print(thisnumberstring)
        print(type(thisnumberstring))
        # factorwithoutunderscores = fac.replace("_", " ")

        #####
        facwithoutunderscores = fac.replace('_',' ')
        text_str = facwithoutunderscores +': ' + format(row.loc[fac], '.1f')
        ax.text(text_x + 0.5*XYRange, text_y - (0.3*XYRange) + (0.08*XYRange) + (n*(0.08*XYRange)), 
                text_str, #These weird numbers were worked out manually
                color='k', fontsize=PLOT_TEXT_SIZE, fontdict = None) #spidermoose )

        #####


        # ax.text(text_x + 0.6*XYRange, text_y - 47 + (0.08*XYRange) + (n*(0.08*XYRange)), text_str, #These weird numbers were worked out manually
        #         color='k', fontsize=30,size = 36, fontdict = None)

                # + format(cell_df.iloc[i_step][fac], '.1f')

    # for n, fac in enumerate(mig_display_factors): #Positioning is currently relative to data. Can it be relative to plot?
    #
    #     ax.text(text_x + 0.6*XYRange,text_y + (0.08*XYRange) + (n*(0.08*XYRange)), fac +': '+ format(row.loc[fac], '.1f'), #These weird numbers were worked out manually
    #             color='tab:blue', fontsize=30,size = 36, fontdict = None)
    #
    #
    # for n, fac in enumerate(shape_display_factors):
    #
    #     ax.text(text_x + 0.6*XYRange,text_y -  (n*(0.08*XYRange)), fac +': '+ format(row.loc[fac], '.1f'),
    #             color='tab:orange', fontsize=30,size = 36, fontdict = None)

    # General plot improvements
    plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
    ax.set_title(plottitle, fontname="Arial",fontsize=30)
    ax.set_xlabel('x (px)', fontname="Arial",fontsize=30)
    ax.set_ylabel("y (px)", fontname="Arial",fontsize=30)
    ax.tick_params(axis='both', labelsize=30)
    ax.set_aspect('equal')
    ax.set_adjustable("datalim")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if boxoff:
        ax.axis('off')

    # plt.autoscale()
    # sns.despine(left=True)

    # ax.set_yticklabels(['eNK','eNK+CytoD'])
    return fig #delete?



################################################

 ##### Using as of 1-18-2023 #####
 # 
def disambiguate_timepoint_deprecated(df, exemps, top_dictionary, XYRange=100,boxoff=True):


    for n in exemps.index:

        row=exemps.iloc[n] #extract an example exemplar row

        # Use exemplar row to look up the corresponding cell TRACK from the cell_df
        cell_df = df[(df['Condition']==row['Condition']) &
                        (df['Replicate_ID']==row['Replicate_ID']) &
                        (df['particle']==row['particle'])]
        cell_df = cell_df.reset_index(drop=True)

        # This looks up the exemplar point, based on all of these metrics, so that the correct exemplar point is displayed in the visualization
        # exemplarpoint = cell_df.index[(cell_df['area']==row['area']) &
        #                 (cell_df['speed']==row['speed']) &
        #                 (cell_df['frame']==row['frame']) &
        #                 (cell_df['label']==row['label'])]
        exemplarpoint = cell_df.index[(cell_df['uniq_id']==row['uniq_id']) &
                        (cell_df['frame']==row['frame']) &
                        (cell_df['label']==row['label'])]
        CLUSTERID = row['label']

        # f=cp.plot_cell_metrics(cell_df, exemplarpoint[0]) #mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS
        f=plot_cell_metrics_timepoint(cell_df, exemplarpoint[0], XYRange, boxoff, top_dictionary, CLUSTERID) #mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS

        ######
        f2,f3,f4 = plot_single_cell_factor(cell_df, top_dictionary, CLUSTERID, PLOT_PLASTICITY = True, thisistavg=False)



        # f.savefig( CLUST_DISAMBIG_DIR_TAVG + '\Contour ' + str(cell_df['Condition_shortlabel'].iloc[0]) +
            #   '. TAVG_Cluster ID = ' + str(cell_df['tavg_label'].iloc[0]) +'__disambiguated__R'+str(XYRange)+'_'+str(n)+'.png'  ,dpi=300,bbox_inches="tight")
        f.savefig( CLUST_DISAMBIG_DIR + '\Contour__CLUSID_' + str(CLUSTERID) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', dpi=300,bbox_inches="tight")  

        f2.write_image( CLUST_DISAMBIG_DIR + '\Metrics__CLUSID_' + str(CLUSTERID) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 

        f3.write_image( CLUST_DISAMBIG_DIR + '\Plasticity_TWindow_CLUSID_' + str(CLUSTERID) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 
        
        f4.write_image( CLUST_DISAMBIG_DIR + '\Plasticity_Cumulative_CLUSID_' + str(CLUSTERID) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 

        ######
        print("Saving to" + CLUST_DISAMBIG_DIR + '\Disambiguated_ClusterID_'+str(CLUSTERID)+'_'+str(n))
        # f.savefig( CLUST_DISAMBIG_DIR + '\ClusterID__'+str(ClusterID)+'__disambiguated__R'+str(XYRange)+'_'+str(n)  ,dpi=300,bbox_inches="tight")

    return   

####### new 1-18-2023 #######

def plot_single_cell_factor_deprecated(cell_df, top_dictionary, CLUSTERID, PLOT_PLASTICITY = True, thisistavg = True):
    # labels=wholetrack_exemplar_df['label'].unique()
    # totaltime = totalframes*SAMPLING_INTERVAL
    import matplotlib.cm as cm
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    df=cell_df
    # for the tavg version, work out the cluster ID from the tavg_label column

    if thisistavg == True:
        clusterID=cell_df['tavg_label'].unique()[0]
        plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"

    else:
        clusterID=CLUSTERID
        plottitle="Cluster ID: " + str(clusterID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"

    factors = top_dictionary[clusterID]
    n=len(factors)
    factors=factors[::-1]

    # for the non-tavg version, work out the cluster ID from the label column
    
    
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False, subplot_titles=factors) 
    # add a new column to the cell_df that is the frame * SAMPLING_INTERVAL
    cell_df['time']=cell_df['frame']*SAMPLING_INTERVAL

    for i in range(n):
        factor=factors[i]
        
        df=df.sort_values(by='frame')

        fig.add_trace(go.Scatter(x=df['time'], y=df[factor], mode='lines', name=factor), row=i+1, col=1)

        # change font size of the trace names
        fig.update_layout(font_size=PLOT_TEXT_SIZE)
        fig.update_annotations(font_size=PLOT_TEXT_SIZE)

        # make the range of the x axis the min and max of the time column
        # fig.update_xaxes(range=[0, totaltime], row=i+1, col=1)
        mintime=df['time'].min()
        maxtime=df['time'].max()

        # change the color of the grid lines to light grey  
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', title_text='time (mins)', nticks=10, range = [mintime-1, maxtime+1])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey')
        #change each y axis title to the factor name in the for loop
        fig.update_yaxes(title_text=factor, row=i+1, col=1)

    
     # make sure the x axis has ranges from 0 to totalframes
    # fig.update_xaxes(range=[0, totaltime])
    # fig.update_xaxes(range=[0, totaltime])
    # fig.update_layout(height=n*300, width=900, title_text='', font_size=24, plot_bgcolor='white', xaxis_showgrid=True, yaxis_showgrid=True, title_font_size=24, hovermode="closest")
    fig.update_layout(height=n*450, width=900, font_size=PLOT_TEXT_SIZE, plot_bgcolor='white', xaxis_showgrid=True, yaxis_showgrid=True, title_font_size=PLOT_TEXT_SIZE, hovermode="x unified", colorway=px.colors.qualitative.Dark24)
    # change the overall title to the label and condition
    # plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
    # plottitle="Cluster ID: " + str(clusterID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
    fig.update_layout(title_text=plottitle, title_font_size=PLOT_TEXT_SIZE)
    #remove the legend
    fig.update_layout(showlegend=False)

   
    # fig.update_layout(legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.02,
    #     xanchor="right",
    #     x=3))
    

    # fig.show()

    #####
    if PLOT_PLASTICITY == True:
        plasticity_factors=['label', 'tavg_label','twind_n_changes', 'tavg_twind_n_changes', 'twind_n_labels', 'tavg_twind_n_labels']
        n=len(plasticity_factors)
        # nticks should be only integers
        # layout(yaxis=list(tickformat=',d'))
        fig2 = make_subplots(rows=n, cols=1, shared_xaxes=False,  vertical_spacing = 0.1, horizontal_spacing = 0.2) #subplot_titles=plasticity_factor_titles,
        for i in range(n):
            plasticity_factor=plasticity_factors[i]       
            # print('The plasticity factor is: ' + plasticity_factor) 
            df=df.sort_values(by='frame')
            # if i <= 2:
            #     columntoplot = 1
            #     rowtoplot = i+1
            # else:
            #     columntoplot = 2
            #     rowtoplot = i-2
            fig2.add_trace(go.Scatter(x=df['time'], y=df[plasticity_factor], mode='lines', name=plasticity_factor), row=i+1, col=1)
            # Capitalize the plasticity factor and remove the underscore
            plasticity_factor_title=plasticity_factor.replace('_', ' ').capitalize()
            # print ('The plasticity factor title is: ' + plasticity_factor_title)   
            # change font size of the trace names
            fig2.update_layout(font_size=PLOT_TEXT_SIZE)
            fig2.update_annotations(font_size=PLOT_TEXT_SIZE)
            # make the range of the x axis the min and max of the time column
            mintime=df['time'].min()
            maxtime=df['time'].max()
            # change the color of the grid lines to light grey  
            fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', title_text='time (mins)', nticks=10, range = [mintime-1, maxtime+1])
            fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', nticks = 5, title_text=plasticity_factor_title, row=i+1, col=1)

        fig2.update_layout(
            yaxis = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis2 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis3 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis4 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis5 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis6 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            )
        )
            #change each y axis title to the factor name in the for loop
            # fig2.update_yaxes(title_text=plasticity_factor_title, row=rowtoplot, col=columntoplot)
        fig2.update_layout(height=n*450, width=900, font_size=PLOT_TEXT_SIZE, plot_bgcolor='white', xaxis_showgrid=True,
                            yaxis_showgrid=True,  hovermode="x unified", colorway=px.colors.qualitative.Dark24) #title_font_size=PLOT_TEXT_SIZE,
    # change the overall title to the label and condition
        # plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
        # plottitle="Cluster ID: " + str(clusterID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
        fig2.update_layout(title_text=plottitle, title_font_size=PLOT_TEXT_SIZE)
        # fig2.update_layout(yaxis=list(tickformat ='d')) #yaxis_tickformat =',d', yaxis_tickformat ='d',
        #remove the legend
        fig2.update_layout(showlegend=False)    

        ########### DEV
        plasticity_factors=['label', 'tavg_label','cum_n_changes', 'tavg_cum_n_changes', 'cum_n_labels', 'tavg_cum_n_labels']
        n=len(plasticity_factors)
        fig3 = make_subplots(rows=n, cols=1, shared_xaxes=False,  vertical_spacing = 0.1, horizontal_spacing = 0.2) #subplot_titles=plasticity_factor_titles,
        for i in range(n):
            plasticity_factor=plasticity_factors[i]       
            # print('The plasticity factor is: ' + plasticity_factor) 
            df=df.sort_values(by='frame')
            # if i <= 2:
            #     columntoplot = 1
            #     rowtoplot = i+1
            # else:
            #     columntoplot = 2
            #     rowtoplot = i-2
            fig3.add_trace(go.Scatter(x=df['time'], y=df[plasticity_factor], mode='lines', name=plasticity_factor), row=i+1, col=1)
            # Capitalize the plasticity factor and remove the underscore
            plasticity_factor_title=plasticity_factor.replace('_', ' ').capitalize()
            # print ('The plasticity factor title is: ' + plasticity_factor_title)   
            # change font size of the trace names
            fig3.update_layout(font_size=PLOT_TEXT_SIZE)
            fig3.update_annotations(font_size=PLOT_TEXT_SIZE)
            # make the range of the x axis the min and max of the time column
            mintime=df['time'].min()
            maxtime=df['time'].max()
            # change the color of the grid lines to light grey  
            fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', title_text='time (mins)', nticks=10, range = [mintime-1, maxtime+1])
            fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', nticks = 5, title_text=plasticity_factor_title, row=i+1, col=1)
            #change each y axis title to the factor name in the for loop
            # fig2.update_yaxes(title_text=plasticity_factor_title, row=rowtoplot, col=columntoplot)
        fig3.update_layout(height=n*450, width=900, font_size=PLOT_TEXT_SIZE, plot_bgcolor='white', xaxis_showgrid=True,
                            yaxis_showgrid=True,  hovermode="x unified", colorway=px.colors.qualitative.Dark24) #title_font_size=PLOT_TEXT_SIZE, 
        
        fig3.update_layout(
            yaxis = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis2 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis3 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis4 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis5 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis6 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            )
        )
    # change the overall title to the label and condition
        # plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
        # plottitle="Cluster ID: " + str(clusterID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
        fig3.update_layout(title_text=plottitle, title_font_size=PLOT_TEXT_SIZE)
        # fig2.update_layout(yaxis_tickformat =',d', yaxis=dict(tickformat ='d'))
        #remove the legend
        fig3.update_layout(showlegend=False)   


        ########### DEV

    ######

    return fig, fig2, fig3


def disambiguate_tavg(df, exemp_df, top_dictionary, XYRange=100,boxoff=True): #separate function for disambiguating the tavg clusters

    # This part extracts the whole cell tracks (from df) represented by single exemplars (from exemp_df)
    wholetrack_exemplararray=np.ones(len(df.axes[1])) # Makes a first row of ones to initialize the array

    for n in exemp_df.index:
        row=exemp_df.iloc[n] #extract an example exemplar row
        # Use exemplar row to look up the corresponding cell TRACK from the cell_df
        cell_df = df[(df['Condition']==row['Condition']) &
                        (df['Replicate_ID']==row['Replicate_ID']) &
                        (df['particle']==row['particle']) &
                        (df['ntpts']==row['ntpts'])]
        print('Condition = ' + str(cell_df['Condition_shortlabel'].iloc[0]) +
              '. TAVG_Cluster ID = ' + str(cell_df['tavg_label'].iloc[0]) +
              '. Cell_df track length = ' + str(len(cell_df['ntpts'])) + ' and ntpts from tavg_exemplar_df = ' + str(row['ntpts']))
        cell_array = cell_df.to_numpy()
        wholetrack_exemplararray=np.vstack((wholetrack_exemplararray, cell_array))

    
        # print(top_dictionary)
        CLUSTERID=1
        f=plot_cell_metrics_tavg(cell_df, XYRange, boxoff, row, top_dictionary) #mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS

        f2,f3 = plot_single_cell_factor(cell_df, top_dictionary, CLUSTERID, PLOT_PLASTICITY = True)

        print("Saving to" + CLUST_DISAMBIG_DIR_TAVG + '\Disambiguated_WHOLETRACK_Condition = ' + str(cell_df['Condition_shortlabel'].iloc[0]) +
              '. TAVG_Cluster ID = ' + str(cell_df['tavg_label'].iloc[0]) +'__disambiguated__R'+str(XYRange)+'_'+str(n))

        # f.savefig( CLUST_DISAMBIG_DIR_TAVG + '\Contour ' + str(cell_df['Condition_shortlabel'].iloc[0]) +
            #   '. TAVG_Cluster ID = ' + str(cell_df['tavg_label'].iloc[0]) +'__disambiguated__R'+str(XYRange)+'_'+str(n)+'.png'  ,dpi=300,bbox_inches="tight")
        f.savefig( CLUST_DISAMBIG_DIR_TAVG + '\Contour__TAVG_CLUSID_' + str(cell_df['tavg_label'].iloc[0]) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', dpi=300,bbox_inches="tight")  

        f2.write_image( CLUST_DISAMBIG_DIR_TAVG + '\Metrics__TAVG_CLUSID_' + str(cell_df['tavg_label'].iloc[0]) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 

        f3.write_image( CLUST_DISAMBIG_DIR_TAVG + '\Plasticity__TAVG_CLUSID_' + str(cell_df['tavg_label'].iloc[0]) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 

    wholetrack_exemplararray=np.delete(wholetrack_exemplararray, obj=0, axis=0) # delete the initialization row of ones
    colsare=df.columns.tolist()
    wholetrack_exemplar_df = pd.DataFrame(wholetrack_exemplararray, columns=colsare)



    return(wholetrack_exemplar_df)






#########



def clustering_heatmap(df_in, dr_factors=DR_FACTORS): #New function 12-14-2022

    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    fig, ax = plt.subplots(figsize=(24,20))

    df_in_labels=df_in['label']
    CLUSTERON=dr_factors #what columns you want included in the heatmap?

    sub_set = df_in[CLUSTERON] #makes a subset_df of just the measurements (no DR, no labels)

    Z = StandardScaler().fit_transform(sub_set)
    X = MinMaxScaler().fit_transform(Z)
    sub_set_scaled_df=pd.DataFrame(data=X, columns = CLUSTERON)
    lab_sub_set_scaled_df = pd.concat([sub_set_scaled_df,df_in_labels], axis=1)

    lab_sub_set_scaled_df.set_index("label", inplace = True)
    lab_sub_set_scaled_df.sort_index(ascending=True, inplace=True)

    sns.heatmap(lab_sub_set_scaled_df,vmin=0, vmax=1,fmt='.2g',ax=ax, cmap='crest') #center=0.5 #, square=True
    ax.invert_yaxis()
    ax.set_ylabel("Cluster", fontsize=24)
    ax.tick_params(axis='both', labelsize=24)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)

    fig.savefig( CLUST_DISAMBIG_DIR + '\heatmap_disambiguation',dpi=600,bbox_inches="tight")

    return

def correlation_matrix_heatmap(df_in, factors = ALL_FACTORS): #Function added new 12-14-2022

    cmap = 'crest'
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    # df_in_labels=df_in['label']
    # CLUSTERON=factors #what columns you want included in the heatmap?
    sub_set = df_in[factors] #makes a subset_df of just the measurements (no DR, no labels)
    Z = StandardScaler().fit_transform(sub_set)
    X = MinMaxScaler().fit_transform(Z)
    sub_set_scaled_df=pd.DataFrame(data=X, columns = factors)
    # print(sub_set_scaled_df)
    # lab_sub_set_scaled_df = pd.concat([sub_set_scaled_df,df_in_labels], axis=1)
    #Correlation matrix
    corr = sub_set_scaled_df.corr('pearson') # 'pearson','kendall','spearman' ‘spearman’
    #Figure set up
    f, ax = plt.subplots(figsize=(24, 20))
    #Make mask to remove duplicates
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, ax=ax)
    #Plot settings
    ax.set_ylabel("", fontsize=24)
    ax.tick_params(axis='both', labelsize=24)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b,t)
    g, a = plt.xlim()
    # plt.show()
    f.savefig( CLUST_DISAMBIG_DIR + '\correlation_matrix',dpi=300,bbox_inches="tight")
    return

def contribution_to_clusters_deprecated(df_in, threshold_value=0.0001, dr_factors=DR_FACTORS): #New function 21-14-2022

    ### 12-12-2022 ##### DEV THIS ONE

    # Part 1: take the metrics and center scales them, then puts them back into a DF
    # Part 2: Find the median value per cluster for each metric using groupby
    # Part 3: Makes some iterables for the parts below.
    # Part 4: Makes a Variance DF that describes the variance of each metric BETWEEN CLUSTERS
    # Part 5: Makes a boolean mask of variances based on that threshold value, and a dataframe that contains values if true, and NaN if not
    # Part 6: Prints the names of the important values per cluster. TBH, printing the df might be better.
    # Part 7: print a boolean_df?
    # Part 8: exports a df that can be used to select what metrics you want to show?

    # from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler

    # df_in = tptlabel_dr_df
    # threshold_value = 0.0001
    CLUSTERON = dr_factors

    # Part 1: take the metrics and center scales them, then puts them back into a DF
    sub_set = df_in[CLUSTERON]
    X = MinMaxScaler().fit_transform(sub_set)
    thelabels = df_in['label']
    scaled_df_in = pd.DataFrame(data=X, columns = CLUSTERON)
    df_out = pd.concat([scaled_df_in,thelabels], axis=1)#Put this back into a DF, with the col names and labels.
    # df_out

    # Part 2: Find the median value per cluster for each metric using groupby
    clusteraverage_df = df_out.groupby('label').median()#.reset_index(drop=True)

    # Part 3: Makes some iterables for the parts below.
    numberofclusters=len(clusteraverage_df.axes[0])
    iterable_clusters=np.arange(len(clusteraverage_df.axes[0])) # Iterable across rows
    iterable_metrics=np.arange(len(clusteraverage_df.axes[1])) # Iterable across columns (metrics)
    metric_means = clusteraverage_df.mean(axis=0).tolist()

    # Part 4: Makes a Variance DF that describes the variance of each metric BETWEEN CLUSTERS
    variance_between_clusters=[]
    variance_between_clusters_array = []
    variance_between_clusters_array_total=[]

    ###############################################################################
    # Calculates the variance of each value compared to the other values (discounting current value in calculation of mean) ((x - mean)^2)/numberofvalues
    for metric in iterable_metrics:
        # print(metric)
        variance_between_clusters_array=[]
        for cluster in iterable_clusters:
            list1 = clusteraverage_df.iloc[cluster,metric]
            list2=clusteraverage_df.iloc[:,metric].tolist()
            list2.remove(list1)
            meanminusthatvalue=np.mean(list2)
            variance_between_clusters = ((clusteraverage_df.iloc[cluster,metric] - meanminusthatvalue)**2)/(numberofclusters)
            variance_between_clusters_array.append(variance_between_clusters)
        variance_between_clusters_array_total.append(variance_between_clusters_array)
    variance_df = pd.DataFrame(data = variance_between_clusters_array_total, columns = clusteraverage_df.index, index = CLUSTERON )
    display(variance_df)
    ###############################################################################
    # # OLD WAY - CONSIDERS ALL THE VALUES FOR CALCULATING THE MEAN

    # for metric in iterable_metrics:
    #     # print(metric)
    #     variance_between_clusters_array=[]
    #     for cluster in iterable_clusters:
    #         variance_between_clusters = ((clusteraverage_df.iloc[cluster,metric] - metric_means[metric])**2)/(numberofclusters)
    #         variance_between_clusters_array.append(variance_between_clusters)
    #     variance_between_clusters_array_total.append(variance_between_clusters_array)
    # variance_df = pd.DataFrame(data = variance_between_clusters_array_total, columns = clusteraverage_df.index, index = CLUSTERON )
    #############################################################################

    # Part 5: Makes a boolean mask of variances based on that threshold value, and a dataframe that contains values if true, and NaN if not
    high_mask = variance_df > threshold_value
    trueones=variance_df[high_mask]
    # display(high_mask)

    # Part 6: Prints the names of the important values per cluster. TBH, printing the df might be better.
    for clusterlabel in iterable_clusters:
        clusteryeah=trueones.iloc[:,clusterlabel]
        clusterboolean=clusteryeah.notnull()

        highmetrics=clusteryeah[clusterboolean]
        clusternames=trueones.columns.tolist()
        print("Cluster " + str(clusternames[clusterlabel]) + " has the following main contributors: " + str(highmetrics.index.tolist()))

    # Part 7: print a boolean_df?

    # Part 8: exports a df that can be used to select what metrics you want to show?
    return(variance_df)

def contribution_to_clusters_topdictionary_deprecated(df_in, threshold_value=0.0001, dr_factors=DR_FACTORS, howmanyfactors=6, scalingmethod = SCALING_METHOD): #New function 21-14-2022

    ### 12-12-2022 ##### 

    # Part 1: take the metrics and center scales them, then puts them back into a DF
    # Part 2: Find the median value per cluster for each metric using groupby
    # Part 3: Makes some iterables for the parts below.
    # Part 4: Makes a Variance DF that describes the variance of each metric BETWEEN CLUSTERS
    # Part 5: Makes a boolean mask of variances based on that threshold value, and a dataframe that contains values if true, and NaN if not
    # Part 6: Prints the names of the important values per cluster. TBH, printing the df might be better.
    # Part 7: print a boolean_df?
    # Part 8: exports a df that can be used to select what metrics you want to show?

    # from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PowerTransformer

    # df_in = tptlabel_dr_df
    # threshold_value = 0.0001
    CLUSTERON = dr_factors

    # Part 1: take the metrics and center scales them, then puts them back into a DF
    sub_set = df_in[CLUSTERON]
    Z = sub_set.values
    if scalingmethod == 'minmax': #log2minmax minmax powertransformer
        X = MinMaxScaler().fit_transform(Z)
        correctcolumns = CLUSTERON
    elif scalingmethod == 'log2minmax':
        negative_FACTORS = []
        positive_FACTORS = []
        for factor in dr_factors:
            if np.min(df_in[factor]) < 0:
                print('factor ' + factor + ' has negative values')
                negative_FACTORS.append(factor)
                
            else:
                print('factor ' + factor + ' has no negative values')
                positive_FACTORS.append(factor)
        
        
        pos_df = df_in[positive_FACTORS]
        pos_x = pos_df.values
        neg_df = df_in[negative_FACTORS]
        neg_x = neg_df.values
        neg_x_ = MinMaxScaler().fit_transform(neg_x)
        pos_x_constant = pos_x + 0.000001
        pos_x_log = np.log2(pos_x + pos_x_constant)
        pos_x_ = MinMaxScaler().fit_transform(pos_x_log)
        X = np.concatenate((pos_x_, neg_x_), axis=1)
        correctcolumns=positive_FACTORS + negative_FACTORS

    elif scalingmethod == 'powertransformer':    
        
        pt = PowerTransformer(method='yeo-johnson')
        X = pt.fit_transform(Z)
        correctcolumns=CLUSTERON

        #########

  
    elif scalingmethod == 'choice': 
        print('Factors to be scaled using log2 and then minmax:')
        FactorsNOTtotransform = ['arrest_coefficient', 'rip_L', 'rip_p', 'rip_K', 'eccentricity', 'orientation', 'directedness', 'turn_angle', 'dir_autocorr', 'glob_turn_deg']
        FactorsNottotransform_actual=[]
        FactorsToTransform_actual=[]
        for factor in dr_factors:
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
        X=x_
        correctcolumns=newcols
        # plt.show()
        # plt.savefig(savedir+ str(scalingmethod) +'.png')
        # df_out = pd.concat([scaled_df_here,df_in[[x,y,z]]], axis=1)    

    elif scalingmethod == None:
        X = Z
        correctcolumns=CLUSTERON

    # X = MinMaxScaler().fit_transform(sub_set)
    thelabels = df_in['label']
    scaled_df_in = pd.DataFrame(data=X, columns = correctcolumns)
    df_out = pd.concat([scaled_df_in,thelabels], axis=1)#Put this back into a DF, with the col names and labels.

   

    ####### Here starts the new bit for the scaled numbers on the disambiguates #######

    # Isolate the columsn of the cell_df that are not the newcols as a list
    cols_to_keep = [col for col in df_in.columns if col not in correctcolumns]
    # Extract a sub df from the cell_df that contains only the columns in cols_to_keep
    scaled_df = pd.concat([df_in[cols_to_keep], scaled_df_in], axis=1)


    # Part 2: Find the median value per cluster for each metric using groupby
    clusteraverage_df = df_out.groupby('label').median()#.reset_index(drop=True)

    

    ###

    # Part 3: Makes some iterables for the parts below.
    numberofclusters=len(clusteraverage_df.axes[0])
    iterable_clusters=np.arange(len(clusteraverage_df.axes[0])) # Iterable across rows
    iterable_metrics=np.arange(len(clusteraverage_df.axes[1])) # Iterable across columns (metrics)
    metric_means = clusteraverage_df.mean(axis=0).tolist()

    # Part 4: Makes a Variance DF that describes the variance of each metric BETWEEN CLUSTERS
    variance_between_clusters=[]
    variance_between_clusters_array = []
    variance_between_clusters_array_total=[]

    ###############################################################################
    # Calculates the variance of each value compared to the other values (discounting current value in calculation of mean) ((x - mean)^2)/numberofvalues
    for metric in iterable_metrics:
        # print(metric)
        variance_between_clusters_array=[]
        for cluster in iterable_clusters:
            list1 = clusteraverage_df.iloc[cluster,metric]
            list2=clusteraverage_df.iloc[:,metric].tolist()
            list2.remove(list1)
            meanminusthatvalue=np.mean(list2)
            variance_between_clusters = ((clusteraverage_df.iloc[cluster,metric] - meanminusthatvalue)**2)/(numberofclusters)
            variance_between_clusters_array.append(variance_between_clusters)
        variance_between_clusters_array_total.append(variance_between_clusters_array)
    variance_df = pd.DataFrame(data = variance_between_clusters_array_total, columns = clusteraverage_df.index, index = CLUSTERON )
    # display(variance_df)


    ##########
    trans_variance_df=variance_df.T
    df= trans_variance_df

    topfactors = []
    contributors=[]

    for ind in df.index:
        col=trans_variance_df.loc[ind,:] #using loc as the index is label.
        sortedcol=col.sort_values(ascending=False)
        # thename=sortedcol.name
        #Select top 4 of these
        contributors=(sortedcol[0:howmanyfactors]).index.tolist()
        # contributors.insert(0, thename)
        topfactors.append(contributors)

    #Make a dictionary of these results

    top_dictionary = {}
    keys = df.index.tolist()
    for i in keys:
        top_dictionary[i] = topfactors[i]

    ###############################################################################
    # # OLD WAY - CONSIDERS ALL THE VALUES FOR CALCULATING THE MEAN

    # for metric in iterable_metrics:
    #     # print(metric)
    #     variance_between_clusters_array=[]
    #     for cluster in iterable_clusters:
    #         variance_between_clusters = ((clusteraverage_df.iloc[cluster,metric] - metric_means[metric])**2)/(numberofclusters)
    #         variance_between_clusters_array.append(variance_between_clusters)
    #     variance_between_clusters_array_total.append(variance_between_clusters_array)
    # variance_df = pd.DataFrame(data = variance_between_clusters_array_total, columns = clusteraverage_df.index, index = CLUSTERON )
    #############################################################################

    # Part 5: Makes a boolean mask of variances based on that threshold value, and a dataframe that contains values if true, and NaN if not
    high_mask = variance_df > threshold_value
    trueones=variance_df[high_mask]
    # display(high_mask)

    # Part 6: Prints the names of the important values per cluster. TBH, printing the df might be better.
    for clusterlabel in iterable_clusters:
        clusteryeah=trueones.iloc[:,clusterlabel]
        clusterboolean=clusteryeah.notnull()

        highmetrics=clusteryeah[clusterboolean]
        clusternames=trueones.columns.tolist()
        # print("Cluster " + str(clusternames[clusterlabel]) + " has the following main contributors: " + str(highmetrics.index.tolist()))

    # Part 7: print a boolean_df?

    # Part 8: exports a df that can be used to select what metrics you want to show?
    return top_dictionary, clusteraverage_df, scaled_df



def plot_cluster_averages_deprecated(top_dictionary, df, scaled_df): # New 3-7-2023
    num_plots = len(top_dictionary)
    # print(top_dictionary)
    # Reverse the order of the values in the dictionary so that the biggest contributor to variability is on top
    top_dictionary = {k: v[::-1] for k, v in top_dictionary.items()}
    # print(top_dictionary)
    ### Make a totally non-normalized version of the clusteraverage_df
    cluster_average_df = df.groupby('label').median()#.reset_index(drop=True)
    cluster_average_scaled_df = scaled_df.groupby('label').median()#.reset_index(drop=True)

    # Create a grid of subplots with one row and num_plots columns
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(15*num_plots,10))
    
    # Loop over the cluster IDs and corresponding values in the dictionary
    for i, (cluster_id, value) in enumerate(top_dictionary.items()):
        # Get the row in the dataframe that corresponds to the current cluster ID
        cluster_row = cluster_average_df.loc[cluster_id]
        scaled_cluster_row=cluster_average_scaled_df.loc[cluster_id]
        
        # Loop over the column names for the current key and create a text string
        text_str = ""
        scaled_text_str = ""
        for column_name in value:
            # Get the value in the specified column for the current cluster
            column_value = round(cluster_row[column_name], 4)
            scaled_column_value = round(scaled_cluster_row[column_name], 1)
            
            # Add the string and the corresponding value to the text string
            text_str += f"{column_name.title()}: {column_value} (s={scaled_column_value})\n"
            text_str = text_str.replace('_',' ')

            # scaled_text_str += f"{column_name.title()}: {scaled_column_value}\n"
            # scaled_text_str = f"{scaled_column_value}\n"
            # scaled_text_str = scaled_text_str.replace('_',' ')
        
        # Plot the text string as text in the current subplot
        axs[i].text(0.5, 0.5, text_str.strip(), ha='center', va='center', fontsize=30)
        # axs[i].text(0.8, 0.5, scaled_text_str.strip(), ha='center', va='center', fontsize=30)
        
        # Set the title of the current subplot to the current cluster ID
        axs[i].set_title(f"Cluster {cluster_id}", fontsize = 30)

        axs[i].set_xticks([])
        axs[i].set_yticks([])
        # Remove the lines around the subplot
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
    
    
    # Add an overall title to the figure
    fig.suptitle("Average of top contributors per cluster ID", fontsize=36)
    # Save figure as a png
    # plt.savefig("cluster_average.png")
    fig.savefig( CLUST_DISAMBIG_DIR + '\cluster_average.png',dpi=300,bbox_inches="tight") 
 
    
    plt.show()


#####
def plot_cell_trajectories(cell_df, dr_df, dr_method='tSNE',contour_scale=1,cmap_by=None):#'area'):

    '''
    Compare cell trajectories through physical (xy) and Dimension Reduced (dr) space.

    Input:
        cell_df: DataFrame containing the cell to be plotted.
        dr_df: dimension reduced dataframe (for plot context, grayed out - full data set.)
        dr_method: tSNE, PCA, umap
        contour_scale: factor to scale the contours when plotting them in DR space
        cmap_by: factor to colormap the tSNE scatter. Default: area.
                Set to 'label' if lab_dr_df supplied as input

TO TEST: Does this handle multiple ones?? Maybe it should...
INCOMPLETE: Still need to colormap the trajectory segments AND contours by time.




    '''

    if(dr_method == 'tsne' or dr_method == 'tSNE'):

            x_lab = 'tSNE1'
            y_lab = 'tSNE2'

    elif dr_method == 'PCA':

            x_lab = 'PC1'
            y_lab = 'PC2'

    elif dr_method == 'umap':

            x_lab = 'UMAP1'
            y_lab = 'UMAP2'

    contour_list = get_cell_contours(cell_df)
    assert len(contour_list) == len(cell_df.index), 'Contours doesnt match input data'

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

    ax1.title.set_text('Cell trajectory: xy space')
    ax2.title.set_text('Cell contour: xy space')
    ax3.title.set_text('Cell trajectory: ' + dr_method + ' space')
    ax4.title.set_text('Cell contour:' + dr_method + ' space')

    ax1.set_aspect('equal')
    ax1.set_adjustable("datalim")
    ax2.set_aspect('equal')
    ax2.set_adjustable("datalim")

    # Plots that don't need to be redrawn for each contour
    ax1.scatter(x=dr_df['x'],y=dr_df['y'],c=dr_df['rip_L'], alpha=0.5, s=10) # 'gray'
    # ax2.scatter(x=dr_df['x'],y=dr_df['y'],c=dr_df['rip_L'], alpha=0.5, s=10) #'gray'

    '''Consider scaling the scatter window to a function of cell diameters.'''
    scat_wind = 100

    ax1.set_xlim(cell_df['x_pix'].values[0] - scat_wind/2, cell_df['x_pix'].values[0] + scat_wind/2)
    ax1.set_ylim(cell_df['y_pix'].values[0] - scat_wind/2, cell_df['y_pix'].values[0] + scat_wind/2)

    ax2.plot(cell_df['x_pix']*2,cell_df['y_pix']*2,'-o',markersize=3,c='gray')
    '''
    Why does 2* work for scaling???
    '''
    # Calculate rgb values of colormap according to PCs 1-3.
    pc_colors = colormap_pcs(dr_df, cmap='rgb')
    # pc_colors = np.asarray(dr_df[['PC1','PC2','PC3']])
    # scaler = MinMaxScaler()
    # scaler.fit(pc_colors)
    # pc_colors= scaler.transform(pc_colors)

    # ax3.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c=dr_df[cmap_by], alpha=0.1, s=2)
    ax3.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c=pc_colors, alpha=0.1, s=2)

    if 'label' in dr_df.columns:

        # print('Label included, drawing cluster hulls.')
        draw_cluster_hulls(dr_df,cluster_by=dr_method,ax=ax3)

    ax3.plot(cell_df[x_lab],cell_df[y_lab],'-o',markersize=3,c='red')

    ax4.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c='gray', alpha=0.3, s=1)

    assert len(cell_df['x']) == len(contour_list)

    # Colormap the contours
    PALETTE = 'flare'
    colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(contour_list)))
    cm_data = np.asarray(colors)

    for i,contour in enumerate(contour_list):

        rgb=cm_data[i,:]

        this_colour = rgb#'red' # Eventually calculate color along colormap

        if i > 0:
            # segment of the trajectory
            this_seg = cell_df[['x_pix','y_pix']].values[i-1:i+1]
            ax1.plot(this_seg[:,0],this_seg[:,1],'-o',markersize=3,c=this_colour)

            # segment of the DR trajectory
            dr_seg = cell_df[[x_lab,y_lab]].values[i-1:i+1]
            ax3.plot(dr_seg[:,0],dr_seg[:,1],'-o',markersize=3,c=this_colour)

        contour_arr = np.asarray(contour).T


        x = cell_df['x_pix'].values[i]# - window / 2
        y = cell_df['y_pix'].values[i]# - window / 2

        # Cel contour change relative to centroid (less relevant)
        if not np.isnan(np.sum(contour_arr)):
            ax2.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c=this_colour)

        # Need to cell contours to be centered on the cells position within the image
        x_dr = cell_df[x_lab].values[i] - x# - window / 2
        y_dr = cell_df[y_lab].values[i] - y# - window / 2
        ax4.scatter(x=x+x_dr,y=y+y_dr,c='red', alpha=1, s=2)

        # Cell contour relative to tSNE positions
        if not np.isnan(np.sum(contour_arr)):
            ax4.plot((x_dr+contour_arr[:,0])*contour_scale,(y_dr+contour_arr[:,1])*contour_scale,'-o',markersize=1,c=this_colour)

        return fig


def visualize_cell_t_window(cell_df, dr_df, t, t_window, contour_list=None, dr_method='tSNE', cid='test', contour_scale=1,cmap_by = 'area'):

    '''

    NOTE; THis function still uses a magic factor of 2 when drawing the trajectories relative to the contours.
    So, I wouldn't trust the absolute axis values for the cell location,

    TO DO: find unique identifier for this cell.


    To better understand the measurements being included for each cell at each timepoint,.
    A plot to represent the cell shape through space and time, that is representative
    of the measures included in the time window.

    Namely, the path centered at that point.
    The cell contour at that timepoint, but also faint representations of the othrer timepoints
    that are being averaged across.

    '''

    df = cell_df.copy()

    if(dr_method == 'tsne' or dr_method == 'tSNE'):

            x_lab = 'tSNE1'
            y_lab = 'tSNE2'

    elif dr_method == 'PCA':

            x_lab = 'PC1'
            y_lab = 'PC2'

    elif dr_method == 'umap':

            x_lab = 'UMAP1'
            y_lab = 'UMAP2'

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10,5))

    # ax1: xy space
    ax1.title.set_text('Cell contour + trajectory in time window: xy space' )
    ax2.title.set_text('Cell contour + trajectory: ' + dr_method + ' space')

    ax1.set_aspect('equal')
    ax1.set_adjustable("datalim")

    scale_factor = 2

    scat_wind = int(60 / MICRONS_PER_PIXEL)#100
    min_x, max_x = (np.min(cell_df['x_pix']),np.max(cell_df['x_pix']))
    min_y, max_y = (np.min(cell_df['y_pix']),np.max(cell_df['y_pix']))

    ax1.set_xlim(min_x* scale_factor-scat_wind/2, max_x* scale_factor+scat_wind/2)
    ax1.set_ylim(min_y* scale_factor-scat_wind/2, max_y* scale_factor+scat_wind/2)

    # Add select metrics as text.
    text_x = min_x * scale_factor-scat_wind/1.6 # Still a magic number, replace with calibration value ??
    text_y = np.mean(cell_df['y_pix']) * scale_factor#+scat_wind/2# Still a magic number, replace with calibration value ??


    # ax2: Low-dimensional space
    ax2.set_xlim(np.min(dr_df[x_lab])-abs(np.min(dr_df[x_lab])/2),
                 np.max(dr_df[x_lab])+ abs(np.max(dr_df[x_lab])/2))
    ax2.set_ylim(np.min(dr_df[y_lab])-abs(np.min(dr_df[y_lab])/2),
                 np.max(dr_df[y_lab]) + abs(np.max(dr_df[y_lab])/2))

    ax2.set_aspect('equal')
    # ax2.set_adjustable("datalim")

    if contour_list is None: # Allow animate_t_window() to pass the contours instead of reloading
        contour_list = get_cell_contours(cell_df)

    assert len(contour_list) == len(cell_df.index), 'Contours doesnt match input data'

    frames = cell_df['frame']
    frame_i = int(t * (len(cell_df['frame'])-1))
    frame = cell_df.iloc[frame_i]['frame']

    identifier = str(cid) +'_'+ dr_method +'_t_'+str(t)
    fig.suptitle('Cell '+str(cell_df['uniq_id'].values[0])+' behaviour at ' +
            str(format(frame * SAMPLING_INTERVAL, '.2f')) + ' mins', fontsize=12)
    # Time-windowed dataframe
    if t_window is not None:

        # get a subset of the dataframe across the range of frames
        cell_t_df = cell_df[(cell_df['frame']>=frame - t_window/2) &
                      (cell_df['frame']<frame + t_window/2)]
    else:

        cell_t_df = cell_df.copy()

    # Colormap the contours (to the time window)
    PALETTE = 'flare'
    colors = np.asarray(sns.color_palette(PALETTE, n_colors=t_window))
    cm_data = np.asarray(colors)

    # Get these contours and the track.
    '''Should be able to index the contour list generated above using the frame_i'''

    this_contour = contour_list[frame_i]
    t_window_contours = contour_list[int(frame_i-t_window/2):int(frame_i+t_window/2)]

    # Plot the spatial trajectory.
    ax1.plot(cell_t_df['x_pix']*scale_factor,cell_t_df['y_pix']*scale_factor,'-o',markersize=3,c='gray')


    # Plot the dimension reduction plot
    # ax2.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c=dr_df[cmap_by], alpha=0.1, s=2)
    pc_colors = colormap_pcs(dr_df, cmap='rgb')
    ax2.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c=pc_colors, alpha=0.3, s=1)
    ax2.plot(cell_t_df[x_lab],cell_t_df[y_lab],'-o',markersize=2,c='black')

    # Plot the contours (All indexing relative to time_window.)
    for i,contour in enumerate(t_window_contours):

        rgb=cm_data[i,:]

        this_colour = rgb#'red' # Eventually calculate color along colormap

        contour_arr = np.asarray(contour).T# * contour_scale

        # Get the x,y pixel position of the cell at this frame
        x = cell_t_df['x_pix'].values[i]# - window / 2
        y = cell_t_df['y_pix'].values[i]# - window / 2

        # Need to cell contours to be centered on the cells position within the image
        x_dr = cell_t_df[x_lab].values[i] - x# - window / 2
        y_dr = cell_t_df[y_lab].values[i] - y# - window / 2


#         # Translate and scale the contour. (NOT YET WORKING CORRECTLY)
#         rel_contour_arr = contour_arr.copy()
#         rel_contour_arr[:,0] = contour_arr[:,0] - x_dr
#         rel_contour_arr[:,1] = contour_arr[:,1] - y_dr
#         rel_contour_arr = rel_contour_arr * contour_scale

        twind_traj = cell_t_df[['x_pix','y_pix']].values


        # Cell contour change relative to centroid (less relevant)
        if not np.isnan(np.sum(contour_arr)):

            if(i == int(t_window/ 2)): # If center of the t_window, this frame

                # XY-space visualization

                ax1.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=5,linewidth=5,c=this_colour, alpha=1)
                this_seg = cell_t_df[['x_pix','y_pix']].values[i-1:i+1]
                ax1.plot(this_seg[:,0]*scale_factor,this_seg[:,1]*scale_factor,'-o',markersize=5,linewidth=5,c=this_colour)

#                 # Add select metrics as text (moving with the cell)
#                 text_x = x*scale_factor # Still a magic number, replace with calibration value ??
#                 text_y = y*scale_factor # Still a magic number, replace with calibration value ??

                for n, fac in enumerate(MIG_DISPLAY_FACTORS):

                    ax1.text(text_x + 15,text_y +5 *  (n+1), fac +': '+ format(cell_t_df.iloc[i][fac], '.1f'),
                            color='tab:blue', fontsize=10)

                for n, fac in enumerate(SHAPE_DISPLAY_FACTORS):

                    ax1.text(text_x + 15, text_y - 5 * (n+1), fac +': '+ format(cell_t_df.iloc[i][fac], '.1f'),
                            color='tab:orange', fontsize=10)


                # Low-D space visualization

                # Draw a brief trajectory through DR space
                ax2.scatter(x=cell_t_df[x_lab].values[i],y=cell_t_df[y_lab].values[i],color=this_colour, alpha=1, s=50)
                # Also draw this contour directly onto the DR plot.
                ax2.plot(x_dr+contour_arr[:,0],y_dr+contour_arr[:,1],'-o',markersize=1,c=this_colour)
                # Draw the trajectory segment for the time window included in the current data

                ax2.plot((x_dr+twind_traj[:,0]),
                         (y_dr+twind_traj[:,1]),'-o',markersize=3,c=this_colour,linewidth=1)
                ax2.plot((x_dr+this_seg[:,0]),
                         (y_dr+this_seg[:,1]),'-o',markersize=3,c=this_colour,linewidth=3)


            else:
                ax1.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c=this_colour, alpha=0.5)

    return fig

def plot_traj_cluster_avg(traj_list, cluster_list, label):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''

    color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                 'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])

    cluster_count = np.max(cluster_list) + 1

    for cluster_id in np.unique(cluster_list):


        clust_inds = np.argwhere(cluster_list==cluster_id)
        clust_inds = np.squeeze(clust_inds).astype(int)

        this_traj_list = []

        for ind in clust_inds:
            this_traj = traj_list[ind]
            this_traj_list.append(this_traj)

#         # Plot the trajectories without any averging(if not the same length)
#         for cluster_traj in this_traj_list:
#             plt.plot(cluster_traj[:,0], cluster_traj[:,1], c=color_lst[cluster_id % len(color_lst)], alpha=0.1)

        # Find the average along the trajectory or segment (must be same length)

        cluster_traj = np.asarray(this_traj_list)
        clust_avg_traj = np.mean(cluster_traj, axis=0)

        if cluster_id == -1:
            # Means it it a noisy trajectory, paint it black
            plt.plot(cluster_traj[:, :,0], cluster_traj[:, :,1], c='k', linestyle='dashed', alpha=0.01)

        else:
            plt.plot(cluster_traj[:, :,0], cluster_traj[:, :,1], c=color_lst[cluster_id % len(color_lst)], alpha=0.1)
            plt.plot(clust_avg_traj[:, 0], clust_avg_traj[:, 1], c=color_lst[cluster_id % len(color_lst)], alpha=1,mec='black')



    if STATIC_PLOTS:

        plt.savefig(CLUST_DIR+'average_trajectories.png', dpi=300)


def trajectory_cluster_vis(traj_clust_df,traj_factor, scatter=False):

    if (traj_factor=='tSNE' or traj_factor=='tsne'):
        if scatter:
            traj_clust_df.plot.scatter(x='tSNE1',y='tSNE2',c='traj_id',colormap='viridis')
        x_label = 'tSNE1'
        y_label = 'tSNE2'

    elif (traj_factor=='UMAP' or traj_factor=='umap'):
        if scatter:
            traj_clust_df.plot.scatter(x='UMAP1',y='UMAP2',c='traj_id',colormap='viridis')
        x_label = 'UMAP1'
        y_label = 'UMAP2'

    # Define a custom colormap for the trajectory clusters (To match what's done in draw_cluster_hulls())
    cluster_colors = []
    labels = list(set(traj_clust_df['traj_id'].unique()))
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))

    fig, ax = plt.subplots()
    ax.set_title('Cell trajectory clustering through low-dimensional space', fontsize=18)
    ax.scatter(x=traj_clust_df[x_label], y=traj_clust_df[y_label], s=0.5,alpha=0.5,
            c='gray')

    # labels = list(set(traj_clust_df['traj_id'].unique())) # Set helps return a unioque ordeed list.

    for i, traj in enumerate(labels[:-1]): # Same as when drawing contours
        traj_sub_df = traj_clust_df[traj_clust_df['traj_id'] == traj]

        # Draw lines and scatters individually for each label
        ax.plot(traj_sub_df[x_label], traj_sub_df[y_label], alpha=0.2, c=cluster_colors[i],linewidth=0.5)
        ax.scatter(x=traj_sub_df[x_label], y=traj_sub_df[y_label], s=0.8,alpha=0.5,
            color=cluster_colors[i])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    draw_cluster_hulls(traj_clust_df,cluster_by=traj_factor, cluster_label='traj_id', ax=ax, color_by='cluster',legend=True)

    if STATIC_PLOTS:

        plt.savefig(CLUST_DIR + 'trajectory_clusters.png', format='png', dpi=600)

    # return fig






##### DEBUGGING ######

##### THE plot_cell_metrics_timepoint FUNCTION ######

def plot_cell_metrics_timepoint_dev_deprecated(cell_df, i_step, scaled_cell_df, XYRange,boxoff, top_dictionary, Cluster_ID, mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS, dr_factors=DR_FACTORS): #spoof
    '''
    For a selected cell, at a selected step of its trajectory, make a plot
    '''

    # Using the same sample cell trajectory, print out some measurements
    fig, (ax) = plt.subplots(1,1, figsize=(0.08*XYRange,0.08*XYRange))
    # print(top_dictionary)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Arial'] + plt.rcParams['font.serif'] #Times New Roman
    plt.rcParams['mathtext.default'] = 'regular'

    ax.title.set_text('Cell contour: xy space')
    ax.plot(cell_df['x_pix'],cell_df['y_pix'],'-o',markersize=3,c='black')

    ################
    xminimum=cell_df['x_pix'].min()
    xmaximum=cell_df['x_pix'].max()
    xmid = np.median([xmaximum, xminimum])
    xmin=xmid-XYRange/2
    xmax=xmid+XYRange/2

    yminimum=cell_df['y_pix'].min()
    ymaximum=cell_df['y_pix'].max()
    ymid = np.median([ymaximum, yminimum])
    ymin=ymid-XYRange/2
    ymax=ymid+XYRange/2

    #######################

    contour_list = get_cell_contours(cell_df)
    # Draw all contours faintly as image BG
    for i,contour in enumerate(contour_list):

        rgb = np.random.rand(3,)
        this_colour = rgb#'red' # Eventually calculate color along colormap
        contour_arr = np.asarray(contour).T


        if not np.isnan(np.sum(contour_arr)): # If the contour is not nan

            x = cell_df['x_pix'].values[i]# - window / 2
            y = cell_df['y_pix'].values[i]# - window / 2
            # Cell contour relative to x,y positions
            '''Want to double check that the x,y positions not mirrored from the contour function'''
            ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c='gray', alpha=0.2)

    # Draw this contour, highlighted.
    contour = contour_list[i_step]
    contour_arr = np.asarray(contour).T
    x = cell_df['x_pix'].values[i_step]# - window / 2
    y = cell_df['y_pix'].values[i_step]# - window / 2

    # Cell contour relative to x,y positions
    '''Want to double check that the x,y positions not mirrored from the contour function'''
    if not np.isnan(np.sum(contour_arr)):
        ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c='tab:orange',linewidth=5, alpha=1)

        # Current segment
        if i_step > 0:
            x_seg = cell_df['x_pix'].values[i_step-1:i_step+1]# - window / 2
            y_seg = cell_df['y_pix'].values[i_step-1:i_step+1]# - window / 2
            # ax.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c='red', alpha=1)
            # ax.plot(x_seg*2,y_seg*2,'-o',markersize=10,c='tab:blue', linewidth=4)
            ax.plot(x_seg,y_seg,'-o',markersize=10,c='tab:blue', linewidth=4)

    text_x = xmid
    text_y = ymid
    # SPIDER #
    # p=cell_df.iloc[i_step]
    # ClusterID = cell_df['label'].iloc[i_step]
    # print('TRYING TO MAKE THIS A CLUSTER ID TO USE ON THE DICTIONARY >>>>>>>>' )
    # display(ClusterID)
    # # ClusterID=p['label']
    # print(row['label']) #here, the row label is something different
    # ClusterID = row['label']
    display_factors = top_dictionary[Cluster_ID]
    # You want the list of factors in top_dictionary that relates to the cluster in question.
    # IN this case, the cluster in question is the one at iloc[i_step]

    for n, fac in enumerate(display_factors): #Positioning is currently relative to data. Can it be relative to plot?
        facwithoutunderscores = fac.replace('_',' ')
        text_str = facwithoutunderscores +': '+ format(cell_df.iloc[i_step][fac], '.1f')
        text_str_2 = format(scaled_cell_df.iloc[i_step][fac], '.1f')
        ax.text(text_x + 0.5*XYRange, text_y - (0.3*XYRange) + (0.08*XYRange) + (n*(0.08*XYRange)), 
                text_str, #These weird numbers were worked out manually
                color='k', fontsize=PLOT_TEXT_SIZE, fontdict = None) #spidermoose )
        ax.text(text_x + 0.45*XYRange, text_y - (0.3*XYRange) + (0.08*XYRange) + (n*(0.08*XYRange)), 
                text_str_2, #These weird numbers were worked out manually
                color='grey', fontsize=PLOT_TEXT_SIZE, fontdict = None) # bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10} #spidermoose )        

        

        # text_str_2 = format(cell_df_scaled_final.iloc[i_step][fac], '.1f')
        # print('CHECK THAT THIS IS THE NORMALIZED VALUES FOR THE ONE ABOVE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print(text_str_2)

        ################################
        #Original line#
        
        # ax.text(text_x + 0.6*XYRange, text_y - 47 + (0.08*XYRange) + (n*(0.08*XYRange)), facwithoutunderscores +': '+ format(cell_df.iloc[i_step][fac], '.1f'), #These weird numbers were worked out manually
        #         color='k', fontsize=30, fontdict = None) #spidermoose )

        ################################


    # for n, fac in enumerate(shape_display_factors):
    #
    #     ax.text(text_x + 0.6*XYRange,text_y -  (n*(0.08*XYRange)), fac +': '+ format(cell_df.iloc[i_step][fac], '.1f'),
    #             color='tab:orange', fontsize=30,size = 36, fontdict = None)

    # General plot improvements
    plottitle="Cluster ID: " + str(cell_df['label'].iloc[i_step]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[i_step] + ")"
    # plottitle="Cluster ID: " + str(Cluster_ID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[i_step] + ")"
    ax.set_title(plottitle, fontname="Arial",fontsize=PLOT_TEXT_SIZE) #TEMPORARILY COMMENTED OUT
    ax.set_xlabel('x (px)', fontname="Arial",fontsize=PLOT_TEXT_SIZE)
    ax.set_ylabel("y (px)", fontname="Arial",fontsize=PLOT_TEXT_SIZE)
    ax.tick_params(axis='both', labelsize=PLOT_TEXT_SIZE)
    ax.set_aspect('equal')
    ax.set_adjustable("datalim")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if boxoff:
        ax.axis('off')

    # plt.autoscale()
    # sns.despine(left=True)

    # ax.set_yticklabels(['eNK','eNK+CytoD'])
    return fig#delete?

##### THE plot_single_cell_factor FUNCTION ######

def plot_single_cell_factor(cell_df, top_dictionary, CLUSTERID, PLOT_PLASTICITY = True, thisistavg = True):
    # labels=wholetrack_exemplar_df['label'].unique()
    # totaltime = totalframes*SAMPLING_INTERVAL
    import matplotlib.cm as cm
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    df=cell_df
    # for the tavg version, work out the cluster ID from the tavg_label column

    if thisistavg == True:
        clusterID=cell_df['tavg_label'].unique()[0]
        plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"

    else:
        clusterID=CLUSTERID
        plottitle="Cluster ID: " + str(clusterID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"

    factors = top_dictionary[clusterID]
    n=len(factors)
    # factors=factors[::-1] #Here, you don't reverse them, so they plot in order. The reversal is because when matplotlib does text, it does the last one first.

    # for the non-tavg version, work out the cluster ID from the label column
    
    
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False, subplot_titles=factors) 
    # add a new column to the cell_df that is the frame * SAMPLING_INTERVAL
    cell_df['time']=cell_df['frame']*SAMPLING_INTERVAL

    for i in range(n):
        factor=factors[i]
        
        df=df.sort_values(by='frame')

        fig.add_trace(go.Scatter(x=df['time'], y=df[factor], mode='lines', name=factor), row=i+1, col=1)

        # change font size of the trace names
        fig.update_layout(font_size=PLOT_TEXT_SIZE)
        fig.update_annotations(font_size=PLOT_TEXT_SIZE)

        # make the range of the x axis the min and max of the time column
        # fig.update_xaxes(range=[0, totaltime], row=i+1, col=1)
        mintime=df['time'].min()
        maxtime=df['time'].max()

        # change the color of the grid lines to light grey  
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', title_text='time (mins)', nticks=10, range = [mintime-1, maxtime+1])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey')
        #change each y axis title to the factor name in the for loop
        fig.update_yaxes(title_text=factor, row=i+1, col=1)

    
     # make sure the x axis has ranges from 0 to totalframes
    # fig.update_xaxes(range=[0, totaltime])
    # fig.update_xaxes(range=[0, totaltime])
    # fig.update_layout(height=n*300, width=900, title_text='', font_size=24, plot_bgcolor='white', xaxis_showgrid=True, yaxis_showgrid=True, title_font_size=24, hovermode="closest")
    fig.update_layout(height=n*450, width=900, font_size=PLOT_TEXT_SIZE, plot_bgcolor='white', xaxis_showgrid=True, yaxis_showgrid=True, title_font_size=PLOT_TEXT_SIZE, hovermode="x unified", colorway=px.colors.qualitative.Dark24)
    # change the overall title to the label and condition
    # plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
    # plottitle="Cluster ID: " + str(clusterID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
    fig.update_layout(title_text=plottitle, title_font_size=PLOT_TEXT_SIZE)
    #remove the legend
    fig.update_layout(showlegend=False)

   
    # fig.update_layout(legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.02,
    #     xanchor="right",
    #     x=3))
    

    # fig.show()

    #####
    if PLOT_PLASTICITY == True:
        plasticity_factors=['label', 'tavg_label','twind_n_changes', 'tavg_twind_n_changes', 'twind_n_labels', 'tavg_twind_n_labels']
        n=len(plasticity_factors)
        # nticks should be only integers
        # layout(yaxis=list(tickformat=',d'))
        fig2 = make_subplots(rows=n, cols=1, shared_xaxes=False,  vertical_spacing = 0.1, horizontal_spacing = 0.2) #subplot_titles=plasticity_factor_titles,
        for i in range(n):
            plasticity_factor=plasticity_factors[i]       
            # print('The plasticity factor is: ' + plasticity_factor) 
            df=df.sort_values(by='frame')
            # if i <= 2:
            #     columntoplot = 1
            #     rowtoplot = i+1
            # else:
            #     columntoplot = 2
            #     rowtoplot = i-2
            fig2.add_trace(go.Scatter(x=df['time'], y=df[plasticity_factor], mode='lines', name=plasticity_factor), row=i+1, col=1)
            # Capitalize the plasticity factor and remove the underscore
            plasticity_factor_title=plasticity_factor.replace('_', ' ').capitalize()
            # print ('The plasticity factor title is: ' + plasticity_factor_title)   
            # change font size of the trace names
            fig2.update_layout(font_size=PLOT_TEXT_SIZE)
            fig2.update_annotations(font_size=PLOT_TEXT_SIZE)
            # make the range of the x axis the min and max of the time column
            mintime=df['time'].min()
            maxtime=df['time'].max()
            # change the color of the grid lines to light grey  
            fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', title_text='time (mins)', nticks=10, range = [mintime-1, maxtime+1])
            fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', nticks = 5, title_text=plasticity_factor_title, row=i+1, col=1)

        fig2.update_layout(
            yaxis = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis2 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis3 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis4 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis5 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis6 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            )
        )
            #change each y axis title to the factor name in the for loop
            # fig2.update_yaxes(title_text=plasticity_factor_title, row=rowtoplot, col=columntoplot)
        fig2.update_layout(height=n*450, width=900, font_size=PLOT_TEXT_SIZE, plot_bgcolor='white', xaxis_showgrid=True,
                            yaxis_showgrid=True,  hovermode="x unified", colorway=px.colors.qualitative.Dark24) #title_font_size=PLOT_TEXT_SIZE,
    # change the overall title to the label and condition
        # plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
        # plottitle="Cluster ID: " + str(clusterID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
        fig2.update_layout(title_text=plottitle, title_font_size=PLOT_TEXT_SIZE)
        # fig2.update_layout(yaxis=list(tickformat ='d')) #yaxis_tickformat =',d', yaxis_tickformat ='d',
        #remove the legend
        fig2.update_layout(showlegend=False)    

        ########### DEV
        plasticity_factors=['label', 'tavg_label','cum_n_changes', 'tavg_cum_n_changes', 'cum_n_labels', 'tavg_cum_n_labels']
        n=len(plasticity_factors)
        fig3 = make_subplots(rows=n, cols=1, shared_xaxes=False,  vertical_spacing = 0.1, horizontal_spacing = 0.2) #subplot_titles=plasticity_factor_titles,
        for i in range(n):
            plasticity_factor=plasticity_factors[i]       
            # print('The plasticity factor is: ' + plasticity_factor) 
            df=df.sort_values(by='frame')
            # if i <= 2:
            #     columntoplot = 1
            #     rowtoplot = i+1
            # else:
            #     columntoplot = 2
            #     rowtoplot = i-2
            fig3.add_trace(go.Scatter(x=df['time'], y=df[plasticity_factor], mode='lines', name=plasticity_factor), row=i+1, col=1)
            # Capitalize the plasticity factor and remove the underscore
            plasticity_factor_title=plasticity_factor.replace('_', ' ').capitalize()
            # print ('The plasticity factor title is: ' + plasticity_factor_title)   
            # change font size of the trace names
            fig3.update_layout(font_size=PLOT_TEXT_SIZE)
            fig3.update_annotations(font_size=PLOT_TEXT_SIZE)
            # make the range of the x axis the min and max of the time column
            mintime=df['time'].min()
            maxtime=df['time'].max()
            # change the color of the grid lines to light grey  
            fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', title_text='time (mins)', nticks=10, range = [mintime-1, maxtime+1])
            fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='lightgrey', nticks = 5, title_text=plasticity_factor_title, row=i+1, col=1)
            #change each y axis title to the factor name in the for loop
            # fig2.update_yaxes(title_text=plasticity_factor_title, row=rowtoplot, col=columntoplot)
        fig3.update_layout(height=n*450, width=900, font_size=PLOT_TEXT_SIZE, plot_bgcolor='white', xaxis_showgrid=True,
                            yaxis_showgrid=True,  hovermode="x unified", colorway=px.colors.qualitative.Dark24) #title_font_size=PLOT_TEXT_SIZE, 
        
        fig3.update_layout(
            yaxis = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis2 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis3 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis4 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis5 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            ),
            yaxis6 = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            )
        )
    # change the overall title to the label and condition
        # plottitle="Cluster ID: " + str(cell_df['tavg_label'].iloc[0]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
        # plottitle="Cluster ID: " + str(clusterID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[0] + ")"
        fig3.update_layout(title_text=plottitle, title_font_size=PLOT_TEXT_SIZE)
        # fig2.update_layout(yaxis_tickformat =',d', yaxis=dict(tickformat ='d'))
        #remove the legend
        fig3.update_layout(showlegend=False)   


        ########### DEV

    ######

    return fig, fig2, fig3


##### THE DISAMBIGAUTE FUNCTION ######

def disambiguate_timepoint(df,  exemps, scaled_df, top_dictionary, XYRange=100,boxoff=True, trajectory=False):

    '''
    This function uses several other functions to make plots of single cells. 
    If trajectory is True, it plots the whole cell trajectory. If trajectory is False, it plots a single timepoint.

    Expansion: make an option to also output a corresponding raw cell image with contour overlaid.

    Inputs:
    The main df, with at least 'label' which is the cluster ID. 
    Exemps is a dataframe with the same columns as the main df, but with only one row per cell because it denotes exemplar cells at given timepoints.
    scaled_df is produced in the 'contribution to clusters' function and is just scaled version of the main df.
    top_dictionary is the dictionary of top factors for each cluster, produced in the 'contribution to clusters' function.
    XYRange is the size of the plot in pixels.

    '''

    top_dictionary = {k: [metric for metric, variance in v] for k, v in top_dictionary.items()} # This actually removes the variances from the dictionary


    for n in exemps.index:

        row=exemps.iloc[n] #extract an example exemplar row

        # Use exemplar row to look up the corresponding cell TRACK from the cell_df
        ### Old way beginning...###
        # cell_df = df[(df['Condition']==row['Condition']) &
        #                 (df['Replicate_ID']==row['Replicate_ID']) &
        #                 (df['particle']==row['particle'])]
        
        ### Old way end...###
        ### new way beginning...###
        # cell_df = df[(df['Condition']==row['Condition']) &
        cell_df = df[(df['uniq_id']==row['uniq_id'])]
        ### new way end...###
        cell_df = cell_df.reset_index(drop=True)

        # Do the same for the scaled_df
        print(row['Condition'], row['Replicate_ID'], row['particle'])

        # scaled_cell_df = scaled_df[(scaled_df['Condition']==row['Condition']) &
        #                 (scaled_df['Replicate_ID']==row['Replicate_ID']) &
        #                 (scaled_df['particle']==row['particle'])]
        scaled_cell_df = scaled_df[(scaled_df['uniq_id']==row['uniq_id'])]
        scaled_cell_df = scaled_cell_df.reset_index(drop=True)

        # This looks up the exemplar point, based on all of these metrics, so that the correct exemplar point is displayed in the visualization 
        exemplarpoint = cell_df.index[(cell_df['uniq_id']==row['uniq_id']) &
                        (cell_df['frame']==row['frame']) &
                        (cell_df['label']==row['label'])]
        CLUSTERID = row['label']

        ##
        cluster_colors = []
        labels = list(set(df['label'].unique()))
        numofcolors = len(labels)
        cmap = cm.get_cmap(CLUSTER_CMAP)
        for i in range(numofcolors):
            cluster_colors.append(cmap(i))
            
        ###
        if trajectory:
            do_trajectory = 'DoingTrajectories'
        else:
            do_trajectory = 'NotDoingTrajectories'

        f=plot_cell_metrics_timepoint(cell_df, exemplarpoint[0], scaled_cell_df, XYRange, boxoff, top_dictionary, CLUSTERID, cluster_colors, do_trajectory) #cilantro #mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS scaled_cell_df,

        ######
        f2,f3,f4 = plot_single_cell_factor(cell_df, top_dictionary, CLUSTERID, PLOT_PLASTICITY = True, thisistavg=False)

        cell_ID = str(cell_df['uniq_id'].iloc[0])


        # if trajectory id is in the cell_df:
        if trajectory:
            f.savefig( TRAJECTORY_DISAMBIG_DIR + '\Contour_Cell' + cell_ID + '_TRAJECTORY_ID_' + str(cell_df['trajectory_id'].iloc[0]) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', dpi=300,bbox_inches="tight")

            f2.write_image( TRAJECTORY_DISAMBIG_DIR + '\Metrics_Cell' + cell_ID + '_TRAJECTORY_ID_' + str(cell_df['trajectory_id'].iloc[0]) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
            '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 

            f3.write_image( TRAJECTORY_DISAMBIG_DIR + '\Plasticity_Cell' + cell_ID + '_TRAJECTORY_ID_' + str(cell_df['trajectory_id'].iloc[0]) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
                '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 
            
            f4.write_image( TRAJECTORY_DISAMBIG_DIR + '\Plasticity_Cell' + cell_ID + '_TRAJECTORY_ID_' + str(cell_df['trajectory_id'].iloc[0]) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
                '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1)       
            
            print("Saving to" + TRAJECTORY_DISAMBIG_DIR + '\Disambiguated_TrajectoryID_'+str(cell_df['trajectory_id'].iloc[0])+'_'+str(n))
        else: 
          
            f.savefig( CLUST_DISAMBIG_DIR + '\Contour_Cell' + cell_ID + '_CLUSID_' + str(CLUSTERID) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
                '__R'+str(XYRange)+'_'+str(n)+'.png', dpi=300,bbox_inches="tight")          

            f2.write_image( CLUST_DISAMBIG_DIR + '\Metrics_Cell' + cell_ID + '_CLUSID_' + str(CLUSTERID) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
                '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 

            f3.write_image( CLUST_DISAMBIG_DIR + '\Plasticity_Cell' + cell_ID + 'TWindow_CLUSID_' + str(CLUSTERID) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
                '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 
            
            f4.write_image( CLUST_DISAMBIG_DIR + '\Plasticity_Cell' + cell_ID + 'Cumulative_CLUSID_' + str(CLUSTERID) + '__Cond_' + str(cell_df['Condition_shortlabel'].iloc[0]) + 
                '__R'+str(XYRange)+'_'+str(n)+'.png', scale = 1) 

        ######
            print("Saving to" + CLUST_DISAMBIG_DIR + '\Disambiguated_ClusterID_'+str(CLUSTERID)+'_'+str(n))
        # f.savefig( CLUST_DISAMBIG_DIR + '\ClusterID__'+str(ClusterID)+'__disambiguated__R'+str(XYRange)+'_'+str(n)  ,dpi=300,bbox_inches="tight")

    return   

########## 10-6-2023 ##########

def plot_cell_metrics_timepoint(cell_df, i_step, scaled_cell_df, XYRange,boxoff, top_dictionary, Cluster_ID, cluster_colors, do_trajectory, mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS, dr_factors=DR_FACTORS): 
    '''
    For a selected cell, at a selected step of its trajectory, make a plot
    '''
    if do_trajectory == 'DoingTrajectories':
        print('This is running the trajectory version of the function')
    else:
        print('Not doing trajectories')

    # Using the same sample cell trajectory, print out some measurements
    fig, (ax) = plt.subplots(1,1, figsize=(0.08*XYRange,0.08*XYRange))
    # print(top_dictionary)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Arial'] + plt.rcParams['font.serif'] #Times New Roman
    plt.rcParams['mathtext.default'] = 'regular'

    ax.title.set_text('Cell contour: xy space')
    if do_trajectory == 'DoingTrajectories':
        ax.plot(cell_df['x_pix'],cell_df['y_pix'],'-o',markersize=3,c='black')
    else:
        print('Not doing trajectories so not drawing all contours')

    #######################
    # cluster_colors = []
    # labels = list(set(lab_dr_df['label'].unique()))
    # cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    # for i in range(cmap.N):
    #     cluster_colors.append(cmap(i))

    #######################

    ################
    xminimum=cell_df['x_pix'].min()
    xmaximum=cell_df['x_pix'].max()
    xmid = np.median([xmaximum, xminimum])
    xmin=xmid-XYRange/2
    xmax=xmid+XYRange/2

    yminimum=cell_df['y_pix'].min()
    ymaximum=cell_df['y_pix'].max()
    ymid = np.median([ymaximum, yminimum])
    ymin=ymid-XYRange/2
    ymax=ymid+XYRange/2

    #######################

    contour_list = get_cell_contours(cell_df)
    # Draw all contours faintly as image BG
    if do_trajectory == 'DoingTrajectories':
        for i,contour in enumerate(contour_list):

            rgb = np.random.rand(3,)
            this_colour = rgb#'red' # Eventually calculate color along colormap
            contour_arr = np.asarray(contour).T


            if not np.isnan(np.sum(contour_arr)): # If the contour is not nan

                x = cell_df['x_pix'].values[i]# - window / 2
                y = cell_df['y_pix'].values[i]# - window / 2
                # Cell contour relative to x,y positions
                '''Want to double check that the x,y positions not mirrored from the contour function'''
                ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c='gray', alpha=0.2)
    else:
        print('Not doing trajectories so not drawing all contours')
        

    # Draw this contour, highlighted.
    contour = contour_list[i_step]
    contour_arr = np.asarray(contour).T
    x = cell_df['x_pix'].values[i_step]# - window / 2
    y = cell_df['y_pix'].values[i_step]# - window / 2
    

    # Cell contour relative to x,y positions
    '''Want to double check that the x,y positions not mirrored from the contour function'''
    if not np.isnan(np.sum(contour_arr)):
        ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c=cluster_colors[Cluster_ID],linewidth=5, alpha=1) #cilantro

        # Current segment
        if i_step > 0:
            x_seg = cell_df['x_pix'].values[i_step-1:i_step+1]# - window / 2
            y_seg = cell_df['y_pix'].values[i_step-1:i_step+1]# - window / 2
            # ax.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c='red', alpha=1)
            # ax.plot(x_seg*2,y_seg*2,'-o',markersize=10,c='tab:blue', linewidth=4)
            ax.plot(x_seg,y_seg,'-o',markersize=10,c='tab:blue', linewidth=4)

    text_x = xmid
    text_y = ymid

    display_factors = top_dictionary[Cluster_ID]
    # reverse the display factors so they display properly on the plot
    display_factors = display_factors[::-1]

    for n, fac in enumerate(display_factors): #Positioning is currently relative to data. Can it be relative to plot?
        facwithoutunderscores = fac.replace('_',' ')
        text_str = facwithoutunderscores +': '+ format(cell_df.iloc[i_step][fac], '.1f')
        text_str_2 = format(scaled_cell_df.iloc[i_step][fac], '.1f')

        ############# Adds a category for low, med, high ############
        # Q1 = scaled_df[column_name].quantile(0.33)
        # Q2 = scaled_df[column_name].quantile(0.5)
        # Q3 = scaled_df[column_name].quantile(0.66)

        # # q1 = quartiles.xs(column_name, level=1).xs(cluster_id, level=0)[0.25]
        # # q3 = quartiles.xs(column_name, level=1).xs(cluster_id, level=0)[0.75]
        # if scaled_column_value < Q1:
        #     c = 'L'
        # elif scaled_column_value <= Q3:
        #     c = 'M'
        # elif scaled_column_value > Q3:
        #     c = 'H'
        # else:
        #     c = 'O'

        # text_str_2 = format(scaled_cell_df.iloc[i_step][fac], '.1f') + ' (' + c + ')'

        ################

        ax.text(text_x + 0.5*XYRange, text_y - (0.3*XYRange) + (0.08*XYRange) + (n*(0.08*XYRange)), 
                text_str, #These weird numbers were worked out manually
                color='k', fontsize=PLOT_TEXT_SIZE, fontdict = None) #spidermoose )
        ax.text(text_x + 0.45*XYRange, text_y - (0.3*XYRange) + (0.08*XYRange) + (n*(0.08*XYRange)), 
                text_str_2, #These weird numbers were worked out manually
                color='grey', fontsize=PLOT_TEXT_SIZE, fontdict = None) # bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10} #spidermoose )        


    # General plot improvements
    if 'trajectory_id' in cell_df.columns:
        plottitle = f'TRAJECTORY ID is {cell_df["trajectory_id"].iloc[0]}'
    else:
        plottitle="Cluster ID: " + str(cell_df['label'].iloc[i_step]) + " (condition: " + cell_df['Condition_shortlabel'].iloc[i_step] + ")"
    # plottitle="Cluster ID: " + str(Cluster_ID) + " (condition: " + cell_df['Condition_shortlabel'].iloc[i_step] + ")"
    ax.set_title(plottitle, fontname="Arial",fontsize=PLOT_TEXT_SIZE) #TEMPORARILY COMMENTED OUT
    ax.set_xlabel('x (px)', fontname="Arial",fontsize=PLOT_TEXT_SIZE)
    ax.set_ylabel("y (px)", fontname="Arial",fontsize=PLOT_TEXT_SIZE)
    ax.tick_params(axis='both', labelsize=PLOT_TEXT_SIZE)
    ax.set_aspect('equal')
    ax.set_adjustable("datalim")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if boxoff:
        ax.axis('off')

    # plt.autoscale()
    # sns.despine(left=True)

    # ax.set_yticklabels(['eNK','eNK+CytoD'])
    return fig

### Function to take ANY exemplar df or df of chosen cells (from single timepoints), and find the full tracks from an input dataframe, outputting those as a dataframe

# Those cells are then used for plotting.

def cell_tracks_from_chosen_cells(df_in, chosen_cells_df):

    particles = chosen_cells_df['uniq_id'].unique()
    # print(particles)
    # Use this list to filter the tptlabel_dr_df so it only contains rows in which particle = one of the elements in particles
    exemplar_cell_tracks_df = df_in[df_in['uniq_id'].isin(particles)]
    print("The number of datapoints in the original dataframe was " + str(len(df_in)))
    print("The number of cells in the original dataframe was " + str(len(df_in['uniq_id'].unique())))
    print("The number of datapoints in the new dataframe is " + str(len(exemplar_cell_tracks_df)))
    print("The number of cells in the new dataframe is " + str(len(exemplar_cell_tracks_df['uniq_id'].unique())))

    return exemplar_cell_tracks_df


# This version uses the uniq_id instead of the particle number
# This is used for the small multiples and for the disambiguations, and it chooses exemplars with a good number of timepoints so as to sample single cells well

def filter_exemplars(whole_df, exemplar_df, numberofdesiredtimepoints = 200, numberofcellspercluster = 40, override = 100):

    print('Filtering exemplars so that only those with > ' + str(numberofdesiredtimepoints) + ' timepoints are included')

    print('Aiming for ' + str(numberofcellspercluster) + ' cells per cluster')

    num_clusters_whole_dataset = len(whole_df['label'].unique())

    print('Which means ' + str(num_clusters_whole_dataset*numberofcellspercluster) + ' cells in total')

    print('But we can accept 70 percent! Which means we are only aiming for ' + str((num_clusters_whole_dataset*numberofcellspercluster)*0.7) + ' cells in total')

    print('Number of unique cells in exemplar_df: ', len(exemplar_df['uniq_id'].unique()))    

    exemplar_df_filt = exemplar_df[exemplar_df['ntpts'] > numberofdesiredtimepoints] # filter out cells that don't have enough timepoints

    print('Number of unique cells in exemplar_df after filtering: ', len(exemplar_df_filt['uniq_id'].unique()))
    # Draw a histogram of which frames are most common in the exemplar_df
    exemplar_df['frame'].hist(bins=100)
    # do the same for the exemplar_df_filt
    exemplar_df_filt['frame'].hist(bins=100)
    # Save the exemplar_df_filt
    
    num_clusters_exemplars = len(exemplar_df_filt['label'].unique())
    print('The number of clusters in whole dataset is: ' + str(num_clusters_whole_dataset) + ' whereas represented in the filtered and ready exemplar = ' + str(num_clusters_exemplars))

    exemplar_df_filt_selected = exemplar_df_filt.groupby('label').apply(lambda x: x.sample(min(numberofcellspercluster,len(x)))).reset_index(drop=True)

    numberofcellsindf = len(exemplar_df_filt_selected['uniq_id'].unique())

    numberofuniquecells_intended = numberofcellspercluster*num_clusters_whole_dataset

    print('Intended number of unique cells in the final df = ' + str(numberofuniquecells_intended) + ' , and the actual number is ' + str(numberofcellsindf))


    # while numberofuniquecells_intended != numberofcellsindf:
    while override >= numberofcellsindf:

        print('Intended number of unique cells in the final df = ' + str(numberofuniquecells_intended) + ' , and the actual number is ' + str(numberofcellsindf))
        
        exemplar_df_filt_selected = exemplar_df_filt.groupby('label').apply(lambda x: x.sample(min(numberofcellspercluster,len(x)))).reset_index(drop=True)
        numberofcellsindf = len(exemplar_df_filt_selected['uniq_id'].unique())
        numberofuniquecells_intended = numberofcellspercluster*num_clusters_whole_dataset
    else:
        print('Intended number of unique cells in the final df = ' + str(numberofuniquecells_intended) + ' , and the actual number is ' + str(numberofcellsindf))

        exemplar_df_filt = exemplar_df_filt_selected

    exemplar_cell_tracks_df = cell_tracks_from_chosen_cells(df_in=whole_df, chosen_cells_df=exemplar_df_filt)

    exemplar_df_filt.to_csv(SAVED_DATA_PATH + 'exemplar_df_filt.csv', index=False)
    exemplar_cell_tracks_df.to_csv(SAVED_DATA_PATH + 'exemplar_cell_tracks_df.csv', index=False)

    return exemplar_df_filt, exemplar_cell_tracks_df

################################################

def plot_trajectories(df, global_y=True, global_x=True):

    
    # Calculate the 'timeminutes' variable
    df['timeminutes'] = df['frame'] * SAMPLING_INTERVAL
    
    # Sort the DataFrame by 'Condition_shortlabel' and 'frame'
    df_sorted = df.sort_values(by=['Condition_shortlabel', 'frame'])

    # Group data by 'trajectory_id'
    grouped = df_sorted.groupby('trajectory_id')


    if global_y:

        # Find the global minimum and maximum values for the 'y' axis
        y_min = df['label'].min()
        y_max = df['label'].max()
    else:
        y_min = y_max = None

    if global_x:
        # Find the global minimum and maximum values for the 'x' axis
        global_x_min = df['timeminutes'].min()
        global_x_max = df['timeminutes'].max()
    else:
        print('global_x is False')
    # Determine unique conditions and corresponding colors
    conditions = df_sorted['Condition_shortlabel'].unique()
    color_map = plt.cm.get_cmap(CONDITION_CMAP, len(conditions))
    # Set the font size for all subplots on all axes in all figures to be larger
    plt.rcParams.update({'font.size': FONT_SIZE})  

    for trajectory_id, group in grouped:
        num_rows = len(group['uniq_id'].unique())
        figsize = (6, num_rows * 2)  # Adjust the figure size as needed

        fig, ax = plt.subplots(num_rows, 1, figsize=figsize, sharex=False)
        fig.suptitle(f'Trajectory ID: {trajectory_id}', y=0.95, fontsize=PLOT_TEXT_SIZE)

        for i, uniq_id in enumerate(group['uniq_id'].unique()):
            sub_df = group[group['uniq_id'] == uniq_id]
            condition = sub_df['Condition_shortlabel'].iloc[0]
            color = color_map(conditions.tolist().index(condition))

            ax[i].plot(sub_df['timeminutes'], sub_df['label'], label=f'{condition} ID: {uniq_id}', color=color)
            ax[i].set_ylabel('Label',fontsize=FONT_SIZE)
            if global_y:
                ax[i].set_yticks(np.arange(y_min, y_max, 1))
                ax[i].set_ylim(y_min-0.5, y_max+0.5)
            else:
                print('')
                # Display every y tick value
                y_min = []
                y_max = []
                y_min, y_max = ax[i].get_ylim()
                ax[i].set_yticks(np.arange(y_min, y_max, 1))
                ax[i].set_ylim(y_min-0.5, y_max+0.5)
            if global_x:
                ax[i].set_xlim(global_x_min-0.5, global_x_max+0.5)
            else:
                ax[i].set_xlim(sub_df['timeminutes'].min()-0.5, sub_df['timeminutes'].max()+0.5)
                ax[i].set_xlabel('Time (min)',fontsize=FONT_SIZE)

            ax[i].legend()

        # Adjust layout and spacing
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Show or save the figure (replace 'savefig' with 'plt.show()' to display)
        plt.savefig(TRAJECTORY_DISAMBIG_DIR + f'trajectory_{trajectory_id}_plots.png')
        # fig.write_image(TRAJECTORY_DISAMBIG_DIR + f'trajectory_{trajectory_id}_plots.png')

    plt.show()  # If you want to display the plots


    ######


def contribution_to_clusters(df_in, threshold_value=0.0001, dr_factors=DR_FACTORS, howmanyfactors=6, scalingmethod = SCALING_METHOD): #New function 21-14-2022

    ### 10-6-2023 ##### 

    '''
    Steps:
    1. Take the metrics and center scales them, then puts them back into a DF
    2. Find the median value per cluster for each metric using groupby
    3. Makes some iterables for the parts below.
    4. Makes a Variance DF that describes the variance of each metric BETWEEN CLUSTERS
    5. Makes a boolean mask of variances based on that threshold value, and a dataframe that contains values if true, and NaN if not
    6. exports a df that can be used to select what metrics you want to show?

    '''

    # from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PowerTransformer

    # df_in = tptlabel_dr_df
    # threshold_value = 0.0001
    CLUSTERON = dr_factors

    # Part 1: take the metrics and center scales them, then puts them back into a DF
    sub_set = df_in[CLUSTERON]
    Z = sub_set.values
    if scalingmethod == 'minmax': #log2minmax minmax powertransformer
        X = MinMaxScaler().fit_transform(Z)
        correctcolumns = CLUSTERON
    elif scalingmethod == 'log2minmax':
        negative_FACTORS = []
        positive_FACTORS = []
        for factor in dr_factors:
            if np.min(df_in[factor]) < 0:
                print('factor ' + factor + ' has negative values')
                negative_FACTORS.append(factor)
                
            else:
                print('factor ' + factor + ' has no negative values')
                positive_FACTORS.append(factor)
        
        pos_df = df_in[positive_FACTORS]
        pos_x = pos_df.values
        neg_df = df_in[negative_FACTORS]
        neg_x = neg_df.values
        neg_x_ = MinMaxScaler().fit_transform(neg_x)
        pos_x_constant = pos_x + 0.000001
        pos_x_log = np.log2(pos_x + pos_x_constant)
        pos_x_ = MinMaxScaler().fit_transform(pos_x_log)
        X = np.concatenate((pos_x_, neg_x_), axis=1)
        correctcolumns=positive_FACTORS + negative_FACTORS

    elif scalingmethod == 'powertransformer':    
        
        pt = PowerTransformer(method='yeo-johnson')
        X = pt.fit_transform(Z)
        correctcolumns=CLUSTERON

    elif scalingmethod == 'choice': 
        print('Factors to be scaled using log2 and then minmax:')
        FactorsNOTtotransform = ['arrest_coefficient', 'rip_L', 'rip_p', 'rip_K', 'eccentricity', 'orientation', 'directedness', 'turn_angle', 'dir_autocorr', 'glob_turn_deg']
        FactorsNottotransform_actual=[]
        FactorsToTransform_actual=[]
        for factor in dr_factors:
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
        
        trans_x_log = np.log2(trans_x_constant) # This is what it should be.
        trans_x_=MinMaxScaler().fit_transform(trans_x_log)
        nontrans_x_=MinMaxScaler().fit_transform(nontrans_x)
        x_=np.concatenate((trans_x_, nontrans_x_), axis=1)
        newcols=FactorsToTransform_actual + FactorsNottotransform_actual
        scaled_df_here = pd.DataFrame(x_, columns = newcols)
        scaled_df_here.hist(column=newcols, bins = 160, figsize=(20, 10),color = "black", ec="black")
        plt.tight_layout()
        plt.title('Transformed data')
        X=x_
        correctcolumns=newcols

    elif scalingmethod == None:
        X = Z
        correctcolumns=CLUSTERON

    thelabels = df_in['label']
    scaled_df_in = pd.DataFrame(data=X, columns = correctcolumns)
    df_out = pd.concat([scaled_df_in,thelabels], axis=1)#Put this back into a DF, with the col names and labels.

    ####### Here starts the new bit for the scaled numbers on the disambiguates #######

    # Isolate the columsn of the cell_df that are not the newcols as a list
    cols_to_keep = [col for col in df_in.columns if col not in correctcolumns]
    # Extract a sub df from the cell_df that contains only the columns in cols_to_keep
    scaled_df = pd.concat([df_in[cols_to_keep], scaled_df_in], axis=1)

    # Part 2: Find the median value per cluster for each metric using groupby
    clusteraverage_df = df_out.groupby('label').median()#.reset_index(drop=True)
    ###

    # Part 3: Makes some iterables for the parts below.
    numberofclusters=len(clusteraverage_df.axes[0])
    iterable_clusters=np.arange(len(clusteraverage_df.axes[0])) # Iterable across rows
    iterable_metrics=np.arange(len(clusteraverage_df.axes[1])) # Iterable across columns (metrics)
    metric_means = clusteraverage_df.mean(axis=0).tolist()

    # Part 4: Makes a Variance DF that describes the variance of each metric BETWEEN CLUSTERS
    variance_between_clusters=[]
    variance_between_clusters_array = []
    variance_between_clusters_array_total=[]

    ###############################################################################
    # Calculates the variance of each value compared to the other values (discounting current value in calculation of mean) ((x - mean)^2)/numberofvalues
    # for metric in iterable_metrics:
    #     # print(metric)
    #     variance_between_clusters_array=[]
    #     for cluster in iterable_clusters:
    #         list1 = clusteraverage_df.iloc[cluster,metric]
    #         list2=clusteraverage_df.iloc[:,metric].tolist()
    #         list2.remove(list1)
    #         meanminusthatvalue=np.mean(list2)
    #         variance_between_clusters = ((clusteraverage_df.iloc[cluster,metric] - meanminusthatvalue)**2)/(numberofclusters)
    #         variance_between_clusters_array.append(variance_between_clusters)
    #     variance_between_clusters_array_total.append(variance_between_clusters_array)
    # variance_df = pd.DataFrame(data = variance_between_clusters_array_total, columns = clusteraverage_df.index, index = CLUSTERON )


    ###############################################################################
    # CONSIDERS ALL THE VALUES FOR CALCULATING THE MEAN

    for metric in iterable_metrics:
        # print(metric)
        variance_between_clusters_array=[]
        for cluster in iterable_clusters:
            variance_between_clusters = ((clusteraverage_df.iloc[cluster,metric] - metric_means[metric])**2)/(numberofclusters)
            variance_between_clusters_array.append(variance_between_clusters)
        variance_between_clusters_array_total.append(variance_between_clusters_array)
    variance_df = pd.DataFrame(data = variance_between_clusters_array_total, columns = clusteraverage_df.index, index = CLUSTERON )
    #############################################################################

    # This part then makes the top factors into a dictionary
    trans_variance_df=variance_df.T
    df= trans_variance_df

    topfactors = []
    contributors=[]

    for ind in df.index:
        col=trans_variance_df.loc[ind,:]
        sortedcol=col.sort_values(ascending=False)
        # Get the top factors and their variances
        topfactor_variances = sortedcol[0:howmanyfactors]
        # Convert the series to a list of tuples where each tuple is (factor, variance)
        contributors = [(factor, variance) for factor, variance in topfactor_variances.items()]
        topfactors.append(contributors)

    #Make a dictionary of these results
    top_dictionary = {}
    keys = df.index.tolist()
    for i in keys:
        top_dictionary[i] = topfactors[i]

    # Part 5: Makes a boolean mask of variances based on that threshold value, and a dataframe that contains values if true, and NaN if not
    high_mask = variance_df > threshold_value
    trueones=variance_df[high_mask]

    # Part 6: Prints the names of the important values per cluster. 
    for clusterlabel in iterable_clusters:
        clusteryeah=trueones.iloc[:,clusterlabel]
        clusterboolean=clusteryeah.notnull()
        highmetrics=clusteryeah[clusterboolean]
        clusternames=trueones.columns.tolist()

    # Part 8: exports a df that can be used to select what metrics you want to show?
    return top_dictionary, clusteraverage_df, scaled_df



def plot_cluster_averages_dev_deprecated(top_dictionary, df, scaled_df): # New 3-7-2023
    num_plots = len(top_dictionary)
    # print(top_dictionary)
    # Reverse the order of the values in the dictionary so that the biggest contributor to variability is on top
    # top_dictionary = {k: v[::-1] for k, v in top_dictionary.items()}
    # print(top_dictionary)
    ### Make a totally non-normalized version of the clusteraverage_df
    cluster_average_df = df.groupby('label').median()#.reset_index(drop=True)
    cluster_average_scaled_df = scaled_df.groupby('label').median()#.reset_index(drop=True)

    # Create a grid of subplots with one row and num_plots columns
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(15*num_plots,10))

    ##################
    for i, (cluster_id, value) in enumerate(top_dictionary.items()):
        # Rest of your code...
        # Get the row in the dataframe that corresponds to the current cluster ID
        cluster_row = cluster_average_df.loc[cluster_id]
        scaled_cluster_row=cluster_average_scaled_df.loc[cluster_id]
        # Loop over the column names for the current key and create a text string
        text_str = ""
        scaled_text_str = ""

        for column_name, variance in value:
            column_value = round(cluster_row[column_name], 4)
            scaled_column_value = round(scaled_cluster_row[column_name], 1)

            # Include the variance in the text string
            text_str += f"{column_name.title()}: {column_value} (s={scaled_column_value})\n" #v={variance}
            text_str = text_str.replace('_',' ')

    ######################

    # old:
    
    # Loop over the cluster IDs and corresponding values in the dictionary
    # for i, (cluster_id, value) in enumerate(top_dictionary.items()):
    #     # Get the row in the dataframe that corresponds to the current cluster ID
    #     cluster_row = cluster_average_df.loc[cluster_id]
    #     scaled_cluster_row=cluster_average_scaled_df.loc[cluster_id]
        
    #     # Loop over the column names for the current key and create a text string
    #     text_str = ""
    #     scaled_text_str = ""
    #     for column_name in value:
    #         # Get the value in the specified column for the current cluster
    #         column_value = round(cluster_row[column_name], 4)
    #         scaled_column_value = round(scaled_cluster_row[column_name], 1)
            
    #         # Add the string and the corresponding value to the text string
    #         text_str += f"{column_name.title()}: {column_value} (s={scaled_column_value})\n"
    #         text_str = text_str.replace('_',' ')
    ########################

            # scaled_text_str += f"{column_name.title()}: {scaled_column_value}\n"
            # scaled_text_str = f"{scaled_column_value}\n"
            # scaled_text_str = scaled_text_str.replace('_',' ')
        
        # Plot the text string as text in the current subplot
        axs[i].text(0.5, 0.5, text_str.strip(), ha='center', va='center', fontsize=30)
        # axs[i].text(0.8, 0.5, scaled_text_str.strip(), ha='center', va='center', fontsize=30)
        
        # Set the title of the current subplot to the current cluster ID
        axs[i].set_title(f"Cluster {cluster_id}", fontsize = 30)

        axs[i].set_xticks([])
        axs[i].set_yticks([])
        # Remove the lines around the subplot
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
    
    
    # Add an overall title to the figure
    fig.suptitle("Average of top contributors per cluster ID", fontsize=36)
    # Save figure as a png
    # plt.savefig("cluster_average.png")
    fig.savefig( CLUST_DISAMBIG_DIR + '\cluster_average.png',dpi=300,bbox_inches="tight") 
 
    
    plt.show()

def plot_cluster_averages(top_dictionary, df, scaled_df): 

    '''
    This calculated the quartiles, splitting scaled data into 3 parts around the median, and then labels each metric as 'low', 'medium', or 'high'.
    It also spits out median values for each metric, and the scaled median values for each metric.
    '''
    num_plots = len(top_dictionary)

    cluster_average_df = df.groupby('label').median()
    cluster_average_scaled_df = scaled_df.groupby('label').median()

    # Calculate the quartiles for each metric
    quartiles = df.groupby('label').quantile([0.25, 0.5, 0.75])

    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(15*num_plots,10))
    
    for i, (cluster_id, value) in enumerate(top_dictionary.items()):
        cluster_row = cluster_average_df.loc[cluster_id]
        scaled_cluster_row=cluster_average_scaled_df.loc[cluster_id]
        
        text_str = ""
        for column_name, variance in value:  # We still get the variance but don't use it
            column_value = round(cluster_row[column_name], 4)
            scaled_column_value = round(scaled_cluster_row[column_name], 1)
            
            # Calculate the quartiles for the current metric
            Q1 = scaled_df[column_name].quantile(0.33)
            Q2 = scaled_df[column_name].quantile(0.5)
            Q3 = scaled_df[column_name].quantile(0.66)

            if scaled_column_value < Q1:
                c = 'L'
            elif scaled_column_value <= Q3:
                c = 'M'
            elif scaled_column_value > Q3:
                c = 'H'
            else:
                c = 'O'

            text_str += f"{column_name.title()}: {column_value} (s={scaled_column_value}, c={c})\n"
            text_str = text_str.replace('_',' ')

        axs[i].text(0.5, 0.5, text_str.strip(), ha='center', va='center', fontsize=30)
        axs[i].set_title(f"Cluster {cluster_id}", fontsize = 30)

        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)

    fig.suptitle("Average of top contributors per cluster ID", fontsize=36)
    fig.savefig( CLUST_DISAMBIG_DIR + '\cluster_average.png',dpi=300,bbox_inches="tight") 

    plt.show()

def create_cluster_averages_table(top_dictionary, df, scaled_df): 

    '''
    Makes a nice table showing the median values for each metric 
    '''

    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.table import Table
    from matplotlib.font_manager import FontManager, FontProperties
    # Calculate the quartiles for each metric
    quartiles = df.groupby('label').quantile([0.25, 0.5, 0.75])

    cluster_average_df = df.groupby('label').median()
    cluster_average_scaled_df = scaled_df.groupby('label').median()

    # Initialize an empty DataFrame to store the results
    # result_df = pd.DataFrame(columns=['ClusterID', 'Metric', 'Median', 'Scaled', 'Category'])
    result_df = pd.DataFrame(columns=['ClusterID', 'Metric', 'Median', 'Category'])

    # Loop over each cluster and metric
    for cluster_id, value in top_dictionary.items():
        cluster_row = cluster_average_df.loc[cluster_id]
        scaled_cluster_row=cluster_average_scaled_df.loc[cluster_id]

        for column_name, variance in value:
            column_value = round(cluster_row[column_name], 4)
            scaled_column_value = round(scaled_cluster_row[column_name], 1)

            # Calculate the quartiles for the current metric
            Q1 = scaled_df[column_name].quantile(0.33)
            Q3 = scaled_df[column_name].quantile(0.66)

            # Classify the median value as 'low', 'medium', or 'high'
            if scaled_column_value < Q1:
                category = 'Low'
            elif scaled_column_value <= Q3:
                category = 'Medium'
            else:
                category = 'High'

            # Remove underscores from the column name
            column_name = column_name.replace('_', ' ')
            # Capitalize the first letter of each word
            column_name = column_name.title()
            # If the column name is MSD, capitazlie all letters
            if column_name == 'Msd':
                column_name = column_name.upper()

            # Append the result to the DataFrame
            # result_df = result_df.append({'ClusterID': cluster_id, 'Metric': column_name, 
            #                               'Median': column_value, 'Scaled': scaled_column_value, 
            #                               'Category': category}, ignore_index=True)
            result_df = result_df.append({'ClusterID': cluster_id, 'Metric': column_name, 
                                          'Median': column_value, #'Scaled': scaled_column_value, 
                                          'Category': category}, ignore_index=True)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')

    font = FontProperties(family='DejaVu Sans', size=12)
    bold_font = FontProperties(family='DejaVu Sans', size=12, weight='bold')


    # Create a table
    table = ax.table(cellText=result_df.values, colLabels=result_df.columns, loc='center', cellLoc='center') #fontproperties=font

    table.auto_set_column_width(col=list(range(result_df.shape[1])))

    # # Set the font properties of each cell THAT WORKS
    for cell in table._cells.values():
        cell.set_text_props(fontproperties=font)
        cell.set_height(0.07)

    # These are the dictionary keys for the column heads. They are OK hardcoded because they will always be the same
    colhead_dict_keys = [(0, 0), (0, 1), (0, 2), (0, 3)] 
    # Makes the column heads bold
    for cell in colhead_dict_keys:
        ok = table._cells[cell]
        print(ok)
        ok.set_text_props(fontproperties=bold_font)

    # Add the table to the axes
    ax.add_table(table)

    # Save the figure as a PNG image
    plt.savefig(CLUST_DISAMBIG_DIR + 'clusterIDtable.png', dpi=300, bbox_inches="tight")

    return result_df


def fingerprintplot_clusters_per_trajectory(df):
    # Calculate frequencies
    counts = df.groupby(['trajectory_id', 'label']).size()
    fontsize = PLOT_TEXT_SIZE
    plt.rc('font', size=fontsize) 
    plt.rc('axes', titlesize=fontsize) 
    plt.rc('axes', labelsize=fontsize) 
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize) 
    plt.rc('legend', fontsize=fontsize) 
    plt.rc('figure', titlesize=fontsize)  

    # Convert frequencies to percentages
    percentages = counts.groupby(level=0).apply(lambda x: 100 * x / x.sum())

    cluster_colors = []
    labels = list(set(df['label'].unique()))
    numofcolors = len(labels)
    cmap = cm.get_cmap(CLUSTER_CMAP)
    for i in range(numofcolors):
        cluster_colors.append(cmap(i))

    # Reset the index of the DataFrame and pivot it for the stacked bar plot
    percentages = percentages.reset_index().pivot(columns='label', index='trajectory_id', values=0)

    # Create a dictionary mapping labels to colors
    color_dict = dict(zip(labels, cluster_colors))

    # Create a list of colors for each label in the DataFrame
    colors = percentages.columns.map(color_dict)

    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(20, 20))
    percentages.plot(kind='bar', stacked=True, ax=ax, color=colors)
    # Get the number of unique trajectory IDs
    num_traj = len(df['trajectory_id'].unique())
    # print(f'num_traj = {num_traj}')

    # print(f'The length of the ax patches is {len(ax.patches)}' )

    # Make a list to alternate between 0 and 0.5
    positionmodifier = [0.5 if i % 8 >= 4 else 0 for i in range(len(ax.patches))]

    # print(positionmodifier)
    # print(len(positionmodifier))

    # Label the percentages with a shift according to positionmodifier
    for p, pos in zip(ax.patches, positionmodifier):
        # print(f'The p from the ax.patches is {p}')
        # print(f'The pos from the positionmodifier is {pos}')
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.0f}%', (x + width/2 + pos, y + height/2), ha='center', va='center', fontsize=fontsize)

    # Move the legend to the right-hand side and give it a title
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Cluster ID')
    # Label the y-axis percentage
    ax.set_ylabel('Percentage cluster ID per trajectory ID')
    # rotate x ticks 90 
    plt.xticks(rotation=90)
    # Set the x-axis label
    ax.set_xlabel('Trajectory ID')
    # Save the plot in the trajectory disambig dir
    plt.savefig(TRAJECTORY_DISAMBIG_DIR + 'fingerprint_clusters_per_trajectory.png', dpi=300, bbox_inches="tight")

    plt.show()

    return

def plasticity_per_trajectory(df):

    # Set the style and color palette
    sns.set_style("whitegrid")
    max_values = df.groupby(['trajectory_id', 'uniq_id'])['cum_n_changes'].max().reset_index()
    display(max_values)
    median_values = max_values.groupby(['trajectory_id'])['cum_n_changes'].median().reset_index()

    colors = []
    cmap = cm.get_cmap(CLUSTER_CMAP)
    numcolors=len(df['trajectory_id'].unique())
    for i in range(numcolors):
        colors.append(cmap(i))    

    plt.figure(figsize=(15,10))  # Adjust the size of the plot

    barplot = sns.barplot(x='trajectory_id', y='cum_n_changes', data=median_values, color='grey', alpha = 0.5)
    sns.violinplot(x='trajectory_id', y='cum_n_changes', data=max_values, palette=colors)

    # Add the median values to the plot
    for i, bar in enumerate(barplot.patches):
        barplot.annotate(format(median_values['cum_n_changes'].values[i], '.2f'), 
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, -10),  
                        textcoords = 'offset points',
                        fontsize = 35)
    # Increase the font size
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    # Increase font size of x and y labels
    plt.xlabel('Trajectory ID', fontsize=35)
    plt.ylabel('Cumulative cluster switches', fontsize=35)
    # save it in the trajectory disambig folder
    plt.savefig(TRAJECTORY_DISAMBIG_DIR + 'plasticity_per_trajectory.png', dpi=300)

    plt.show()
    
    return

import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def animate(step, ax, xmin, xmax, ymin, ymax, cell_df, contour_list, unique_id, trajectory_colors, cluster_colors, colormode):


    font_size = 60
    # Update the default text sizes
    plt.rcParams.update({
        'font.size': font_size,       # controls default text sizes
        'axes.titlesize': font_size,  # fontsize of the axes title
        'axes.labelsize': font_size,  # fontsize of the x and y labels
        'xtick.labelsize': font_size, # fontsize of the tick labels
        'ytick.labelsize': font_size, # fontsize of the tick labels
        'legend.fontsize': font_size, # legend fontsize
        'figure.titlesize': font_size # fontsize of the figure title
    })
    ax.clear()
    for i in range(step+1):  # loop over all preceding steps
        contour = contour_list[i]
        contour_arr = np.asarray(contour).T
        x = cell_df['x_pix'].values[i]
        y = cell_df['y_pix'].values[i]   
        Cluster_ID = cell_df['label'].iloc[i]
        traj_id = cell_df['trajectory_id'].iloc[0]
        '''Want to double check that the x,y positions not mirrored from the contour function'''
        if not np.isnan(np.sum(contour_arr)):
            # Check the colormode and set the color accordingly
            if colormode == 'trajectory':
                if i == step:
                    color = trajectory_colors[traj_id]
                    alpha = 1
                else:
                    color = trajectory_colors[traj_id]
                    alpha = 0.3
            elif colormode == 'cluster':
                if i == step:
                    color = cluster_colors[Cluster_ID]
                    alpha = 1
                else:
                    color = cluster_colors[Cluster_ID]
                    alpha = 0.3
            elif colormode == 'singlecluster':
                if i == step:
                    color = cluster_colors[Cluster_ID]
                    alpha = 1
                else:
                    color = 'gray'
                    alpha = 0.3
            ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c=color,linewidth=5, alpha=alpha)
            if i > 0:
                x_seg = cell_df['x_pix'].values[i-1:i+1]# - window / 2
                y_seg = cell_df['y_pix'].values[i-1:i+1]# - window / 2
                ax.plot(x_seg,y_seg,'-o',markersize=10,c='black', linewidth=4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Add a scale bar
    scaleinmicrons = 20
    scalebar = AnchoredSizeBar(ax.transData,
                               scaleinmicrons / MICRONS_PER_PIXEL, f'', 'lower left', 
                               pad=0.1,
                               color='black',
                               borderpad=0,
                               frameon=False,
                               size_vertical=5,
                               fontproperties=fm.FontProperties(size=font_size))
    ax.add_artist(scalebar)
    # Add a text box for the scale bar text
    text = ax.text(xmin, ymin, f'{scaleinmicrons} um', fontproperties=fm.FontProperties(size=font_size))
    # Add a title with the trajectory ID
    ax.set_title('Trajectory ID: ' + str(traj_id) +'   Cell ID: ' + str(unique_id))
    # Add a timer in the top right corner
    ax.text(xmax, ymin, f'Time: {step*SAMPLING_INTERVAL:.2f} mins', horizontalalignment='right', verticalalignment='bottom', fontsize=font_size)
    ax.axis('off')
    return ax,

def make_trajectory_animations(df, exemplar_df_trajectories, number_of_trajectories=3, colormode='cluster', XYRange =300):
    trajectory_ids = df['trajectory_id'].unique()
    # repeat each element of trajectory_ids by the number_of_trajectories and make that a list
    trajectory_ids = np.repeat(trajectory_ids, number_of_trajectories).tolist()
    
    cluster_colors = []
    labels = list(set(df['label'].unique()))
    numofcolors = len(labels)
    cmap = cm.get_cmap(CLUSTER_CMAP)
    for i in range(numofcolors):
        cluster_colors.append(cmap(i))
    # match the cluster colors to each cluster ID
    cluster_colors = dict(zip(labels, cluster_colors))

    # Trajectory_IDs = tptlabel_dr_df_filt_clusteredtrajectories['trajectory_id'] 
    # depending on the cluster ID, make a list of colors for each cluster that is the same length of the list of the labels
    trajectory_colors = []
    trajlabels = list(set(df['trajectory_id'].unique()))
    numofcolors = len(trajlabels)
    cmap = cm.get_cmap(CLUSTER_CMAP)
    for i in range(numofcolors):
        trajectory_colors.append(cmap(i))
    # match the cluster colors to each cluster ID
    trajectory_colors = dict(zip(trajlabels, trajectory_colors))

    redundancy_list = []

    for trajectory_id_choice in trajectory_ids:
        uniq_id_choices = exemplar_df_trajectories[exemplar_df_trajectories['trajectory_id']==trajectory_id_choice]['uniq_id'].values
        uniq_id_choice = np.random.choice(uniq_id_choices)

        while uniq_id_choice in redundancy_list:
            uniq_id_choice = np.random.choice(uniq_id_choices)
            
        # append each choice to a redundancy list
        redundancy_list.append(uniq_id_choice)



        cell_df = df[df['uniq_id']==uniq_id_choice]
        contour_list = []
        contour_list=get_cell_contours(cell_df) # CHANGE CP 

        ### This part makes sure each plot is going to be on the same scale ###
        xminimum=cell_df['x_pix'].min() 
        xmaximum=cell_df['x_pix'].max()
        xmid = np.median([xmaximum, xminimum])
        xmin=xmid-XYRange/2
        xmax=xmid+XYRange/2

        yminimum=cell_df['y_pix'].min()
        ymaximum=cell_df['y_pix'].max()
        ymid = np.median([ymaximum, yminimum])
        ymin=ymid-XYRange/2
        ymax=ymid+XYRange/2

        ##############################

        # print(contour_list)

        print(f'The colour mode is {colormode}')

        fig, (ax) = plt.subplots(1,1, figsize=(0.08*XYRange,0.08*XYRange))
        unique_id=cell_df['uniq_id'].iloc[0]
        traj_id=cell_df['trajectory_id'].iloc[0]
        ani = animation.FuncAnimation(fig, animate, fargs=(ax, xmin, xmax, ymin, ymax, cell_df, contour_list, 
            unique_id, trajectory_colors, cluster_colors, colormode,), frames=len(cell_df), interval=200)
        writer = animation.PillowWriter(fps=15)
        ani.save(TRAJECTORY_DISAMBIG_DIR + f'{colormode}_animation_traj_ID_{traj_id}_cellID_{unique_id}_.gif', writer=writer)

    return redundancy_list
    

def get_single_cell_raw_image(cell_df, XYRange, cell_follow):

# Step 1) Find the raw data folder and print it

    df_out = cell_df.copy()

    cond = df_out['Condition'].values[0]
    exp = df_out['Replicate_ID'].values[0]
    exp = exp.replace('_tracks','') # remove _tracks from exp
    load_path = os.path.join(DATA_PATH,cond, exp)


    # 2) Based on the uniq_id of the cell_df, extract a list of the coordinates of the cell and the frame numbers

    # sort the df by frame first
    cell_df = cell_df.sort_values(by=['frame'])
    coordinates = cell_df[['x', 'y']].values.tolist()
    frame_numbers = cell_df['frame'].values.tolist()

    #3 ) FIRST DEAL WITH OPENING THE IMAGE OF THE RIGHT FRAMES

    # turn the frame_numbers into integers
    frame_numbers = [int(i) for i in frame_numbers]

    # If any of those frame numbers have less than 3 digits, add a 0 to the front of them
    frame_numbers = ['0' + str(i) if len(str(i)) == 2 else '00' + str(i) if len(str(i)) == 1 else str(i) for i in frame_numbers]

    # list the files in the load_path
    files = os.listdir(load_path)

    # basename = files[0].split('_T')[0]
    basename = files[0].split('T0')[0]

    xstart_list = []
    ystart_list = []
    xshift_list = []
    yshift_list = []
    xend_list = []
    yend_list = []

    padding = XYRange // 2
    # 4) prior to making a cropping loop, make the cropping tings

    if cell_follow == 'follow':
        print('Following the cell')

        cropped_image_list = []

        for i, currentframe in enumerate(frame_numbers):
            # print(f' this is i {i}' )
            # print(f' this is currentframe{currentframe}')


            x = coordinates[i][0] # first dim should be i for the i'th image if you want the version that follows the cell!
            y = coordinates[i][1] # 


            # Calculate the top-left corner of the crop region
            x_start = x - (XYRange // 2)
            y_start = y - (XYRange // 2)
            # Also extract an xshift and a yshift
            xshift = x_start
            yshift = y_start

            # calculate the end of the crop region
            x_end = x_start + XYRange
            y_end = y_start + XYRange
            # make them all integers
            x_start = int(x_start+padding)
            y_start = int(y_start+padding)
            x_end = int(x_end+padding)
            y_end = int(y_end+padding)

            xstart_list.append(x_start)
            ystart_list.append(y_start)
            xshift_list.append(xshift)
            yshift_list.append(yshift)
            xend_list.append(x_end)
            yend_list.append(y_end)
            

            # imagename = basename + '_T0{}.tif'.format(currentframe)
            imagename = basename + 'T0{}.tif'.format(currentframe)
            image = io.imread(os.path.join(load_path, imagename))
            image = image[0,:,:] # At the moment, only one channel is imported
            padded_image = np.pad(image, pad_width=((padding, padding), (padding, padding)), mode='constant')
            padded_image = padded_image.astype(np.uint16)
            # Standardize the image to between 
            crop_img = padded_image[y_start:y_end, x_start:x_end]

            # # Standardize the image to between 
            # image = image[0,:,:] # At the moment, only one channel is imported
            # crop_img = image[y_start:y_end, x_start:x_end]
            # Append this thing to a list
            cropped_image_list.append(crop_img)

    else:
        # print('Not following the cell')
        x = coordinates[0][0] # 
        y = coordinates[0][1] # 

        # Calculate the top-left corner of the crop region
        x_start = x - (XYRange // 2)
        y_start = y - (XYRange // 2)
        # Also extract an xshift and a yshift
        xshift = x_start
        yshift = y_start

        # calculate the end of the crop region
        x_end = x_start + XYRange
        y_end = y_start + XYRange

        # make them all integers
        x_start = int(x_start+padding)
        y_start = int(y_start+padding)
        x_end = int(x_end+padding)
        y_end = int(y_end+padding)

        cropped_image_list = []
    # 4) Now, load the image of the first frame

        for currentframe in frame_numbers:

            imagename = basename + 'T0{}.tif'.format(currentframe)
            image = io.imread(os.path.join(load_path, imagename))# 
            image = image[0,:,:] # At the moment, only one channel is imported
            # Standardize the image to between 0 and 1, using the max val
            padded_image = np.pad(image, pad_width=((padding, padding), (padding, padding)), mode='constant')
            padded_image = padded_image.astype(np.uint16)
            # Crop the image
            # crop_img = image[y_start:y_end, x_start:x_end]
            crop_img = padded_image[y_start:y_end, x_start:x_end]
            # Append this thing to a list
            cropped_image_list.append(crop_img)
            # Save the image as a gif

    if cell_follow == 'follow':
        return cropped_image_list, xshift_list, yshift_list
    else:
        return cropped_image_list, xshift, yshift


import imageio

def make_raw_cell_pngstacks(df,chosen_uniq_ids,XYRange = 300, follow_cell = False, invert=True, LUTlow=10, LUThi=140): #cluster or trajectory
    
    # These uniq_ids are the ones to be used here

    font_size = 15
    dpi = 150 # This is the resolution of the figure
    width = 1200 # This is the width of the figure in pixels
    height = 1200 # This is the height of the figure in pixels

    if follow_cell:
        cell_follow = 'follow'
    else:
        cell_follow = 'no_follow'

    # make a cell df based off a uniq_id
    for uniq_id in chosen_uniq_ids:
        print(f'Processing cell {uniq_id}')
        cell_df = df[df['uniq_id'] == uniq_id]    
        cropped_cell_list, xshift, yshift = get_single_cell_raw_image(cell_df, XYRange, cell_follow)

        traj_id = cell_df['trajectory_id'].iloc[0]


        # element of the list 
        if follow_cell:
            nameforfolder = f'Raw_followed_cell_{uniq_id}'
        else:
            nameforfolder = f'Raw_static_cell_{uniq_id}'
        # Define the directory
        output_dir = os.path.join(TRAJECTORY_DISAMBIG_DIR, nameforfolder)
        os.makedirs(output_dir, exist_ok=True)
        # Create a new figure with a specific size in inches

        # Get the maximum value across all images
        if invert:
            print('Inverting the image')
            max_value = np.max([np.max(img) for img in cropped_cell_list])
            # Invert the colors of the images
            inverted_image_list = [max_value - img for img in cropped_cell_list]
            cropped_cell_list = inverted_image_list
            color = 'k'

        else:
            # print('Not inverting the image')
            color = 'w'

        max_value = np.max([np.max(img) for img in cropped_cell_list])
        min_value = np.min([np.min(img) for img in cropped_cell_list])

        if follow_cell:

            for i, img in enumerate(cropped_cell_list):

                fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
                # Apply the 'hot' colormap
                # plt.imshow(img, cmap='Greys')
                if LUTlow is not None and LUThi is not None:
                    plt.imshow(img, cmap='gray', vmin=max_value - LUThi, vmax=max_value - LUTlow)
                else:
                    plt.imshow(img, cmap='gray')
                ax = plt.gca()
                # plt.clim(0, 1) # Set the color limits
                plt.axis('off') # Turn off the axis
                xcoords = cell_df['x'].values
                ycoords = cell_df['y'].values
                shifted_xcoords = xcoords - xshift[i]
                shifted_ycoords = ycoords - yshift[i]

                for j in range(i+1):
                    x_seg = shifted_xcoords[j:j+2]
                    y_seg = shifted_ycoords[j:j+2]
                    ax.plot(x_seg,y_seg,'-o',markersize=2,c='black', linewidth=1)

                ax.set_title(f'Trajectory ID: {traj_id}    Cell_ID: {uniq_id}', x=0.5, y=0.95, pad=-10, fontsize=font_size, c='k')
                
                plt.savefig(os.path.join(output_dir, f'followed_cell_{i}.png'), bbox_inches='tight', pad_inches=0)
                # imageio.imwrite(os.path.join(output_dir, f'followed_raw_{i}.png'), img.astype(np.uint16))
                # cv2.imwrite(os.path.join(output_dir, f'followed_raw_{i}.png'), img)
                plt.close() # Close the figure to free up memory
                ax.clear()

        else:

            for i, img in enumerate(cropped_cell_list):



                fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
                # Apply the 'hot' colormap

                # plt.imshow(img, cmap='Greys')
                if LUTlow is not None and LUThi is not None:
                    plt.imshow(img, cmap='gray', vmin=max_value - LUThi, vmax=max_value - LUTlow)
                else:
                    plt.imshow(img, cmap='gray')
                ax = plt.gca()
                # plt.clim(0, 1) # Set the color limits
                plt.axis('off') # Turn off the axis
                xcoords = cell_df['x'].values
                ycoords = cell_df['y'].values
                shifted_xcoords = xcoords - xshift
                shifted_ycoords = ycoords - yshift

                ###################################
                for j in range(i+1):
                    x_seg = shifted_xcoords[j:j+2]
                    y_seg = shifted_ycoords[j:j+2]
                    ax.plot(x_seg,y_seg,'-o',markersize=2,c='black', linewidth=1)
                ####################################

                # if i > 0:
                #     x_seg = shifted_xcoords[i-1:i+1]# - window / 2
                #     y_seg = shifted_ycoords[i-1:i+1]# - window / 2
                #     ax.plot(x_seg,y_seg,'-o',markersize=10,c='black', linewidth=4)

                    xmax = XYRange
                    ymin = 0
                    # # Add a timer in the top right corner
                    # if i == 0:
                    #     timetext = 0
                    # else:
                    #     timetext = i*cp.SAMPLING_INTERVAL
                    # ax.text(xmax, ymin, f'Time: {timetext.2f} mins', horizontalalignment='right', verticalalignment='bottom', fontsize=font_size)
                    # ax.axis('off')
                ax.set_title(f'Trajectory ID: {traj_id}    Cell_ID: {uniq_id}', x=0.5, y=0.95, pad=-10, fontsize=font_size, c='k')

                plt.savefig(os.path.join(output_dir, f'static_raw_{i}.png'), bbox_inches='tight', pad_inches=0)
                # imageio.imwrite(os.path.join(output_dir, f'followed_raw_{i}.png'), img.astype(np.uint16))

                plt.close() # Close the figure to free up memory
                ax.clear()

from tqdm import tqdm 

def make_png_behaviour_trajectories(df,chosen_uniq_ids,XYRange = 300, follow_cell = False, invert=True, colormode='trajectory', snapshot=False):
    from tqdm import tqdm 
    
    ###  DEV ONE BURT

    font_size = 15
    dpi = 150 # This is the resolution of the figure
    width = 1200 # This is the width of the figure in pixels
    height = 1200 # This is the height of the figure in pixels

    ############################### Plot settings ##############################
    # font_size = 60
    # Update the default text sizes
    plt.rcParams.update({
        'font.size': font_size,       # controls default text sizes
        'axes.titlesize': font_size,  # fontsize of the axes title
        'axes.labelsize': font_size,  # fontsize of the x and y labels
        'xtick.labelsize': font_size, # fontsize of the tick labels
        'ytick.labelsize': font_size, # fontsize of the tick labels
        'legend.fontsize': font_size, # legend fontsize
        'figure.titlesize': font_size # fontsize of the figure title
    })

    #####################################

    ############################### Colours common to df, not cell_df #######################
    cluster_colors = []
    labels = list(set(df['label'].unique()))
    numofcolors = len(labels)
    cmap = cm.get_cmap(CLUSTER_CMAP)
    for i in range(numofcolors):
        cluster_colors.append(cmap(i))
    # match the cluster colors to each cluster ID
        
    cluster_colors = dict(zip(labels, cluster_colors))

    # Trajectory_IDs = tptlabel_dr_df_filt_clusteredtrajectories['trajectory_id'] 
    # depending on the cluster ID, make a list of colors for each cluster that is the same length of the list of the labels
    trajectory_colors = []
    trajlabels = list(set(df['trajectory_id'].unique()))
    numofcolors = len(trajlabels)
    cmap = cm.get_cmap(CLUSTER_CMAP)
    for i in range(numofcolors):
        trajectory_colors.append(cmap(i))
    # match the cluster colors to each cluster ID
    trajectory_colors = dict(zip(trajlabels, trajectory_colors))

    #################################################################


    if follow_cell:
        cell_follow = 'follow'
    else:
        cell_follow = 'no_follow'

    # make a cell df based off a uniq_id
    for uniq_id in chosen_uniq_ids:
        print(f'Processing cell {uniq_id}')
        # if snapshot then extract the last 8 frames of the cell_df only
        if snapshot:
            cell_df = df[df['uniq_id'] == uniq_id]
            cell_df = cell_df.iloc[-8:]
        else:
            cell_df = df[df['uniq_id'] == uniq_id]   
######################################################
       # Get the initial x and y coordinates
        x0 = cell_df['x_pix'].values[0]
        y0 = cell_df['y_pix'].values[0]

       # Calculate the lower and upper limits
        x_lower = x0 - XYRange / 2
        x_upper = x0 + XYRange / 2
        y_lower = y0 - XYRange / 2
        y_upper = y0 + XYRange / 2
        # get the 
        traject_id = cell_df['trajectory_id'].iloc[0]



###########################################################
        if follow_cell:
            nameforfolder = f'{colormode}_followed_cell_{uniq_id}_traj_{traject_id}'
        else:
            nameforfolder = f'{colormode}_static_cell_{uniq_id}_traj_{traject_id}'
        # Define the directory
        output_dir = os.path.join(TRAJECTORY_DISAMBIG_DIR, nameforfolder)
        os.makedirs(output_dir, exist_ok=True) 

        contour_list=get_cell_contours(cell_df)

        ### This part makes sure each plot is going to be on the same scale ###
        xminimum=cell_df['x_pix'].min() 
        xmaximum=cell_df['x_pix'].max()
        xmid = np.median([xmaximum, xminimum])
        xmin=xmid-XYRange/2
        xmax=xmid+XYRange/2

        yminimum=cell_df['y_pix'].min()
        ymaximum=cell_df['y_pix'].max()
        ymid = np.median([ymaximum, yminimum])
        ymin=ymid-XYRange/2
        ymax=ymid+XYRange/2

        # fig, ax = plt.subplots(1,1, figsize=(0.08*XYRange,0.08*XYRange))
        total_steps=len(cell_df)
        for step in tqdm(range(total_steps)):
            # if not snapshot or (snapshot and step == total_steps - 1):  # Check for the last step if snapshot is True
            # fig, ax = plt.subplots(1,1, figsize=(0.08*XYRange,0.08*XYRange))
            fig, ax = plt.subplots(1,1, figsize=(width/dpi, height/dpi))
#######################################
            # # if step == 0:
            #     ax.set_xlim([x_lower, x_upper])
            #     ax.set_ylim([y_lower, y_upper])


####################################
            
            for i in range(step+1):
                contour = contour_list[i]
                contour_arr = np.asarray(contour).T

                Cluster_ID = cell_df['label'].iloc[i]
                traj_id = cell_df['trajectory_id'].iloc[0]
                if i == step:
                    alpha=1
                else:
                    alpha=0.1
                if not np.isnan(np.sum(contour_arr)):
                    #### dev added 3-20-2024 ####
                    if colormode == 'trajectory':
                        # print('Colouring whole track by trajectory')
                        ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c=trajectory_colors[traj_id],linewidth=1, alpha=alpha) #cilantro
                    else:
                        ax.plot(contour_arr[:,0],contour_arr[:,1],'-o',markersize=1,c=cluster_colors[Cluster_ID],linewidth=1, alpha=alpha) #cilantro
                if i > 0:
                    x_seg = cell_df['x_pix'].values[i-1:i+1]
                    y_seg = cell_df['y_pix'].values[i-1:i+1]
                    # y_seg = cell_df['y_pix'].values[i:i+2]

                    ax.plot(x_seg,y_seg,'-o',markersize=1,c='black', linewidth=1)



        ####
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel('x (px)', fontname="Arial",fontsize=PLOT_TEXT_SIZE)
            ax.set_ylabel("y (px)", fontname="Arial",fontsize=PLOT_TEXT_SIZE)
            ax.tick_params(axis='both', labelsize=PLOT_TEXT_SIZE)
            ax.set_aspect('equal')
            ax.set_adjustable("datalim")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            scaleinmicrons = 20
            scalebar = AnchoredSizeBar(ax.transData,
                                    scaleinmicrons / MICRONS_PER_PIXEL, f'', 'lower left', 
                                    pad=0.1,
                                    color='black',
                                    borderpad=0,
                                    frameon=False,
                                    size_vertical=5,
                                    fontproperties=fm.FontProperties(size=font_size))
            ax.add_artist(scalebar)
            # Add a text box for the scale bar text
            # text = ax.text(xmin, ymin, f'{scaleinmicrons} um', fontproperties=fm.FontProperties(size=font_size))
            text = ax.text(xmin, ymax, f'{scaleinmicrons} um', fontproperties=fm.FontProperties(size=font_size))
            # Add a title with the trajectory ID
            ax.set_title('Trajectory ID: ' + str(traj_id) +'   Cell ID: ' + str(uniq_id), x=0.5, y=0.95, pad=-10, fontsize=font_size)
            # Add a timer in the top right corner
            # ax.text(xmax, ymin, f'Time: {step*SAMPLING_INTERVAL:.2f} mins', horizontalalignment='right', verticalalignment='bottom', fontsize=font_size)
            ax.text(xmax, ymax, f'Time: {step*SAMPLING_INTERVAL:.2f} mins', horizontalalignment='right', verticalalignment='bottom', fontsize=font_size)
            ax.axis('off')
            ax.invert_yaxis() # CILANTRO

            # plt.savefig(os.path.join(output_dir, f'arsebehaviour_{step}.png'), dpi=dpi)
            plt.savefig(os.path.join(output_dir, f'Behaviours_cell_{step}.png'), bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close()
            # clear the plot for the next one
            ax.clear()

            # fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)


    return


###### THIS ONE is DEV ###### AND IT WORKS ######

def make_raw_cell_png_overlaidwith_behaviourcontourandtrack(df, exemplar_df,XYRange = 300, LUTlow = 100, LUThi=1000, follow_cell = False, invert=False):
    from tqdm import tqdm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    chosen_uniq_ids = exemplar_df['uniq_id'].values

    font_size = 15
    dpi = 150 # This is the resolution of the figure
    width = 1200 # This is the width of the figure in pixels
    height = 1200 # This is the height of the figure in pixels

    plt.rcParams.update({
    'font.size': font_size,       # controls default text sizes
    'axes.titlesize': font_size,  # fontsize of the axes title
    'axes.labelsize': font_size,  # fontsize of the x and y labels
    'xtick.labelsize': font_size, # fontsize of the tick labels
    'ytick.labelsize': font_size, # fontsize of the tick labels
    'legend.fontsize': font_size, # legend fontsize
    'figure.titlesize': font_size # fontsize of the figure title
    })


    cell_follow = 'no_follow'
    # Define the directory
    nameforfolder = f'Exemplars_de-abstractified'
    output_dir = os.path.join(CLUST_DISAMBIG_DIR, nameforfolder)
    os.makedirs(output_dir, exist_ok=True)

    # Make the cluster colors
    cluster_colors = []
    labels = list(set(df['label'].unique()))
    numofcolors = len(labels)
    cmap = cm.get_cmap(CLUSTER_CMAP)
    for i in range(numofcolors):
        cluster_colors.append(cmap(i))
    # match the cluster colors to each cluster ID
    cluster_colors = dict(zip(labels, cluster_colors))

    # make a cell df based off a uniq_id
    for uniq_id in tqdm(chosen_uniq_ids):
        # print(f'Processing cell {uniq_id}')
        exemplar_row = exemplar_df[exemplar_df['uniq_id'] == uniq_id] #THERE ARE MANY, LOL. NOT JUST 1 EXEMPLAR FROM EACH CELL.
        if len(exemplar_row) == 0:
            print(f'No exemplar exists for {uniq_id} so moving on to the next one')
            continue
        # Get the frame number
        frame_number = exemplar_row['frame'].values[0]
        # Use the frame number to get the correct row from the df
        cell_df = df[(df['uniq_id'] == uniq_id) & (df['frame'] == frame_number)]
        # Get the raw image as a numpy array and output the xshift and yshift
        cropped_cell_list, xshift, yshift = get_single_cell_raw_image(cell_df, XYRange, cell_follow)   
        # Get the contour list
        contour_list=get_cell_contours(cell_df) 
        # Calculate some min and max values for the cell_df
        x0 = cell_df['x_pix'].values[0]
        y0 = cell_df['y_pix'].values[0]
        xminimum=cell_df['x_pix'].min() 
        xmaximum=cell_df['x_pix'].max()
        xmid = np.median([xmaximum, xminimum])
        xmin=xmid-XYRange/2
        xmax=xmid+XYRange/2
        yminimum=cell_df['y_pix'].min()
        ymaximum=cell_df['y_pix'].max()
        ymid = np.median([ymaximum, yminimum])
        ymin=ymid-XYRange/2
        ymax=ymid+XYRange/2
        # Make some adjustments for inverting the image if that options is selected
        if invert:
            print('Inverting the image')
            max_value = np.max([np.max(img) for img in cropped_cell_list])
            # Invert the colors of the images
            inverted_image_list = [max_value - img for img in cropped_cell_list]
            cropped_cell_list = inverted_image_list
            color = 'k'
        else:
            # print('Not inverting the image')
            color = 'w'

        # Here you extract the next previous row from the exemplar cell, in order to plot the track
        cell_df_row1 = df[(df['uniq_id'] == uniq_id) & (df['frame'] == frame_number)] # you look in the main df for this
        cell_df_row2 = df[(df['uniq_id'] == uniq_id) & (df['frame'] == frame_number - 1)] # then take the previous row
        cell_df_plus = pd.concat([cell_df_row2, cell_df_row1]) # concatenate them
        # Now you start displaying images and plotting:
        fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
        # Get the raw image
        img = cropped_cell_list[0]
        # Compensate for inversion if its done and plot the image
        if invert:
            max_value = np.max(img)  # Assuming img is the inverted image here
            min_value = np.min(img)
            print(f'The max value is {max_value} and the min value is {min_value}')
            ax.imshow(img, cmap='gray', vmin=max_value - LUThi, vmax=max_value - LUTlow)
        else:
            ax.imshow(img, cmap='gray', vmin=LUTlow, vmax=LUThi)
        # Plot the contour on top of the raw image
        contour = contour_list[0]
        contour_arr = np.asarray(contour).T
        contour_arr = contour_arr - np.array([xshift, yshift]) # Shifts the contour to align with the raw image
        Cluster_ID = cell_df['label'].iloc[0]
        # Hide axes and ticks
        ax.axis('off')
        Cluster_ID = cell_df['label'].iloc[0]
        ax.plot(contour_arr[:, 0], contour_arr[:, 1], '-o', markersize=1, linewidth=3, alpha=1, color=cluster_colors[Cluster_ID])
        
        # Finally plot the track on top of the raw image
        xcoords = cell_df_plus['x'].values
        ycoords = cell_df_plus['y'].values
        shifted_xcoords = xcoords - xshift
        shifted_ycoords = ycoords - yshift
        x_seg = shifted_xcoords
        y_seg = shifted_ycoords  
        ax.plot(x_seg,y_seg,'-o',markersize=2,c='k', linewidth=3)     

        # Now do some labelling of the plot:
        ax.set_title(f'Cluster ID {Cluster_ID} + Cell_ID: {uniq_id}', x=0.5, y=0.95, pad=-10, fontsize=font_size, c='k')
        #crucial: the section makes new mins and maxes to align the raw and contour images
        xmin=xmin-xshift
        xmax=xmax-xshift
        ymin=ymin-yshift
        ymax=ymax-yshift
        #crucial: aligns the raw and contour images
        # aaaaand this sets them:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x (px)', fontname="Arial",fontsize=PLOT_TEXT_SIZE)
        ax.set_ylabel("y (px)", fontname="Arial",fontsize=PLOT_TEXT_SIZE)
        ax.tick_params(axis='both', labelsize=PLOT_TEXT_SIZE)
        ax.set_aspect('equal')
        ax.set_adjustable("datalim")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        #make it tight
        plt.tight_layout()
        # Makes a scale bar:
        scaleinmicrons = 20
        scalebar = AnchoredSizeBar(ax.transData,
                                scaleinmicrons / MICRONS_PER_PIXEL, f'', 'lower left', 
                                pad=0.1,
                                color='black',
                                borderpad=0,
                                frameon=False,
                                size_vertical=5,
                                fontproperties=fm.FontProperties(size=font_size))
        ax.add_artist(scalebar)
        # Add a text box for the scale bar text
        ax.text(xmin, ymin+(XYRange*0.05), f'{scaleinmicrons} um', fontproperties=fm.FontProperties(size=font_size))
        #saves the figure
        plt.savefig(os.path.join(output_dir, f'RawContourTrack_{uniq_id}_Cluster_{Cluster_ID}.png'), bbox_inches='tight', pad_inches=0)
        plt.close() # Close the figure to free up memory
        ax.clear()

