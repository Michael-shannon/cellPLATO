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



def plot_cell_metrics_timepoint(cell_df, i_step, XYRange,boxoff, top_dictionary, Cluster_ID, mig_display_factors=MIG_DISPLAY_FACTORS,shape_display_factors=SHAPE_DISPLAY_FACTORS):
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
def disambiguate_timepoint(df, exemps, top_dictionary, XYRange=100,boxoff=True):


    for n in exemps.index:

        row=exemps.iloc[n] #extract an example exemplar row

        # Use exemplar row to look up the corresponding cell TRACK from the cell_df
        cell_df = df[(df['Condition']==row['Condition']) &
                        (df['Replicate_ID']==row['Replicate_ID']) &
                        (df['particle']==row['particle'])]
        cell_df = cell_df.reset_index(drop=True)

        # This looks up the exemplar point, based on all of these metrics, so that the correct exemplar point is displayed in the visualization
        exemplarpoint = cell_df.index[(cell_df['area']==row['area']) &
                        (cell_df['speed']==row['speed']) &
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

def contribution_to_clusters(df_in, threshold_value=0.0001, dr_factors=DR_FACTORS): #New function 21-14-2022

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

def contribution_to_clusters_topdictionary(df_in, threshold_value=0.0001, dr_factors=DR_FACTORS, howmanyfactors=6, scalingmethod = SCALING_METHOD): #New function 21-14-2022

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
    elif scalingmethod == None:
        X = Z
        correctcolumns=CLUSTERON

    # X = MinMaxScaler().fit_transform(sub_set)
    thelabels = df_in['label']
    scaled_df_in = pd.DataFrame(data=X, columns = correctcolumns)
    df_out = pd.concat([scaled_df_in,thelabels], axis=1)#Put this back into a DF, with the col names and labels.
    # df_out

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
    return top_dictionary, clusteraverage_df

def plot_cluster_averages(top_dictionary, df): # New 3-7-2023
    num_plots = len(top_dictionary)
    # print(top_dictionary)
    # Reverse the order of the values in the dictionary so that the biggest contributor to variability is on top
    top_dictionary = {k: v[::-1] for k, v in top_dictionary.items()}
    # print(top_dictionary)
    ### Make a totally non-normalized version of the clusteraverage_df
    cluster_average_df = df.groupby('label').median()#.reset_index(drop=True)

    # Create a grid of subplots with one row and num_plots columns
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(10*num_plots,10))
    
    # Loop over the cluster IDs and corresponding values in the dictionary
    for i, (cluster_id, value) in enumerate(top_dictionary.items()):
        # Get the row in the dataframe that corresponds to the current cluster ID
        cluster_row = cluster_average_df.loc[cluster_id]
        
        # Loop over the column names for the current key and create a text string
        text_str = ""
        for column_name in value:
            # Get the value in the specified column for the current cluster
            column_value = round(cluster_row[column_name], 4)
            
            # Add the string and the corresponding value to the text string
            text_str += f"{column_name.title()}: {column_value}\n"
            text_str = text_str.replace('_',' ')
        
        # Plot the text string as text in the current subplot
        axs[i].text(0.5, 0.5, text_str.strip(), ha='center', va='center', fontsize=30)
        
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

