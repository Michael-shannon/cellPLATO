#shape_calculations.py

from initialization.initialization import *
from initialization.config import *

from data_processing.data_io import btrack_unpack
# from data_processing.data_io import *

from data_processing.data_wrangling import *


import os
import numpy as np
import pandas as pd

import fnmatch
import h5py
from tqdm import tqdm

from skimage import data, measure, io
from skimage.measure import label, regionprops, regionprops_table


def get_cell_contours(data_in, fmt=INPUT_FMT):

    '''
    Take a dataframe containing containing cell location and timestamp,
    return a list of the cell contours. Used for plotting cell contours
    in low-dimensional space.


    Input:
        df_in: Dataframe [N*t,C]
        fmt: string:
            'usiigaci'
            'btrack'

    Output:
        contour_list: list

    '''

    if fmt == 'usiigaci':

        print('get_cell_contours() is not yet able to handle data of ', fmt, ' format.')
        assert isinstance(data_in, pd.DataFrame), 'for usiigaci, data_in must be dataframe'


    elif fmt == 'btrack':

        df_out = data_in.copy()

        props_list = [] # List to store the region properties before DataFrame
        contour_list = []

        # Get numpy array representation fo the dataframe:
        xyzt_arr = df_out[['x', 'y', 'z', 'frame']].to_numpy()
        assert xyzt_arr.shape[0] == len(df_out), 'Wrong shape'

        for i in tqdm(range(len(df_out))):

            # Extract cell position from the data volume
            x,y,z,t = xyzt_arr[i,:]

            # Only if using a new experiment do we need to reload the h5 data.
            if(i == 0):

                cond = df_out['Condition'].values[i]
                exp = df_out['Replicate_ID'].values[i]

                # Unpack h5 file
                load_path = os.path.join(DATA_PATH,cond, exp)
                if(DEBUG):
                    print('loading: ', load_path)
                h5_contents = btrack_unpack(load_path + TRACK_FILENAME)

                coords = np.asarray(h5_contents['coords'])
                seg = np.asarray(h5_contents['segmentation'])

            # If new experiment, load and unpack file
            elif(df_out['Replicate_ID'].values[i] != df_out['Replicate_ID'].values[i-1]):

                cond = df_out['Condition'].values[i]
                exp = df_out['Replicate_ID'].values[i]

                # Unpack h5 file
                load_path = os.path.join(DATA_PATH,cond, exp)

                h5_contents = btrack_unpack(load_path + TRACK_FILENAME)

                coords = np.asarray(h5_contents['coords'])
                seg = np.asarray(h5_contents['segmentation'])

                if(DEBUG):
                    print('loading: ', load_path)
                    print('coords.shape: ',coords.shape)
                    print('seg.shape: ',seg.shape)


            img,(imx, imy) = btrackimage_from_df(x,y,z,t,coords, seg)

            assert img.shape[0] == IMAGE_WIDTH, 'Image width not consistent with segmentations'
            assert img.shape[1] == IMAGE_HEIGHT, 'Image height not consistent with segmentations'

            # Constrain the x and y values to be within the shape of the image (applies to cells at image border)
            imx = np.clip(imx,0,img.shape[0] - 1)
            imy = np.clip(imy,0,img.shape[1] - 1)

            # Catch zero intensity values and replace their regionprops with NaNs
            if(img[imy,imx] == 0):

                # Make a Nan-array
                nan_array = np.empty((1))
                nan_array[:] = np.NaN

                contour_list.append((np.NaN,np.NaN))

            else:


                '''
                Contour findiner code
                from get_contour()
                NOTE: consider updating get_contour to return the x,y coords.
                Note: For now, this part here is being done twice since get_contour()
                is run above.
                '''

                thresh_img = img == img[imy,imx]

                labels = measure.label(thresh_img)
                contours = measure.find_contours(labels == 1, 0.5)

                if(len(contours)>0):
                    contour = measure.find_contours(labels == 1, 0.5)[0]

                else:
                    contour = np.empty((10,2))
                    contour[:] = np.NaN

                y, x = contour.T

                contour_list.append((x,y))
                '''
                End of new contour part.
                '''

    return contour_list



def usiimage_from_df(df,i):

    '''
    Get the image containing the specified cell in the dataframe.

    Input:
        df: dataframe (comb_df, dr_df?)
        i: cell index (ID)

    Returns:
        img: 2D array
        (xpix,ypix): tuple containing cell location in pixels (image coordinates)
    '''

    cond = df['Condition'][i]
    rep = df['Replicate_ID'][i]
    # x = df['x'][i]
    # y = df['y'][i]
    x = df['x_pix'][i]
    y = df['y_pix'][i]
    frame = df['frame'][i]

    # Get mask from 'Id_masks_per_frame'
    load_dir = os.path.join(DATA_PATH,cond,rep,'Id_masks_per_frame')
    file_list = os.listdir(load_dir)

    file_const_str = 'id_masks_'
    img_str = file_const_str + str(int(frame)).zfill(2) + '.png'
    img_path = os.path.join(load_dir,img_str)

    # xpix = int(x / MICRONS_PER_PIXEL)
    # ypix = int(y / MICRONS_PER_PIXEL)
    xpix = int(x)
    ypix = int(y)


    assert os.path.isfile(img_path)

    img = io.imread(img_path)

    return img, (xpix,ypix)

def get_contour(img, val, plot=False):

    '''
    Use skimage to get the contour for the provided 2d image at the provided
    intensity value.

        Input: img:  x*y numpy array representing image (or volume slice)
                val: gray value to extract the contour of.
                plot: Boolean, whether or not to draw the plot. Used in the
                    panel app
        Returns:
                labels: x*y mask of the ROI.
    '''

    thresh_img = img == val

    labels = measure.label(thresh_img)
    contours = measure.find_contours(labels == 1, 0.5)

    if(len(contours)>0):
        contour = measure.find_contours(labels == 1, 0.5)[0]

    else:
        contour = np.empty((10,2))
        contour[:] = np.NaN

    y, x = contour.T

    if plot: #Matplotlib

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(img>0)
        ax1.plot(x, y, c='r')

        # Show crop of the cell
        window = 30
        xi = int(int(imx) - window / 2)
        xii = int(int(imx) + window / 2)
        yi = int(int(imy) - window / 2)#img.shape[1]
        yii = int(int(imy) + window / 2)

        # Get contour for this smaller version of the cell
        sub_img = thresh_img[yi:yii, xi:xii] # Notice the reversal!!!
        sub_labels = measure.label(sub_img)
        sub_contour = measure.find_contours(sub_labels == 1, 0.5)[0]
        sy, sx = sub_contour.T

        # Draw the contour on a crop of the un-thresholded image
        # (So you can see other cells if present)
        crop_img = img[yi:yii, xi:xii] # Notice the reversal!!!
        ax2.imshow(crop_img)
        ax2.plot(sx, sy, c='r')

    # props_table = regionprops_table(labels, img, properties=REGIONPROPS_LIST)
    # props_df = pd.DataFrame(props_table)
    #
    # return (x,y), props_df
    return labels


def btrackimage_from_df(x,y,z,t,coords, seg):

    '''
    Function to load segmentation from h5 file using dataframe as input.

    Input:
        x,y,z,t: location of the cell of interest within the stack.
        coords: numpy array, coordinates from h5 bayesian tracker output
        seg: numpy array, segmentations from h5 bayesian tracker output

    Returns:
        h5_tzslice: 2D slice of the segmentation image that contains the cell of interest.
        (imx, imy): tuple coordinate pairs of cell position within image

    '''

    # Extract from the dataframe values
    imx = int(x)
    imy = int(y)
    imz = int(z / Z_SCALE)
    t = int(t) #This is correct, as the coords from btracker h5 are 0-indexed # - 1

    # constrain values to the dimensions of seg
    # imx = np.clip(imz, 0, seg.shape[2]) # Not sure if they are flipped
    # imy = np.clip(imz, 0, seg.shape[3]) # Not sure if they are flipped
    imz = np.clip(imz, 0, seg.shape[1]-1) # Not clipping could case an error at an edge case of the seg stack
    t = np.clip(t, 0, seg.shape[0])

    # Get slice of coordinates at specified frame
    t_vect = np.unique(coords[:,0])
    t_coords = coords[np.where(coords[:,0] == t_vect[t])]
    # h5_tzslice = seg[t,imz,:,:]

    # seg is expected to be a t*z*x*y array, if input 2D, np.ndim will be 3 if seg is t*x*y only

    if(seg.ndim == 3):
        h5_tzslice = seg[t,:,:] #2D
    else:
        h5_tzslice = seg[t,imz,:,:] #3D

    return h5_tzslice, (imx, imy)



def batch_shape_measurements(data_in, fmt=INPUT_FMT, n_samples='all'):

    '''
    Take a dataframe containing containing cell location and timestamp, return
    dataframe with the cell shape measurements concatenated.
    Where M is a constant list of measurements to be computed

    Input:
        df_in: Dataframe [N*t,C]
        fmt: string:
            'usiigaci'
            'btrack'
        n_samples: int or string:'all'. Allows testing with a smaller set

    Output:
        df_out: Dataframe [N*t,C+M]
    '''

    if fmt == 'usiigaci':

        assert isinstance(data_in, pd.DataFrame), 'for usiigaci, data_in must be dataframe'

        if n_samples == 'all':
            df_out = data_in.copy()
            print('Using all samples from comb_df.')
        else:
            df_out = data_in.head(n_samples)
            print('For testing, using ', n_samples, ' samples from comb_df.')

        props_list = []

        for i in tqdm(range(len(df_out))):

            # Get the image containing the selected Usiigaci cell
            img,(imx, imy) = usiimage_from_df(df_out,i)

            assert img.shape[0] == IMAGE_WIDTH, 'Image width not consistent with segmentations'
            assert img.shape[1] == IMAGE_HEIGHT, 'Image height not consistent with segmentations'

            # seg_coords,prop_row_df = get_contour(img, img[imy,imx], plot=False)
            # props_list.append(prop_row_df) # As a list'

            # # Region props
            # labels = get_contour(img, img[imy,imx], plot=False)
            # props_table = regionprops_table(labels, img, properties=REGIONPROPS_LIST)
            # prop_row_df = pd.DataFrame(props_table)
            # props_list.append(prop_row_df)

            # Catch zero intensity values and replace their regionprops with NaNs
            if(img[imy,imx] == 0):

                # Make a Nan-array
                nan_array = np.empty((1))
                nan_array[:] = np.NaN

                # Create a dataframe of NaNs values for the regionprops.
                props_table = pd.DataFrame()
                for prop in REGIONPROPS_LIST:
                    props_table[prop] = nan_array
            else:

                # Region props
                labels = get_contour(img, img[imy,imx], plot=False)
                props_table = regionprops_table(labels, img, properties=REGIONPROPS_LIST)

            prop_row_df = pd.DataFrame(props_table)
            props_list.append(prop_row_df)


        # Or dataframe
        props_df = pd.concat(props_list)
        props_df.reset_index(inplace=True,drop=True)

        # Asserts to compare the shape properties already in the dataframes
        # With the new version extracted from the contour.
        # Must be equal within the default tolerance level of allclose (1e05-1e08)
        # assert np.allclose(df_out['eccentricity'],props_df['eccentricity'],
        #                      equal_nan=True), 'Eccentricities dont match'
        #
        # assert np.allclose(df_out['solidity'],props_df['solidity'],
        #                      equal_nan=True), 'Solidities dont match'

        '''
        To add: Step to filter the original dataframe
            Want it to not contain duplicate columns,
            AND ideally be consistent with whatever produced by BTracke
            ex:
            sub_df = df[columns_to_keep] # Filter original dataframe by select factors
        '''

        # Combine the dataframes
        df_out = pd.concat([df_out, props_df],
                           axis=1)


    elif fmt == 'btrack':

        if n_samples == 'all':
            df_out = data_in.copy()
            print('Using all samples from comb_df. Warning: for btrack input this may take a long time to process.')
        else:
            df_1 = data_in.head(int(n_samples/2))
            df_2 = data_in.tail(int(n_samples/2))
            df_out = df_1.append(df_2)
            df_out.reset_index(inplace=True, drop=True)
            print('For testing, using ', n_samples, ' samples from comb_df.')

        props_list = [] # List to store the region properties before DataFrame

        # Get numpy array representation fo the dataframe:
        xyzt_arr = df_out[['x', 'y', 'z', 'frame']].to_numpy()
        assert xyzt_arr.shape[0] == len(df_out), 'Wrong shape'

        for i in tqdm(range(len(df_out))):

            # Extract cell position from the data volume
            x,y,z,t = xyzt_arr[i,:]

            # Only if using a new experiment do we need to reload the h5 data.
            if(i == 0):

                cond = df_out['Condition'][i]
                exp = df_out['Replicate_ID'][i]

                # Unpack h5 file
                load_path = os.path.join(DATA_PATH,cond, exp)
                print('loading: ', load_path)
                h5_contents = btrack_unpack(load_path + TRACK_FILENAME)

                coords = np.asarray(h5_contents['coords'])
                seg = np.asarray(h5_contents['segmentation'])

            # If new experiment, load and unpack file
            elif(df_out['Replicate_ID'][i] != df_out['Replicate_ID'][i-1]):

                cond = df_out['Condition'][i]
                exp = df_out['Replicate_ID'][i]

                # Unpack h5 file
                load_path = os.path.join(DATA_PATH,cond, exp)
                print('loading: ', load_path)
                h5_contents = btrack_unpack(load_path + TRACK_FILENAME)

                coords = np.asarray(h5_contents['coords'])
                seg = np.asarray(h5_contents['segmentation'])

                print('coords.shape: ',coords.shape)
                print('seg.shape: ',seg.shape)

            img,(imx, imy) = btrackimage_from_df(x,y,z,t,coords, seg)

            assert img.shape[0] == IMAGE_WIDTH, 'Image width not consistent with segmentations'
            assert img.shape[1] == IMAGE_HEIGHT, 'Image height not consistent with segmentations'

            # Constrain the x and y values to be within the shape of the image (applies to cells at image border)
            imx = np.clip(imx,0,img.shape[0] - 1)
            imy = np.clip(imy,0,img.shape[1] - 1)

            # Catch zero intensity values and replace their regionprops with NaNs
            if(img[imy,imx] == 0):

                # Make a Nan-array
                nan_array = np.empty((1))
                nan_array[:] = np.NaN

                # Create a dataframe of NaNs values for the regionprops.
                props_table = pd.DataFrame()
                for prop in REGIONPROPS_LIST:
                    props_table[prop] = nan_array
            else:

                # Region props
                labels = get_contour(img, img[imy,imx], plot=False)
                props_table = regionprops_table(labels, img, properties=REGIONPROPS_LIST)
            prop_row_df = pd.DataFrame(props_table)
            props_list.append(prop_row_df)


        # List to dataframe
        props_df = pd.concat(props_list)
        props_df.reset_index(inplace=True)

        ''' Add new asserts relevant to h5 data '''

        # Combine the dataframes
        df_out = pd.concat([df_out, props_df],axis=1)



        '''
        Add the section here to apply calibration to the regionprops
        to be sure all the shape measurements are also in microns.
        '''





    return df_out


'''btrack_unpack moved to data_wrangling'''
