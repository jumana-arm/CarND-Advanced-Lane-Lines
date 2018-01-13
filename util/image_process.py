import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def plot_pipeline_step_as_subplots(in_image, out_image, in_title, out_title, out_dir, cmap_in=None, cmap_out=None):
    """
    Helper function to plot an image processed image as subplot of input and output image to visualize the effect of transform
    applied.
    """
    f, (axis1, axis2) = plt.subplots(1, 2, figsize=(15,10))
    axis1.imshow(in_image, cmap=cmap_in)
    axis1.set_title(in_title, fontsize=16)
    axis2.imshow(out_image, cmap=cmap_out)
    axis2.set_title(out_title, fontsize=16)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig('{}/{}.jpg'.format(out_dir, out_title))
       
# Remove distortion from images
def undistort(img, cameraMatrix, distortionCoeffs, image_name='', out_dir=None):
    """
    Undistort the image using Camera calibration parameters.
    
    :returns: undistorted image
    """
    image_size = (img.shape[1], img.shape[0])
    undist = cv2.undistort(img, cameraMatrix, distortionCoeffs, None, cameraMatrix)    
    if out_dir:
        plot_pipeline_step_as_subplots(img, undist, image_name,'{}_undistorted'.format(image_name), out_dir)
    return undist

# Visualize multiple color space channels
def get_color_channels(img, color_space = 'RGB', channel_needed = None, image_name = '', plot = False):
    """
    Convert the input image to various color spaces: RGB, HSV and HLS
    """
    if plot:
        fig, axs = plt.subplots(1,3, figsize=(15,10))
        fig.subplots_adjust(hspace = .2, wspace=.001)
        axs = axs.ravel()
    
    if color_space == 'RGB':
        img_R = img[:,:,0]
        img_G = img[:,:,1]
        img_B = img[:,:,2]
        if plot:
            axs[0].imshow(img_R, cmap='gray')
            axs[0].set_title('{} RGB_R'.format(image_name), fontsize=15)
            axs[1].imshow(img_G, cmap='gray')
            axs[1].set_title('{} RGB_G'.format(image_name), fontsize=15)
            axs[2].imshow(img_B, cmap='gray')
            axs[2].set_title('{} RGB_B'.format(image_name), fontsize=15)
        if channel_needed == 'R':
            return img_R
        if channel_needed == 'G':
            return img_G
        if channel_needed == 'B':
            return img_B
    
    if color_space == 'HSV':
        img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_H = img_HSV[:,:,0]
        img_S = img_HSV[:,:,1]
        img_V = img_HSV[:,:,2]
        if plot:
            axs[0].imshow(img_H, cmap='gray')
            axs[0].set_title('{} HSV_H'.format(image_name), fontsize=15)
            axs[1].imshow(img_S, cmap='gray')
            axs[1].set_title('{} HSV_S'.format(image_name), fontsize=15)
            axs[2].imshow(img_V, cmap='gray')
            axs[2].set_title('{} HSV_V'.format(image_name), fontsize=15)
        if channel_needed == 'H':
            return img_H
        if channel_needed == 'S':
            return img_S
        if channel_needed == 'V':
            return img_V
        
    if color_space == 'HLS':
        img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_H = img_HLS[:,:,0]
        img_L = img_HLS[:,:,1]
        img_S = img_HLS[:,:,2]
        if plot:
            axs[0].imshow(img_H, cmap='gray')
            axs[0].set_title('{} HLS_H'.format(image_name), fontsize=15)
            axs[1].imshow(img_L, cmap='gray')
            axs[1].set_title('{} HLS_L'.format(image_name), fontsize=15)
            axs[2].imshow(img_S, cmap='gray')
            axs[2].set_title('{} HLS_S'.format(image_name), fontsize=15)
        if channel_needed == 'H':
            return img_H
        if channel_needed == 'L':
            return img_L
        if channel_needed == 'S':
            return img_S
    if channel_needed is None:
        return fig
        
def get_binary_thresholded_image(image, image_name='', out_dir=None):
    """
    Get a binary thresholded image that is tuned for road lane detection feature extraction.
    
    :returns: Binary thresholded image
    """
        
    # convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    height, width = gray.shape
    
    # apply sobel gradient threshold on the horizontal gradient
    sx_binary = apply_sobel_absolute(gray, 'x', 10, 200)
    
    # apply sobel gradient direction threshold so that only edges closer to vertical are detected.
    dir_binary = apply_sobel_direction(gray, thresh=(np.pi/6, np.pi/2))
    
    # combine the gradient and direction thresholds.
    dir_horiz_gradient_combined = ((sx_binary == 1) & (dir_binary == 1))
    
    # Combine R and G pixels and threshold them to detect yellow lanes perfectly
    R = get_color_channels(image, color_space = 'RGB', channel_needed = 'R')
    G = get_color_channels(image, color_space = 'RGB', channel_needed = 'G')
    R_G_threshold = 150
    R_G_condition = (R > R_G_threshold) & (G > R_G_threshold)
    
    # Get S channel and threshold it for detecting bright yellow and white lanes
    S = get_color_channels(image, color_space = 'HLS', channel_needed = 'S')
    S_threshold = (100, 255)
    S_condition = (S > S_threshold[0]) & (S <= S_threshold[1])
    
    # Get L channel and threshold it to avoid pixels which have shadows and as a result darker.
    L = get_color_channels(image, color_space = 'HLS', channel_needed = 'L')
    L_threshold = (120, 255)
    L_condition = (L > L_threshold[0]) & (L <= L_threshold[1])

    # combine all the thresholds from the color channels
    color_channel_combined = np.zeros_like(R)
    # A pixel should either be a yellowish or whiteish
    # And it should also have a gradient, as per our thresholds
    color_channel_combined[(R_G_condition & L_condition) & (S_condition | dir_horiz_gradient_combined)] = 1
    
    # Apply the region of interest mask
    ROI_mask = np.zeros_like(color_channel_combined)
    ROI_vertices = np.array([[0,height-1], [width/2, int(0.5*height)], [width-1, height-1]], dtype=np.int32)
    cv2.fillPoly(ROI_mask, [ROI_vertices], 1)
    thresholded = cv2.bitwise_and(color_channel_combined, ROI_mask)
    
    if out_dir:
         plot_pipeline_step_as_subplots(image, thresholded, image_name,'{}_bin_thresholded'.format(image_name), out_dir, cmap_out='gray')
    return thresholded

def apply_sobel_absolute(gray, orient='x', thresh_min=25, thresh_max=255):
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary

def apply_sobel_direction(gray, sobel_kernel=3, thresh=(0, np.pi/2)):    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    binary_output =  np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return binary_output


def apply_warp(img, image_name='', out_dir=None):
    """
    Apply perspective image on a binary thresholded image
    
    :returns: warped image
    """
    image = img.copy()
    image_size = (image.shape[1], image.shape[0])
    
    offset = 0
    
    #These values for source and destination points are derived by trial and error
    source = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    
    destination = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    
    M = cv2.getPerspectiveTransform(source, destination)
    M_inv = cv2.getPerspectiveTransform(destination, source)
    warped = cv2.warpPerspective(image, M, image_size)
    if out_dir:
        plot_pipeline_step_as_subplots(image, warped, '{}_thresholded'.format(image_name), '{}_warped'.format(image_name), out_dir, cmap_in='gray', cmap_out='gray')
    return (warped, M_inv)