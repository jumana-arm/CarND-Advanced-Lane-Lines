import numpy as np
import matplotlib.pyplot as plt
import cv2
from util import image_process as ip

def plot_histogram(warped, image_name='', out_dir=None):
    '''
    Returns histogram peaks along horizontal axis of image, giving one each for left and right half.
    '''
    binary_warped =  warped.copy()
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Peak in the first half indicates the likely position of the left lane
    half_width = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:half_width])

    # Peak in the second half indicates the likely position of the right lane
    rightx_base = np.argmax(histogram[half_width:]) + half_width

    if out_dir:
        plt.figure(figsize=(10,12))
        plt.plot(histogram)
        out_title = '{}_histogram'.format(image_name)
        plt.savefig('{}/{}.jpg'.format(out_dir, out_title))
    return(leftx_base, rightx_base)


def polyfit_sliding_window(warped, image_name='', out_dir=None):
    '''
    Returns pixel points within the search area(left and right half of image) performing a sliding window search on a warped image.
    Also returns a second order polynomial that fits the pixel points identified in the two regions.
    '''
    if out_dir:
        plt.figure(figsize=(10,12))
    binary_warped = warped.copy()
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    leftx_base, rightx_base = plot_histogram(binary_warped)
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
          # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                (0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit_p, right_fit_p = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit_p = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_p = np.polyfit(righty, rightx, 2)
        
    if out_dir:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit_p[0]*ploty**2 + left_fit_p[1]*ploty + left_fit_p[2]
        right_fitx = right_fit_p[0]*ploty**2 + right_fit_p[1]*ploty + right_fit_p[2]
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        out_title = '{}_slidingwindow'.format(image_name)
        plt.savefig('{}/{}.jpg'.format(out_dir, out_title))

    return (left_fit_p, right_fit_p, left_lane_inds, right_lane_inds)

def polyfit_using_previous_fit(warped, left_fit, right_fit, image_name='', out_dir=None):
    '''
    Returns pixel points within the search area(left and right half of image) performing a pixel on a warped image,
    around region spsecified by the fitting polynomials from previous fit.
    Also returns a second order polynomial that fits the pixel points identified in the two regions.
    '''
    if out_dir:
        plt.figure(figsize=(10,12))
        
    binary_warped = warped.copy()
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit_p, right_fit_p = (None, None)
    if len(leftx) != 0:
        left_fit_p = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_p = np.polyfit(righty, rightx, 2)
        
    if out_dir:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit_p[0]*ploty**2 + left_fit_p[1]*ploty + left_fit_p[2]
        right_fitx = right_fit_p[0]*ploty**2 + right_fit_p[1]*ploty + right_fit_p[2]       
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        out_title = '{}_previousfit'.format(image_name)
        plt.savefig('{}/{}.jpg'.format(out_dir, out_title))
   
    return (left_fit_p, right_fit_p, left_lane_inds, right_lane_inds)

def measure_radius_of_curvature(bin_img, left_fit_p, right_fit_p, left_lane_inds, right_lane_inds):
    '''
    Returns radius of curvature of lanes in the input binary image
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
    
     # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials to x,y in world space
    if len(leftx) != 0 and len(rightx) != 0:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        curve_rad = (left_curverad + right_curverad)/2
    else:
        curve_rad = 0
    return curve_rad

def measure_offset_from_center(bin_img, left_fit_p, right_fit_p): 
    '''
    Returns offset distance from center for lanes in the input binary image
    '''
    if left_fit_p is not None and right_fit_p is not None:
        # compute the offset from the center
        h = bin_img.shape[0]
        l_fit_x = left_fit_p[0]*h**2 + left_fit_p[1]*h + left_fit_p[2]
        r_fit_x = right_fit_p[0]*h**2 + right_fit_p[1]*h + right_fit_p[2]
        lane_center = (r_fit_x + l_fit_x) /2
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        center_offset_pixels = abs(bin_img.shape[1]/2 - lane_center)
        center_offset = xm_per_pix*center_offset_pixels
    else:
        center_offset = 0
    return center_offset


def shade_lane_area(img, binary_img, l_fit, r_fit, Minv, image_name='', out_dir=None):
    '''
    Shade the lane area in the original lane image.
    '''
    if l_fit is None or r_fit is None:
        return img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    if out_dir:
        plt.figure(figsize=(10,12))
        out_title = '{}_shadedlane'.format(image_name)
        plt.savefig('{}/{}.jpg'.format(out_dir, out_title))
    return result

def add_distance_text_to_image(img, curv_rad, center_offset, image_name='', out_dir=None):
    '''
    Adds distance measurements as text on top of a lane shaded image
    '''
    height = img.shape[0]
    font = cv2.FONT_HERSHEY_PLAIN
    text = 'Radius of Curvature: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(img, text, (40,70), font, 3, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_offset > 0:
        direction = 'right'
    elif center_offset < 0:
        direction = 'left'
    abs_center_offset = abs(center_offset)
    text = '{:04.3f}'.format(abs_center_offset) + 'm ' + direction + ' of center'
    cv2.putText(img, text, (40,120), font, 3, (200,255,155), 2, cv2.LINE_AA)
    if out_dir:
        out_title = '{}_shadedlane_curvature'.format(image_name)
        plt.savefig('{}/{}.jpg'.format(out_dir, out_title))
    return img

    