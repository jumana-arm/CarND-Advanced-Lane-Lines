import numpy as np
import glob
import cv2
from util import image_process as ip
import matplotlib.image as mpimg

def calibrate_camera(chess_images_dir, nx, ny, nz, out_dir='', plot = False):
    """
    Calibrates camera with some input chess board images of given size
    
    :chess_images_dir: Inpout directory containing chess board images for calibration
    :nx,ny,nz: 
    :out_dir:
    :plot:
    
    :returns: tuple(CameraMatrix, distortionCoeffs)
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(( nx*ny, nz), np.float32)
    objp[:,:2] = np.mgrid[0:ny, 0:nx].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    calib_images = glob.glob(chess_images_dir)

    # Step through the list and search for chessboard corners
    for idx, in_image in enumerate(calib_images):   
        image_name = in_image.split('/')[2].split('.')[0]

        img = cv2.imread(in_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (ny,nx), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw the corners
            cv2.drawChessboardCorners(img, (ny,nx), corners, ret)
            # Plot
            if plot:
                ip.plot_pipeline_step_as_subplots(mpimg.imread(in_image), img, image_name, '{}_corners'.format(image_name), out_dir)        
        
    image_shape = img.shape
    
    # use the object and image points to caliberate the camera and compute the camera matrix and distortion coefficients
    ret, cameraMatrix, distortionCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[:2],None,None)
    return(cameraMatrix, distortionCoeffs)


