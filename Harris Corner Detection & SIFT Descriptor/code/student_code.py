import numpy as np
import cv2 # You must not use cv2.cornerHarris()
# You must not add any other library


### If you need additional helper methods, add those. 
### Write details description of those

"""
  Returns the harris corners,  image derivative in X direction,  and 
  image derivative in Y direction.
  Args
  - image: numpy nd-array of dim (m, n, c)
  - window_size: The shaps of the windows for harris corner is (window_size, wind)
  - alpha: used in calculating corner response function R
  - threshold: For accepting any point as a corner, the R value must be 
   greater then threshold * maximum R value. 
  - nms_size = non maximum suppression window size is (nms_size, nms_size) 
    around the corner
  Returns 
  - corners: the list of detected corners
  - Ix: image derivative in X direction
  - Iy: image derivative in Y direction

"""
def harris_corners(image, window_size=5, alpha=0.04, threshold=1e-2,
                  nms_size=10):

    ### YOUR CODE HERE
    image = cv2.GaussianBlur(image,(9,9),3)
   
    Ix = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    Ixx = cv2.GaussianBlur(Ix*Ix,(window_size,window_size),5)
    Iyy = cv2.GaussianBlur(Iy*Iy,(window_size,window_size),5)
    Ixy = cv2.GaussianBlur(Ix*Iy,(window_size,window_size),5)
    
    det_M=Ixx*Iyy-(Ixy)**2
    trace_M=Ixx+Iyy
    R=det_M-alpha*(trace_M)**2
    R_thresh=np.max(R)*threshold
    R[R<R_thresh]=0
    
    for i in range(len(R)-nms_size):
        for j in range(len(R[1,:])-nms_size):
            temp=R[i:i+nms_size, j:j+nms_size]
            temp[temp!=np.max(temp)]=0
            
    corners = R
    return corners, Ix, Iy

"""
  Creates key points form harris corners and returns the list of keypoints. 
  You must use cv2.KeyPoint() method. 
  Args
  - corners:  list of Normalized corners.  
  - Ix: image derivative in X direction
  - Iy: image derivative in Y direction
  - threshold: only select corners whose R value is greater than threshold
  
  Returns 
  - keypoints: list of cv2.KeyPoint
  
  Notes:
  You must use cv2.KeyPoint() method. You should also pass 
  angle of gradient at the corner. You can calculate this from Ix, and Iy 

"""
def get_keypoints(corners, Ix, Iy, threshold):
    
    ### YOUR CODE HERE
    keypoints=[]
   
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if (corners[i,j]>threshold):
                keypoints.append(cv2.KeyPoint(j,i,1,np.degrees(np.arctan(Iy[i,j]/Ix[i,j])),corners[i,j],0,-1))
    return keypoints

def bin_2_dec(a,b):
        return (2*a)+b

def get_features(image, keypoints, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    #assert len(x) == len(y)
    assert feature_width==16

    image=np.pad(image,((8,8),(8,8)))
    Ix = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    fv=[]
      
    for key in keypoints:
        descriptor=np.zeros(128)
        key_x=int(key.pt[1])
        key_y=int(key.pt[0])
        for x in range(-8,8):
            for y in range(-8,8):
                i=x+8
                j=y+8
                grad_mg=np.sqrt(Ix[key_x+x+8,key_y+y+8]**2+Iy[key_x+x+8,key_y+y+8]**2)
                grad_angle=np.rad2deg(np.arctan2(Iy[key_x+x+8,key_y+y+8], Ix[key_x+x+8,key_y+y+8]))%360
                bd1=int(bin_2_dec(np.floor(i/8),np.floor(j/8))*32)
                bd2=int(bin_2_dec(np.floor((i-(np.floor(i/8)*8))/4),np.floor((j-(np.floor(j/8)*8))/4))*8)
                grad=int(np.floor(grad_angle/45))
                descriptor[bd1+bd2+grad] += grad_mg
        fv.append(descriptor)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv