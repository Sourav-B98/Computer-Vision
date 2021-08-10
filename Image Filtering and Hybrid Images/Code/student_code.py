import numpy as np
#### DO NOT IMPORT cv2 

def my_imfilter(image, filter):
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (k, k)
    Returns
    - filtered_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
    with matrices is fine and encouraged. Using opencv or similar to do the
    filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
    it may take an absurdly long time to run. You will need to get a function
    that takes a reasonable amount of time to run so that I can verify
    your code works.
    - Remember these are RGB images, accounting for the final image dimension.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    f1,f2=filter.shape
    p1=f1//2
    p2=f2//2
    pad0 = np.pad(image[:,:,0],((p1,p1),(p2,p2)),mode='reflect')
    pad1 = np.pad(image[:,:,1],((p1,p1),(p2,p2)),mode='reflect')
    pad2 = np.pad(image[:,:,2],((p1,p1),(p2,p2)),mode='reflect')
    img_pad2 = np.dstack([pad0,pad1,pad2])
    h2,w2,c=img_pad2.shape
    filtered_image=np.zeros((h2-f1+1,w2-f2+1,3))
    for i in range(p1,h2-p1):
        for j in range(p2,w2-p2):
            filtered_image[i-p1][j-p2][0]=np.sum(img_pad2[i-p1:i+p1+1,j-p2:j+p2+1,0]*filter)
            filtered_image[i-p1][j-p2][1]=np.sum(img_pad2[i-p1:i+p1+1,j-p2:j+p2+1,1]*filter)
            filtered_image[i-p1][j-p2][2]=np.sum(img_pad2[i-p1:i+p1+1,j-p2:j+p2+1,2]*filter)
    ### END OF STUDENT CODE ####
    ############################
    return filtered_image

def create_hybrid_image(image1, image2, filter):
    
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
    in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### TODO: YOUR CODE HERE ###
    low_frequencies=my_imfilter(image1,filter)
    high_frequencies=image2-my_imfilter(image2,filter)
    hybrid_image=low_frequencies+high_frequencies
    ### END OF STUDENT CODE ####
    ############################
    
    return low_frequencies, high_frequencies, hybrid_image
