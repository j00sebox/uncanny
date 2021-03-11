import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import ndimage
import math
import cv2
from skimage.exposure import rescale_intensity

# perform convolution on an image with a given filter
def convolve(im, kernel):
    
    imHeight = im.shape[1]
    imWidth = im.shape[0]

    kHeight = kernel.shape[1]
    kWidth = kernel.shape[0]

    # if the kernel has a dimension of 1 then we shouldn't subtract from the padding
    if (kWidth == 1):
        padx = 1
    else:
        padx = kWidth-1
    
    if (kHeight == 1):
        pady = 1
    else:
        pady = kHeight-1

    # this adds zeros around the image so we can keep the original dimensions
    im = cv2.copyMakeBorder(im, padx, padx, pady, pady, cv2.BORDER_CONSTANT, 0)

    res = np.zeros( (imWidth, imHeight) )

    # move the window along the x direction until it reaches the end
    for x in np.arange(0, imWidth):

        for y in np.arange(0, imHeight):

            # current window of interest
            window = im[x:x+padx+1, y:y+pady+1]

            res[x, y] = (window * kernel).sum()
    
    
    res = rescale_intensity( res, in_range=(0, 255) )
    res = res * 255
	
    return res

# construct a gaussian gradient kernel of a certain size
# return the spatially separated kernel of the gaussian derivative
def gaussian_kernel_d(sigma, sz):

    rng = math.floor(sz/2)

    y,x = np.ogrid[-rng:rng+1, -rng:rng+1]

    normal = 1 / (2*np.pi*sigma**2)

    dGx = -( x / sigma**2 ) * np.exp( -(x**2) / (2.0 * sigma**2) )
    dGy = -( y / sigma**2 ) * np.exp( (y**2) / (2.0 * sigma**2) )

    return dGx, dGy

# thins out edge lines
def non_max_spr(G, theta):

    W, H = G.shape[:2]

    res = np.zeros( (W, H) )

    # convert angle matrix to degrees
    theta = theta * 180.0 / np.pi

    tolerance = 25
    halfT = tolerance/2

    for i in range(0, W-1):
        for j in range(0, H-1):

            try:
                # these are the interpolated pixels on the line that are in front and behind the pixel being checked
                q = 0
                r = 0
                
                # angle is around 0 degrees
                if( (0 <= theta[i][j] < tolerance) or ( (180-tolerance) <= theta[i][j] < 180) ):
                    q = G[i, j+1]
                    r = G[i, j-1]
                # angle is around 45 degrees
                elif( 45-halfT <= theta[i][j] < 45+halfT ):
                    q = G[i+1][j-1]
                    r = G[i-1][j+1]
                # angle is around 90 degrees
                elif( 90-halfT <= theta[i][j] < 90+halfT ):
                    q = G[i+1][j]
                    r = G[i-1][j]
                # angle is around 135
                elif( 135-halfT <= theta[i][j] < 135+halfT ): 
                    q = G[i-1][j-1]
                    r = G[i+1][j+1]
                
                # if the pixel in question is greater than the iterpolated pixels then it will retain value
                if( G[i,j] >= q and G[i,j] >= r ):
                    res[i][j] = G[i][j]
                else:
                    res[i][j] = 0
            
            except IndexError as err:
                pass
    
    return res

# determines which pixels matter the most
def thresholding(img, L, H):

    M, N = img.shape[:2]

    for i in range(0, M-1):
        for j in range(0, N-1):

            try:

                # if it is in between the threshold values then check surrounding pixels
                # if any of the surrounding pixels are greater than the high threshold then the current pixel can become apart of the strong edge
                if(L <= img[i, j] and img[i, j] <= H):
                    if( (img[i+1, j-1] >= H) or (img[i+1, j] >= H) or (img[i+1, j+1] >= H) 
                        or (img[i, j-1] >= H) or (img[i, j+1] >= H)
                        or (img[i-1, j-1] >= H) or (img[i-1, j] >= H) or (img[i-1, j+1] >= H) ):
                        img[i, j] = 255
                    else:
                        img[i, j] = 0
                # if a pixel is below threshold then it can be discarded
                elif(img[i, j] <= L):
                    img[i, j] = 0
                elif(img[i, j] >= H):
                    img[i, j] = 255

            except IndexError as err:
                pass
    
    return img

def main():
    parser = argparse.ArgumentParser()

    # filname of image 
    parser.add_argument('-f', action='store', dest='fname', help='Image to detect edges on.', required=True)

    # user inputs what sigma value they want to use for Gaussian kernel
    parser.add_argument('-s', action='store', dest='sigma', type=float, help='Sigma value for the Gaussian kernel.', required=True)

    # low value for hysteresis thresholding
    parser.add_argument('-L', action='store', dest='low', type=int, help='Lower end of the threshold.', required=True)

    # high value for hysteresis thresholding
    parser.add_argument('-H', action='store', dest='high', type=int, help='Higher end of the threshold.', required=True)

    # desired dimensions of the kernel
    parser.add_argument('-S', action='store', dest='size', type=int, default=5, help='Size of the Gaussian kernel. Default is 5x5 kernel.')

    args = parser.parse_args()

    if(args.low >= args.high):
        raise ValueError("High value threshold must be greater than low value!")

    if(args.high > 255):
        raise ValueError("High threshold cannot be greater than 255!")

    if(args.low < 0):
        raise ValueError("Low threshold cannot be lower than 0!")

    img = Image.open(args.fname)

    # need image to be gray scale for algorithm to work
    img = np.asarray( ImageOps.grayscale(img) )

    # get spatially separated kernel
    dGx, dGy = gaussian_kernel_d(args.sigma, args.size)

    # get x and y gradients
    Ix = convolve(img, dGx)
    Iy = convolve(img, dGy)

    # calclate the magnitudes of the gradient at each pixel
    dG = np.sqrt(Ix**2 + Iy**2)

    # this calculates the angle of the gradient at each pixel
    angles = np.arctan2(Iy, Ix)

    # we only want the largest value along the gradient to be visible
    nm = non_max_spr(dG, angles)

    # exacts the most important pixels based on the threshold values
    thresholding(nm, args.low, args.high)

    pilImg = Image.fromarray(nm)

    pilImg.show()


if __name__ == '__main__':
    main()