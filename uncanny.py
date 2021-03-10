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

    pad = kWidth-1

    im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)

    res = np.zeros( (imWidth, imHeight) )

    for x in np.arange(0, imWidth):

        for y in np.arange(0, imHeight):

            window = im[x:x+pad+1, y:y+pad+1]

            res[x, y] = (window * kernel).sum()
    
    
    res = rescale_intensity( res, in_range=(0, 255) )
    res = res * 255
	
    return res

# construct a gaussian gradient kernel of a certain size
def gaussian_kernel_d(sz=5, sigma=5):

    rng = math.floor(sz/2)

    y,x = np.ogrid[-rng:rng+1, -rng:rng+1]

    normal = 1 / (2*np.pi*sigma**2)

    dGx = -( x / sigma**2 ) * np.exp( -(x**2 + y**2) / (2.0 * sigma**2) )
    dGy = -( y / sigma**2 ) * np.exp( -(x**2 + y**2) / (2.0 * sigma**2) )

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


def main():
    parser = argparse.ArgumentParser()

    # filname of image 
    parser.add_argument('-f', action='store', dest='fname', help='Image to detect edges on.')

    # user inputs what sigma value they want to use for Gaussian kernel
    parser.add_argument('-s', action='store', dest='sigma', help='Sigma value for the Gaussian kernel.')

    # desired dimensions of the kernel
    parser.add_argument('-S', action='store', dest='size', help='Size of the Gaussian kernel.')

    args = parser.parse_args()

    img = Image.open(args.fname)

    img = np.asarray( ImageOps.grayscale(img) )

    dGx, dGy = gaussian_kernel_d()

    Ix = convolve(img, dGx)
    Iy = convolve(img, dGy)

    dG = np.sqrt(Ix**2 + Iy**2)

    angles = np.arctan2(Iy, Ix)

    nm = non_max_spr(dG, angles)

    pilImg = Image.fromarray(nm)

    pilImg.show()


if __name__ == '__main__':
    main()