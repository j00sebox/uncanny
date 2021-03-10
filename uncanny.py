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

    y,x = np.mgrid[-rng:rng+1, -rng:rng+1]

    normal = 1 / (2*np.pi*sigma**2)

    dGx = -( x / sigma**2 ) * np.exp( -(x**2 + y**2) / (2.0 * sigma**2) )
    dGy = -( y / sigma**2 ) * np.exp( -(x**2 + y**2) / (2.0 * sigma**2) )

    return dGx, dGy

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

    pilImg = Image.fromarray(Ix)

    pilImg.show()


if __name__ == '__main__':
    main()