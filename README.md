# uncanny

## Description 
The Canny Edge Detector is a very popular and reliable method for detecting edges within images. This project is an implementation of this algorithm as well as an implementation of the Harris Corner Detector for detecting corners.

## How it Works

### Canny Edge Detector

First thing that needs to happen is to convert the iamge to grayscale. This is because the algorithm needs the image to be in one channel. 

Original Image:

![Orig](/screenshots/orig_im.PNG)

Next we need a kernel to convolve to image with. For this project I used a Gaussian kernel that was spatially separated to make it faster to compute. Then we end up with Ix and Iy which represent the intensity of the gradient in the x and  y directions. In order to calucalte the gradient intensity matrix we caculate the magnitude of the gradients at each point. 

Image after convolution of gaussian gradient:

![Gradient Conv](/screenshots/gauss_grad.PNG)

As you can see it looks like it detected the edges correctly. Now we want to thin out the edge lines by using non-maximum suppression. This works by checking interpolated pixels along the gradient line and only keeping the largest one.

Image after non-maximum suppression:

![Non Max](/screenshots/nm_sup.PNG)

Next we need to only keep the strongest edges by using hysteresis thresholding. This works by defining a high value threshold and a low value threshold. If an edge is less than the low value than it is discarded, if it is larger than the high value then we can set it to 255. If an edge is weak, meaning it lies between the two thresholds then we keep it if it is connected to another strong edge and we discard it if not.

Image after Thresholding:

![Thresh](/screenshots/threshold.PNG)

### Harris Corner Detector

This works by using a window to detect differences of the gradients from one pixel to the next. At each pixel we construct a matrix based on the x and y gradients of the image. Using this matrix we can calculate the R score of each pixel using the determinant of the matrix minus the square of the trace times a constant k. A large R score means that both the eigencalues of the matrix are large so there it is a corner. If R is very negative then one of the eigenvalues is a lot larger than the other so it is an edge. After we have the R scores we define some threshold to check the R score against. If it is greater than the threshold then it is determined to be a corner. 

Image after Corner Detector:

![Corner](/screenshots/corners.PNG)

## How to Use

This project comes with a set of arguments that need to be given to the program for it to work.

| Command | Description |
| --- | --- |
| -f | The filname of the image to perform the algorithms on |
| -s | Sigma value of the gaussian kernel. Larger sigma results in less detail of the edges. |
| -L | Low value used for the hysteresis thresholding |
| -H | High value used for the hysteresis thresholding |
| -R | Threshold for the R score |
| -S | Optional argument. Defines the size of the kernel. Must be an odd number. |

