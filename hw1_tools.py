import numpy as np
import cv2
'''
All funcs for hw1.
'''

def linear_correction(img):
    '''
    Performs linear correction upon a given image
    in np.array uint8 format.
    
    Args:
         img - 2D or 3D numpy array.
    
    Returns:
            np.array of inputs shape and type.
    '''
    shape = img.shape
    img = img.astype(float)
    if len(shape) > 1 :
        maxes = np.amax(img)
        mins = np.amin(img)

        print(mins,maxes)
        corrected = 255*(img - mins)/(maxes - mins)
        corrected = corrected.astype(np.uint8)
    else:
        raise ValueError('Wrong data format.')
    return corrected

def to_grayscale(img):
    '''
    Converts to gray scale.
    
    Args:
         img - 3D numpy array.
    Returns:
            3D numpy array of inputs shape.
    '''
    grayscale = img[:,:,0]*0.2125 + img[:,:,1]*0.7154 + img[:,:,2]*0.0721
    grayscale = grayscale.astype(np.uint8)
    grayscale = np.stack((grayscale,grayscale,grayscale), axis=2)
    
    return grayscale


def crop(img, top_left_point,size):
    '''
    Crops subImage from original image.
    
    Args:
         img - original img 2D or 3D np.array.
         top_left_point - 2-tuple of ints
                          top left corner coodinates of subimage 
                          in original image.
         size - 2-tuple of ints, size of subImage to crop.
    
    Returns:
            2D or 3D numpy array of subImage of size = size.
    '''
    cropped = img[top_left_point[0]:top_left_point[0]+size[0],top_left_point[1]:top_left_point[1]+size[1]]
    return cropped


def blur(img,mode='avg',kernel_side=3):
    '''
    Most naive blur with averaging filter.
    
    Args:
         img - original img 2D or 3D np.array.
         
    Returns:
            2D or 3D numpy array of inputs size.
    '''
    if mode == 'avg':
        kernel = np.ones((kernel_side,kernel_side), dtype=float)
        kernel = kernel/(kernel_side**2)
    dst = cv2.filter2D(img, -1,kernel) 
    
    return dst

def WB(img,mode='ww'):
    
    if mode == 'ww':
        img_gr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(float)
        max_ind = np.unravel_index(img_gr.argmax(), img_gr.shape)
    
        max_pixel = img[max_ind[0],max_ind[1]]
        
        img = (255/max_pixel)*img
    elif mode == 'gw':
        img = img.astype(float)
        means = np.mean(img,axis=(0,1))
        mean = np.mean(means)
        
        img = img * (mean/means)
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype(np.uint8)
    return img

def nonlinear_corr(img,gamma):
    img = img.astype(float)
    img = ((img/255)**gamma)*255
    img = img.astype(np.uint8)
    return img

def invert(img):
    img = 255 - img
    return img