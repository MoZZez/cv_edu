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


def add_pads(img,ker_size):
    
    ker_rad_y = ker_size[0] // 2
    ker_rad_x = ker_size[1] // 2
    
    shape = (img.shape[0]+ker_rad_y*2,img.shape[1]+ker_rad_x*2,img.shape[2])
    
    with_pads = np.zeros(shape,dtype=np.uint8)
    
    vert_pads_shape = (with_pads.shape[0],ker_rad_y,with_pads.shape[2])
    hor_pads_shape = (ker_rad_x,with_pads.shape[1],with_pads.shape[2])
    
    top_add = np.zeros(hor_pads_shape,dtype=np.uint8)
    top_add[:,ker_rad_x:-ker_rad_x] = np.array(img[:ker_rad_y])
    #print(top_add[:,ker_rad_x:-ker_rad_x].shape,img[-ker_rad_y:].shape)
   
    
    bot_add = np.zeros(hor_pads_shape,dtype=np.uint8)
    bot_add[:,ker_rad_x:-ker_rad_x] = np.array(img[-ker_rad_y:])
    
    
    left_add = np.zeros(vert_pads_shape,dtype=np.uint8)
    left_add[ker_rad_y:-ker_rad_y,:] = np.array(img[:,:ker_rad_x])
    
    right_add = np.zeros(vert_pads_shape,dtype=np.uint8)
    right_add[ker_rad_y:-ker_rad_y,:] = np.array(img[:,-ker_rad_x-1:-1])
    
    with_pads[:,:ker_rad_x] = left_add
    with_pads[:,-ker_rad_x-1:-1] = right_add
    
    with_pads[:ker_rad_y,:] = top_add
    with_pads[-ker_rad_y-1:-1,:] = bot_add
    
    with_pads[ker_rad_y:-ker_rad_y,ker_rad_x:-ker_rad_x] = np.array(img)
    
    return with_pads

def cut_pads(img, ker_size):
    ker_rad_y = ker_size[0] // 2
    ker_rad_x = ker_size[1] // 2
    
    img_cut = np.array(img[ker_rad_y:-ker_rad_y,ker_rad_x:-ker_rad_x])
    return img_cut

def convolve(img, kernel, median=False):
    padded_img = add_pads(img, kernel.shape)
    #cv2.imshow('original',padded_img)
    #cv2.imshow('cut',out)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    ker_rad_y = kernel.shape[0] // 2
    ker_rad_x = kernel.shape[1] // 2
    
    out = np.zeros(padded_img.shape, dtype=float)
    img = img.astype(float)
    kernel = np.expand_dims(kernel,axis=2)
    #center_cords = [ker_rad_y,ker_rad_x]
    
    max_x_to = 0
    for i in range(ker_rad_y, img.shape[0]+ker_rad_y):
        for j in range(ker_rad_x,img.shape[1]+ker_rad_x):
                center_cords = [i,j]
            
                y_from = center_cords[0] - ker_rad_y
                y_to = center_cords[0] + ker_rad_y
                
                x_from = center_cords[1] - ker_rad_x
                x_to = center_cords[1] + ker_rad_x
                if median == True:
                    new_value = np.median(padded_img[y_from:y_to+1, x_from:x_to+1],axis=(0,1))
                else:
                    if x_to > max_x_to:
                        max_x_to = x_to
                    conv_res = kernel * padded_img[y_from:y_to+1, x_from:x_to+1]
                    new_value = np.sum(conv_res,axis=(0,1))
                #print('Performing dilation!')
                out[i,j] = new_value
    out = out.astype(np.uint8)
    #print(max_x_to,img.shape[1],padded_img.shape[1])
    out = cut_pads(out,kernel.shape)
    #cv2.imshow('original',out_cut)
    #cv2.imshow('cut',out)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return out

def get_gauss_kernel(gamma):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    #not really sure how to handle one-cell-kernel convolution
    if gamma < 2:
        gamma = 3
    l = int(gamma)
    sig = gamma / 3

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)


def blur(img,mode='avg',kernel_side=3, gamma=3.):
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
        dst = convolve(img, kernel) 
        
    elif mode == 'gauss':
        gamma = float(gamma)
        kernel = get_gauss_kernel(gamma)
        dst = convolve(img, kernel)
        
    elif mode == 'median':
        kernel = np.ones((kernel_side,kernel_side), dtype=float)
        dst = convolve(img, kernel, median=True)
    return dst