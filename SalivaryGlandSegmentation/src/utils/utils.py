import os
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import nibabel as nib
from tf_metrics_mod import ssim, psnr
from tensorflow.python.ops import math_ops
from scipy.ndimage import rotate, shift 
import matplotlib.pyplot as plt

#from train import train

def scheduler(epoch, lr):

    if epoch < 4:
        lr = lr

    else:
        k = 0.1
        lr = lr * tf.math.exp(-k)

    tf.summary.scalar('learning rate', data=lr, step=epoch)
    
    return lr

# this is function to load weight file for resuming training from last epoch (https://github.com/sanpreet/Loading-saved-keras-model-and-continue-training-from-last-epoch/blob/526f4fcb9e7219e530e3a550909decc78ca7e086/load_saved_keras_model.py#L38)
def checkpoint_function(checkpoint_path):
    # if there is no file being saved in the checkpoint path then routine will go to loop model.fit(X, y, epochs=150, batch_size=1, callbacks=[ckpt_callback])
    # this signifies that no file is saved in the checkpoint path and let us begin the training from the first epoch.
    if not os.listdir(checkpoint_path):
        return
    # this loop will fetch epochs number in a list    
    files_int = list()
    for i in os.listdir(checkpoint_path):
        epoch = int(i.split('epochs:')[1].split('-')[0])
        files_int.append(epoch)
    # getting the maximum value for an epoch from the list
    # this is reference value and will help to find the file with that value    
    max_value = max(files_int)
    # conditions are applied to find the file which has the maximum value of epoch
    # such file would be the last file where the training is stopped and we would like to resume the training from that point.
    for i in os.listdir(checkpoint_path):
        epoch = int(i.split('epochs:')[1].split('-')[0])
        if epoch == max_value:
            final_file = i    
    # in the end we will get the epoch as well as file         
    return final_file, max_value

# @tf.function
def normalize_intensity_4metrics(img, mask):

    if mask.shape != img.shape:
        if tf.is_tensor(img):
            mask = tf.repeat(mask, tf.shape(img)[-1], axis=-1) 
        else:
            mask = np.repeat(mask, img.shape[-1], axis=-1) 
    
    # Calculate max and min value only inside the brain 
    MAX, MIN =   np.max(img[(mask != 0)])\
    ,   np.min(img[(mask != 0)]) 
    
    # Normalize the image
    img_norm = (img - MIN) / (MAX - MIN)

    # Multiply with brain mask to put voxels outside the brain to zero
    img_norm_mask = img_norm * mask

    assert np.sum(img_norm_mask[mask==0]) == 0, "background contains non-zero values"
    assert np.sum(img_norm_mask*mask) == np.sum(img_norm_mask), "masked image don't match foreground data"
    # Return image with voxels outside the brain mask to zero, and inside the brain normalized between 0 and 1
    return img_norm_mask 

def normalize_intensity_4loss(img, mask):

    if mask.shape != img.shape:
        mask = tf.repeat(mask, tf.shape(img)[-1], axis=-1) 
        
    # Calculate max and min value only inside the brain 
    MAX, MIN =   tf.math.reduce_max(img[(mask != 0)])\
    ,   tf.math.reduce_min(img[(mask != 0)]) 
    
    # Normalize the image
    img_norm = (img - MIN) / (MAX - MIN)

    # Multiply with brain mask to put voxels outside the brain to zero
    img_norm_mask = img_norm * mask

    # assert tf.math.reduce_sum(img_norm_mask[mask==0]) == 0, "background contains non-zero values"
    # assert tf.math.reduce_sum(img_norm_mask*mask) == tf.math.reduce_sum(img_norm_mask), "masked image don't match foreground data"
    # Return image with voxels outside the brain mask to zero, and inside the brain normalized between 0 and 1
    return img_norm_mask 

############################### 
#  Metrics functions
###############################

# @tf.function
def SSIM_metric_DWI(y_true, y_pred):
    mask = y_true[:, :, :, :, -1:]
    y_true = y_true[:, :, :, :, 1:-1]
    y_true_masked_norm = normalize_intensity_4metrics(y_true,mask)
    y_pred = y_pred[:, :, :, :, 1:-1]
    y_pred_masked_norm = normalize_intensity_4metrics(y_pred, mask)
    
    part1 = 0
    part2 = 0

    #val = tf.reduce_mean(tf.image.ssim(y_true_masked_norm, y_pred_masked_norm, 1.0))
    for vol in range(0,y_true.shape[-1]):
        # Use the modified version of the ssim function from tf that retrieves a ssim value per voxel
        ssim_pervoxel = ssim(y_true_masked_norm[:,:,:,:,vol], y_pred_masked_norm[:,:,:,:,vol], 1.0)
        kernel = np.ones((11,11,1,1))
        kernel /= (11*11)
        
        conv_mask = tf.nn.conv2d(tf.transpose(np.squeeze(mask, axis=-1), [3,1,2,0]),kernel,strides=[1, 1, 1, 1], padding='VALID')
        conv_mask = tf.cast(tf.transpose(conv_mask, [3,1,2,0]), tf.float32)
        
        # Multiply by the mask
        ssim_mask = ssim_pervoxel*conv_mask
        
        # Calculate the mean value inside the brain as the sum over all volumes / (number of voxels inside de brain mask * #volumes)
        part1 += math_ops.reduce_sum(ssim_mask)
        part2 += math_ops.reduce_sum(conv_mask)

    mean_ssim_inBrain = math_ops.divide(part1, part2)

    return mean_ssim_inBrain

# @tf.function
def SSIM_metric_B0(y_true, y_pred):
    ''' Implementation inside the brain '''
    mask = np.squeeze(y_true[:, :, :, :, -1:],axis=-1)
    y_true = y_true[:, :, :, :, 0]
    y_true_masked_norm = normalize_intensity_4metrics(y_true, mask)
    y_pred = y_pred[:, :, :, :, 0]
    y_pred_masked_norm = normalize_intensity_4metrics(y_pred,mask)
    
    #val = tf.reduce_mean(tf.image.ssim(y_true_masked_norm, y_pred_masked_norm, 1.0))
    
    # Use the modified version of the ssim function from tf that retrieves a ssim value per voxel
    ssim_pervoxel = ssim(y_true_masked_norm, y_pred_masked_norm, 1.0)
    kernel = np.ones((11,11,1,1))
    kernel /= (11*11)
    conv_mask = tf.nn.conv2d(tf.transpose(mask, [3,1,2,0]),kernel,strides=[1, 1, 1, 1], padding='VALID')
    # Since image is smaller due to the windows used for calculations,pad the image to get same size as for mask
    #pad_ssim = tf.pad(ssim_pervoxel,paddings=[[0, 0],[5, 5],[5, 5],[0, 0]], mode='SYMMETRIC')
    # Multiply by the mask
    conv_mask = tf.cast(tf.transpose(conv_mask, [3,1,2,0]), tf.float32)
    ssim_mask = ssim_pervoxel*conv_mask
    # Calculate the mean value inside the brain as the sum over all volumes / (number of voxels inside de brain mask * #volumes)
    mean_ssim_inBrain = math_ops.divide( math_ops.reduce_sum(ssim_mask),\
          math_ops.reduce_sum(conv_mask))

    return mean_ssim_inBrain

def SSIM_metric_wholeImage_B0(y_true, y_pred):
    ''' Naive Implementation with tensorflow'''
    mask = np.squeeze(y_true[:, :, :, :, -1:],-1)
    y_true_masked = y_true[:, :, :, :, 0] * mask
    y_true_masked_norm = normalize_intensity_4metrics(y_true_masked,mask)
    y_pred_masked = y_pred[:, :, :, :, 0] * mask
    y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked,mask)
    
    return tf.reduce_mean(tf.image.ssim(y_true_masked_norm, y_pred_masked_norm, 1.0))

def SSIM_metric_wholeImage_DWI(y_true, y_pred):
    mask = y_true[:, :, :, :, -1:]
    y_true_masked = y_true[:, :, :, :, 1:-1] * mask
    y_true_masked_norm = normalize_intensity_4metrics(y_true_masked,mask)
    y_pred_masked = y_pred[:, :, :, :, 1:-1] * mask
    y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked,mask)
    
    return tf.reduce_mean(tf.image.ssim(y_true_masked_norm, y_pred_masked_norm, 1.0))


# @tf.function
def PSNR_metric_DWI(y_true, y_pred):
    mask = y_true[:, :, :, :, -1:]
    y_true_masked = y_true[:, :, :, :, 1:-1] * mask
    y_true_masked_norm = normalize_intensity_4metrics(y_true_masked,mask)
    y_pred_masked = y_pred[:, :, :, :, 1:-1] * mask
    y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked,mask)

    # Calculate psnr per voxel with the modified psnr function
    psnr_inBrain_dwi = psnr(y_true_masked_norm, y_pred_masked_norm, max_val=1.0)
    
    return tf.reduce_mean(psnr_inBrain_dwi)

# @tf.function
def PSNR_metric_B0(y_true, y_pred):
    mask = np.squeeze(y_true[:, :, :, :, -1:],-1)
    y_true_masked = y_true[:, :, :, :, 0] * mask
    y_true_masked_norm = normalize_intensity_4metrics(y_true_masked,mask)
    y_pred_masked = y_pred[:, :, :, :, 0] * mask
    y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked,mask)

    # Calculate psnr per voxel with the modified psnr function
    psnr_inBrain = psnr(y_true_masked_norm, y_pred_masked_norm, max_val=1.0)
    
    return tf.reduce_mean(psnr_inBrain)

def PSNR_metric_wholeImage_B0(y_true, y_pred):
    mask = np.squeeze(y_true[:, :, :, :, -1:],-1)
    y_true_masked = y_true[:, :, :, :, 0] * mask
    y_true_masked_norm = normalize_intensity_4metrics(y_true_masked,mask)
    y_pred_masked = y_pred[:, :, :, :, 0] * mask
    y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked,mask)
    
    return tf.reduce_mean(tf.image.psnr(y_true_masked_norm, y_pred_masked_norm, max_val=1.0))

def PSNR_metric_wholeImage_DWI(y_true, y_pred):
    mask = y_true[:, :, :, :, -1:]
    y_true_masked = y_true[:, :, :, :, 1:-1] * mask
    y_true_masked_norm = normalize_intensity_4metrics(y_true_masked,mask)
    y_pred_masked = y_pred[:, :, :, :, 1:-1] * mask
    y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked,mask)
    
    return tf.reduce_mean(tf.image.psnr(y_true_masked_norm, y_pred_masked_norm, max_val=1.0))


def average_differenceImage(x_input, y_pred):
    residuals = x_input[:,:,:,:,:-1] - y_pred[:,:,:,:,:-1]
    
    return np.mean(residuals)

# Calculate mean and std of the predictions
@tf.function
def std_predictionB0(y_true, y_pred):
    mask = tf.squeeze(y_true[:, :, :, :, -1:],-1)
    # Multiply by the mask
    y_pred_masked = y_pred[:, :, :, :, 0] * mask

    # Calculate the mean value inside the brain as the sum over all volumes / (number of voxels inside de brain mask * #volumes)
    means = math_ops.reduce_mean(y_pred_masked, axis=-1, keepdims=True)
    diff = y_pred_masked - means
    squared_deviations = math_ops.square(diff)

    part1 = math_ops.reduce_sum(  math_ops.reduce_sum(squared_deviations, axis=-1))
    part2 = math_ops.reduce_sum(tf.image.convert_image_dtype(mask, y_pred.dtype))
    
    mean_inBrain =   math_ops.divide(  part1, part2)

    return mean_inBrain

@tf.function
def mean_predictionB0(y_true, y_pred):
    mask = tf.squeeze(y_true[:, :, :, :, -1:],4)
    # Multiply by the mask
    y_pred_masked = y_pred[:, :, :, :, 0] * mask

    # Calculate the mean value inside the brain as the sum over all volumes / (number of voxels inside de brain mask * #volumes)
    part1 = math_ops.reduce_sum(  math_ops.reduce_sum(y_pred_masked, axis=-1))
    part2 = math_ops.reduce_sum(tf.image.convert_image_dtype(mask, y_pred.dtype))
    mean_inBrain =   math_ops.divide(  part1, part2)

    return mean_inBrain

@tf.function
def std_predictionDWI(y_true, y_pred):
    mask = y_true[:, :, :, :, -1:]
    # Multiply by the mask
    y_pred_masked = y_pred[:, :, :, :, 1:-1] * mask

    # Calculate the mean value inside the brain as the sum over all volumes / (number of voxels inside de brain mask * #volumes)
    means = math_ops.reduce_mean(y_pred_masked, axis=-1, keepdims=True)
    diff = y_pred_masked - means
    squared_deviations = math_ops.square(diff)

    part1 = math_ops.reduce_sum( math_ops.reduce_sum(squared_deviations, axis=-1))
    part2 = math_ops.multiply(math_ops.reduce_sum(tf.image.convert_image_dtype(mask, y_pred.dtype)), 30)
    
    mean_inBrain =   math_ops.divide(  part1, part2)

    return mean_inBrain

@tf.function
def mean_predictionDWI(y_true, y_pred):
    mask = y_true[:, :, :, :, -1:]
    # Multiply by the mask
    y_pred_masked = y_pred[:, :, :, :, 1:-1] * mask

    # Calculate the mean value inside the brain as the sum over all volumes / (number of voxels inside de brain mask * #volumes)
    part1 = math_ops.reduce_sum(  math_ops.reduce_sum(y_pred_masked, axis=-1))
    part2 = math_ops.multiply(math_ops.reduce_sum(tf.image.convert_image_dtype(mask, y_pred.dtype)),30)
    mean_inBrain =   math_ops.divide(  part1, part2)

    return mean_inBrain

############################### 
#  Data Augmentation functions
###############################

def translateit(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    return shift(image, (int(offset[0]), int(offset[1]), int(offset[2]), 0), order=order, mode='nearest')

def rotateit(image, theta, isseg=False):
    order = 0 if isseg == True else 5
        
    return rotate(image, float(theta), reshape=False, order=order, mode='nearest')

def flipit(image, axes):
    
    if axes[0]:
        image = np.fliplr(image)
    if axes[1]:
        image = np.flipud(image)
    
    return image

def scaleit(image, factor, isseg=False):
    order = 0 if isseg == True else 3

    height, width, depth= image.shape
    zheight             = int(np.round(factor * height))
    zwidth              = int(np.round(factor * width))
    zdepth              = depth

    if factor < 1.0:
        newimg  = np.zeros_like(image)
        row     = (height - zheight) // 2
        col     = (width - zwidth) // 2
        layer   = (depth - zdepth) // 2
        newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = interpolation.zoom(image, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]

        return newimg

    elif factor > 1.0:
        row     = (zheight - height) // 2
        col     = (zwidth - width) // 2
        layer   = (zdepth - depth) // 2

        newimg = zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')  
        
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        return newimg

    else:
        return image

def plotit(self,images):
    f, axes = plt.subplots(3, 2)
    for i, axis, image in zip(range(len(images)), axes.ravel(), images):
        if i == 0 or i == 1:
            axis.imshow(image[:,:,30,0]) 
        elif i == 2 or i == 3:
            axis.imshow(image[:,:,30,0], cmap='gray')
        elif i == 4 or i == 5:
            axis.imshow(image[:,:,30,0]) 
    plt.show()

    # f.savefig(f'augmentations_Images/augmentationdata_{self.patientID}')

def data_aug(self,img1, img_mask, img2): 
    ''' DATA AUGMENTATION PARAMETERS
    rotations - theta ∈[−10.0,10.0] degrees
    scaling - factor ∈[0.9,1.1] i.e. 10% zoom-in or zoom-out
    intensity - factor ∈[0.8,1.2] i.e. 20% increase or decrease
    translation - offset ∈[−5,5] pixels
    margin - I tend to set at either 5 or 10 pixels.'''
    original_img1 = img1
    original_img_mask = img_mask
    original_img2 = img2

    np.random.seed()
    numTrans     = np.random.randint(1, 4, size=1)        
    allowedTrans = [0, 3, 4]
    whichTrans   = np.random.choice(allowedTrans, numTrans, replace=False)

    print(f'Doing transformations numbers: {whichTrans}')

    if 0 in whichTrans:
        theta   = float(np.around(np.random.uniform(-10.0,10.0, size=1), 2))
        img1  = rotateit(img1, theta)
        img_mask = rotateit(img_mask, theta, isseg=True)
        img2  = rotateit(img2, theta)

    # if 1 in whichTrans:
    #     scalefactor  = float(np.around(np.random.uniform(0.9, 1.1, size=1), 2))
    #     img1  = scaleit(img1, scalefactor)
    #     img_mask = scaleit(img_mask, scalefactor, isseg=True)
    #     img2  = scaleit(img2, scalefactor)

    # if 2 in whichTrans:
    #     factor  = float(np.around(np.random.uniform(0.8, 1.2, size=1), 2))
    #     thisim  = intensifyit(thisim, factor)
    #     #no intensity change on segmentation

    if 3 in whichTrans:
        axes    = list(np.random.choice(2, 1, replace=True))
        img1  = flipit(img1, axes+[0])
        img_mask = flipit(img_mask, axes+[0])
        img2  = flipit(img2, axes+[0])

    if 4 in whichTrans:
        offset  = list(np.random.randint(-5,5, size=3))
        img1  = translateit(img1, offset)
        img_mask = translateit(img_mask, offset, isseg=True)
        img2  = translateit(img2, offset)
    
    # plotit(self,[original_img1, img1, original_img_mask, img_mask, original_img2, img2])
    
    return img1, img_mask, img2

# def SSIM_metric_DWI(y_true, y_pred):
#     mask = y_true[:, :, :, :, -1:]
#     y_true_masked = y_true[:, :, :, :, 1:-1] * mask
#     y_true_masked_norm = normalize_intensity_4metrics(y_true_masked)
#     y_pred_masked = y_pred[:, :, :, :, 1:-1] * mask
#     y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked)

#     return tf.reduce_mean(tf.image.ssim(y_true_masked_norm, y_pred_masked_norm, 1.0))


# def SSIM_metric_B0(y_true, y_pred):
#     mask = tf.squeeze(y_true[:, :, :, :, -1:],4)
#     y_true_masked = y_true[:, :, :, :, 0] * mask
#     y_true_masked_norm = normalize_intensity_4metrics(y_true_masked)
#     y_pred_masked = y_pred[:, :, :, :, 0] * mask
#     y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked)

#     return tf.reduce_mean(tf.image.ssim(y_true_masked_norm, y_pred_masked_norm, 1.0))

# def PSNR_metric_DWI(y_true, y_pred):
#     mask = y_true[:, :, :, :, -1:]
#     y_true_masked = y_true[:, :, :, :, 1:-1] * mask
#     y_true_masked_norm = normalize_intensity_4metrics(y_true_masked)
#     y_pred_masked = y_pred[:, :, :, :, 1:-1] * mask
#     y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked)

#     return tf.reduce_mean(tf.image.psnr(y_true_masked_norm, y_pred_masked_norm, max_val=1.0))


# def PSNR_metric_B0(y_true, y_pred):
#     mask = tf.squeeze(y_true[:, :, :, :, -1:],4)
#     y_true_masked = y_true[:, :, :, :, 0] * mask
#     y_true_masked_norm = normalize_intensity_4metrics(y_true_masked)
#     y_pred_masked = y_pred[:, :, :, :, 0] * mask
#     y_pred_masked_norm = normalize_intensity_4metrics(y_pred_masked)

#     return tf.reduce_mean(tf.image.psnr(y_true_masked_norm, y_pred_masked_norm, max_val=1.0))
