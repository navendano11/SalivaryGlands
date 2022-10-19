# qtlib.py
#
# Utiliti functions for training CNN in DeepDTI.
#
# (c) Qiyuan Tian, Harvard, 2021

import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from utils import normalize_intensity_4metrics, normalize_intensity_4loss
import nibabel as nib

def extract_block(data, inds):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]
    
    blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block))
    
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]
    
    return blocks


def merge_block(data, inds):
    
    data_merged = np.zeros((145, 174, 145, 1))
    
    for ii in np.arange(inds.shape[0]):

        data_merged[inds[ii,0]: inds[ii,1]+1, inds[ii,2]: inds[ii,3]+1, inds[ii,4]: inds[ii,5]+1, :] = data[ii,:,:,:,:]

    return data_merged


def mean_squared_error_weighted_byten(y_true, y_pred):
    ''' Due to small differences between input and gt, we enforced the loss to be higher.
        This step is similar to raise the learning rate.'''
        
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights # use last channel from grouth-truth data to weight loss from each voxel from first n-1 channels
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights

    return 10*K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)


def mean_squared_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights # use last channel from grouth-truth data to weight loss from each voxel from first n-1 channels
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights

    return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)

def mean_absolute_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    
    return K.mean(K.abs(y_pred_weighted - y_true_weighted), axis=-1)

def mean_gt_minus_pred(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights # use last channel from grouth-truth data to weight loss from each voxel from first n-1 channels
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights

    voxels_inside_mask = K.sum(loss_weights)

    val = K.sum(y_pred_weighted - y_true_weighted) / voxels_inside_mask

    return val

def ssim_L1(y_true, y_pred):
    mask = y_true[:, :, :, :, -1:]
    y_true = y_true[:, :, :, :, :-1]
    y_true_masked_norm = normalize_intensity_4loss(y_true,mask)
    y_pred = y_pred[:, :, :, :, :-1]
    y_pred_masked_norm = normalize_intensity_4loss(y_pred, mask)
    

    ssim_loss =  1 - tf.reduce_mean(tf.image.ssim(y_true_masked_norm, y_pred_masked_norm, 1.0))
    L1_loss = K.mean(K.abs(y_true_masked_norm - y_pred_masked_norm), axis=-1) 

    alpha = 0.84
    partA = tf.multiply(alpha, ssim_loss)
    partB = tf.multiply((1-alpha), L1_loss)

    loss = tf.add(partA, partB) 

    return loss