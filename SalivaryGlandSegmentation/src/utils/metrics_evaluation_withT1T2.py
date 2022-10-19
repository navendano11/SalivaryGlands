import os 
import numpy as np
import nibabel as nib
import qtlib as qtlib
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import models
from utils import PSNR_metric_wholeImage_B0, PSNR_metric_wholeImage_DWI, SSIM_metric_B0, PSNR_metric_B0, SSIM_metric_DWI, PSNR_metric_DWI, SSIM_metric_wholeImage_B0, SSIM_metric_wholeImage_DWI
from pandas import DataFrame


# tf.config.run_functions_eagerly(True)

def metrics2use(numch):

    if numch == 32:  
        metrics = {
            'ssim_dwi': SSIM_metric_DWI,
            'psnr_dwi': PSNR_metric_DWI,
            'ssim_wholeImage_dwi': SSIM_metric_wholeImage_DWI,
            'psnr_wholeImage_dwi': PSNR_metric_wholeImage_DWI,
        }

        results_gt_input = {
            'patient_ID':[],
            'ssim_dwi': [],
            'psnr_dwi': [],
            'ssim_wholeImage_dwi': [],
            'psnr_wholeImage_dwi': [],
        }

        results_gt_pred = {
            'patient_ID':[],
            'ssim_dwi': [],
            'psnr_dwi': [],
            'ssim_wholeImage_dwi': [],
            'psnr_wholeImage_dwi': [],
        }

    elif numch > 3 :
        metrics = {
            'ssim_b0': SSIM_metric_B0,
            'psnr_b0': PSNR_metric_B0,
            'ssim_wholeImage_b0': SSIM_metric_wholeImage_B0,
            'psnr_wholeImage_b0': PSNR_metric_wholeImage_B0,
            'ssim_dwi': SSIM_metric_DWI,
            'psnr_dwi': PSNR_metric_DWI,
            'ssim_wholeImage_dwi': SSIM_metric_wholeImage_DWI,
            'psnr_wholeImage_dwi': PSNR_metric_wholeImage_DWI,
        }

        results_gt_input = {
            'patient_ID':[],
            'ssim_b0': [],
            'psnr_b0': [],
            'ssim_dwi': [],
            'psnr_dwi': [],
            'ssim_wholeImage_b0': [],
            'psnr_wholeImage_b0': [],
            'ssim_wholeImage_dwi': [],
            'psnr_wholeImage_dwi': [],
        }

        results_gt_pred = {
            'patient_ID':[],
            'ssim_b0': [],
            'psnr_b0': [],
            'ssim_dwi': [],
            'psnr_dwi': [],
            'ssim_wholeImage_b0': [],
            'psnr_wholeImage_b0': [],
            'ssim_wholeImage_dwi': [],
            'psnr_wholeImage_dwi': [],
        }

    else: 

        metrics = {
            'ssim': SSIM_metric_B0,
            'psnr': PSNR_metric_B0,
            'ssim_wholeImage': SSIM_metric_wholeImage_B0,
            'psnr_wholeImage': PSNR_metric_wholeImage_B0,
        }

        results_gt_input = {
            'patient_ID':[],
            'ssim': [],
            'psnr': [],
            'ssim_wholeImage': [],
            'psnr_wholeImage': [],
        }

        results_gt_pred = {
            'patient_ID':[],
            'ssim': [],
            'psnr': [],
            'ssim_wholeImage': [],
            'psnr_wholeImage': [],
        }

    return metrics, results_gt_input, results_gt_pred

def save_csv(dir, set_measured, results_gt_input, results_gt_pred):
    
    df_in = pd.DataFrame.from_dict(results_gt_input) 
    df_in.to_csv(os.path.join(dir,f'{set_measured}-metrics-input_gt.csv'))

    if results_gt_pred is not None:
        df_pred =  pd.DataFrame.from_dict(results_gt_pred) 
        df_pred.to_csv(os.path.join(dir,f'{set_measured}-metrics-pred_gt.csv'))
        print(f'Average metrics for Prediction vs Ground Truth: {df_pred.mean()}')
        print(f'Standard Dev metrics for Prediction vs Ground Truth:  {df_pred.std()}')

    ######################################################
    #####  Mean & std of GT vs Input
    print(f'Average metrics for Input vs Ground Truth: {df_in.mean()}')
    print(f'Standard Dev metrics for Input vs Ground Truth:  {df_in.std()}')

    # print(f'Average metrics for Prediction vs Ground Truth: {df_pred.mean()}')
    # print(f'Standard Dev metrics for Prediction vs Ground Truth:  {df_pred.std()}')


def evaluate_metrics(data_generator, set_measured, residual_learning, out_path, checkpoint, channels_in,\
    model = None, mask_block_train = None, imgres_block_train = None ):


    if model is None:
        # ============ Load Checkpoint ============

        print('[INFO] Loading checkpoint name ' + checkpoint)

        dncnn = models.load_model(checkpoint,
                                    custom_objects={'mean_squared_error_weighted': qtlib.mean_squared_error_weighted,
                                                    'mean_gt_minus_pred': qtlib.mean_gt_minus_pred,
                                                    'mean_absolute_error_weighted': qtlib.mean_absolute_error_weighted})
        # dncnn = models.load_model(checkpoint,
        #                             custom_objects={'ssim_L1': qtlib.ssim_L1,
        #                                             'mean_squared_error_weighted': qtlib.mean_squared_error_weighted,
        #                                             'mean_gt_minus_pred': qtlib.mean_gt_minus_pred,
        #                                             'mean_absolute_error_weighted': qtlib.mean_absolute_error_weighted})
    else:

        dncnn = model

    def calculate_metrics(img1,img2, metrics):
        # Get SSIM from the different functions between the GT & Input images
        # DWI 
        values = {}

        for key, metric in metrics.items():

            # print(f'- Doing metric: {key}')

            values[key] = metric(img1, img2)

        return values

    #######################################################
    # ----- Metrics evaluation --------
    #######################################################
    metrics, results_gt_input, results_gt_pred = metrics2use(channels_in)


    for i, sample in enumerate(data_generator):

        if len(sample) > 2: 
            img_block_train = data_generator
            img_block = np.expand_dims(img_block_train[i,:,:,:,:], axis = 0)
            mask_block = np.expand_dims(mask_block_train[i,:,:,:,:], axis = 0)
            imgres_block = np.expand_dims(imgres_block_train[i,:,:,:,:], axis = 0)
        else: 
            img_block = sample[0][0]
            mask_block = sample[0][1]
            # imgres_block = sample[1]

        
        if residual_learning: 
            print("Residual learning Evaluation")
            imgres_block_pred = dncnn.predict([img_block, mask_block], workers=1, use_multiprocessing=False); # predicted residual images
            print(f'\nOutput shape: {imgres_block_pred.shape}')  
            # img_block = img_block[:, :, :, :, :-2] # remove T1 and T2 images for metric calculations
            img_block = img_block[:, :, :, :, :-1] # remove T1 images for metric calculations (klinikum)
            imgres_block_pred = imgres_block_pred[:, :, :, :, :-1] * mask_block; # remove last channel
            img_block_pred = (img_block + imgres_block_pred) * mask_block; # denoised images
            img_block_pred = np.concatenate((img_block_pred,mask_block),axis=-1) # concatenate mask to calculate metrics only inside the brain
            # img_block_gt = img_block + imgres_block[:, :, :, :, :-1]; # ground-truth images
            # img_block_gt = np.concatenate((img_block_gt,mask_block),axis=-1) # concatenate mask to calculate metrics only inside the brain
            
            img_block = np.concatenate((img_block,mask_block),axis=-1)# concatenate mask to calculate metrics only inside the brain
            
        else: 
            print("Normal learning Evaluation")
            img_block_pred = dncnn.predict([img_block, mask_block],  workers=1, use_multiprocessing=False); # predicted residual images
            img_block = img_block[:, :, :, :, :-2] # remove T1 and T2 images for metric calculations
            img_block_pred = img_block_pred[:, :, :, :, :-1] * mask_block; # remove last channel
            img_block_pred = np.concatenate((img_block_pred,mask_block),axis=-1) # concatenate mask to calculate metrics only inside the brain
            img_block_gt = imgres_block[:, :, :, :, :-1]; # ground-truth images
            img_block_gt = np.concatenate((img_block_gt,mask_block),axis=-1) # concatenate mask to calculate metrics only inside the brain
            
            img_block = np.concatenate((img_block,mask_block),axis=-1)# concatenate mask to calculate metrics only inside the brain

        results_gt_input['patient_ID'].append(i)
        # results_gt_pred['patient_ID'].append(i)

        for block in range(0,img_block.shape[0]):

            #results_gt_input['patient_ID'].append(i*img_block.shape[0] + block)
            #results_gt_pred['patient_ID'].append(i*img_block.shape[0] + block)
            
            # img_gt = img_block_gt[block:block+1,:,:,:,:]
            img = img_block[block:block+1,:,:,:,:]
            img_pred = img_block_pred[block:block+1,:,:,:,:]

            print('# Calculating metrics for ground truth and input images')
            # values_input = calculate_metrics(img_gt, img, metrics)
            values_input = calculate_metrics(img, img_pred, metrics)
            print(f'Results for the metrics calculation is {values_input}')

            #tmp_input = {}
            for key, value in values_input.items():
                try:
                    results_gt_input[key].append(value.numpy())
                    #tmp_input[key].append(value.numpy())
                except:
                    results_gt_input[key].append(value)
                    #tmp_input[key].append(value)


            # print('# Calculating metrics for ground truth and predicted images')
            # values_pred = calculate_metrics(img_gt, img_pred, metrics)
            # print(f'Results for the metrics calculation is {values_pred}')

            #tmp_pred = {}
            # for key, value in values_pred.items():
              #   try:
                     # results_gt_pred[key].append(value.numpy())
                    #tmp_pred[key].append(value.numpy())
                # except:
                  #   results_gt_pred[key].append(value)
                    #tmp_pred[key].append(value)
        
        for key in results_gt_input.keys():
            # Compute the mean only considering the metric values of current subject
            mean_input = np.mean(results_gt_input[key][i:])
            # mean_pred = np.mean(results_gt_pred[key][i:])
            # Delete all metric values of current subject
            del results_gt_input[key][i:]
            # del results_gt_pred[key][i:]
            # Append the mean to the dictionary
            results_gt_input[key].append(mean_input)
            # results_gt_pred[key].append(mean_pred)
            #results_gt_input[key].append(np.mean(results_gt_input[key]))
            #results_gt_pred[key].append(np.mean(results_gt_pred[key]))
        
    
    save_csv(out_path, set_measured, results_gt_input, results_gt_pred)

def evaluate_metrics_dwi_sequential(data_generator, set_measured, residual_learning, out_path, checkpoint, channels_in,\
    model = None, mask_block_train = None, imgres_block_train = None ):

    if model is None:
        # ============ Load Checkpoint ============

        print('[INFO] Loading checkpoint name ' + checkpoint)

        dncnn = models.load_model(checkpoint,
                                    custom_objects={'mean_squared_error_weighted': qtlib.mean_squared_error_weighted,
                                                    'mean_gt_minus_pred': qtlib.mean_gt_minus_pred,
                                                    'mean_absolute_error_weighted': qtlib.mean_absolute_error_weighted})
        # dncnn = models.load_model(checkpoint,
        #                             custom_objects={'ssim_L1': qtlib.ssim_L1,
        #                                             'mean_squared_error_weighted': qtlib.mean_squared_error_weighted,
        #                                             'mean_gt_minus_pred': qtlib.mean_gt_minus_pred,
        #                                             'mean_absolute_error_weighted': qtlib.mean_absolute_error_weighted})
    else:

        dncnn = model

    def calculate_metrics(img1,img2, metrics):
        # Get SSIM from the different functions between the GT & Input images
        # DWI 
        values = {}

        for key, metric in metrics.items():

            # print(f'- Doing metric: {key}')

            values[key] = metric(img1, img2)

        return values

    #######################################################
    # ----- Metrics evaluation --------
    #######################################################
    metrics, results_gt_input, results_gt_pred = metrics2use(channels_in)


    for i, sample in enumerate(data_generator):

        
        img_block_dwi = sample[0][0]
        mask_block = sample[0][1]
        # imgres_block_dwi = sample[1]

        # for volume in range(0,img_block_dwi.shape[-1] - 2)
        for volume in range(0,img_block_dwi.shape[-1] - 1):  # for klinikum data is -1 -> only T1 added 
            # Concatenate 1 DWI + T1 + T2 for prediction
            img_block = np.concatenate([img_block_dwi[:,:,:,:,volume:volume+1],img_block_dwi[:,:,:,:,-1:]], axis=-1)
            mask_block = mask_block
            # imgres_block = np.concatenate([imgres_block_dwi[:,:,:,:,volume:volume+1],mask_block], axis=-1)

            print("Residual learning Evaluation DWI Sequential")
            imgres_block_pred = dncnn.predict([img_block, mask_block], workers=1, use_multiprocessing=False); # predicted residual images
            # img_block = img_block[:, :, :, :, :-2] # remove T1 and T2 images for metric calculations
            img_block = img_block[:, :, :, :, :-1] # remove T1 images for metric calculations (klinikum)
            imgres_block_pred = imgres_block_pred[:, :, :, :, :-1] * mask_block; # remove last channel
            img_block_pred = (img_block + imgres_block_pred) * mask_block; # denoised images
            img_block_pred = np.concatenate((img_block_pred,mask_block),axis=-1) # concatenate mask to calculate metrics only inside the brain
            # img_block_gt = img_block + imgres_block[:, :, :, :, :-1]; # ground-truth images
            img_block_gt = np.concatenate((img_block_gt,mask_block),axis=-1) # concatenate mask to calculate metrics only inside the brain
            
            img_block = np.concatenate((img_block,mask_block),axis=-1)# concatenate mask to calculate metrics only inside the brain

            results_gt_input['patient_ID'].append(i)
            # results_gt_pred['patient_ID'].append(i)

            for block in range(0,img_block.shape[0]):
                
                # img_gt = img_block_gt[block:block+1,:,:,:,:]
                img = img_block[block:block+1,:,:,:,:]
                img_pred = img_block_pred[block:block+1,:,:,:,:]

                print('# Calculating metrics for ground truth and input images')
                values_input = calculate_metrics(img, img_pred, metrics)
                print(f'Results for the metrics calculation is {values_input}')

                for key, value in values_input.items():
                    try:
                        results_gt_input[key].append(value.numpy())
                    except:
                        results_gt_input[key].append(value)


                # print('# Calculating metrics for ground truth and predicted images')
                #values_pred = calculate_metrics(img_gt, img_pred, metrics)
                #print(f'Results for the metrics calculation is {values_pred}')

                #for key, value in values_pred.items():
                 #    try:
                   #      results_gt_pred[key].append(value.numpy())
                    # except:
                      #   results_gt_pred[key].append(value)

            for key in results_gt_input.keys():
                # Compute the mean only considering the metric values of current subject
                mean_input = np.mean(results_gt_input[key][i:])
                # mean_pred = np.mean(results_gt_pred[key][i:])
                # Delete all metric values of current subject
                del results_gt_input[key][i:]
                # del results_gt_pred[key][i:]
                # Append the mean to the dictionary
                results_gt_input[key].append(mean_input)
                # results_gt_pred[key].append(mean_pred)
                #results_gt_input[key].append(np.mean(results_gt_input[key]))
                #results_gt_pred[key].append(np.mean(results_gt_pred[key]))

    save_csv(out_path, set_measured, results_gt_input, results_gt_pred)

