import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import skimage.transform as skTrans
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import os
from polyaxon_client.tracking import Experiment

#from sklearn.utils import resample
from functions import *

def resample_bb(new_sampling,path, files, outbound = 2):

    ####################
    ### Resampling
    ####################

    max_bb_r =np.zeros(3).astype(int)
    max_bb_l =np.zeros(3).astype(int)

    complete_data = []
    #print(files)
    #print(path)
    for i in files:
        #print('hello world')
        dirname = path+"/"+i+"/NII"
        path_ct = dirname+"/img.nii.gz"
        path_rsg = dirname+ "/Labels/Parotid_R.nii.gz"
        path_lsg = dirname+ "/Labels/Parotid_L.nii.gz"
        #print(dirname)
        # Read NIFTI imgs
        volume = sitk.ReadImage(path_ct)
        volume_labelR = sitk.ReadImage(path_rsg)
        volume_labelL = sitk.ReadImage(path_lsg)
        

        # Resample Imgs
        vol_ct_isotropic = resample_img(volume, new_sampling)
        vol_rsg_isotropic = resample_img(volume_labelR,new_sampling)
        vol_lsg_isotropic = resample_img(volume_labelL,new_sampling)

        # Convert img into array
        ct = sitk.GetArrayFromImage(vol_ct_isotropic)
        rsg = sitk.GetArrayFromImage(vol_rsg_isotropic)
        lsg = sitk.GetArrayFromImage(vol_lsg_isotropic)

        
        # Get Bounding Box from annotation
        rsg_bb, shape_rsg_bb = bbox2_3D(rsg)
        lsg_bb, shape_lsg_bb = bbox2_3D(lsg)
        
        bool_bb_r = max_bb_r< shape_rsg_bb
        bool_bb_l = max_bb_l< shape_lsg_bb
    
        if bool_bb_r.any():
            max_bb_r[bool_bb_r] = shape_rsg_bb[bool_bb_r]

        if bool_bb_l.any():
            max_bb_l[bool_bb_l] = shape_lsg_bb[bool_bb_l]

            
        complete_data.append({'id': i, 'ct': ct, 'rsg': rsg, 'lsg': lsg, 'bb_rsg':rsg_bb, 'bb_lsg':lsg_bb,'bb_rsg_shape': shape_rsg_bb, 'bb_lsg_shape': shape_lsg_bb})

    
    # Convert odd number shape in even number
    is_odd_bb_r =  max_bb_r % 2 == 1
    is_odd_bb_l =  max_bb_l % 2 == 1

    if is_odd_bb_r.any():
        max_bb_r[is_odd_bb_r] = max_bb_r[is_odd_bb_r]+1
        
    if is_odd_bb_l.any():
        max_bb_l[is_odd_bb_l] = max_bb_l[is_odd_bb_l]+1

    # Setting final bb shape
    final_outbound = np.ones(3)*outbound

    if new_sampling == [1,1,1]:
        bb_r_final_shape = max_bb_r + final_outbound.astype(int) + [0, 6, 12]
        bb_l_final_shape = max_bb_l + final_outbound.astype(int) + [0,4,8]
    else:
        bb_r_final_shape = max_bb_r + final_outbound.astype(int) + [0, 6, 10]
        bb_l_final_shape = max_bb_l + final_outbound.astype(int) + [0,4,8]


    print(bb_r_final_shape)
    print(bb_l_final_shape)

    
    resample_folder = [str(item) for item in new_sampling]
    resample_folder = "".join(resample_folder)
    resample_folder = str(int(resample_folder))
    print(resample_folder)

    # Setting same bounding box for all items

    for i in complete_data:#[43:49]:
        img_id = i['id'] 
        
        dirname = path+"/"+img_id+"/NII"
        #print(dirname)
        ct = i['ct']  
        rsg = i['rsg']  
        lsg = i['lsg'] 
        bb_rsg = i['bb_rsg'] 
        bb_lsg = i['bb_lsg'] 
        bb_rsg_shape = i['bb_rsg_shape']  
        bb_lsg_shape = i['bb_lsg_shape'] 

    
        out_bound_rsg_min, out_bound_rsg_max = bb_outbound_difference(bb_rsg_shape, bb_r_final_shape)
        out_bound_lsg_min, out_bound_lsg_max = bb_outbound_difference(bb_lsg_shape, bb_l_final_shape)
        
        out_bound_rsg_min[0] = 0
        out_bound_rsg_max[0] = 0
        
        out_bound_lsg_min[0] = 0
        out_bound_lsg_max[0] = 0

        bb_r_final,bb_r_shape = bbox2_3D(bb_rsg, out_bound_rsg_min , out_bound_rsg_max )
        bb_l_final,bb_l_shape = bbox2_3D(bb_lsg, out_bound_lsg_min , out_bound_lsg_max )


        # Save bounding box in NIFTI
        volume_rsg_bb = sitk.GetImageFromArray(bb_r_final)
        volume_lsg_bb = sitk.GetImageFromArray(bb_l_final)
        
        volume_ct = sitk.GetImageFromArray(ct)
        volume_rsg = sitk.GetImageFromArray(rsg)
        volume_lsg = sitk.GetImageFromArray(lsg)
        
    
            
        # create folder Resampling
        if not os.path.exists(dirname + "/Resample"):
            os.makedirs(dirname + "/Resample")
        if not os.path.exists(dirname + "/Resample/"+resample_folder):
            os.makedirs(dirname + "/Resample/"+resample_folder)

        # create folder Bounding Box
        if not os.path.exists(dirname + "/Resample/"+resample_folder+"/Bounding Box"):
            os.makedirs(dirname + "/Resample/"+resample_folder+"/Bounding Box")
        # create folder Labels
        if not os.path.exists(dirname + "/Resample/"+resample_folder+"/Labels"):
            os.makedirs(dirname + "/Resample/"+resample_folder+"/Labels")


        sitk.WriteImage(volume_ct, dirname + "/Resample/"+resample_folder+"/ct.nii.gz")
        sitk.WriteImage(volume_rsg, dirname + "/Resample/"+resample_folder+"/Labels/rsg.nii.gz")
        sitk.WriteImage(volume_lsg, dirname + "/Resample/"+resample_folder+"/Labels/lsg.nii.gz")   

        sitk.WriteImage(volume_rsg_bb, dirname + "/Resample/"+resample_folder+"/Bounding Box/rsg_bb.nii.gz")
        sitk.WriteImage(volume_lsg_bb, dirname + "/Resample/"+resample_folder+"/Bounding Box/lsg_bb.nii.gz") 
        
        


# ---------------------
# RUN Resampling and bounding box
# ---------------------
path = '/natalia_ap/original_data/HeadNeck_MICCAI2015/' #'MICCAI/Samples'
experiment = Experiment()
data_paths = experiment.get_data_paths()

files = os.listdir(data_paths['data1']+path)

resample_bb(new_sampling = [1,1,1], path = data_paths['data1']+path, files=files,outbound = 2)