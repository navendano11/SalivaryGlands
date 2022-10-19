from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import SimpleITK as sitk
import skimage.transform as skTrans
import torch


#######################################################################################################
######################################       Saving Slices       ######################################
#######################################################################################################

# Get pixels of Bounding Box in each axis
def bbox2_3D(img):

    bb_array = np.where(img==1) # look position of the values equal to 1
    xmin, xmax = np.min(bb_array[0]) , np.max(bb_array[0]) + 1 # Get min and max position in axial axis
    ymin, ymax = np.min(bb_array[1]), np.max(bb_array[1]) + 1 # Get min and max position in coronal axis
    zmin, zmax = np.min(bb_array[2]) , np.max(bb_array[2]) + 1 # Get min and max position in sagital axis

    return xmin,xmax, ymin,ymax,zmin,zmax

# Get Slices of the Data
def Slicing_Dataset( root_dir ,split = 'train', cut_head=True, net2 = False, side= 'right', resample_data = '333', transform = None):
  
    # Get root directory and list of the folders
    root_dir = Path(root_dir)
    folders = os.listdir(root_dir)

    
    # Choosing data for trains, val and test
    if split == 'train':
        files = folders[:28]#[:28]
    elif split == 'val':
        files = folders[28:38]#[28:38]
    elif split == 'test':
        files = folders[38:48]#[38:48]
    
    ct_slices= []
    rsg_slices = []
    lsg_slices = []
    image_name_slice = []
    
    # Going thru each Volume path and BB, and Mask
    for image_folder_sample in  files:

        # Loading CT Data
        path_ct = root_dir /image_folder_sample/'NII'/'Resample'/resample_data/'ct.nii.gz'
        
        volume_ct = sitk.ReadImage(str(path_ct))
        
        ct = sitk.GetArrayFromImage(volume_ct)
        ct = np.flip(ct,0)
    
        # Loading Bounding Box data
        path_bb_rsg = root_dir /image_folder_sample/'NII'/'Resample'/resample_data/'Bounding Box'/'rsg_bb.nii.gz'
        path_bb_lsg = root_dir /image_folder_sample/'NII'/'Resample'/resample_data/'Bounding Box'/'lsg_bb.nii.gz'       
        
        volume_bb_rsg = sitk.ReadImage(str(path_bb_rsg))
        volume_bb_lsg = sitk.ReadImage(str(path_bb_lsg))
                
        bb_rsg = sitk.GetArrayFromImage(volume_bb_rsg)
        bb_rsg = np.flip(bb_rsg,0)
        bb_lsg = sitk.GetArrayFromImage(volume_bb_lsg)
        bb_lsg = np.flip(bb_lsg,0)

         # Cutting the head and neck region for Network 1
        if cut_head:
            # Finding min and max values on each axis
            if resample_data == '111':
                axial_ax_max = 228
                coronal_ax_min = np.ceil(ct.shape[1]/2).astype(int) - 192
                coronal_ax_max = np.ceil(ct.shape[1]/2).astype(int) + 192
                sagital_ax_min = np.ceil(ct.shape[2]/2).astype(int) - 192
                sagital_ax_max = np.ceil(ct.shape[2]/2).astype(int) + 192

            elif resample_data == '333':

                # if ct.shape[1]<=129:
                #     ct = ct[:76,:128,:128]
                #     bb_rsg = bb_rsg[:76,:128,:128]
                #     bb_lsg = bb_lsg[:76,:128,:128]
                # else:
                #     ct = ct[:76,15:143,20:148]
                #     bb_rsg = bb_rsg[:76,15:143,20:148]
                #     bb_lsg = bb_lsg[:76,15:143,20:148] 

                axial_ax_max = 76
                coronal_ax_min = np.ceil(ct.shape[1]/2).astype(int) - 64
                coronal_ax_max = np.ceil(ct.shape[1]/2).astype(int) + 64
                sagital_ax_min = np.ceil(ct.shape[2]/2).astype(int) - 64
                sagital_ax_max = np.ceil(ct.shape[2]/2).astype(int) + 64

            #Cutting the values
            ct = ct[:axial_ax_max,coronal_ax_min:coronal_ax_max,sagital_ax_min:sagital_ax_max]
            bb_rsg = bb_rsg[:axial_ax_max,coronal_ax_min:coronal_ax_max,sagital_ax_min:sagital_ax_max]
            bb_lsg = bb_lsg[:axial_ax_max,coronal_ax_min:coronal_ax_max,sagital_ax_min:sagital_ax_max]

            if split == 'train' and transform == True:
                ct_flip = ct.copy()
                ct_flip = np.flip(ct_flip,2)
                bb_lsg_flip = bb_rsg.copy()
                bb_lsg_flip = np.flip(bb_lsg_flip,2)
                bb_rsg_flip = bb_lsg.copy()
                bb_rsg_flip = np.flip(bb_rsg_flip,2)


        # Choosing the data for the Second Network for the pixelwise segmentation
        if net2:
            
            # Loading pixel-wise annotated data
            path_rsg = root_dir /image_folder_sample/'NII'/'Resample'/resample_data/'Labels'/'rsg.nii.gz'
            path_lsg = root_dir /image_folder_sample/'NII'/'Resample'/resample_data/'Labels'/'lsg.nii.gz'
            
            volume_rsg = sitk.ReadImage(str(path_rsg))
            volume_lsg = sitk.ReadImage(str(path_lsg))
            
            rsg = sitk.GetArrayFromImage(volume_rsg)
            rsg = np.flip(rsg,0)
            lsg = sitk.GetArrayFromImage(volume_lsg)
            lsg = np.flip(lsg,0)

            # Get position of bounding box, ROI to segment in Network 2
            if side == 'right':
                xmin,xmax, ymin,ymax,zmin,zmax = bbox2_3D(bb_rsg)
            elif side == 'left':
                xmin,xmax, ymin,ymax,zmin,zmax = bbox2_3D(bb_lsg)
            
            ct = ct[xmin:xmax, ymin:ymax,zmin:zmax]
            rsg = rsg[xmin:xmax, ymin:ymax,zmin:zmax]
            lsg = lsg[xmin:xmax, ymin:ymax,zmin:zmax]       
        
        # bb_psg = bb_rsg + bb_lsg
 
        # Appending all data in just one list - Save data in 2D slices
        for i in range(ct.shape[0]):
            ct_slices.append(ct[i,:,:])
            image_name_slice.append([image_folder_sample+"_"+str(i)])  
            if net2:
                rsg_slices.append(rsg[i,:,:])
                lsg_slices.append(lsg[i,:,:])
            else:
                rsg_slices.append(bb_rsg[i,:,:])
                lsg_slices.append(bb_lsg[i,:,:])
            # bb_psg_slices.append(bb_psg[i,:,:])

            if split == 'train' and transform == True:
                ct_slices.append(ct_flip[i,:,:])
                rsg_slices.append(bb_rsg_flip[i,:,:])
                lsg_slices.append(bb_lsg_flip[i,:,:])
                image_name_slice.append([image_folder_sample+"_flip"+str(i)])

    return ct_slices, rsg_slices, lsg_slices, image_name_slice



#######################################################################################################
#####################################      Load and Save Data     #####################################
#######################################################################################################

class SalivaryGlandsDataset(Dataset):
    def __init__(self, root_dir = '../MICCAI/Samples', split = 'train', transform = None, 
                 cut_head = True, net2 = False, side = 'right', resample ='333'):
        
        assert split in ['train', 'val', 'test']  # Verify that the split value is correct
        assert side in ['right', 'left']  # Verify that the side to segment value is correct
        
        self.root_dir = Path(root_dir)
        self.folders = os.listdir(self.root_dir)
        self.split = split

        # Net2 = True 
        self.net2 = net2
        self.cut_head = cut_head

        # If Net2 is True the cut_head value must be false and otherwise
        if self.net2 == True:
           assert self.cut_head in [False] 
        if self.cut_head == True:
           assert net2 in [False] 

        self.side = side #Side of segmentation
        self.resample = resample #Type of resample data to segment
         
        self.transform = transform
        # Get all the slices in an array
        self.ct,self.rsg, self.lsg, self.image_slice = Slicing_Dataset(root_dir,
                                                                        self.split, 
                                                                        self.cut_head, 
                                                                        self.net2, 
                                                                        self.side, 
                                                                        self.resample, 
                                                                        self.transform)
        
         
        
    def __len__(self):
        return len(self.ct)
    
    # Go thru each slice and save it in a tensor
    def __getitem__(self, index):
        
        ct = self.ct[index]
        rsg = self.rsg[index]
        lsg = self.lsg[index]
        # bb_psg = self.bb_psg[index]
        image_name_slice = self.image_slice[index]

        
        ct_tensor = torch.from_numpy(ct.copy()).unsqueeze(dim=0)
        rsg_tensor = torch.from_numpy(rsg.copy()).unsqueeze(dim=0)
        lsg_tensor = torch.from_numpy(lsg.copy()).unsqueeze(dim=0)
        # bb_psg_tensor = torch.from_numpy(bb_psg.copy()).unsqueeze(dim=0)

        return {'ct': ct_tensor, 'mask_rsg': rsg_tensor, 'mask_lsg': lsg_tensor, 'img_name': image_name_slice}
    