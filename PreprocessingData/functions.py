import SimpleITK as sitk
import numpy as np

def resample_img(image, new_spacing=[1,1,1]):
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)

    orig_size = np.array(image.GetSize(), dtype=int)
    orig_spacing = image.GetSpacing()
    new_size = np.array([x * (y / z) for x, y, z in zip(orig_size, orig_spacing, new_spacing)])
    new_size = np.floor(new_size).astype(int)
    new_size = [int(s) for s in new_size]
    resampler.SetSize(new_size)

    isotropic_img = resampler.Execute(image)
    return isotropic_img

def bbox2_3D(img, out_bound_min=np.zeros(3).astype(int), out_bound_max=np.zeros(3).astype(int)):

    bb_array = np.where(img==1)

    xmin, xmax = np.min(bb_array[0])-out_bound_min[0] , np.max(bb_array[0]) + 1+ out_bound_max[0]
    ymin, ymax = np.min(bb_array[1])-out_bound_min[1] , np.max(bb_array[1]) + 1+ out_bound_max[1]
    zmin, zmax = np.min(bb_array[2])-out_bound_min[2] , np.max(bb_array[2]) + 1+ out_bound_max[2]
    
    BB = np.zeros_like(img)
    BB[xmin:xmax, ymin:ymax,zmin:zmax]=1
    bb_shape = np.array(BB[xmin:xmax, ymin:ymax,zmin:zmax].shape).astype(int)
    return BB,bb_shape

def bb_outbound_difference(original_bb_shape, final_bb_shape):
    
    out_bound = np.ceil(abs(final_bb_shape - original_bb_shape)/2).astype(int)
    out_bound_min =np.zeros(3).astype(int)
    out_bound_max =np.zeros(3).astype(int)
    residual = (final_bb_shape - original_bb_shape) %2 == 0
    
    if residual.any():
        out_bound_min[residual] = out_bound[residual]
        out_bound_max[residual] = out_bound[residual] 

    if np.invert(residual).any():
        out_bound_min[np.invert(residual)] = out_bound[np.invert(residual)]
        out_bound_max[np.invert(residual)] = out_bound[np.invert(residual)] - 1
    return out_bound_min,out_bound_max