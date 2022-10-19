# SalivaryGlandsSegmentation

The architecture is implemented in [MICCAI challenge 2015](https://www.imagenglab.com/newsite/pddca/) data set

## Preprocessing Data

- Code to change the format from nrrd -> NIFTI (apply in local not in polyaxon): [01. Change img format.ipynb](https://gitlab.lrz.de/CAMP_IFL/salivaryglandssegmentation/-/blob/main/PreprocessingData/notebook/01.%20Change%20img%20format.ipynb)
- Resampling and Bounding Box Generation (apply in polyaxon): [Resample_BB.py](https://gitlab.lrz.de/CAMP_IFL/salivaryglandssegmentation/-/blob/main/PreprocessingData/Resample_BB.py)


## Salivary Glands Architecture

- [Data Generator](https://gitlab.lrz.de/CAMP_IFL/salivaryglandssegmentation/-/blob/main/SalivaryGlandSegmentation/src/data/data_generator.py): Creation of 2D slices from 3D CT slices.
    - Network 1 - BB Segmentation: Cropping face of the 2D CT slice.
    - Network 2 -  Pixel Wise Segmentation: Cropping bounding box coordinates in the CT slice.
- [Segmentation](https://gitlab.lrz.de/CAMP_IFL/salivaryglandssegmentation/-/blob/main/SalivaryGlandSegmentation/src/models/segmentation.py): Unet apply for the segmentation of the BB and pixel wise. More [information](https://gitlab.lrz.de/CAMP_IFL/salivaryglandssegmentation/-/blob/main/Presentation/03._Final_Presentation_2.0.pdf).

