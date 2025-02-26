Python script to convert the Ultrasound dicom images to nifti format. It converts the volume slice-by-slice and rearranges it together using the spacing mentioned in the dicom. 

Usage:

```
python convert_to_nifti.py /path/to/dicom/folder /path/to/output/folder
```

This script also converts the segmentation annotations to binary segmentation masks for each annotated organ. 

The output is saved in following directory structure

```
output_folder
│
├───images
│       patient_ID.nii.gz
│
├───labels_bladder
│       patient_ID_bladder_mask.nii.gz
│
├───labels_prostate
│       patient_ID_prostate_mask.nii.gz
│
├───labels_rectum
│       patient_ID_rectum_mask.nii.gz
│
└───labels_urethra
        patient_ID_urethra_mask.nii.gz

```