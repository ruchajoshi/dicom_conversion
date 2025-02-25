import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import nibabel as nib

def load_dicom_series(dicom_folder):
    """Load and sort DICOM image slices from a folder and detect the RTSS file."""
    dicom_files = []
    rtss_file = None
    
    for file in os.listdir(dicom_folder):
        path = os.path.join(dicom_folder, file)
        try:
            dicom = pydicom.dcmread(path)
            
            if dicom.Modality == "RTSTRUCT":
                rtss_file = dicom  # Store the RT Structure Set file
            elif hasattr(dicom, "ImagePositionPatient"):  # Only keep image slices
                dicom_files.append(dicom)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    dicom_files.sort(key=lambda d: d.ImagePositionPatient[2])  # Sort by slice position
    return dicom_files, rtss_file

def get_contour_data(rtss, structure_name=None):
    """Extract contour data for a given structure from RTSS. If structure_name is None, get data for all structures."""
    contours_per_structure = {}
    
    for roi in rtss.StructureSetROISequence:
        structure_name = roi.ROIName  # Structure name
        roi_number = roi.ROINumber

        contours_per_slice = {}
        
        # Iterate through the ROIContourSequence to find contour data for the structure
        for roi_contour in rtss.ROIContourSequence:
            if roi_contour.ReferencedROINumber == roi_number:
                
                # Check if 'ContourSequence' exists and extract data if present
                if hasattr(roi_contour, 'ContourSequence'):
                    for contour in roi_contour.ContourSequence:
                        z_pos = round(contour.ContourData[2], 2)  # Z position
                        points = np.array(contour.ContourData).reshape(-1, 3)[:, :2]  # X, Y only
                        contours_per_slice.setdefault(z_pos, []).append(points)

        # If the structure has contours, add it to the results
        if contours_per_slice:
            contours_per_structure[structure_name] = contours_per_slice

    return contours_per_structure

def create_binary_mask(dicom, contours, boundary_margin=5):
    """Convert contours into a binary mask matching DICOM image dimensions."""
    image_shape = dicom.pixel_array.shape
    mask = np.zeros(image_shape, dtype=np.uint8)

    if contours:
        pixel_spacing = dicom.PixelSpacing
        origin = dicom.ImagePositionPatient[:2]
        

        for contour in contours:
            pixel_points = []
            for point in contour:
                x = (point[0] - origin[0]) / pixel_spacing[0]
                y = (point[1] - origin[1]) / pixel_spacing[1]


                # Clip points to stay within image bounds
                x = np.clip(int(x), 0, image_shape[1] - 1)
                y = np.clip(int(y), 0, image_shape[0] - 1)

                pixel_points.append((x, y))

            if pixel_points:  # Only proceed if there are valid points
                pixel_points = np.array(pixel_points, np.int32)
                pixel_points = pixel_points.reshape((-1, 1, 2))  # Format for OpenCV

                # Fill the polygon into the mask
                cv2.fillPoly(mask, [pixel_points], 255)

    return mask

def get_slice_spacing(dicom_series):
    """Calculate slice spacing using ImagePositionPatient."""
    if len(dicom_series) > 1:
        z_positions = [float(d.ImagePositionPatient[2]) for d in dicom_series]
        z_positions.sort()  # Ensure slices are ordered
        slice_spacing = np.mean(np.diff(z_positions))  # Compute mean spacing
    else:
        slice_spacing = 1.0  # Default fallback if only one slice is present
    return slice_spacing

def save_as_nifti(image_data, output_path, affine=np.eye(4)):
    """Save the image data as a NIfTI file."""
    img = nib.Nifti1Image(image_data, affine)
    nib.save(img, output_path)

# Paths
dicom_folder = r"C:\Users\r.joshi\Downloads\01_11_2024\7139000004\COMBI\L"

# Load dicom series and RTSS
dicom_series, rtss = load_dicom_series(dicom_folder)

# Create the DICOM volume, taking only US images
dicom_volume = []
for dicom in dicom_series:
    if dicom.Modality == 'US':
        dicom_volume.append(dicom.pixel_array)


# Convert list of slices to a numpy array (3D volume)
dicom_volume = np.array(dicom_volume)
dicom_volume = np.transpose(dicom_volume, (2, 1, 0))

reference_slice = dicom_series[0]
# Get Pixel Spacing
pixel_spacing = np.array(reference_slice.PixelSpacing, dtype=np.float32)  # [row_spacing, col_spacing]

# Get Slice Spacing
slice_spacing = get_slice_spacing(dicom_series)

# Create Affine Matrix
affine = np.eye(4)
affine[0, 0] = pixel_spacing[0]  # X-axis spacing
affine[1, 1] = pixel_spacing[1]  # Y-axis spacing
affine[2, 2] = slice_spacing     # Z-axis spacing (computed from slice positions)

# Save the original DICOM volume as a NIfTI file
output_dicom_path = os.path.join(dicom_folder, "original_dicom_volume.nii.gz")
save_as_nifti(dicom_volume, output_dicom_path, affine)
print(f"Saved original DICOM volume to {output_dicom_path}")

# Extract contour data for all structures
contours_per_structure = get_contour_data(rtss)

# Create and save masks for all structures
for structure_name, contours_per_slice in contours_per_structure.items():
    print(f"Processing structure: {structure_name}")
    
    # Initialize mask_volume with the shape of the first slice as reference
    mask_volume = []
    slice_shape = reference_slice.pixel_array.shape  # Shape of the first slice

    for dicom in dicom_series:
        if dicom.Modality == 'US':
            z_pos = round(dicom.ImagePositionPatient[2], 2)
            contours = contours_per_slice.get(z_pos, [])

            # Create a binary mask for this slice
            mask = create_binary_mask(dicom, contours)

            # Ensure mask matches the shape of the first slice (in case the dimensions change)
            if mask.shape != slice_shape:
                mask = np.zeros(slice_shape, dtype=np.uint8)  # Fill with zeros if there's a mismatch

            mask_volume.append(mask)

    # Convert list of masks to a numpy array (volume)
    mask_volume = np.array(mask_volume)
    mask_volume = np.transpose(mask_volume, (2, 1, 0))

    # Save the mask as a NIfTI file
    output_mask_path = os.path.join(dicom_folder, f"{structure_name}_mask.nii.gz")
    save_as_nifti(mask_volume, output_mask_path, affine)
    print(f"Saved mask for {structure_name} to {output_mask_path}")

# Optionally, show a sample mask for verification
sample_structure = list(contours_per_structure.keys())[0]
sample_z = list(contours_per_structure[sample_structure].keys())[len(contours_per_structure[sample_structure]) // 2]  # Pick a middle slice
sample_mask = contours_per_structure[sample_structure].get(sample_z, [])
sample_mask_image = create_binary_mask(dicom_series[len(contours_per_structure[sample_structure]) // 2], sample_mask)

plt.subplot(1, 2, 1)
plt.imshow(dicom_series[len(contours_per_structure[sample_structure]) // 2].pixel_array, cmap="gray")
plt.title("DICOM Image")

plt.subplot(1, 2, 2)
plt.imshow(sample_mask_image, cmap="gray")
plt.title("Segmentation Mask")

plt.show()
