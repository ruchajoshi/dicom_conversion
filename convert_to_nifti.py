import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import nibabel as nib

def load_dicom_series(dicom_folder):
    """Load and sort DICOM image slices from a folder."""
    dicom_files = []
    for file in os.listdir(dicom_folder):
        path = os.path.join(dicom_folder, file)
        try:
            dicom = pydicom.dcmread(path)
            if hasattr(dicom, "ImagePositionPatient"):  # Only keep image slices
                dicom_files.append(dicom)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    dicom_files.sort(key=lambda d: d.ImagePositionPatient[2])  # Sort by slice position
    return dicom_files

def load_rtss(rtss_path):
    """Load the RT Structure Set (RTSS) file."""
    return pydicom.dcmread(rtss_path)

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

                if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                    pixel_points.append((x, y))

            if pixel_points:  # Only proceed if there are valid points
                pixel_points = np.array(pixel_points, np.int32)
                pixel_points = pixel_points.reshape((-1, 1, 2))  # Format for OpenCV

                # Check if the contour is too close to the image boundaries
                min_x, min_y = np.min(pixel_points, axis=0)[0]
                max_x, max_y = np.max(pixel_points, axis=0)[0]

                # If the contour touches the boundaries of the image, skip it
                if min_x <= boundary_margin or min_y <= boundary_margin or \
                   max_x >= image_shape[1] - boundary_margin or max_y >= image_shape[0] - boundary_margin:
                    print(f"Skipping contour near boundary (min_x: {min_x}, min_y: {min_y}, max_x: {max_x}, max_y: {max_y})")
                    continue  # Skip this contour

                # Fill the polygon into the mask
                cv2.fillPoly(mask, [pixel_points], 255)

    return mask

def save_as_nifti(image_data, output_path, affine=np.eye(4)):
    """Save the image data as a NIfTI file."""
    img = nib.Nifti1Image(image_data, affine)
    nib.save(img, output_path)

# Paths
dicom_folder = r"C:\Users\r.joshi\Downloads\01_11_2024\7139000004\COMBI\L"
rtss_path = r"C:\Users\r.joshi\Downloads\01_11_2024\7139000004\COMBI\L\rtss.dcm"

# Load DICOM series and RTSS
dicom_series = load_dicom_series(dicom_folder)
rtss = load_rtss(rtss_path)

# Find the shape of the first DICOM slice to use as reference
reference_slice = dicom_series[0]
reference_shape = reference_slice.pixel_array.shape

# Create the DICOM volume, resizing slices to match the reference shape
dicom_volume = []
for dicom in dicom_series:
    slice_data = dicom.pixel_array
    if slice_data.shape != reference_shape:
        # Resize slices to match the reference shape (using interpolation)
        slice_data = cv2.resize(slice_data, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # If slice has an extra channel (e.g., (634, 719, 93)), we reduce it to (634, 719)
    if len(slice_data.shape) == 3 and slice_data.shape[2] > 1:
        slice_data = slice_data[:, :, 0]  # Take only the first channel (usually grayscale)
    
    dicom_volume.append(slice_data)

# Ensure all slices are of the same shape by checking the final shapes
final_shapes = [slice.shape for slice in dicom_volume]
if len(set(final_shapes)) > 1:
    print("Warning: Slices have inconsistent shapes after resizing. You may need to adjust the resizing logic.")
    print("Shapes of slices: ", final_shapes)

# Convert list of slices to a numpy array (3D volume)
dicom_volume = np.array(dicom_volume)

# Save the original DICOM volume as a NIfTI file
output_dicom_path = os.path.join(dicom_folder, "original_dicom_volume.nii.gz")
save_as_nifti(dicom_volume, output_dicom_path)
print(f"Saved original DICOM volume to {output_dicom_path}")

# Extract contour data for all structures
contours_per_structure = get_contour_data(rtss)

# Create and save masks for all structures
for structure_name, contours_per_slice in contours_per_structure.items():
    print(f"Processing structure: {structure_name}")
    
    # Initialize mask_volume with the shape of the first slice as reference
    mask_volume = []
    slice_shape = dicom_series[0].pixel_array.shape  # Shape of the first slice

    for dicom in dicom_series:
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

    # Save the mask as a NIfTI file
    output_mask_path = os.path.join(dicom_folder, f"{structure_name}_mask.nii.gz")
    save_as_nifti(mask_volume, output_mask_path)
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
