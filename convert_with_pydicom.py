import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

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

# def list_rtss_structures(rtss):
#     """Print all structure names available in the RT Structure Set."""
#     print("\nAvailable structures in RTSS:")
#     for roi in rtss.StructureSetROISequence:
#         print(f" - {roi.ROIName}")

def load_rtss(rtss_path):
    """Load the RT Structure Set (RTSS) file."""
    return pydicom.dcmread(rtss_path)

def get_contour_data(rtss, structure_name):
    """Extract contour data for a given structure from RTSS."""
    roi_number = None
    available_names = {roi.ROIName.lower(): roi.ROIName for roi in rtss.StructureSetROISequence}

    # Find the closest match (case-insensitive)
    if structure_name.lower() in available_names:
        structure_name = available_names[structure_name.lower()]
    else:
        raise ValueError(f"Structure '{structure_name}' not found. Available: {list(available_names.values())}")

    for roi in rtss.StructureSetROISequence:
        if roi.ROIName == structure_name:
            roi_number = roi.ROINumber
            break

    contours_per_slice = {}
    for roi_contour in rtss.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            for contour in roi_contour.ContourSequence:
                z_pos = round(contour.ContourData[2], 2)  # Z position
                points = np.array(contour.ContourData).reshape(-1, 3)[:, :2]  # X, Y only
                contours_per_slice.setdefault(z_pos, []).append(points)

    return contours_per_slice


def create_binary_mask(dicom, contours):
    """Convert contours into a binary mask matching DICOM image dimensions."""
    image_shape = dicom.pixel_array.shape
    mask = np.zeros(image_shape, dtype=np.uint8)

    if contours:
        pixel_spacing = dicom.PixelSpacing
        origin = dicom.ImagePositionPatient[:2]

        for contour in contours:
            pixel_points = []
            for point in contour:
                x = int((point[0] - origin[0]) / pixel_spacing[0])
                y = int((point[1] - origin[1]) / pixel_spacing[1])
                pixel_points.append((x, y))

            cv2.fillPoly(mask, [np.array(pixel_points, np.int32)], 255)

    return mask

# Paths
dicom_folder = r"C:\Users\r.joshi\Downloads\01_11_2024\7139000003\COMBI\L"
rtss_path = r"C:\Users\r.joshi\Downloads\01_11_2024\7139000003\COMBI\L\rtss.dcm"
structure_name = "bladder"

# Load DICOM series and RTSS
dicom_series = load_dicom_series(dicom_folder)
rtss = load_rtss(rtss_path)

# Extract contour data
contours_per_slice = get_contour_data(rtss, structure_name)

# Generate masks for each slice
masks = {}
for dicom in dicom_series:
    z_pos = round(dicom.ImagePositionPatient[2], 2)
    contours = contours_per_slice.get(z_pos, [])
    mask = create_binary_mask(dicom, contours)
    masks[z_pos] = mask

# Display a sample slice
sample_z = list(masks.keys())[len(masks) // 2]  # Pick a middle slice
plt.subplot(1, 2, 1)
plt.imshow(dicom_series[len(masks) // 2].pixel_array, cmap="gray")
plt.title("DICOM Image")

plt.subplot(1, 2, 2)
plt.imshow(masks[sample_z], cmap="gray")
plt.title("Segmentation Mask")

plt.show()
