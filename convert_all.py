import os
import subprocess

# Define the base directory and output directory
base_dir = r"C:\Users\r.joshi\Downloads\01_11_2024"  # Change this to your actual base directory
output_dir = r"C:\Users\r.joshi\Downloads\processed_data"  # Change this to your desired output directory
script_path = r"convert_to_nifti.py"  # Adjust if needed (e.g., full path to script)

# Walk through the directory
for root, dirs, files in os.walk(base_dir):
    if root.endswith(("L", "V")) and "COMBI" in root:
        if not files:
            print(f'Empty folder {root}')
            continue

        print(f"Processing: {root}")  # Optional: Print the directory being processed
        try:
            command = ["python", script_path, root, output_dir]
            # print("Executing:", " ".join(command))
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error processing {root}: {e}\n')


print('Data conversion completed!')
