import os
import pandas as pd
from tqdm import tqdm

# Load the CSV file with image paths and labels
df = pd.read_csv('dataset/stage_2_train_labels.csv')

# Paths to the original image directory and output directory
original_image_dir = './dataset/stage_2_train_images'
output_dir = './dataset_formatted/stage_2_train_images'
notFoundCount = 0
# Iterate through each row in the CSV file
for index, row in tqdm(df.iterrows()):
    image_path = os.path.join(original_image_dir, str(row['patientId']) + '.dcm')
    label = row['Target']

    # Check if the file exists before moving
    if os.path.exists(image_path):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        new_image_path = os.path.join(label_dir, os.path.basename(image_path))
        os.rename(image_path, new_image_path)
    else:
        notFoundCount += 1
        print(f"File not found: {image_path}")  # Optional: Print a message for missing files

print(f"Total number of images: {len(df)}")
print(f"Total number of missing images: {notFoundCount}")
print(f"Total number of images copied: {len(df) - notFoundCount}")

