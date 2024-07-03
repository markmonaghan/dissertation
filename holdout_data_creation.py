# This code will be used to take 10% of the images from each of the three 'fibrosis', 'no_finding' and 'non_fibrosis' directories and move them to the holdout directories.

import os
import shutil

# Location of the image files
image_base_dir = r'C:\Users\Mark\Documents\rp\nih_dataset'

# List of directories to store the images based on their labels
# The directories are created in the data_organiser.py script
original_directories = ['fibrosis', 'no_finding', 'non_fibrosis']
target_directories = ['fibrosis_holdout', 'no_finding_holdout', 'non_fibrosis_holdout']

def move_images_to_holdout_folders(original_directories, target_directories):
    """
    Function to move 10% of the images from each of the three classes to the holdout folders.
    
    Args:
        original_directories (list): List of directories where the original 'fibrosis', 'no_finding' and 'non_fibrosis' images are stored.
        target_directories (list): List of "holdout" directories where the images will be moved to.
    """
    
    for original_directory, target_directory in zip(original_directories, target_directories):
        original_directory_path = os.path.join(image_base_dir, original_directory)
        target_directory_path = os.path.join(image_base_dir, target_directory)
        if not os.path.exists(target_directory_path):
            os.makedirs(target_directory_path)
        image_files = os.listdir(original_directory_path)
        num_files = len(image_files)
        num_files_to_move = int(num_files * 0.1)
        for i in range(num_files_to_move):
            source_file = os.path.join(original_directory_path, image_files[i])
            target_file = os.path.join(target_directory_path, image_files[i])
            shutil.move(source_file, target_file)
            print(f"Moving file: {image_files[i]} from {original_directory} to {target_directory}")
    print("All images have been moved to the holdout folders.")

# Call the function to move the images to the holdout folders
# move_images_to_holdout_folders(original_directories, target_directories)