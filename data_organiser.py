import pandas as pd
import os
import shutil
from tqdm import tqdm

data_dictionary = r'C:\Users\Mark\Documents\rp\nih_xrays_data_dictionary.csv'

df = pd.read_csv(data_dictionary)

# Location of the CXR image files
image_directory = r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_all'

# List of directories to store the images based on their labels
directories = ['fibrosis', 'no_finding', 'non_fibrosis']

def create_directories(directories):
    """
    Function to create directories for the images based on their labels.
    
    Args:
        directories (list): Names of the directories to be created.

    """
    for directory in directories:
        full_path = os.path.join(image_directory, directory)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Directory '{full_path}' created.")
        else:
            print(f"Directory '{full_path}' already exists.")

    print("All directories have been created.")

# Call the function to create the directories
# create_directories(directories)

def count_image_labels(df):
    """
    Function to count the number of images based on their labels.
    
    Args:
        df (DataFrame): DataFrame containing the data dictionary.
    """
    fibrosis_count = 0
    no_finding_count = 0
    non_fibrosis_count = 0

    for index, row in df.iterrows():       
        if 'Fibrosis' in row['Finding Labels']:
            fibrosis_count += 1
        elif 'No Finding' in row['Finding Labels']:
            no_finding_count += 1
        else:
            non_fibrosis_count += 1

    print(f"Number of images with the label 'Fibrosis': {fibrosis_count}")
    print(f"Number of images with the label 'No Finding': {no_finding_count}")
    print(f"Number of images with the label 'Non-Fibrosis': {non_fibrosis_count}")

# Call the function to count the number of images based on their labels
# count_image_labels(df)

# List of directories on the local filesystem where the NIH CXR images are stored
# images_directory = [
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_001\images' 
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_002\images', 
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_003\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_004\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_005\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_006\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_007\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_008\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_009\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_010\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_011\images',
    # r'C:\Users\Mark\Documents\rp\nih_dataset\images_012\images' ]

def process_images_in_directories(image_directory):
    """
    Function to process the images in the directories and move them to the appropriate folders based on their labels.
    
    Args:
        images_directories (list): List of directories where the images are stored.

    """
    for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)   
                
    #     for index, row in df.iterrows():       
    #         if filename == row['Image Index']:
    #             if 'Fibrosis' in row['Finding Labels']:
    #                 # print(f"Moving image: {row['Image Index']} to fibrosis folder")
    #                 shutil.move(image_path, os.path.join(image_directory, 'fibrosis', filename))
    #                 # print(f"Fibrosis image found: {row['Finding Labels']}, {row['Image Index']}, Patient ID: {row['Patient ID']}")
    #             elif 'No Finding' in row['Finding Labels']:
    #                 # print(f"Moving image: {row['Image Index']} to no_finding folder")
    #                 shutil.move(image_path, os.path.join(image_directory, 'no_finding', filename))
    #             else:
    #                 # print(f"Moving image: {row['Image Index']} to non_fibrosis folder")
    #                 shutil.move(image_path, os.path.join(image_directory, 'non_fibrosis', filename))

    # print("All images have been processed.")

# Call the function to process the images
process_images_in_directories(image_directory)