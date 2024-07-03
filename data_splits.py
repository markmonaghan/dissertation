import pandas as pd
import numpy as np
import os
import random

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Define the path to the data dictionary CSV file
data_dictionary = r'C:\Users\Mark\Documents\rp\nih_xrays_data_dictionary.csv'

# Read the CSV file into a DataFrame
all_images = pd.read_csv(data_dictionary)

# Define a function for labeling
def determine_label(finding):
    """
    Function to assign a label based on the 'Finding Labels' column.
    """
    
    if 'Fibrosis' in finding:
        return 'Fibrosis'
    elif 'No Finding' in finding:
        return 'No Finding'
    else:
        return 'Non-Fibrosis'

# Define a function for class assignment
def determine_class(finding):
    """
    Function to assign a class based on the 'Finding Labels' column.
    """
    
    if 'Fibrosis' in finding:
        return 'Fibrosis'
    else:
        return 'Non-Fibrosis'

# Add 2 new columns to the all_images DataFrame
all_images['Label'] = None
all_images['Class'] = None

# Get the index of the 'Finding Labels' column and add the 2 new columns next to it
finding_labels_index = all_images.columns.get_loc('Finding Labels')
columns = list(all_images.columns)
columns.insert(finding_labels_index + 1, columns.pop(columns.index('Label')))
columns.insert(finding_labels_index + 2, columns.pop(columns.index('Class')))

# Reorder the columns in the DataFrame
all_images = all_images[columns]

# Populating the 'Label' and 'Class' columns based on conditions using the determine_label and determine_class functions
all_images['Label'] = all_images['Finding Labels'].apply(determine_label)
all_images['Class'] = all_images['Finding Labels'].apply(determine_class)

def count_image_labels(dataframe):

    """
    Function to count the number of images based on their labels.
    
    Args:
        dataframe (DataFrame): name of DataFrame containing the data dictionary.
    """
    
    fibrosis_count = 0
    no_finding_count = 0
    non_fibrosis_count = 0
    total_non_fibrosis_count = no_finding_count + non_fibrosis_count

    for index, row in dataframe.iterrows():       
        if 'Fibrosis' in row['Finding Labels']:
            fibrosis_count += 1
        elif 'No Finding' in row['Finding Labels']:
            no_finding_count += 1
        else:
            non_fibrosis_count += 1
            
    total_non_fibrosis_count = no_finding_count + non_fibrosis_count
                    
    # print(f"\nTotal number of images: {fibrosis_count + no_finding_count + non_fibrosis_count}")
    # print(f"Number of 'Fibrosis' images: {fibrosis_count}")
    # print(f"Number of 'No Finding' images: {no_finding_count}")
    # print(f"Number of 'Non-Fibrosis' images: {non_fibrosis_count}")
    # print(f"Total number of 'Non-Fibrosis' images: {total_non_fibrosis_count}\n")
    
    return fibrosis_count, no_finding_count, non_fibrosis_count, total_non_fibrosis_count

# Call the count_image_labels function and pass it the dataframe    
count_image_labels(all_images)

fibrosis_count, no_finding_count, non_fibrosis_count, total_non_fibrosis_count = count_image_labels(all_images)

def count_image_views(dataframe):
    """
    Function to count the number of images based on their view position.
    
    Args:
        dataframe (DataFrame): name of DataFrame containing the data dictionary.
    """
    
    # Initialize counters for each combination of 'View Position' and 'Label'
    ap_count = 0
    pa_count = 0
    ap_fibrosis_count = 0
    ap_non_fibrosis_count = 0
    pa_fibrosis_count = 0
    pa_non_fibrosis_count = 0

    # Iterate through the DataFrame to count occurrences
    for _, row in dataframe.iterrows():
        view_position = row['View Position']
        label = row['Label']
        
        if view_position == 'AP':
            if label == 'Fibrosis':
                ap_fibrosis_count += 1
            elif label == 'Non-Fibrosis' or label == 'No Finding':
                ap_non_fibrosis_count += 1
        elif view_position == 'PA':
            if label == 'Fibrosis':
                pa_fibrosis_count += 1
            elif label == 'Non-Fibrosis' or label == 'No Finding':
                pa_non_fibrosis_count += 1
                
    ap_count = dataframe['View Position'].value_counts().get('AP', 0)
    pa_count = dataframe['View Position'].value_counts().get('PA', 0)

    # Print the results
    # print(f"Number of times 'PA' appears: {pa_count}")
    # print(f"Number of times 'PA' with 'Fibrosis' appears: {pa_fibrosis_count}")
    # print(f"Number of times 'PA' with 'Non-Fibrosis' appears: {pa_non_fibrosis_count}\n")
    # print(f"Number of times 'AP' appears: {ap_count}")
    # print(f"Number of times 'AP' with 'Fibrosis' appears: {ap_fibrosis_count}")
    # print(f"Number of times 'AP' with 'Non-Fibrosis' appears: {ap_non_fibrosis_count}\n")
    
    return ap_count, pa_count, ap_fibrosis_count, ap_non_fibrosis_count, pa_fibrosis_count, pa_non_fibrosis_count
 
# Call the count_image_views function and pass it the dataframe    
ap_count, pa_count, ap_fibrosis_count, ap_non_fibrosis_count, pa_fibrosis_count, pa_non_fibrosis_count = count_image_views(all_images)

def split_view_positions(dataframe):
    """
    Splits the input DataFrame into two separate DataFrames based on the 'View Position' column.
    
    Args:
        dataframe (pd.DataFrame): name of input DataFrame that contains a column 'View Position'.
    
    Returns:
        pa_images (pd.DataFrame): DataFrame containing rows where 'View Position' is 'PA'.
        ap_images (pd.DataFrame): DataFrame containing rows where 'View Position' is 'AP'.
    """
    
    # Ensure the 'View Position' column exists in the DataFrame
    if 'View Position' not in dataframe.columns:
        raise ValueError("The DataFrame must contain a 'View Position' column.")
    
    # Filter rows where 'View Position' is 'PA'
    pa_images = dataframe[dataframe['View Position'] == 'PA']
    
    # Filter rows where 'View Position' is 'AP'
    ap_images = dataframe[dataframe['View Position'] == 'AP']
    
    # Save the new dataframes to new CSV files    
    pa_images.to_csv(r'C:\Users\Mark\Documents\rp\data_dictionaries\pa_view_data_dictionary.csv', index=False)
    ap_images.to_csv(r'C:\Users\Mark\Documents\rp\data_dictionaries\ap_view_data_dictionary.csv', index=False)
    
    # print("All images DataFrame shape:", all_images.shape)
    # print("PA DataFrame shape:", pa_images.shape)
    # print("AP DataFrame shape:", ap_images.shape)
    
    return pa_images, ap_images

# Call the function to split the DataFrame and unpack the results into two separate DataFrames
pa_images, ap_images = split_view_positions(all_images)

def balance_the_classes(dataframe):
    """
    Function to balance the classes in the input DataFrame by taking the number of 'Fibrosis' images and creating a second class the same size made up of an even split between 'Non-Fibrosis' and 'No Finding'.
    
    Args:
        dataframe (pd.DataFrame): name of input DataFrame that contains a column 'Label'.
        
    Returns:
        balanced_df (pd.DataFrame): DataFrame with balanced classes.
    """
        
    # Separate the DataFrame by classes
    fibrosis_df = dataframe[dataframe['Label'] == 'Fibrosis']
    non_fibrosis_df = dataframe[dataframe['Label'] == 'Non-Fibrosis']
    no_finding_df = dataframe[dataframe['Label'] == 'No Finding']

    # Determine the number of samples needed from 'Non-Fibrosis' and 'No Finding'
    num_samples_per_class = fibrosis_count // 2

    # Resample 'Non-Fibrosis' and 'No Finding' to half the number of 'Fibrosis' images
    non_fibrosis_sampled = resample(non_fibrosis_df, n_samples=num_samples_per_class, random_state=42)
    no_finding_sampled = resample(no_finding_df, n_samples=num_samples_per_class, random_state=42)

    # Combine the balanced classes
    balanced_df = pd.concat([fibrosis_df, non_fibrosis_sampled, no_finding_sampled])

    # Shuffle the DataFrame to mix the classes
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the new dataframe to a new CSV file
    balanced_df.to_csv(rf'C:\Users\Mark\Documents\rp\data_dictionaries\balanced_classes_data_dictionary.csv', index=False)
    
    return balanced_df

balanced_df = balance_the_classes(all_images)



def split_data(dataframe, dataframe_name):
    """
    Function to split the data into training, evaluation, and test sets,
    ensuring that each patient ID appears only in one set and the splits
    are evenly distributed across gender and age groups.
    
    Args:
        dataframe (DataFrame): DataFrame containing the data dictionary.
        dataframe_name (str): Name of the DataFrame.
    """
    # Define split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Create a column for stratification
    dataframe['stratify_col'] =  dataframe['Patient Gender'].astype(str) + '_' + dataframe['Patient Age'].astype(str)

    # Identify unique patient IDs and their stratification group
    unique_patient_ids = dataframe[['Patient ID', 'stratify_col']].drop_duplicates(subset=['Patient ID'])

    # Shuffle the patient IDs
    np.random.seed(42)  # For reproducibility
    random.seed(42)  # For reproducibility
    # unique_patient_ids = unique_patient_ids.sample(frac=1).reset_index(drop=True)

    # Prepare lists to collect patient IDs for each split
    train_patient_ids = []
    val_patient_ids = []
    test_patient_ids = []


    # Separate patient IDs based on stratified groups
    for stratify_val, group in unique_patient_ids.sort_values('stratify_col').groupby('stratify_col'):
        patient_ids = group['Patient ID'].values
        n_samples = len(patient_ids)

        if n_samples == 1:
            # If there is only one sample, put it in the training set
            train_patient_ids.extend(patient_ids)
        else:
            # Split the group into train and temp sets
            train_ids, temp_ids = train_test_split(
                patient_ids, 
                test_size=(val_ratio + test_ratio), 
                random_state=42, 
                stratify=[stratify_val] * len(patient_ids)
            )
            
            if len(temp_ids) == 1:
                if random.random() < 0.5:
                    val_ids = temp_ids
                    test_ids = []
                else: 
                    test_ids = temp_ids
                    val_ids = []
            else:
                # Calculate proportion of validation size to temp size
                temp_val_size = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0

                # Split the temp set into validation and test sets
                val_ids, test_ids = train_test_split(
                    temp_ids, 
                    test_size=temp_val_size, 
                    random_state=42, 
                    stratify=[stratify_val] * len(temp_ids)
                )

            # Add to respective lists
            train_patient_ids.extend(train_ids)
            val_patient_ids.extend(val_ids)
            test_patient_ids.extend(test_ids)

    # Convert lists to sets to ensure there are no duplicates
    # train_patient_ids = list(set(train_patient_ids))
    # val_patient_ids = list(set(val_patient_ids))
    # test_patient_ids = list(set(test_patient_ids))

    # Check for overlapping IDs
    assert len(set(train_patient_ids) & set(val_patient_ids)) == 0, "Patient IDs overlap between train and val datasets."
    assert len(set(train_patient_ids) & set(test_patient_ids)) == 0, "Patient IDs overlap between train and test datasets."
    assert len(set(val_patient_ids) & set(test_patient_ids)) == 0, "Patient IDs overlap between val and test datasets."

    # Create new DataFrames based on the splits
    train_data = dataframe[dataframe['Patient ID'].isin(train_patient_ids)]
    val_data = dataframe[dataframe['Patient ID'].isin(val_patient_ids)]
    test_data = dataframe[dataframe['Patient ID'].isin(test_patient_ids)]

    # Drop the temporary stratification column
    train_data = train_data.drop(columns=['stratify_col'])
    val_data = val_data.drop(columns=['stratify_col'])
    test_data = test_data.drop(columns=['stratify_col'])

    # Save the splits to new CSV files
    train_data.to_csv(f'C:\\Users\\Mark\\Documents\\rp\\data_dictionaries\\{dataframe_name}_train_data_dictionary.csv', index=False)
    val_data.to_csv(f'C:\\Users\\Mark\\Documents\\rp\\data_dictionaries\\{dataframe_name}_eval_data_dictionary.csv', index=False)
    test_data.to_csv(f'C:\\Users\\Mark\\Documents\\rp\\data_dictionaries\\{dataframe_name}_test_data_dictionary.csv', index=False)

    print("**************************************************")
    print(f"Balanced DataFrame shape for {dataframe_name}:", dataframe.shape)
    print(f"Train DataFrame shape for {dataframe_name}:", train_data.shape)
    print(f"Eval DataFrame shape for {dataframe_name}:", val_data.shape)
    print(f"Test DataFrame shape for {dataframe_name}:", test_data.shape)
    print("**************************************************")
    
    return train_data, val_data, test_data


# Call the split_data function and pass it the dataframe and the name of the dataframe as a string to name the new CSV files
train_data, eval_data, test_data = split_data(balanced_df, "balanced_df")

# List of directories on the local filesystem where the NIH CXR images are stored
images_directory = [
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_1',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_2',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_3',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_4',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_5',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_6',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_7',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_8',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_9',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_10',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_11',
    r'C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir\dir_12'
    ]

def find_image_paths(df, column_name, directories):
    """
    Function to find the paths of images listed in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the image names.
    column_name (str): The column name in the DataFrame that contains the image names.
    directories (list): List of directories to search for the images.
    
    Returns:
    None: Prints the paths to the images.
    """
    # Iterate over each image index in the DataFrame
    for image_name in df[column_name]:
        # Initialize a flag to check if the image is found
        image_found = False
        
        # Iterate over each directory
        for directory in directories:
            # Construct the full path of the image
            image_path = os.path.join(directory, image_name)
            
            # Check if the image exists at the constructed path
            if os.path.isfile(image_path):
                # print(f"Image found: {image_path}")
                image_found = True
                break  # Exit the loop once the image is found
        
        # If the image is not found in any directory, print a message
        if not image_found:
            print(f"Image not found: {image_name}")

# Call the function to find the paths of images
# find_image_paths(balanced_df, 'Image Index', images_directory)


def print_stats():
    print("\n******* Printing Stats *******\n")
    
    print(f"Total number of images: {fibrosis_count + no_finding_count + non_fibrosis_count}")
    print(f"Number of 'Fibrosis' images: {fibrosis_count}")
    print(f"Number of 'No Finding' images: {no_finding_count}")
    print(f"Number of 'Non-Fibrosis' images: {non_fibrosis_count}")
    print(f"Total number of 'Non-Fibrosis' images: {total_non_fibrosis_count}\n")
    
    print(f"Number of times 'PA' appears: {pa_count}")
    print(f"Number of times 'PA' with 'Fibrosis' appears: {pa_fibrosis_count}")
    print(f"Number of times 'PA' with 'Non-Fibrosis' appears: {pa_non_fibrosis_count}\n")
    print(f"Number of times 'AP' appears: {ap_count}")
    print(f"Number of times 'AP' with 'Fibrosis' appears: {ap_fibrosis_count}")
    print(f"Number of times 'AP' with 'Non-Fibrosis' appears: {ap_non_fibrosis_count}\n")
    
    print("All images DataFrame shape:", all_images.shape)
    print("PA DataFrame shape:", pa_images.shape)
    print("AP DataFrame shape:", ap_images.shape)
    
    print("\nTrain DataFrame shape:", train_data.shape)
    print("Eval DataFrame shape:", eval_data.shape)
    print("Test DataFrame shape:", test_data.shape)
    
    # Print the number of rows in the new balanced classes DataFrame for each class
    print("\nBalanced Classes Dataframe Counts\n")
    print(f"Number of 'Fibrosis' rows in balanced_df:", (balanced_df['Label'] == 'Fibrosis').sum())
    print(f"Number of 'Non-Fibrosis' rows in balanced_df:", (balanced_df['Label'] == 'Non-Fibrosis').sum())
    print(f"Number of 'No Finding' rows in balanced_df:", (balanced_df['Label'] == 'No Finding').sum())
    print(f"Number of 'Non-Fibrosis + No Finding' rows in balanced_df:", (balanced_df['Label'].isin(['Non-Fibrosis', 'No Finding'])).sum())

print_stats()



def check_for_duplicate_patient_ids():
    """
    Function to check for duplicate patient IDs in the data dictionary.
    """
    print("Checking for duplicate Patient IDs...")
    
    # Convert Patient ID columns to sets
    train_ids = set(train_data['Patient ID'])
    eval_ids = set(eval_data['Patient ID'])
    test_ids = set(test_data['Patient ID'])

    # Check for intersections between sets
    train_eval_intersection = train_ids.intersection(eval_ids)
    train_test_intersection = train_ids.intersection(test_ids)
    eval_test_intersection = eval_ids.intersection(test_ids)

    # Collect all duplicate Patient IDs
    duplicate_patient_ids = train_eval_intersection.union(train_test_intersection).union(eval_test_intersection)

    if duplicate_patient_ids:
        print(f"Duplicate Patient IDs found: {duplicate_patient_ids}")
    else:
        print("No duplicate Patient IDs found.")

# check_for_duplicate_patient_ids()







