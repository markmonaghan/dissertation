import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def copy_file(src, dst):
    """Copy a file from src to dst."""
    shutil.copy2(src, dst)

def copy_images(source_dir, destination_dir, num_images=1000, num_threads=10):
    """Copy num_images from source_dir to destination_dir using num_threads."""
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get list of image files
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Select the first num_images files
    files_to_copy = files[:num_images]

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a list of tasks
        tasks = []
        for file_name in files_to_copy:
            src = os.path.join(source_dir, file_name)
            dst = os.path.join(destination_dir, file_name)
            tasks.append(executor.submit(copy_file, src, dst))

        # Wait for all tasks to complete
        for task in tasks:
            task.result()

if __name__ == "__main__":
    source_directory = r"C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_all"
    destination_directory = r"C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_1000"
    copy_images(source_directory, destination_directory)
