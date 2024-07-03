import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def copy_file(src, dst):
    """Copy a file from src to dst."""
    shutil.copy2(src, dst)

def distribute_images(source_dir, base_destination_dir, num_dirs=12, num_threads=10):
    """Distribute images from source_dir into num_dirs directories within base_destination_dir."""
    # Ensure base destination directory exists
    if not os.path.exists(base_destination_dir):
        os.makedirs(base_destination_dir)

    # Create the destination directories if they do not exist
    destination_dirs = []
    for i in range(num_dirs):
        dir_path = os.path.join(base_destination_dir, f"dir_{i+1}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        destination_dirs.append(dir_path)

    # Get list of image files
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    total_files = len(files)
    files_per_dir = total_files // num_dirs

    # Distribute files equally among the destination directories
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a list of tasks
        tasks = []
        for i, file_name in enumerate(files):
            src = os.path.join(source_dir, file_name)
            dst_dir = destination_dirs[i % num_dirs]
            dst = os.path.join(dst_dir, file_name)
            tasks.append(executor.submit(copy_file, src, dst))

        # Wait for all tasks to complete
        for task in tasks:
            task.result()

if __name__ == "__main__":
    source_directory = r"C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_all"
    base_destination_directory = r"C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir"
    distribute_images(source_directory, base_destination_directory)
