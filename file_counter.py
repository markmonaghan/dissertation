import os

def count_images_in_dirs(base_destination_dir, num_dirs=12):
    """Count the number of images in num_dirs within base_destination_dir."""
    total_images = 0
    
    for i in range(num_dirs):
        dir_path = os.path.join(base_destination_dir, f"dir_{i+1}")
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            total_images += len(files)
    
    return total_images

if __name__ == "__main__":
    base_destination_directory = r"C:\Users\Mark\Documents\rp\nih_dataset\cxr_images_dir"
    total_images = count_images_in_dirs(base_destination_directory)
    print(f"Total number of images across {12} directories: {total_images}")
