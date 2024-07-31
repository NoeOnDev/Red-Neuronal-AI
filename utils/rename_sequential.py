import os
import re

def sort_images_numerically(img_list):
    """Ordenar la lista de imágenes numéricamente."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(img_list, key=alphanum_key)

def rename_images_in_directory(directory):
    """Renombrar las imágenes en el directorio dado para que estén en orden secuencial."""
    img_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    img_files = sort_images_numerically(img_files)
    
    for count, img_file in enumerate(img_files):
        new_name = f"img_{count:03d}.jpg"
        old_path = os.path.join(directory, img_file)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

def rename_images_in_all_directories(main_directory):
    for root, dirs, files in os.walk(main_directory):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            rename_images_in_directory(subdir_path)

main_directory = 'images_train'
rename_images_in_all_directories(main_directory)
