import os

def count_images_in_directories(main_directory):
    for root, dirs, files in os.walk(main_directory):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            image_count = len([file for file in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, file))])
            print(f'{subdir}: {image_count} imágenes')

main_directory = 'images_train'
count_images_in_directories(main_directory)
