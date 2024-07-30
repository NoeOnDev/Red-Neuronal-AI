import os
import uuid

def rename_images(directory):
    files = os.listdir(directory)
    
    for filename in files:
        file_extension = os.path.splitext(filename)[1]
        new_filename = str(uuid.uuid4()) + file_extension
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        os.rename(old_file, new_file)
        print(f'Renamed {old_file} to {new_file}')

image_directory = 'images_train/1_as_de_picas'
rename_images(image_directory)