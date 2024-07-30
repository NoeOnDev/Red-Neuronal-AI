import cv2
import os
import time

def capture_images_from_camera(output_dir, img_count, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error opening the camera")
        return img_count
    
    print("Presiona 's' para comenzar a capturar imágenes...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Press "s" to start capturing images', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    
    paused = False
    last_capture_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        if not paused and (current_time - last_capture_time) >= 1:
            img_path = os.path.join(output_dir, f"img_{img_count:02d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved image {img_path}")
            img_count += 1
            last_capture_time = current_time

        cv2.imshow('Capturing Images', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Paused. Press 'p' again to resume.")
            else:
                print("Resumed capturing images.")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_count} images from the camera")
    return img_count

if __name__ == "__main__":
    output_directory = "images_train/13_k_de_picas/"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    existing_images = [f for f in os.listdir(output_directory) if f.endswith('.jpg')]
    img_count = len(existing_images)
    
    camera_index = int(input("Ingrese el índice de la cámara que desea usar: "))
    
    img_count = capture_images_from_camera(output_directory, img_count, camera_index)
