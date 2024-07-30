import cv2
import os

def capture_images_from_camera(output_dir, img_count, capture_interval=20, camera_index=0):
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
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % capture_interval == 0:
            img_path = os.path.join(output_dir, f"img_{img_count:02d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved image {img_path}")
            img_count += 1
        
        frame_count += 1

        cv2.imshow('Capturing Images', frame)
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_count} images from the camera")
    return img_count

if __name__ == "__main__":
    output_directory = "images_train/tmp/"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    img_count = 0
    capture_interval = 2  
    
    camera_index = int(input("Ingrese el índice de la cámara que desea usar: "))
    
    img_count = capture_images_from_camera(output_directory, img_count, capture_interval, camera_index)