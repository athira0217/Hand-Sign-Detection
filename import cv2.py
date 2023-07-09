import cv2
import os

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def capture_images(name, num_images):
    # Create folder for the person's images
    create_folder(name)
    
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    
    # Create a cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image_count = 0
    while image_count < num_images:
        # Read a frame from the camera
        ret, frame = camera.read()
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Crop the face region
            face = frame[y:y+h, x:x+w]
            
            # Resize the face to a standard size
            face = cv2.resize(face, (150, 150))
            
            # Save the face image in the person's folder
            filename = f"{name}/{name}_{image_count}.jpg"
            cv2.imwrite(filename, face)
            
            image_count += 1
            print(f"Captured image {image_count}/{num_images}")
        
        # Display the frame with bounding boxes around the detected faces
        cv2.imshow('Capture Images', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

# Prompt the user to enter the name of the individual
name = input("Enter the name of the individual: ")

# Capture 200 images
num_images = 200

# Call the function to capture images
capture_images(name, num_images)
