'''Face landmark detection and Face direction detection together''' '''---TASK 3---'''

import cv2
import mediapipe as mp
import os


def draw_and_classify_face(image_path, output_folder="Results"):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils # For drawing landmarks
    mp_drawing_styles = mp.solutions.drawing_styles # using drawing stayles inside the medipipe lib
    
    image = cv2.imread(image_path)
    if image is None: return "No readings"

    # Create an output folder (skip if exists)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return "Cannot detect face"

        # Drawing the landmarks
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # 1. Draw the mesh(lines)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # 2. draw eyes, lips and nose
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Save the image
        file_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_folder, f"result_{file_name}"), annotated_image)

        # Direction detection
        nose = face_landmarks.landmark[1].x
        left_edge = face_landmarks.landmark[234].x
        right_edge = face_landmarks.landmark[454].x
        center_score = (nose - left_edge) / (right_edge - left_edge)

        # Center Score is our threshold to decide the direction of the face
        # if nose is close to the left side of the picture ---> LEFT side (nose - left_edge will be shorter) Center score will be low.
        if center_score < 0.4: return "Left"
        elif center_score > 0.6: return "Right"
        else: return "Straight"

#Creating the landmark images here by calling the functions
def process_folder(folder_name):
    if not os.path.exists(folder_name):
        folder_name = os.path.join("..", folder_name) 
    
    if not os.path.exists(folder_name):
        print(f"Error: '{folder_name}' Cannot be found!")
        return

    print(f"{'File Name':<25} | {'Direction'}")
    print("-" * 45)

    for filename in os.listdir(folder_name):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_name, filename)
            
            # Find the full path of the images(FaceData Folder)
            result = draw_and_classify_face(full_path, output_folder="Results") #add the drawings into the Results folder
            print(f"{filename:<25} | {result}")

if __name__ == "__main__":
    target_folder = "FaceData" 
    process_folder(target_folder) #---> CAll the function here