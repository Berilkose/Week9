'''Pose landmark detection''' '''---TASK 2---'''

import cv2
import mediapipe as mp
import os

def draw_and_classify_pose(image_path, output_folder="Results_Pose"):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils # function inside the mediapipe library
    mp_drawing_styles = mp.solutions.drawing_styles

    image = cv2.imread(image_path)
    if image is None: return "no readings"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # BGR -> RGB transformation and implementation (with openCV)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return "Cannot detect a body"

        # Drawing on the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks( #special function for mediapipe
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Save the landmark image
        file_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_folder, f"pose_{file_name}"), annotated_image)

        # arm positions: 
        # Landmark 15: left wrist, 11: left shoulder
        # Landmark 16: right wrist, 12: right shoulder
        lm = results.pose_landmarks.landmark
        
        left_up = lm[15].y < lm[11].y
        right_up = lm[16].y < lm[12].y

        if left_up and right_up: return "Both arms up"
        elif left_up: return "Left arm up"
        elif right_up: return "Right arm up"
        else: return "Arms Down"

def process_pose_folder(folder_name):
    # check the folder path
    if not os.path.exists(folder_name):
        folder_name = os.path.join("..", folder_name)
    
    if not os.path.exists(folder_name):
        print(f"Error: '{folder_name}' cannot be found !")
        return

    print(f"{'File name':<25} | {'Position'}")
    print("-" * 50)

    for filename in os.listdir(folder_name):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_name, filename)
            
            result = draw_and_classify_pose(full_path)
            print(f"{filename:<25} | {result}")

if __name__ == "__main__":
    target_folder = "TestData" 
    process_pose_folder(target_folder)