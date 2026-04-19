import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from traceback import print_exc

# Import our previous functions & ROS2 environment
from dataset_builder import normalize_landmarks
from stable_baselines3 import PPO
from robot_env import Op3GymEnv

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    print("Initializing Live Demo Pipeline...")
    
    # 1. Load the KNN Classifer (For Action + 'OTHERS' tagging)
    with open("fast_knn_classifier.pkl", "rb") as f:
        knn_data = pickle.load(f)
        knn_model = knn_data["model"]
        label_map = knn_data["labels"]
        
    print("✅ Loaded KNN Classifier for Live Recognition.")
    
    # 2. Load the GAIL Generator (Robot Policy)
    robot_policy = None
    try:
        robot_policy = PPO.load("op3_gail_policy.zip")
        # We initialize the custom environment purely to pass commands to WSL2 via TCP
        robot_env = Op3GymEnv()
        print("✅ Loaded GAIL Controller Policy & Windows->WSL TCP Bridge.")
    except Exception as e:
        print("⚠️ Could not load `op3_gail_policy.zip`. Running Vision & KNN node solely.")
    
    # 3. Initialize MediaPipe
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return
        
    print("📸 Starting Webcam Stream. Press 'Q' to exit.")

    # Frame skipping / smoothing constraints
    last_act_time = time.time()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Performance improvement: pass by reference RGB
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            display_text = "OTHERS"
            color = (0, 0, 255) # Red for OTHERS by default
            
            # If we see a human
            if results.pose_world_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract 132 dim coords + visibility from real-time webcam identical to Data Prep phase
                flat_lm = []
                for lm in results.pose_world_landmarks.landmark:
                    flat_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                # Reshape for Normalization & Inference
                current_obs = np.array([flat_lm], dtype=np.float32)
                normalized_obs = normalize_landmarks(current_obs)
                
                # ------ INFERENCE NODE (Phase 4 Logic) ------
                # Ask KNN the nearest neighbor class probabilities
                probs = knn_model.predict_proba(normalized_obs)[0]
                max_prob = np.max(probs)
                predicted_class = np.argmax(probs)
                
                # 🧠 Threshold Logic! 
                # If probability isn't absolutely confident (say < 0.65), tag as "OTHERS"
                if max_prob > 0.65:
                    display_text = label_map[predicted_class]
                    color = (0, 255, 0) # Green for known classes
                
                # ------ GAIL CONTROLLER NODE ------
                # Feed normalized target pose to the policy to mimic real-time joints
                if robot_policy is not None and (time.time() - last_act_time) > 0.05: # Cap ROS 20hz frequency to avoid spamming joint_states
                    # The GAIL model was trained on DummyVecEnv so predict returns vectorized action
                    action, _states = robot_policy.predict(normalized_obs, deterministic=True)
                    
                    # Ensure action is exactly a 1D array of 23 elements!
                    action_flat = np.array(action).flatten()
                        
                    # Pass the 23-dim joint array to Gazebo
                    robot_env.step(action_flat)
                    last_act_time = time.time()
                    
            # ------ HUD/UI NODE (Phase 5) ------
            cv2.rectangle(image, (0, 0), (600, 80), (0, 0, 0), -1)
            cv2.putText(
                image, 
                f"ACTION: {display_text}", 
                (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.3, 
                color, 
                3, 
                cv2.LINE_AA
            )
            cv2.putText(
                image, 
                f"GAIL Active: {robot_policy is not None}", 
                (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255,255,255), 
                1
            )

            cv2.imshow("Hackathon Execution Node & UI", image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if robot_policy is not None:
        robot_env.close()

if __name__ == "__main__":
    main()
