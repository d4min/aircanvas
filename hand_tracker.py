import cv2
import mediapipe as mp 
import numpy as np 
from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_HANDS,
    FLIP_CAMERA
)

class HandTracker:
    def __init__(self):
        # initialise MediaPipe Hands solution 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        # For drawing the hand landmarks 
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # List of landmark positions 
        self.landamark_positions = []

        # Fingertip IDs (index, middle, ring, pinky, thumb)
        self.fingertips = [8, 12, 16, 20, 4]
        self.finger_bases = [5, 9, 13, 17, 2]

    def process_frame (self, frame):
        """Process a frame and detect hands"""
        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        # Covnert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Reset landmark positions
        self.landmark_positions = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Store landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    # Convert normalised coordinates to pixel coordinates
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))

                self.landmark_positions.append(landmarks)

                # Draw hand landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        # Debug: Print if landmarks were found
        if self.landmark_positions:
            print(f"Hand tracker found {len(self.landmark_positions)} hands with {len(self.landmark_positions[0])} landmarks")

        return frame
    
    def get_landmark_positions(self):
        """Return the current landmark positions"""
        return self.landamark_positions
    
    def fingertip_positions(self):
        """Return positions of just the fingertips"""
        fingertip_positions = []

        for hand_landmarks in self.landmark_positions:
            fingertips = [hand_landmarks[tip_id] for tip_id in self.fingertips]
            fingertip_positions.append(fingertips)

        return fingertip_positions

    def count_fingers_up(self):
        """Count how many fingers are up"""
        if not self.landmark_positions:
            return 0 
        
        fingers_up = 0
        hand_landmarks = self.landmark_positions[0] 

        # Check thumb
        if hand_landmarks[self.fingertips[4]][0] < hand_landmarks[self.finger_bases[4]][0]:
            fingers_up += 1
            # Check other fingers
            for idx in range(4):
                # If fingertip is higher (lower y value) than base joint
                if hand_landmarks[self.fingertips[idx]][1] < hand_landmarks[self.finger_bases[idx]][1]:
                    fingers_up += 1
        
            return fingers_up
        
    def is_pinch_gesture(self):
        """Detect pinch gesture (index finger and thumb)"""
        if not self.landmark_positions:
            return False
        
        hand_landmarks = self.landmark_positions[0]
        # Calculate distance between index fingertip and thumb tip 
        index_tip = hand_landmarks[self.fingertips[0]]
        thumb_tip = hand_landmarks[self.fingertips[4]]

        distance = np.sqrt((index_tip[0] - thumb_tip[0])**2 + (index_tip[1] - thumb_tip[1])**2)

        # If distance is less than 40 pixels, consider it a pinch
        return distance < 40
    
    def release(self):
        """Release resources"""
        self.hands.close()

