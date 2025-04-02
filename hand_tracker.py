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

        return frame
