import cv2
import numpy as np 
import math 
from enum import Enum

class GestureType(Enum):
    """Enum for different gesture types"""
    NONE = 0
    DRAW = 1
    ERASE = 2
    SELECT = 3
    MOVE = 4
    CLEAR = 5

class GestureRecogniser:
    def __init__(self):
        # Gesture state tracking 
        self.current_gesture = GestureType.NONE
        self.previous_gesture = GestureType.NONE
        self.gesture_start_time = 0
        self.gesture_duration = 0

        # Stability counters (to prevent flickering)
        self.gesture_stability_count = 0
        self.required_stability_count = 5

        # Gesture positions
        self.gesture_position = None
        self.previous_positions = []
        self.max_position_history = 5

        # Drawing state
        self.is_drawing = False
        self.draw_start_position = None

    def recognise_gesture(self, hand_landmarks, frame_time):
        """
        Recognise gestures from hand landmarks
        Returns the current gesture type and position 
        """
        if not hand_landmarks or len(hand_landmarks) == 0:
            return self._update_gesture_state(GestureType.NONE, None, frame_time)
        
        # Use first hand's landmarks
        landmarks = hand_landmarks[0]

        # Extract key points
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        thumb_tip = landmarks[4]
        wrist = landmarks[0]

        # Calculate palm center
        palm_points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        palm_center = self._calculate_center_point(palm_points)

        # Calculate key distances 
        index_thumb_distance = self._calculate_distance(index_tip, thumb_tip)
        palm_size = self._calculate_distance(landmarks[5], landmarks[17])

        # Normalise distances relative to palm size to account for distance from camera
        normalised_index_thumb = index_thumb_distance / palm_size if palm_size > 0 else float('inf')

        # Find current gesture
        gesture_type = GestureType.NONE
        gesture_position = None

        # Detect drawing gesture (pinch)
        if normalised_index_thumb < 0.3:
            gesture_type = GestureType.DRAW
            gesture_position = self._midpoint(index_tip, thumb_tip)

        # Detect eraser gesture (open palm with fingers up)
        elif (self._is_finger_up(landmarks, 8, 5) and 
              self._is_finger_up(landmarks, 12, 9) and 
              self._is_finger_up(landmarks, 16, 13) and 
              self._is_finger_up(landmarks, 20, 17)):
            gesture_type = GestureType.ERASE
            gesture_position = palm_center

        # Detect select gesture (index finger up, others down)
        elif (self._is_finger_up(landmarks, 8, 5) and 
             not self._is_finger_up(landmarks, 12, 9) and 
             not self._is_finger_up(landmarks, 16, 13) and 
             not self._is_finger_up(landmarks, 20, 17)):
            gesture_type = GestureType.SELECT
            gesture_position = index_tip
            
        # Detect clear gesture (fist with thumb out)
        elif (not self._is_finger_up(landmarks, 8, 5) and 
              not self._is_finger_up(landmarks, 12, 9) and 
              not self._is_finger_up(landmarks, 16, 13) and 
              not self._is_finger_up(landmarks, 20, 17) and
              self._calculate_distance(thumb_tip, wrist) > palm_size):
            gesture_type = GestureType.CLEAR
            gesture_position = palm_center
        
        # Update and return gesture state
        return self._update_gesture_state(gesture_type, gesture_position, frame_time)