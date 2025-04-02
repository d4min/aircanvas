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
        print(f"Received landmarks: {type(hand_landmarks)}, Length: {len(hand_landmarks) if hand_landmarks else 0}")

        if not hand_landmarks or len(hand_landmarks) == 0:
            print("No hand landmarks received in recognize_gesture")
            return self._update_gesture_state(GestureType.NONE, None, frame_time)
        
        # Use first hand's landmarks
        landmarks = hand_landmarks[0]
        print(f"First hand has {len(landmarks)} landmarks")

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
    
    def _update_gesture_state(self, detected_gesture, position, frame_time):
        """Update gesture sttae with stability checks"""
        # If gesture changed, start stability counter 
        if detected_gesture != self.previous_gesture:
            self.gesture_stability_count = 1
            self.previous_gesture = detected_gesture   
            return self.current_gesture, self.gesture_position 
        
        # If same gesture, increment stabililty counter
        self.gesture_stability_count += 1

        # Only update current gesture if stable enough 
        if self.gesture_stability_count >= self.required_stability_count:
            # If gesture changed, update start time and duration 
            if self.current_gesture != detected_gesture:
                self.gesture_start_time = frame_time
                self.gesture_duration = 0
            else:
                self.gesture_duration = frame_time - self.gesture_start_time

            self.current_gesture = detected_gesture

            # Update position with smoothing 
            if position is not None:
                if self.gesture_position is None:
                    self.gesture_position = position
                else:
                    # Apply smoothing 
                    self.previous_positions.append(position)
                    if len(self.previous_positions) > self.max_position_history:
                        self.previous_positions.pop(0)

                    # Average the positions for stability
                    avg_x = sum(pos[0] for pos in self.previous_positions) / len(self.previous_positions)
                    avg_y = sum(pos[1] for pos in self.previous_positions) / len(self.previous_positions)
                    self.gesture_position = (int(avg_x), int(avg_y))
            else:
                self.gesture_position = None
                self.previous_positions = []
        
        return self.current_gesture, self.gesture_position
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _midpoint(self, point1, point2):
        """Calculate the midpoint between two points"""
        return (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2))
    
    def _calculate_center_point(self, points):
        """Calculate the center point of a set of points"""
        if not points:
            return None 
        
        sum_x = sum(point[0] for point in points)
        sum_y = sum(point[1] for point in points)
        return (int(sum_x / len(points)), int(sum_y / len(points)))
    
    def _is_finger_up(self, landmarks, fingertip_idx, base_idx):
        """Check if a finger is pointing up (y-coordinate is less than base)"""
        return landmarks[fingertip_idx][1] < landmarks[base_idx][1]
    
    def get_gesture_info(self):
        """Get current gesture information"""
        return {
            'type': self.current_gesture,
            'position': self.gesture_position,
            'duration': self.gesture_duration,
            'is_drawing': self.is_drawing
        }
    
    def update_drawing_state(self):
        """Update drawing state based on current gesture"""
        # Start drawing when gesture changes to DRAW
        if self.current_gesture == GestureType.DRAW and self.is_drawing:
            self.is_drawing = True
            self.draw_start_position = self.gesture_position

        #Stop drawing when gesture is no longer DRAW
        elif self.current_gesture != GestureType.DRAW and self.is_drawing:
            self.is_drawing = False
            self.draw_start_position = None

        return self.is_drawing, self.draw_start_position

