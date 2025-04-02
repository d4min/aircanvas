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
