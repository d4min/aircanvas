import cv2
import numpy as np 
import time 
from enum import Enum
from config import CANVAS_WIDTH, CANVAS_HEIGHT, BRUSH_THICKNESS, ERASER_THICKNESS, DEFAULT_BRUSH_COLOR

class ToolType(Enum):
    """Enum for different drawing tools"""
    PEN = 0
    ERASER = 1
    LINE = 2
    RECTANGLE = 3
    CIRCLE = 4

class Canvas:
    def __init__(self):
        """Initialise the canvas and tools"""
        # Create a transparent canvas
        self.canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.uint8)

        # Drawing properties
        self.current_tool = ToolType.PEN
        self.brush_color = DEFAULT_BRUSH_COLOR
        self.brush_thickness = BRUSH_THICKNESS
        self.eraser_thickness = ERASER_THICKNESS

        # Tracking variables for drawing
        self.prev_point = None
        self.start_point = None
        self.shape_preview = None

        # Color palette
        self.colors = [
            (0, 0, 255),    # Red (BGR)
            (0, 165, 255),  # Orange
            (0, 255, 255),  # Yellow
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (255, 0, 255),  # Magenta
            (0, 0, 0),      # Black
            (255, 255, 255) # White
        ]

        # Tool selection and control areas
        self.color_box_size = 30
        self.tool_box_size = 40
        self.setup_ui_areas()

        # History for undo/redo
        self.history = [self.canvas.copy()]
        self.history_index = 0
        self.max_history = 20

    def setup_ui_areas(self):
        """Set up UI element areas (color palette, tools)"""
        # Define color palette area - top row
        self.color_palette_area = []
        for i, color in enumerate(self.colors):
            x1 = 10 + i * (self.color_box_size + 5)
            y1 = 10
            x2 = x1 + self.color_box_size
            y2 = y1 + self.color_box_size
            self.color_palette_area.append((x1, y1, x2, y2, color))

        # Define tool selection area - left column
        self.tool_selection_area = []
        tool_icons = [
            "PEN", "ERASER", "LINE", "RECT", "CIRCLE"
        ]
        for i, tool in enumerate(tool_icons):
            x1 = 10
            y1 = 50 + i * (self.tool_box_size + 5)
            x2 = x1 + self.tool_box_size
            y2 = y1 + self.tool_box_size
            self.tool_selection_area.append((x1, y1, x2, y2, tool))