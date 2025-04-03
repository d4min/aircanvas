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

    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        ui_frame = frame.copy()
        
        # Draw color palette
        for x1, y1, x2, y2, color in self.color_palette_area:
            cv2.rectangle(ui_frame, (x1, y1), (x2, y2), color, -1)
            # Highlight selected color
            if color == self.brush_color:
                cv2.rectangle(ui_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 2)
        
        # Draw tool selection
        for i, (x1, y1, x2, y2, tool) in enumerate(self.tool_selection_area):
            # Tool background
            cv2.rectangle(ui_frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
            
            # Tool name
            text_size = cv2.getTextSize(tool[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (self.tool_box_size - text_size[0]) // 2
            text_y = y1 + (self.tool_box_size + text_size[1]) // 2
            cv2.putText(ui_frame, tool[0], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Highlight selected tool
            if i == self.current_tool.value:
                cv2.rectangle(ui_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 2)
        
        return ui_frame
    
    def check_ui_interaction(self, point):
        """Check if interaction is with UI elements"""
        if point is None:
            return False
        
        x, y = point
        
        # Check color palette
        for i, (x1, y1, x2, y2, color) in enumerate(self.color_palette_area):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.brush_color = color
                return True
        
        # Check tool selection
        for i, (x1, y1, x2, y2, tool) in enumerate(self.tool_selection_area):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.current_tool = ToolType(i)
                # Reset tracking variables when tool changes
                self.prev_point = None
                self.start_point = None
                return True
        
        return False
    
    def draw(self, point, is_drawing):
        """Draw on canvas based on current tool and point"""
        if point is None:
            # If point is None, reset drawing state
            self.prev_point = None
            return self.canvas
        
        # Check if interaction is with UI
        if self.check_ui_interaction(point):
            # Reset drawing
            self.prev_point = None
            return self.canvas
        
        # Make a copy to avoid modifying the original during preview
        canvas_copy = self.canvas.copy()
        
        # Handle drawing based on tool
        if self.current_tool == ToolType.PEN:
            self._draw_pen(canvas_copy, point, is_drawing)
        elif self.current_tool == ToolType.ERASER:
            self._draw_eraser(canvas_copy, point, is_drawing)
        elif self.current_tool == ToolType.LINE:
            self._draw_line(canvas_copy, point, is_drawing)
        elif self.current_tool == ToolType.RECTANGLE:
            self._draw_rectangle(canvas_copy, point, is_drawing)
        elif self.current_tool == ToolType.CIRCLE:
            self._draw_circle(canvas_copy, point, is_drawing)
        
        # Only save to history when drawing is complete
        if not is_drawing and self.prev_point is not None:
            self._add_to_history(canvas_copy)
        
        return canvas_copy
    
    def _draw_pen(self, canvas, point, is_drawing):
        """Draw with pen tool"""
        if self.prev_point is None:
            # First point, just store it
            self.prev_point = point
        else:
            # Draw line between previous and current point
            cv2.line(canvas, self.prev_point, point, self.brush_color, self.brush_thickness)
            # Update previous point
            if is_drawing:
                self.prev_point = point
            else:
                self.prev_point = None
    
    def _draw_eraser(self, canvas, point, is_drawing):
        """Erase at the current point"""
        if self.prev_point is None:
            # First point, just store it
            self.prev_point = point
        else:
            # Draw white line (erase) between previous and current point
            cv2.line(canvas, self.prev_point, point, (255, 255, 255), self.eraser_thickness)
            # Update previous point
            if is_drawing:
                self.prev_point = point
            else:
                self.prev_point = None
    
    def _draw_line(self, canvas, point, is_drawing):
        """Draw a straight line"""
        if self.start_point is None:
            # Set starting point
            self.start_point = point
            self.prev_point = point
        else:
            # Create a temporary preview of the line
            temp_canvas = self.canvas.copy()
            cv2.line(temp_canvas, self.start_point, point, self.brush_color, self.brush_thickness)
            
            if is_drawing:
                # Just show preview while still drawing
                canvas[:] = temp_canvas[:]
            else:
                # Complete the drawing when released
                cv2.line(canvas, self.start_point, point, self.brush_color, self.brush_thickness)
                self.start_point = None
                self.prev_point = None
    
    def _draw_rectangle(self, canvas, point, is_drawing):
        """Draw a rectangle"""
        if self.start_point is None:
            # Set starting point
            self.start_point = point
            self.prev_point = point
        else:
            # Create a temporary preview of the rectangle
            temp_canvas = self.canvas.copy()
            cv2.rectangle(temp_canvas, self.start_point, point, self.brush_color, self.brush_thickness)
            
            if is_drawing:
                # Just show preview while still drawing
                canvas[:] = temp_canvas[:]
            else:
                # Complete the drawing when released
                cv2.rectangle(canvas, self.start_point, point, self.brush_color, self.brush_thickness)
                self.start_point = None
                self.prev_point = None
    
    def _draw_circle(self, canvas, point, is_drawing):
        """Draw a circle"""
        if self.start_point is None:
            # Set starting point (center)
            self.start_point = point
            self.prev_point = point
        else:
            # Calculate radius
            radius = int(((point[0] - self.start_point[0])**2 + 
                          (point[1] - self.start_point[1])**2)**0.5)
            
            # Create a temporary preview of the circle
            temp_canvas = self.canvas.copy()
            cv2.circle(temp_canvas, self.start_point, radius, self.brush_color, self.brush_thickness)
            
            if is_drawing:
                # Just show preview while still drawing
                canvas[:] = temp_canvas[:]
            else:
                # Complete the drawing when released
                cv2.circle(canvas, self.start_point, radius, self.brush_color, self.brush_thickness)
                self.start_point = None
                self.prev_point = None
    
    def _add_to_history(self, canvas):
        """Add current canvas state to history"""
        # Trim history if we're in the middle of an undo sequence
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        # Add current state
        self.history.append(canvas.copy())
        self.history_index = len(self.history) - 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
        
        # Update canvas
        self.canvas = canvas.copy()
    
    def undo(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.canvas = self.history[self.history_index].copy()
    
    def redo(self):
        """Redo previously undone action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.canvas = self.history[self.history_index].copy()
    
    def clear(self):
        """Clear the canvas"""
        self.canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255
        self._add_to_history(self.canvas)
    
    def save(self, filename="drawing.jpg"):
        """Save the canvas to a file"""
        cv2.imwrite(filename, self.canvas)
        return filename