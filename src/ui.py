import cv2
import numpy as np 

class UIManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.colours = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "purple": (255, 0, 255),
            "orange": (0, 165, 255),
            "black": (0, 0, 0),
            "white": (255, 255, 255)
        }

        self.colour_box_size = 50
        self.padding = 10
        self.selected_colour = "black"

        self.colour_boxes = self._create_colour_boxes()

    def _create_colour_boxes(self):
        boxes = {}
        x_start = self.width = (self.colour_box_size + self.padding)

        for i, (colour_name, colour_value) in enumerate(self.colours.items()):
            y = self.padding + i * (self.colour_box_size + self.padding)
            boxes[colour_name] = {
                'rect': (x_start, y, self.colour_box_size, self.colour_box_size),
                'colour': colour_value
            }

        return boxes
    
    def draw_ui(self, frame):
        for colour_name, box in self.colour_boxes.items():
            x, y, w, h = box['rect']
            colour = box['colour']

            # Draw colour box
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, -1)
            
            # Draw white/black border for visibility
            border_colour = (0, 0, 0) if sum(colour) > 380 else (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_colour, 1)
            
            # Highlight selected colour
            if colour_name == self.selected_colour:
                # Draw double border for selected colour
                cv2.rectangle(frame, (x-2, y-2), (x + w+2, y + h+2), (255, 255, 255), 2)
                cv2.rectangle(frame, (x-3, y-3), (x + w+3, y + h+3), (0, 0, 0), 1)
                
                # Draw current colour indicator in top-left
                cv2.rectangle(frame, (10, 10), (40, 40), colour, -1)
                cv2.rectangle(frame, (10, 10), (40, 40), border_colour, 1)
                cv2.putText(frame, "Current", (50, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
    def check_colour_selection(self, point):
        for colour_name, box in self.colour_boxes.items():
            x, y, w, h = box['rect']
            if (x <= point[0] <= x + w) and (y <= point[1] <= y + h):
                self.selected_color = colour_name
                return True, self.colors[colour_name]
        return False, None