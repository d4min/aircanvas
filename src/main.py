import cv2
import numpy as np
from config import *
from hand_tracker import HandTracker
from gesture import GestureRecogniser, GestureType
from drawing import DrawingCanvas
from ui import UIManager

def initialise_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    return cap

def main():
    cap = initialise_camera()
    tracker = HandTracker()
    gesture_recogniser = GestureRecogniser()
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT)
    ui_manager = UIManager(CAMERA_WIDTH, CAMERA_HEIGHT)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to get frame from camera")
            break
    
        # flip frame if enabled because i look ugly mirrored
        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        # find and draw hands
        frame = tracker.find_hands(frame)
        landmark_list = tracker.get_hand_position(frame)

        # recognise gesture
        gesture = gesture_recogniser.recognise_gesture(landmark_list)
        index_finger = tracker.get_finger_position(frame, 8) if landmark_list else None


        # Handle drawing actions
        if index_finger:
            if gesture == GestureType.DRAW:
                if not canvas.drawing:
                    canvas.start_drawing(index_finger)
                else:
                    canvas.draw(index_finger)
                    
            elif gesture == GestureType.ERASE:
                # Draw eraser preview circle
                cv2.circle(frame, 
                          index_finger, 
                          canvas.eraser_thickness // 2,  # Radius is half the thickness
                          (255, 0, 0),  # Red color for eraser
                          2)  # Circle thickness
                
                # Handle eraser drawing
                canvas.set_tool("eraser")
                if not canvas.drawing:
                    canvas.start_drawing(index_finger)
                else:
                    canvas.draw(index_finger)
                    
            elif gesture == GestureType.SELECT:
                color_selected, color = ui_manager.check_color_selection(index_finger)
                if color_selected:
                    canvas.set_color(color)
                
        # Combine canvas with camera feed
        # Draw canvas content
        drawing_display = canvas.get_display()
        # Only show camera feed where there is no drawing
        mask = cv2.cvtColor(drawing_display, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        
        # Combine the camera feed and drawing
        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        drawing_fg = cv2.bitwise_and(drawing_display, drawing_display, mask=mask)
        frame = cv2.add(frame_bg, drawing_fg)
        
        # Add UI elements
        ui_manager.draw_ui(frame)
        

        cv2.imshow('AirCanvas', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()