import cv2
import time
from hand_tracker import HandTracker
from gesture_recogniser import GestureRecogniser, GestureType
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    # Create hand tracker and gesture recogniser
    hand_tracker = HandTracker()
    gesture_recogniser = GestureRecogniser()
    
    # Lower the stability count for testing
    gesture_recogniser.required_stability_count = 2
    
    print("Gesture recognition test started. Press 'q' to quit.")
    
    # Colors for different gestures
    gesture_colors = {
        GestureType.NONE: (128, 128, 128),    # Gray
        GestureType.DRAW: (0, 0, 255),        # Red
        GestureType.ERASE: (255, 0, 0),       # Blue
        GestureType.SELECT: (0, 255, 0),      # Green
        GestureType.MOVE: (255, 0, 255),      # Magenta
        GestureType.CLEAR: (0, 255, 255)      # Yellow
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process the frame with hand tracker
        processed_frame = hand_tracker.process_frame(frame)
        
        # Get hand landmarks
        landmarks = hand_tracker.get_landmark_positions()
        
        # Direct debug of landmarks
        if landmarks:
            print(f"Main loop: Found {len(landmarks)} hands")
            
            # Try to recognise gestures
            current_time = time.time()
            # DEBUG: Directly check what's happening with landmarks
            try:
                gesture_type, gesture_position = gesture_recogniser.recognise_gesture(landmarks, current_time)
                print(f"Recognised gesture: {gesture_type.name}")
            except Exception as e:
                print(f"ERROR in gesture recognition: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Main loop: No landmarks found")
        
        # Show hand landmarks for debugging
        if landmarks and len(landmarks) > 0:
            for point in landmarks[0]:
                cv2.circle(processed_frame, point, 3, (255, 0, 0), -1)
            
            # Mark specific points in different colors
            if len(landmarks[0]) >= 21:
                # Thumb tip (point 4)
                cv2.circle(processed_frame, landmarks[0][4], 5, (0, 255, 255), -1)
                # Index tip (point 8)
                cv2.circle(processed_frame, landmarks[0][8], 5, (255, 0, 255), -1)
        
        # Update drawing state
        is_drawing, draw_start = gesture_recogniser.update_drawing_state()
        
        # Display gesture information
        cv2.putText(
            processed_frame,
            f"Gesture: {gesture_type.name if 'gesture_type' in locals() else 'NONE'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        
        # Show gesture position
        if 'gesture_position' in locals() and gesture_position:
            cv2.circle(
                processed_frame,
                gesture_position,
                10,
                gesture_colors[gesture_type],
                -1
            )
            
            # Add drawing indicator
            if is_drawing:
                cv2.putText(
                    processed_frame,
                    "Drawing",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
        
        # Show the frame
        cv2.imshow('Gesture Recognition Test', processed_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    hand_tracker.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()