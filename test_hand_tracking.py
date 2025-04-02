import cv2
from hand_tracker import HandTracker
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT

def main():
    # Initialize the webcam
    print(f"Attempting to open camera with ID: {CAMERA_ID}")
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID: {CAMERA_ID}")
        return
    
    print(f"Camera opened successfully")
    
    # Try to set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    # Check what properties were actually set
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Requested resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Actual resolution: {actual_width}x{actual_height}")
    
    # First, check if we can read a frame before creating the hand tracker
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame, exiting")
        cap.release()
        return
    
    # Show the initial frame
    cv2.imshow('Initial Frame Test', frame)
    cv2.waitKey(1000)  # Display for 1 second
    
    # Create hand tracker
    print("Creating hand tracker...")
    hand_tracker = HandTracker()
    
    print("Hand tracking test started. Press 'q' to quit.")
    
    frame_count = 0
    while frame_count < 1000:  # Limit to 1000 frames for safety
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame at count {frame_count}")
            break
        
        frame_count += 1
        
        # Process the frame with hand tracker
        processed_frame = hand_tracker.process_frame(frame)
        
        # Add finger count information
        fingers_up = hand_tracker.count_fingers_up()
        cv2.putText(
            processed_frame, 
            f"Fingers up: {fingers_up}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Add pinch gesture detection
        is_pinching = hand_tracker.is_pinch_gesture()
        cv2.putText(
            processed_frame, 
            f"Pinch: {'Yes' if is_pinching else 'No'}", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Add frame counter for debugging
        cv2.putText(
            processed_frame, 
            f"Frame: {frame_count}", 
            (10, 110), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Show the frame
        cv2.imshow('Hand Tracking Test', processed_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    print("Cleaning up resources...")
    hand_tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    main()