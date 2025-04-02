import cv2

def test_camera(flip_horizontal=True):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    
    print("Camera opened successfully.")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    test_camera()