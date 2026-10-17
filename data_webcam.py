import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open the webcam.")

print("Press 'q' in the video window to quit.")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read a frame.")
            break

        # Flip the frame horizontally for a natural mirror view and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)
        gesture_text = "Show Rock, Paper, or Scissors"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton connections
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                
                # Finger tip and pip joint y-coordinates to check if fingers are open or closed
                # Tips: Index(8), Middle(12), Ring(16), Pinky(20)
                # PIP joints: Index(6), Middle(10), Ring(14), Pinky(18)
                index_extended = landmarks[8].y < landmarks[6].y
                middle_extended = landmarks[12].y < landmarks[10].y
                ring_extended = landmarks[16].y < landmarks[14].y
                pinky_extended = landmarks[20].y < landmarks[18].y
                
                # Basic gesture rule logic
                if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                    gesture_text = "ROCK"
                elif index_extended and middle_extended and ring_extended and pinky_extended:
                    gesture_text = "PAPER"
                elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                    gesture_text = "SCISSORS"
                else:
                    gesture_text = "Unknown Gesture"

        # Display the recognized gesture on the screen
        cv2.putText(frame, f"Gesture: {gesture_text}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Rock Paper Scissors - MediaPipe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()