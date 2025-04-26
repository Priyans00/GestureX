import cv2
import numpy as np
from supabase import create_client, Client
import subprocess
from dotenv import load_dotenv
import os
import time
load_dotenv()

# Set up Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define skin color range in HSV
lower_skin = np.array([0, 48, 80], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# System state variables
is_running = False
last_action_time = 0
cooldown_period = 2  # seconds to wait after executing an action

# Define detection region (top right corner)
def get_detection_region(frame):
    height, width = frame.shape[:2]
    region_size = min(width, height) // 2  # Size of the square region
    x1 = width - region_size
    y1 = 0
    x2 = width
    y2 = region_size
    return (x1, y1, x2, y2)

def count_fingers(hand_contour, frame):
    # Find the convex hull
    hull = cv2.convexHull(hand_contour, returnPoints=False)
    
    # Find convexity defects
    defects = cv2.convexityDefects(hand_contour, hull)
    
    if defects is None:
        return 0
    
    # Initialize finger count
    finger_count = 0
    fingertips = []
    
    # Process each defect
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(hand_contour[s][0])
        end = tuple(hand_contour[e][0])
        far = tuple(hand_contour[f][0])
        
        # Calculate distances
        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        
        # Calculate angle
        angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        
        # If angle is less than 90 degrees or greter than a small point, it's likely a finger
        if angle <= 90 and angle >=5:
            finger_count += 1
            # Store potential fingertip points
            if d > 100:  # Depth threshold
                fingertips.append(far)
    
    # Draw crosses on the fingertips
    cross_size = 5  # Size of the cross
    for tip in fingertips:
        x, y = tip
        # Draw the cross
        cv2.line(frame, (x - cross_size, y - cross_size), (x + cross_size, y + cross_size), (0, 0, 255), 2)
        cv2.line(frame, (x - cross_size, y + cross_size), (x + cross_size, y - cross_size), (0, 0, 255), 2)
    
    
    return finger_count 

def draw_status(frame, is_running, cooldown_remaining):
    # Draw status bar at the top
    status_color = (0, 255, 0) if is_running else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    status_text = "Running" if is_running else "Paused"
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Draw cooldown indicator if applicable
    if cooldown_remaining > 0:
        cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (frame.shape[1] - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw controls help
    controls_text = "Press 's' to start/stop, 'q' to quit"
    cv2.putText(frame, controls_text, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def process_frame(frame):
    # Get detection region
    x1, y1, x2, y2 = get_detection_region(frame)
    
    # Draw detection region
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "Show fingers here", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # Apply Gaussian blur
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Create a mask for the detection region
    region_mask = np.zeros_like(mask)
    region_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, region_mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Process largest contour (hand)
    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand_contour) > 1000:
            # Count fingers
            num_fingers = count_fingers(hand_contour, frame)
            gesture = f"{num_fingers} fingers"
            
            # Draw hand contour
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
            
            # Display finger count
            cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return num_fingers, gesture
    return None, None

print("Gesture Control System")
print("Press 's' to start/stop detection")
print("Press 'q' to quit")

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate cooldown remaining
    current_time = time.time()
    cooldown_remaining = max(0, cooldown_period - (current_time - last_action_time))

    # Draw status and controls
    draw_status(frame, is_running, cooldown_remaining)

    # Process frame and get finger count (always do this)
    num_fingers, gesture = process_frame(frame)

    # Execute action only if running and not in cooldown
    if is_running and cooldown_remaining == 0 and gesture:
        response = supabase.table("gesture_control").select("action_script").eq("gesture_name", gesture).execute()
        if response.data:
            action_script = response.data[0]["action_script"]
            try:
                subprocess.run(["cmd", "/c", action_script], check=True)
                # Set cooldown after successful execution
                last_action_time = current_time
                # Automatically pause after executing an action
                is_running = False
            except subprocess.CalledProcessError as e:
                print(f"Error executing script {action_script}: {e}")

    # Display frame
    cv2.imshow('Gesture Recognition', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_running = not is_running
        print("System", "started" if is_running else "paused")

# Cleanup
cap.release()
cv2.destroyAllWindows()