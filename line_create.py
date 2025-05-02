import cv2
import numpy as np

def load_first_frame(video_path):
    """Load first frame from video"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read video frame")
    
    # Resize to standard DeepStream dimensions
    frame = cv2.resize(frame, (1280, 720))
    return frame

def draw_line_crossing_config(video_path):
    """
    Interactive line configuration for DeepStream
    
    :param video_path: Path to input video
    :return: List of 4 points (8 coordinates)
    """
    points = []
    frame = load_first_frame(video_path)
    display_frame = frame.copy()
    
    def mouse_handler(event, x, y, flags, param):
        nonlocal points, display_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Limit to 4 points
            if len(points) < 4:
                points.append((x, y))
                
                # Draw point
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Line Configuration', display_frame)
                
                # Draw lines and arrows
                if len(points) == 2:
                    # Direction vector (green arrow)
                    cv2.arrowedLine(display_frame, points[0], points[1], 
                                     (0, 255, 0), 2)
                    cv2.imshow('Line Configuration', display_frame)
                
                if len(points) == 4:
                    # Crossing line (red)
                    cv2.line(display_frame, points[2], points[3], 
                             (0, 0, 255), 2)
                    cv2.imshow('Line Configuration', display_frame)
    
    # Create window
    cv2.namedWindow('Line Configuration')
    cv2.setMouseCallback('Line Configuration', mouse_handler)
    
    # Show initial frame
    cv2.imshow('Line Configuration', display_frame)
    
    # Instructions
    print("\n--- DeepStream Line Configuration ---")
    print("Instructions:")
    print("1. Click 2 points for direction vector (green arrow)")
    print("2. Click 2 more points for crossing line (red line)")
    print("3. Close window after selecting 4 points")
    
    # Wait until window is closed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print coordinates in DeepStream format
    if len(points) == 4:
        coords_str = ";".join([str(coord) for point in points for coord in point])
        print("\n--- Line Crossing Coordinates ---")
        print(coords_str)
        return points
    
    return None

# Example Usage
if __name__ == "__main__":
    video_path = "1.mp4"  # Your video path
    draw_line_crossing_config(video_path)
