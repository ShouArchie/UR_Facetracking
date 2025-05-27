#!/usr/bin/env python3
"""
Simple camera test script to debug camera issues
"""

import cv2
import time

def test_basic_camera():
    """Test basic camera functionality"""
    print("Testing basic camera access...")
    
    # Try different camera indices
    for cam_index in range(5):
        print(f"Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        
        if cap.isOpened():
            print(f"Camera {cam_index} opened successfully!")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"Camera {cam_index} can capture frames!")
                print(f"Frame shape: {frame.shape}")
                
                # Show frame for 2 seconds
                cv2.imshow(f'Camera {cam_index} Test', frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
                cap.release()
                return cam_index
            else:
                print(f"Camera {cam_index} opened but cannot read frames")
        else:
            print(f"Camera {cam_index} failed to open")
        
        cap.release()
    
    print("No working cameras found!")
    return None

def test_camera_with_settings(cam_index):
    """Test camera with specific settings"""
    print(f"\nTesting camera {cam_index} with settings...")
    
    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print("Camera failed to open")
        return False
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    cap.set(cv2.CAP_PROP_FPS, 13)
    
    print("Camera settings applied, testing capture...")
    
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1} captured successfully")
            cv2.imshow('Camera Test with Settings', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Failed to capture frame {i+1}")
            break
        time.sleep(0.1)
    
    cv2.destroyAllWindows()
    cap.release()
    return True

if __name__ == "__main__":
    print("Camera debugging script")
    print("=" * 40)
    
    # Test basic camera access
    working_cam = test_basic_camera()
    
    if working_cam is not None:
        # Test with settings
        test_camera_with_settings(working_cam)
        print(f"\nWorking camera index: {working_cam}")
        print("You can use this camera index in your main script")
    else:
        print("\nNo working cameras found. Check:")
        print("1. Camera is connected and not used by other apps")
        print("2. Camera drivers are installed")
        print("3. Camera permissions are granted") 