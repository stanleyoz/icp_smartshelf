"""
Development Configuration for Traffic Monitor
Provides fallback modes when hardware is not available
"""
import os

# Development mode settings
DEV_MODE = False  # Enable live mode for real inference
SIMULATE_CAMERA = False  # Use real camera when available
SIMULATE_HAILO = False  # Use real HAILO inference
USE_WEBCAM = True  # Enable webcam for live camera mode

# Paths
MONITOR_PATH = os.path.expanduser("~/g_traffic")
TRACKINGLOG_PATH = os.path.join(MONITOR_PATH, "trackinglog")

# Model settings for development
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3

# Simulation settings
SIMULATE_PERSON_COUNT = 2  # Number of simulated people
SIMULATION_MOVEMENT_SPEED = 5  # Pixels per frame
SIMULATION_FRAME_SIZE = (640, 480)

# Flask settings
FLASK_HOST = "0.0.0.0"  # localhost for development
FLASK_PORT = 5000
FLASK_DEBUG = False  # Set to True for development debugging

# Hardware fallbacks
HARDWARE_AVAILABLE = {
    'picamera2': False,
    'hailo': False,
    'degirum': False
}

def check_hardware():
    """Check what hardware/software is actually available"""
    global HARDWARE_AVAILABLE
    
    # Check for Picamera2
    try:
        from picamera2 import Picamera2
        HARDWARE_AVAILABLE['picamera2'] = True
    except ImportError:
        HARDWARE_AVAILABLE['picamera2'] = False
    
    # Check for HAILO
    try:
        import hailort
        HARDWARE_AVAILABLE['hailo'] = True
    except ImportError:
        HARDWARE_AVAILABLE['hailo'] = False
    
    # Check for DeGirum
    try:
        import degirum
        HARDWARE_AVAILABLE['degirum'] = True
    except ImportError:
        HARDWARE_AVAILABLE['degirum'] = False
    
    return HARDWARE_AVAILABLE

def get_camera_source():
    """Get the best available camera source"""
    # Ensure hardware is checked first
    check_hardware()
    if HARDWARE_AVAILABLE['picamera2'] and not SIMULATE_CAMERA:
        return 'picamera2'
    elif USE_WEBCAM:
        # Check if webcam is actually available
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return 'webcam'
        except:
            pass
        print("⚠️  Webcam not available, using simulation")
        return 'simulation'
    else:
        return 'simulation'

def get_inference_mode():
    """Get the best available inference mode"""
    # Ensure hardware is checked first
    check_hardware()
    if HARDWARE_AVAILABLE['degirum'] and not SIMULATE_HAILO:
        return 'hailo'  # DeGirum can run HAILO models
    else:
        return 'simulation'

def enable_webcam_mode():
    """Enable webcam mode for development"""
    global SIMULATE_CAMERA, USE_WEBCAM
    SIMULATE_CAMERA = False
    USE_WEBCAM = True
    return get_camera_source()