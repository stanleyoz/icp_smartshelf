#!/usr/bin/env python3
"""
HAILO Detection Module using DeGirum PySDK for simplified integration.
"""

import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
import threading
import sys
import traceback

# Import DeGirum PySDK
try:
    import degirum as dg
    HAS_DEGIRUM = True
    print("✅ DeGirum PySDK imported successfully.")
except ImportError as e:
    print(f"❌ DeGirum PySDK import failed: {e}", file=sys.stderr)
    HAS_DEGIRUM = False

# Global references for OpenCV and NumPy, passed from the main script
# These will be assigned values when get_hailo_detector is called.
cv2_ref = None
np_ref = None

@dataclass
class Detection:
    """Detection result from DeGirum PySDK output"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int # COCO class ID (e.g., 0 for person)
    label: str # e.g., 'person'

class HailoDetector:
    """HAILO detector that matches working implementation pattern"""

    # Note: hef_path is changed to model_name to align with DeGirum's load_model API
    def __init__(self, model_path: str, input_size: int = 600, cv2_ref_passed=None, np_ref_passed=None): # ADD cv2_ref_passed, np_ref_passed
        self.model_path = model_path # Store model_name
        self.input_size = input_size # Model input size (e.g., 600 for DeGirum example)
        
        # Store references to cv2 and numpy modules
        global cv2_ref, np_ref # Declare globals to be assigned
        cv2_ref = cv2_ref_passed
        np_ref = np_ref_passed
        
        self.device_type = None # Will store "HAILORT/HAILO8" or "HAILORT/HAILO8L"
        self.model = None # Stores the loaded DeGirum model object
        self.is_initialized = False
        self.lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize HAILO using DeGirum PySDK"""
        if not HAS_DEGIRUM:
            print("❌ DeGirum PySDK not available. Cannot initialize HAILO.", file=sys.stderr)
            return False

        with self.lock:
            try:
                print(f"DEBUG: Initializing DeGirum model: {self.model_path}")

                # 1. Get supported devices from DeGirum
                supported_devices = dg.get_supported_devices(inference_host_address="@local")
                print(f"DEBUG: Supported DeGirum devices: {list(supported_devices)}")

                # 2. Determine appropriate device_type for HAILO8 (or HAILO8L)
                device_type = None
                if "HAILORT/HAILO8" in supported_devices: 
                    device_type = "HAILORT/HAILO8"
                elif "HAILORT/HAILO8L" in supported_devices: 
                    device_type = "HAILORT/HAILO8L"
                else:
                    raise Exception("Hailo device not found or not supported by DeGirum PySDK. Check installation.")

                print(f"DEBUG: Using device type: {device_type}")

                # 3. Load AI model using DeGirum's API
                
                inference_host_address = "@local"
                zoo_url = "degirum/hailo"
                token = ""

                # Set model name and image source
                model_name = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1"
                image_source = "assets/ThreePersons.jpg"

                # Load AI model
                try:
                    self.model = dg.load_model(
                        model_name=model_name,
                        inference_host_address=inference_host_address,
                        zoo_url=zoo_url,
                        token=token,
                        device_type=device_type,
                    )
                except Exception as e:
                    print(f"Error loading model '{model_name}': {e}")
                    sys.exit(1)
                              
                
                print(f"✅ DeGirum model '{self.model_name}' loaded successfully.")

                self.is_initialized = True
                print("✅ HAILO initialization complete using DeGirum PySDK!")
                return True

            except Exception as e:
                print(f"❌ HAILO initialization failed using DeGirum PySDK: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame using DeGirum PySDK"""
        if not self.is_initialized:
            # If detector is not initialized, return empty list (no fallback to simulation,
            # as the main script handles simulation if this returns empty)
            return []

        with self.lock:
            try:
                # DeGirum models can take NumPy arrays directly (like frame from Picamera2).
                # The self.model() call handles necessary preprocessing (resizing, normalization,
                # channel order) internally based on the loaded model's requirements.
                
                # Ensure the input frame is C-contiguous if required by DeGirum.
                # Picamera2 frames are usually contiguous, but good to check.
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np_ref.ascontiguousarray(frame)

                inference_result = self.model(frame) # Perform inference directly on the frame

                detections = []
                # Parse DeGirum's structured output
                # DeGirum returns an iterable of result objects, which have direct attributes.
                for obj in inference_result:
                    # Print raw details for debugging
                    print(f"DEBUG: DeGirum Raw Object: Label={obj.label}, Score={obj.score:.4f}, Bbox={obj.bbox}", file=sys.stderr)

                    # Apply confidence threshold and filter for 'person'
                    if obj.label == "person" and obj.score >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = obj.bbox # Bbox is already in pixel coordinates (e.g., 0-640)
                        
                        # Add a final clip to frame boundaries just in case (as DeGirum might not clip fully)
                        # Ensure these use original frame dimensions (frame.shape[1] for width, frame.shape[0] for height)
                        x1 = max(0, min(int(x1), frame.shape[1]-1))
                        y1 = max(0, min(int(y1), frame.shape[0]-1))
                        x2 = max(0, min(int(x2), frame.shape[1]-1))
                        y2 = max(0, min(int(y2), frame.shape[0]-1))

                        # Ensure valid box after clipping
                        if x2 <= x1 or y2 <= y1:
                            print(f"WARNING: Invalid bbox after clipping for label '{obj.label}'. Skipping.", file=sys.stderr)
                            continue

                        detections.append(Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(obj.score),
                            class_id=obj.category_id, # category_id 0 is typically person in COCO
                            label=obj.label # Store label for debugging/display
                        ))
                
                return detections
                
            except Exception as e:
                print(f"HAILO detection (DeGirum infer) error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return []

    # _preprocess_frame and _postprocess are no longer needed as separate steps for DeGirum inference.
    # The 'detect' method handles everything. They can be removed or left as empty placeholders.
    # I'll leave them here as empty placeholders as they are called in o4_monitor.py
    # If they are called and cause issues, remove them from o4_monitor.py's flow.

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        # This function is not called by the DeGirum path for inference.
        # It's kept as a placeholder to avoid NameErrors if older parts of o4_monitor still call it.
        # DeGirum's model() call handles all preprocessing internally.
        print("WARNING: _preprocess_frame called. This should not happen with DeGirum PySDK.", file=sys.stderr)
        return frame # Return frame as is, DeGirum will handle internal preproc

    def _postprocess(self, outputs: dict, original_frame_shape: Tuple[int, int]) -> List[Detection]:
        # This function is not called by the DeGirum path for inference.
        # It's kept as a placeholder.
        print("WARNING: _postprocess called. This should not happen with DeGirum PySDK.", file=sys.stderr)
        return []
    
    def cleanup(self):
        """Clean up HAILO resources (model release)"""
        with self.lock:
            if self.model: # Check for the DeGirum model object
                try:
                    self.model.release() # DeGirum models have a release method
                    print("DeGirum model released")
                except Exception as e:
                    print(f"Error during DeGirum model release: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            self.is_initialized = False

# Singleton instance
_detector_instance = None
_detector_lock = threading.Lock()

def get_hailo_detector(model_path: str, cv2_ref_passed=None, np_ref_passed=None) -> Optional[HailoDetector]: # Changed model_name to model_path
    global _detector_instance

    with _detector_lock:
        if _detector_instance is None:
            _detector_instance = HailoDetector(model_path, cv2_ref_passed=cv2_ref_passed, np_ref_passed=np_ref_passed) # Pass model_path
            if not _detector_instance.initialize():
                _detector_instance = None

        return _detector_instance

def cleanup_hailo():
    """Clean up HAILO resources using the singleton instance"""
    global _detector_instance
    
    with _detector_lock:
        if _detector_instance:
            _detector_instance.cleanup()
            _detector_instance = None

# Example usage for testing (this block is not used by monitor.py)
if __name__ == "__main__":
    import cv2 # Local import for example usage
    import numpy as np_local # Use np_local to distinguish from outer np_ref

    # Test initialization (use a specific model name for DeGirum)
    test_model_name = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1" # This is the model from test.py
    detector = get_hailo_detector(test_model_name, cv2_ref_passed=cv2, np_ref_passed=np_local) 
    
    if detector:
        print("Testing with camera (standalone mode - DeGirum)...")
        cap = cv2.VideoCapture(0) # Assumes /dev/video0 is available and accessible
        
        if not cap.isOpened():
            print("Error: Could not open video capture device.", file=sys.stderr)
            sys.exit(1)

        try:
            for i in range(300):  # Test 300 frames (~10 seconds at 30 FPS)
                ret, frame = cap.read() # Read frame from OpenCV VideoCapture (BGR format)
                if not ret:
                    print("Failed to grab frame from camera.", file=sys.stderr)
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # DeGirum might expect RGB
                
                detections = detector.detect(frame_rgb)
                print(f"Frame {i}: Detected {len(detections)} persons")
                
                # Draw detections on the original BGR frame for display
                for det in detections:
                    x1, y1, x2, y2 = det.bbox # Detections are already pixel coords
                    color = (0, 255, 0) # Green for person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{det.label} {det.confidence:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imshow("DeGirum HAILO Detector Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cleanup_hailo()
            print("Standalone test finished and cleaned up.")
    else:
        print("Failed to initialize DeGirum HAILO detector for standalone test.", file=sys.stderr)
