#!/usr/bin/env python3

"""
HAILO Detection Module using DeGirum PySDK for simplified integration.
Modified to follow the working test.py pattern exactly.
"""

import numpy as np
import time
import os
from pathlib import Path
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
cv2_ref = None
np_ref = None

# Configuration
CONFIDENCE_THRESHOLD = 0.6  # Only detect persons with high confidence
LOCAL_MODEL_DIR = "./models"  # Default local model directory

@dataclass
class Detection:
    """Detection result from DeGirum PySDK output"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int  # COCO class ID (e.g., 0 for person)
    label: str  # e.g., 'person'

class HailoDetector:
    """HAILO detector that follows working test.py pattern exactly"""

    def __init__(self, model_name: str, input_size: int = 640, cv2_ref_passed=None, np_ref_passed=None):
        self.model_name = model_name  # Store model_name correctly
        self.input_size = input_size
        
        # Store references to cv2 and numpy modules
        global cv2_ref, np_ref
        cv2_ref = cv2_ref_passed
        np_ref = np_ref_passed
        
        self.device_type = None
        self.model = None
        self.is_initialized = False
        self.lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize HAILO using DeGirum PySDK following test.py pattern exactly"""
        if not HAS_DEGIRUM:
            print("❌ DeGirum PySDK not available. Cannot initialize HAILO.", file=sys.stderr)
            return False

        with self.lock:
            try:
                print(f"DEBUG: Initializing DeGirum model: {self.model_name}")

                # 1. Get supported devices from DeGirum (exactly like test.py)
                try:
                    supported_devices = dg.get_supported_devices(inference_host_address="@local")
                    print(f"DEBUG: Supported DeGirum devices: {list(supported_devices)}")
                except Exception as e:
                    print(f"Error fetching supported devices: {e}")
                    return False

                # 2. Determine appropriate device_type (exactly like test.py)
                if "HAILORT/HAILO8L" in supported_devices:
                    device_type = "HAILORT/HAILO8L"
                elif "HAILORT/HAILO8" in supported_devices:
                    device_type = "HAILORT/HAILO8"
                else:
                    print("Hailo device is NOT supported or NOT recognized properly. Please check the installation.")
                    return False

                print(f"DEBUG: Using device type: {device_type}")
                self.device_type = device_type

                # 3. Load model using correct local syntax from Hailo documentation
                inference_host_address = "@local"
                
                # Use zoo_url to point to local model directory
                local_model_dir = os.path.abspath(f"./models/{self.model_name}")
                print(f"DEBUG: Loading model from local directory: {local_model_dir}")
                
                if os.path.exists(local_model_dir):
                    try:
                        # Correct syntax: use zoo_url for local model directory
                        self.model = dg.load_model(
                            model_name=self.model_name,
                            inference_host_address=inference_host_address,
                            zoo_url=local_model_dir,
                            device_type=device_type,
                        )
                        print(f"✅ Local model '{self.model_name}' loaded successfully from {local_model_dir}")
                    except Exception as local_error:
                        print(f"⚠️  Local model loading failed: {local_error}")
                        print("   Attempting to list available models in directory...")
                        try:
                            # Try to connect and list models for debugging
                            zoo_manager = dg.connect(
                                inference_host_address=inference_host_address,
                                zoo_url=local_model_dir
                            )
                            available_models = zoo_manager.list_models()
                            print(f"   Available models in directory: {available_models}")
                            if available_models:
                                # Try loading the first available model
                                first_model = available_models[0]
                                print(f"   Trying to load first available model: {first_model}")
                                self.model = dg.load_model(
                                    model_name=first_model,
                                    inference_host_address=inference_host_address,
                                    zoo_url=local_model_dir,
                                    device_type=device_type,
                                )
                                print(f"✅ Successfully loaded model: {first_model}")
                            else:
                                raise Exception("No models found in local directory")
                        except Exception as debug_error:
                            print(f"   Debug attempt failed: {debug_error}")
                            raise Exception(f"Failed to load any model from {local_model_dir}")
                else:
                    raise Exception(f"Local model directory not found: {local_model_dir}")

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
            return []

        with self.lock:
            try:
                # Ensure the input frame is C-contiguous if required by DeGirum
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np_ref.ascontiguousarray(frame)

                # Perform inference directly on the frame (exactly like test.py)
                inference_result = self.model(frame)
                detections = []

                # Parse DeGirum's structured output
                # Based on debug findings: results are in inference_result.results
                print(f"DEBUG: Inference result type: {type(inference_result)}", file=sys.stderr)
                
                try:
                    # The DetectionResults object has a .results attribute that contains the list
                    if hasattr(inference_result, 'results'):
                        results_list = inference_result.results
                        print(f"DEBUG: Found {len(results_list)} detection results via .results attribute", file=sys.stderr)
                    else:
                        print(f"DEBUG: No .results attribute found. Available attributes: {[attr for attr in dir(inference_result) if not attr.startswith('_')]}", file=sys.stderr)
                        return detections

                    # Process the results - we now know they are dictionaries
                    for i, obj in enumerate(results_list):
                        try:
                            # Detection results are dictionaries with keys: bbox, category_id, label, score
                            if isinstance(obj, dict):
                                label = obj.get('label', 'unknown')
                                score = obj.get('score', 0.0)
                                bbox = obj.get('bbox', None)
                                category_id = obj.get('category_id', 0)
                            else:
                                # Fallback for non-dict objects (shouldn't happen based on debug)
                                label = getattr(obj, 'label', 'unknown')
                                score = getattr(obj, 'score', 0.0)
                                bbox = getattr(obj, 'bbox', None)
                                category_id = getattr(obj, 'category_id', 0)
                            
                            # Only process person detections - skip all other classes
                            if label != "person":
                                continue
                            
                            # Apply confidence threshold for persons only
                            if score >= CONFIDENCE_THRESHOLD:
                                if bbox is None or len(bbox) != 4:
                                    print(f"WARNING: Invalid bbox for detection {i}: {bbox}", file=sys.stderr)
                                    continue
                                    
                                x1, y1, x2, y2 = bbox

                                # Clip to frame boundaries
                                x1 = max(0, min(int(x1), frame.shape[1]-1))
                                y1 = max(0, min(int(y1), frame.shape[0]-1))
                                x2 = max(0, min(int(x2), frame.shape[1]-1))
                                y2 = max(0, min(int(y2), frame.shape[0]-1))

                                # Ensure valid box after clipping
                                if x2 <= x1 or y2 <= y1:
                                    print(f"WARNING: Invalid bbox after clipping for label '{label}': ({x1},{y1},{x2},{y2})", file=sys.stderr)
                                    continue

                                detections.append(Detection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=float(score),
                                    class_id=int(category_id),
                                    label=label
                                ))
                                
                                print(f"✅ Added person detection: confidence={score:.4f}, bbox=({x1},{y1},{x2},{y2})", file=sys.stderr)
                            else:
                                print(f"DEBUG: Person detected but confidence {score:.4f} below threshold {CONFIDENCE_THRESHOLD}", file=sys.stderr)
                            
                        except Exception as e:
                            print(f"WARNING: Error processing detection {i}: {e}", file=sys.stderr)
                            continue

                except Exception as e:
                    print(f"DEBUG: Error parsing results: {e}", file=sys.stderr)
                    return detections

                return detections

            except Exception as e:
                print(f"HAILO detection (DeGirum infer) error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return []

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Placeholder - DeGirum handles preprocessing internally"""
        print("WARNING: _preprocess_frame called. This should not happen with DeGirum PySDK.", file=sys.stderr)
        return frame

    def _postprocess(self, outputs: dict, original_frame_shape: Tuple[int, int]) -> List[Detection]:
        """Placeholder - DeGirum handles postprocessing internally"""
        print("WARNING: _postprocess called. This should not happen with DeGirum PySDK.", file=sys.stderr)
        return []

    def cleanup(self):
        """Clean up HAILO resources"""
        with self.lock:
            if self.model:
                try:
                    # DeGirum models don't have a release() method
                    # Just set to None to allow garbage collection
                    print("DeGirum model cleanup (setting to None)")
                    self.model = None
                except Exception as e:
                    print(f"Error during DeGirum model cleanup: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            self.is_initialized = False

# Singleton instance
_detector_instance = None
_detector_lock = threading.Lock()

def get_hailo_detector(model_name: str, cv2_ref_passed=None, np_ref_passed=None) -> Optional[HailoDetector]:
    """Get or create singleton HAILO detector"""
    global _detector_instance

    with _detector_lock:
        if _detector_instance is None:
            _detector_instance = HailoDetector(model_name, cv2_ref_passed=cv2_ref_passed, np_ref_passed=np_ref_passed)
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

def test_hailo_detector():
    """Test function exactly like test.py"""
    try:
        print("Testing HAILO detector...")
        
        # Test with the same model as test.py
        model_name = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1"
        detector = HailoDetector(model_name)
        
        if detector.initialize():
            print("✅ HAILO detector initialized successfully")
            
            # Create a dummy frame for testing
            import numpy as np
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Try detection
            detections = detector.detect(test_frame)
            print(f"Detection test completed. Found {len(detections)} detections.")
            
            detector.cleanup()
            return True
        else:
            print("❌ HAILO detector initialization failed")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test if module is executed directly
    test_hailo_detector()
