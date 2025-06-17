#!/usr/bin/env python3
"""
Enhanced Integrated Flask Dashboard with Camera, HAILO Inference, and Advanced Person Tracking
Implements proper person detection, tracking, and counting for shopper traffic analysis
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
from datetime import datetime, timedelta
import threading
import os
import sys
import glob
import time
import base64
from collections import deque, defaultdict
import signal
import csv
import traceback
import numpy as np # Keep numpy import here, as it's needed globally
from dataclasses import dataclass # Keep dataclass here, as Detection is defined in this file
from typing import List, Tuple, Dict, Optional
import json

# Import HAILO detection module
# This now uses the hailo_detection_module.py you provided
try:
    from o4_hailo_detection_module import get_hailo_detector, cleanup_hailo, HailoDetector # Import HailoDetector too
    HAS_HAILO_MODULE = True
    print("✅ HAILO detection module imported successfully.")
except ImportError as e:
    print(f"❌ HAILO detection module import failed: {e}", file=sys.stderr)
    HAS_HAILO_MODULE = False

# Camera and CV imports
HAS_CAMERA = False
# Removed HAS_HAILO from here, it's now set based on hailo_detection_module success
picam2 = None

# Initialize cv2_module and np_module as None globally first.
# They will be assigned the actual imported modules within the try-except block below.
cv2_module = None
np_module = None

try:
    # Attempt to import OpenCV and NumPy first
    import cv2 as imported_cv2
    import numpy as imported_np
    cv2_module = imported_cv2
    np_module = imported_np
    print("✅ OpenCV and NumPy imported successfully.")

    try:
        from picamera2 import Picamera2
        HAS_CAMERA = True
        print("✅ Picamera2 imported successfully.")
    except ImportError as e:
        print(f"WARNING: Picamera2 not available. Live camera feed will be disabled. ({e})", file=sys.stderr)

except ImportError as e:
    print(f"CRITICAL ERROR: OpenCV or NumPy not available. ({e})", file=sys.stderr)
    HAS_CAMERA = False

# --- Configuration ---
MONITOR_PATH = os.path.expanduser("~/g_traffic")
TRACKINGLOG_PATH = os.path.join(MONITOR_PATH, "trackinglog")
#HAILO_MODEL_HEF = os.path.join(MONITOR_PATH, "models/yolov11s.hef") # <--- Ensure this path is correct!
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.3  # For tracking association

# HAILO_MODEL_HEF is no longer used for model loading with DeGirum
# Instead, use a model name from DeGirum's Model Zoo or local model path
# CHANGED: Use the same model name as the working test.py
DEGIRUM_MODEL_NAME = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1"  # Same as working test.py
# For local models, you can use: DEGIRUM_MODEL_NAME = "./models/your_model.hef"

# Force cloud loading for now (comment out LOCAL_MODEL_DIR to disable local model search)
# LOCAL_MODEL_DIR = None  # Uncomment this line to force cloud model loading

# --- Global Variables ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Tracking data
current_person_count = 0
historical_data = None
data_lock = threading.Lock()
last_data_update = None
stop_event = threading.Event()

# CSV Logging
csv_filename = None
csv_file = None
csv_writer = None

# HAILO inference
hailo_detector = None # This will hold the HailoDetector instance

# --- Enhanced Person Tracking Classes (Leave these as they are, they are fine) ---
# Your @dataclass Detection and classes TrackedPerson, AdvancedPersonTracker are here.
# Note: The Detection class definition needs to be before its use in Detection: Tuple[int, int, int, int]
# The Detection class from hailo_detection_module.py is also used, ensure consistency.
# If these two Detection classes conflict, you might need to use `as` alias for one.
# For now, let's rename the local Detection class to TrackedDetection to avoid conflict.

@dataclass
class TrackedDetection: # Renamed to avoid conflict with hailo_detection_module.Detection
    """Represents a single person detection for tracking purposes"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    timestamp: float

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class TrackedPerson:
    """Enhanced with memory limits"""
    def __init__(self, track_id: int, initial_detection: TrackedDetection):
        self.track_id = track_id
        # REDUCED: Limit detection history to save memory
        self.detections = deque(maxlen=10)  # Only keep last 10 detections (was 30)
        self.detections.append(initial_detection)
        self.last_seen = initial_detection.timestamp
        self.first_seen = initial_detection.timestamp
        self.counted = False
        self.direction = None
        self.avg_velocity = (0, 0)

    def update(self, detection: TrackedDetection):
        """Update track with new detection"""
        self.detections.append(detection)
        self.last_seen = detection.timestamp
        self._calculate_direction()

    def _calculate_direction(self):
        """Calculate movement direction based on recent positions"""
        if len(self.detections) < 5:
            return

        recent_positions = [d.center for d in list(self.detections)[-5:]]
        dx = recent_positions[-1][0] - recent_positions[0][0]
        dy = recent_positions[-1][1] - recent_positions[0][1]

        self.avg_velocity = (dx / 5, dy / 5)

        # Determine primary direction
        if abs(dx) > abs(dy):
            self.direction = "right" if dx > 0 else "left"
        else:
            self.direction = "down" if dy > 0 else "up"

    @property
    def is_active(self) -> bool:
        """Check if track is still active (seen recently)"""
        return (time.time() - self.last_seen) < 2.0  # 2 second timeout

    def predict_next_position(self) -> Tuple[float, float]:
        """Predict next position based on velocity"""
        if not self.detections:
            return (0, 0)

        last_center = self.detections[-1].center
        return (
            last_center[0] + self.avg_velocity[0],
            last_center[1] + self.avg_velocity[1]
        )

class AdvancedPersonTracker:
    """Enhanced with memory management"""
    def __init__(self):
        self.tracks: Dict[int, TrackedPerson] = {}
        self.next_track_id = 1
        # REDUCED: Limit history to prevent memory bloat
        self.person_count_history = deque(maxlen=60)  # Only 2 seconds at 30fps (was 150)
        self.unique_persons_counted = 0
        self.entry_zones = []
        self.exit_zones = []
        self.counted_track_ids = set()
        
        # Add memory management
        self.last_cleanup_time = time.time()
        self.max_tracks_to_keep = 50  # Limit total tracks in memory

    def update(self, detections: List[TrackedDetection]) -> List[TrackedPerson]:
        """Update tracker with memory management - COMPLETE VERSION"""
        current_time = time.time()

        # ENHANCED: More aggressive cleanup of old tracks
        old_track_count = len(self.tracks)
        # Remove tracks not seen for 5 seconds (was 2 seconds)
        self.tracks = {tid: track for tid, track in self.tracks.items()
                      if (current_time - track.last_seen) < 5.0}

        # ADDED: Limit total number of tracks to prevent memory bloat
        if len(self.tracks) > self.max_tracks_to_keep:
            # Remove oldest tracks first
            sorted_tracks = sorted(self.tracks.items(), key=lambda x: x[1].last_seen)
            tracks_to_remove = len(self.tracks) - self.max_tracks_to_keep
            for i in range(tracks_to_remove):
                track_id, _ = sorted_tracks[i]
                del self.tracks[track_id]
            print(f"DEBUG: Removed {tracks_to_remove} old tracks to free memory", file=sys.stderr)

        # ADDED: Periodic cleanup of counted_track_ids set
        if current_time - self.last_cleanup_time > 300:  # Every 5 minutes
            # Keep only recent track IDs (last 100)
            if len(self.counted_track_ids) > 100:
                # Convert to list, sort, keep recent ones
                recent_ids = sorted(list(self.counted_track_ids))[-100:]
                self.counted_track_ids = set(recent_ids)
                print(f"DEBUG: Cleaned up counted_track_ids, now has {len(self.counted_track_ids)} items", file=sys.stderr)
            self.last_cleanup_time = current_time

        # MISSING LOGIC - ADD THIS:
        if not detections:
            self.person_count_history.append(len(self.tracks))
            return list(self.tracks.values())

        print(f"DEBUG: Processing {len(detections)} new detections, {len(self.tracks)} existing tracks")

        # Associate detections with existing tracks
        matched_pairs, unmatched_detections = self._associate_detections(detections)

        print(f"DEBUG: Matched {len(matched_pairs)} detections, {len(unmatched_detections)} unmatched")

        # Update matched tracks
        for detection, track_id in matched_pairs:
            self.tracks[track_id].update(detection)

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = TrackedPerson(self.next_track_id, detection)
            self.tracks[self.next_track_id] = new_track

            # SIMPLIFIED: Count every new track as unique (for testing)
            if detection.confidence > 0.5:  # Lower threshold
                if self.next_track_id not in self.counted_track_ids:
                    self.unique_persons_counted += 1
                    self.counted_track_ids.add(self.next_track_id)
                    new_track.counted = True
                    print(f"? NEW UNIQUE PERSON: Track ID {self.next_track_id}, Total: {self.unique_persons_counted}", file=sys.stderr)

            self.next_track_id += 1

        # Update count history
        self.person_count_history.append(len(self.tracks))

        return list(self.tracks.values())
    

    def _associate_detections(self, detections: List[TrackedDetection]) -> Tuple[List[Tuple[TrackedDetection, int]], List[TrackedDetection]]:
        """Associate detections with existing tracks using IoU"""
        if not self.tracks:
            return [], detections

        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))

        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                predicted_pos = track.predict_next_position()
                iou = self._calculate_iou_with_prediction(detection, track.detections[-1], predicted_pos)
                iou_matrix[i, j] = iou

        # Simple greedy matching (can be replaced with Hungarian algorithm)
        matched_pairs = []
        used_detections = set()
        used_tracks = set()

        # Sort by IoU in descending order
        indices = np.argwhere(iou_matrix > IOU_THRESHOLD)
        indices = indices[np.argsort(iou_matrix[indices[:, 0], indices[:, 1]])[::-1]]

        for det_idx, track_idx in indices:
            if det_idx not in used_detections and track_idx not in used_tracks:
                matched_pairs.append((detections[det_idx], track_ids[track_idx]))
                used_detections.add(det_idx)
                used_tracks.add(track_idx)

        unmatched_detections = [d for i, d in enumerate(detections) if i not in used_detections]

        return matched_pairs, unmatched_detections

    def _calculate_iou_with_prediction(self, det1: TrackedDetection, det2: TrackedDetection, predicted_center: Tuple[float, float]) -> float:
        """Calculate IoU between two detections with motion prediction"""
        # Adjust det2's position based on predicted movement
        x1_1, y1_1, x2_1, y2_1 = det1.bbox
        x1_2, y1_2, x2_2, y2_2 = det2.bbox

        # Apply predicted offset
        offset_x = predicted_center[0] - det2.center[0]
        offset_y = predicted_center[1] - det2.center[1]

        x1_2 += offset_x
        x2_2 += offset_x
        y1_2 += offset_y
        y2_2 += offset_y

        # Calculate IoU
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _is_entering_zone(self, detection: TrackedDetection) -> bool:
        """Check if detection is in entry zone - SIMPLIFIED for testing"""
        # TEMPORARILY make this always return True to test unique counting
        # This will count every new detection as a unique person
        print(f"DEBUG: _is_entering_zone called for detection with confidence {detection.confidence:.3f}", file=sys.stderr)
        return True  # For testing - count all new detections
        
        # Original logic (comment out for now):
        # x1, y1, x2, y2 = detection.bbox
        # center_x = (x1 + x2) / 2
        # center_y = (y1 + y2) / 2
        # margin = 50
        # frame_width = 640
        # frame_height = 480
        # is_at_edge = (
        #     center_x < margin or
        #     center_x > frame_width - margin or
        #     center_y < margin or
        #     center_y > frame_height - margin
        # )
        # return is_at_edge

    # Also update the unique person counting logic in the update method:

    def get_current_count(self) -> int:
        """Get smoothed current person count"""
        if not self.person_count_history:
            return 0

        # Use median of recent counts for stability
        recent_counts = list(self.person_count_history)[-30:]  # Last second
        return int(np.median(recent_counts)) if recent_counts else 0

    def get_analytics(self) -> Dict:
        """Get tracking analytics"""
        active_tracks = [t for t in self.tracks.values() if t.is_active]

        direction_counts = defaultdict(int)
        for track in active_tracks:
            if track.direction:
                direction_counts[track.direction] += 1

        return {
            'current_count': self.get_current_count(),
            'unique_persons_total': self.unique_persons_counted,
            'active_tracks': len(active_tracks),
            'direction_distribution': dict(direction_counts),
            'avg_dwell_time': np.mean([t.last_seen - t.first_seen for t in self.tracks.values()]) if self.tracks else 0
        }





# Initialize tracker
person_tracker = AdvancedPersonTracker()


# --- HAILO Inference Setup ---

def setup_hailo_inference():

    global hailo_detector # Declare global to assign the detector instance

    if not HAS_HAILO_MODULE: # Check if the module itself imported
        print("DEBUG: HAILO detection module not available. Inference will be disabled.", file=sys.stderr)
        return False

    # Model file path is no longer needed here, DeGirum loads by name
    # if not os.path.exists(HAILO_MODEL_HEF):
    #     print(f"ERROR: HAILO model file not found at {HAILO_MODEL_HEF}", file=sys.stderr)
    #     return False

    try:
        print(f"DEBUG: Setting up HAILO detector with DeGirum model: {DEGIRUM_MODEL_NAME}")

        # Get singleton detector instance from the imported module
        # FIXED: Pass cv2_module and np_module with correct parameter names
        hailo_detector = get_hailo_detector(DEGIRUM_MODEL_NAME, cv2_ref_passed=cv2_module, np_ref_passed=np_module)

        if hailo_detector and hailo_detector.is_initialized:
            print("✅ HAILO detector initialized successfully (DeGirum PySDK)")
            return True
        else:
            print("❌ HAILO detector initialization failed (DeGirum PySDK)", file=sys.stderr)
            return False

    except Exception as e:
        print(f"❌ HAILO setup error (DeGirum PySDK): {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False

def run_hailo_inference(frame_rgb):
    """Run HAILO inference and return Detection objects"""
    current_time = time.time() # Get current time for Detection objects

    # Check if detector is initialized, if not, use simulation
    if not hailo_detector or not hailo_detector.is_initialized:
        # Simulate detections for testing - FIXED to be more stable
        # This simulation now generates Detection objects that match the tracker's expected format
        detections = []
        if hasattr(run_hailo_inference, 'sim_positions'):
            for i, pos in enumerate(run_hailo_inference.sim_positions):
                pos[0] += np_module.random.randint(-5, 6) # Use np_module
                pos[1] += np_module.random.randint(-5, 6) # Use np_module
                pos[0] = max(50, min(550, pos[0]))
                pos[1] = max(50, min(400, pos[1]))

                det = TrackedDetection( # Use TrackedDetection for simulation
                    bbox=(pos[0], pos[1], pos[0] + 80, pos[1] + 120),
                    confidence=0.85,
                    timestamp=current_time
                )
                detections.append(det)
        else:
            run_hailo_inference.sim_positions = [[200, 200], [400, 300]]
        return detections

    try:
        # Run detection using the HailoDetector instance
        # The detect method in hailo_detection_module.py returns a List[Detection]
        hailo_detections = hailo_detector.detect(frame_rgb)

        # Convert HailoDetection objects from the module to TrackedDetection objects for the tracker
        tracked_detections = []
        for det in hailo_detections:
            tracked_detections.append(TrackedDetection( # Create TrackedDetection from HailoDetection
                bbox=det.bbox,
                confidence=det.confidence,
                timestamp=current_time # Use current_time for consistency
            ))

        return tracked_detections

    except Exception as e:
        print(f"HAILO inference execution error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []

# --- Camera & Inference Thread ---

def camera_and_inference_thread():
    """Complete camera and inference thread with memory optimization"""
    global picam2, HAS_CAMERA, current_person_count, cv2_module, np_module, hailo_detector, HAS_HAILO_MODULE

    print("DEBUG: camera_and_inference_thread started.")

    # Memory monitoring variables
    last_memory_check = time.time()
    frame_count = 0
    last_debug_time = 0

    camera_initialized = False
    if HAS_CAMERA:
        print("DEBUG: HAS_CAMERA is True. Initializing camera hardware.")
        camera_initialized = init_camera()
        if not camera_initialized:
            print("DEBUG: Camera initialization failed.")

    if not camera_initialized:
        print("DEBUG: Starting placeholder stream.")
        socketio.start_background_task(target=_send_placeholder_frames)
        return

    # Initialize HAILO
    print("DEBUG: Setting up HAILO inference.")
    hailo_initialized = setup_hailo_inference()
    if not hailo_initialized:
        print("DEBUG: HAILO inference setup failed. Will use simulated detections.")

    print("DEBUG: Entering main camera/inference loop.")
    frame_delay = 1.0 / 30  # 30 FPS
    last_frame_time = time.time()
    last_csv_log_time = time.time()
    last_analytics_log_time = time.time()

    while not stop_event.is_set():
        try:
            current_time = time.time()
            frame_count += 1

            # Memory monitoring every 60 seconds
            if current_time - last_memory_check >= 60:
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"DEBUG: Memory usage: {memory_mb:.1f} MB, Frame count: {frame_count}")
                    
                    # Memory cleanup if usage is high
                    if memory_mb > 500:  # If using more than 500MB
                        print("DEBUG: High memory usage detected, forcing cleanup")
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Clear old tracking data
                        if hasattr(person_tracker, 'person_count_history'):
                            if len(person_tracker.person_count_history) > 30:
                                # Keep only last 30 items
                                recent_items = list(person_tracker.person_count_history)[-30:]
                                person_tracker.person_count_history.clear()
                                person_tracker.person_count_history.extend(recent_items)
                                print("DEBUG: Cleared person count history")
                            
                except ImportError:
                    print("DEBUG: psutil not available for memory monitoring")
                except Exception as e:
                    print(f"DEBUG: Memory check error: {e}")
                
                last_memory_check = current_time

            # Frame rate control
            if current_time - last_frame_time < frame_delay:
                time.sleep(frame_delay - (current_time - last_frame_time))
                current_time = time.time()

            # Capture frame
            frame = picam2.capture_array()  # RGB888

            # Run inference - now returns List[TrackedDetection]
            detections = run_hailo_inference(frame)

            # CRITICAL: If HAILO module not loaded, show warning overlay
            if not HAS_HAILO_MODULE or not (hailo_detector and hailo_detector.is_initialized):
                cv2_module.putText(frame, "HAILO NOT AVAILABLE - USING SIMULATION",
                                 (50, 240), cv2_module.FONT_HERSHEY_SIMPLEX,
                                 0.8, (0, 0, 255), 2)

            # Update tracker
            active_tracks = person_tracker.update(detections)

            # Get analytics
            analytics = person_tracker.get_analytics()
            current_displayed_count = analytics['current_count']

            # Update global count
            with data_lock:
                current_person_count = current_displayed_count

            # Draw visualizations
            try:
                # Draw tracked persons with IDs
                for track in active_tracks:
                    if track.detections:
                        latest_det = track.detections[-1]
                        x1, y1, x2, y2 = latest_det.bbox

                        # Color based on tracking status
                        color = (0, 255, 0) if track.counted else (255, 255, 0)

                        # Draw bbox
                        cv2_module.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Draw track ID and info
                        label = f"Person {track.track_id}"
                        if track.direction:
                            label += f" {track.direction}"

                        # Add confidence score
                        label += f" ({latest_det.confidence:.2f})"

                        cv2_module.putText(frame, label, (x1, y1 - 10),
                                         cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Draw trajectory (last 5 positions)
                        if len(track.detections) > 1:
                            points = [det.center for det in list(track.detections)[-5:]]
                            for i in range(1, len(points)):
                                pt1 = tuple(map(int, points[i-1]))
                                pt2 = tuple(map(int, points[i]))
                                cv2_module.line(frame, pt1, pt2, color, 1)

                # Add overlay information
                cv2_module.putText(frame, f"Count: {current_displayed_count}",
                                 (10, 40), cv2_module.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                cv2_module.putText(frame, f"Unique Total: {analytics['unique_persons_total']}",
                                 (10, 80), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Direction info
                dir_text = f"Dir: {analytics['direction_distribution']}"
                cv2_module.putText(frame, dir_text,
                                 (10, 120), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Timestamp
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                cv2_module.putText(frame, timestamp_str,
                                 (10, 470), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # FPS
                fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
                cv2_module.putText(frame, f"FPS: {fps:.1f}",
                                 (550, 40), cv2_module.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Memory info (optional)
                if frame_count % 1800 == 0:  # Every minute at 30fps
                    try:
                        import psutil
                        process = psutil.Process(os.getpid())
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        cv2_module.putText(frame, f"RAM: {memory_mb:.0f}MB",
                                         (550, 70), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    except:
                        pass

                # Encode and emit frame
                _, buffer = cv2_module.imencode('.jpg', frame, [cv2_module.IMWRITE_JPEG_QUALITY, 85])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': jpg_as_text})

            except Exception as e:
                print(f"Error in visualization: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

            # Log to CSV (every second) - ENHANCED VERSION
            if current_time - last_csv_log_time >= 1.0:
                with data_lock:
                    if csv_writer:
                        try:
                            row_data = [
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                current_displayed_count,
                                analytics['unique_persons_total'],
                                json.dumps(analytics['direction_distribution'])
                            ]
                            csv_writer.writerow(row_data)
                            csv_file.flush()
                            
                            # Debug logging (reduced frequency)
                            if frame_count % 150 == 0:  # Every 5 seconds at 30fps
                                print(f"DEBUG: CSV logged - Count: {current_displayed_count}, Unique: {analytics['unique_persons_total']}", file=sys.stderr)
                                
                        except Exception as e:
                            print(f"ERROR: CSV logging failed: {e}", file=sys.stderr)
                            
                last_csv_log_time = current_time

            # Log detailed analytics (every 5 minutes)
            if current_time - last_analytics_log_time >= 300:
                log_detailed_analytics(analytics)
                last_analytics_log_time = current_time

            # DEBUG: Add metrics debugging every 30 seconds
            if current_time - last_debug_time >= 30:
                try:
                    print(f"DEBUG: Active tracks: {len(active_tracks)}, Memory tracks: {len(person_tracker.tracks)}")
                    print(f"DEBUG: Count history length: {len(person_tracker.person_count_history)}")
                    print(f"DEBUG: Counted track IDs: {len(person_tracker.counted_track_ids)}")
                    
                    # Cleanup old counted track IDs periodically
                    if len(person_tracker.counted_track_ids) > 200:
                        # Keep only recent 100 track IDs
                        recent_ids = sorted(list(person_tracker.counted_track_ids))[-100:]
                        person_tracker.counted_track_ids = set(recent_ids)
                        print(f"DEBUG: Cleaned up counted_track_ids to {len(person_tracker.counted_track_ids)} items")
                        
                except Exception as e:
                    print(f"DEBUG: Debug logging error: {e}")
                    
                last_debug_time = current_time

            last_frame_time = current_time

        except Exception as e:
            print(f"Critical error in camera/inference loop: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break

    # Cleanup on exit
    if not stop_event.is_set():
        print("DEBUG: Starting placeholder stream due to error.", file=sys.stderr)
        socketio.start_background_task(target=_send_placeholder_frames)


def log_detailed_analytics(analytics):
    """Log detailed analytics for LLM analysis"""
    analytics_file = os.path.join(TRACKINGLOG_PATH, f"analytics_{datetime.now().strftime('%Y%m%d')}.json")

    entry = {
        'timestamp': datetime.now().isoformat(),
        'current_count': analytics['current_count'],
        'unique_persons_total': analytics['unique_persons_total'],
        'direction_distribution': analytics['direction_distribution'],
        'avg_dwell_time': analytics['avg_dwell_time'],
        'active_tracks': analytics['active_tracks']
    }

    try:
        # Append to JSON file
        with open(analytics_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')
    except Exception as e:
        print(f"Error logging analytics: {e}", file=sys.stderr)

def init_camera():
    global picam2, HAS_CAMERA

    if not HAS_CAMERA:
        return False

    try:
        picam2 = Picamera2(0)
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        print("✅ Camera initialized successfully.")
        time.sleep(2)
        return True
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        HAS_CAMERA = False
        return False

def _send_placeholder_frames():
    global cv2_module, np_module

    if cv2_module is None or np_module is None:
        return

    while not stop_event.is_set():
        placeholder = np_module.zeros((480, 640, 3), dtype=np_module.uint8)
        placeholder[:] = (50, 50, 50)
        cv2_module.putText(placeholder, "Camera/HAILO Not Available",
                          (50, 240), cv2_module.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, buffer = cv2_module.imencode('.jpg', placeholder)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': jpg_as_text})
        time.sleep(1)

# --- Analytics and Reporting ---

def generate_llm_report():
    """Generate report for LLM analysis"""
    try:
        # Load recent analytics
        analytics_file = os.path.join(TRACKINGLOG_PATH, f"analytics_{datetime.now().strftime('%Y%m%d')}.json")

        if not os.path.exists(analytics_file):
            return None

        # Read last hour of analytics
        analytics_data = []
        with open(analytics_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if (datetime.now() - entry_time).total_seconds() < 3600:  # Last hour
                        analytics_data.append(entry)
                except:
                    continue

        if not analytics_data:
            return None

        # Calculate summary statistics
        counts = [e['current_count'] for e in analytics_data]
        unique_totals = [e['unique_persons_total'] for e in analytics_data]

        # Direction analysis
        all_directions = defaultdict(int)
        for entry in analytics_data:
            for direction, count in entry.get('direction_distribution', {}).items():
                all_directions[direction] += count

        report = {
            'timestamp': datetime.now().isoformat(),
            'period': 'last_hour',
            'summary': {
                'avg_occupancy': round(np.mean(counts), 1),
                'max_occupancy': max(counts),
                'min_occupancy': min(counts),
                'total_unique_visitors': max(unique_totals) if unique_totals else 0,
                'traffic_pattern': 'steady' if np.std(counts) < 2 else 'variable'
            },
            'movement_patterns': dict(all_directions),
            'peak_times': identify_peak_times(analytics_data),
            'recommendations': generate_recommendations(counts, all_directions)
        }

        return report

    except Exception as e:
        print(f"Error generating LLM report: {e}", file=sys.stderr)
        return None

def identify_peak_times(analytics_data):
    """Identify peak traffic times"""
    hourly_max = defaultdict(list)

    for entry in analytics_data:
        timestamp = datetime.fromisoformat(entry['timestamp'])
        hour = timestamp.hour
        minute_block = timestamp.minute // 15  # 15-minute blocks
        hourly_max[f"{hour:02d}:{minute_block*15:02d}"].append(entry['current_count'])

    # Find top 3 peak times
    peak_times = []
    for time_block, counts in hourly_max.items():
        if counts:
            peak_times.append({
                'time': time_block,
                'avg_count': round(np.mean(counts), 1),
                'max_count': max(counts)
            })

    peak_times.sort(key=lambda x: x['max_count'], reverse=True)
    return peak_times[:3]

def generate_recommendations(counts, directions):
    """Generate operational recommendations based on traffic patterns"""
    recommendations = []

    avg_count = np.mean(counts)
    max_count = max(counts)

    # Staffing recommendations
    if max_count > 10:
        recommendations.append({
            'type': 'staffing',
            'priority': 'high',
            'message': f'High traffic detected (max {max_count} people). Consider additional staff at registers.'
        })
    elif avg_count > 5:
        recommendations.append({
            'type': 'staffing',
            'priority': 'medium',
            'message': f'Moderate traffic (avg {avg_count:.1f} people). Ensure adequate checkout coverage.'
        })

    # Flow recommendations
    if directions.get('left', 0) > directions.get('right', 0) * 1.5:
        recommendations.append({
            'type': 'layout',
            'priority': 'medium',
            'message': 'Predominantly leftward movement detected. Consider adjusting promotional displays.'
        })

    # Cleaning recommendations
    if sum(counts) > 100:  # High cumulative traffic
        recommendations.append({
            'type': 'maintenance',
            'priority': 'medium',
            'message': 'High foot traffic in the past hour. Schedule floor cleaning during next lull.'
        })

    return recommendations

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('dashboard_final.html', has_camera=HAS_CAMERA)

@app.route('/api/metrics')
def get_metrics():
    print("DEBUG: /api/metrics endpoint called")
    
    with data_lock:
        analytics = person_tracker.get_analytics()
    
    print(f"DEBUG: Analytics from tracker: {analytics}")

    current_time = datetime.now()

    # Load historical data for period metrics
    df = load_tracking_data()
    metrics = calculate_metrics(df, current_time)

    # Add real-time analytics
    metrics['unique_visitors'] = analytics['unique_persons_total']
    metrics['movement_patterns'] = analytics['direction_distribution']
    metrics['active_tracks'] = analytics['active_tracks']

    print(f"DEBUG: Final metrics being returned: {metrics}")

    return jsonify({
        'metrics': metrics,
        'timestamp': current_time.isoformat(),
        'last_update': last_data_update.isoformat() if last_data_update else None
    })


@app.route('/api/llm_report')
def get_llm_report():
    """Endpoint for LLM integration"""
    report = generate_llm_report()

    if report is None:
        return jsonify({'error': 'No data available for report'}), 404

    return jsonify(report)

@app.route('/api/analytics')
def get_analytics():
    """Real-time analytics endpoint"""
    with data_lock:
        analytics = person_tracker.get_analytics()

    return jsonify(analytics)

@app.route('/api/reset_counter', methods=['POST'])
def reset_counter():
    """Reset the unique person counter"""
    global person_tracker
    with data_lock:
        person_tracker.unique_persons_counted = 0
        person_tracker.counted_track_ids.clear()
        person_tracker.tracks.clear()
        person_tracker.next_track_id = 1

    return jsonify({'status': 'success', 'message': 'Counter reset successfully'})

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {threading.current_thread().name}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {threading.current_thread().name}")

# --- Helper Functions ---

def load_tracking_data():
    """Fixed version with memory limits"""
    try:
        csv_files = glob.glob(os.path.join(TRACKINGLOG_PATH, "person_tracking_*.csv"))
        print(f"DEBUG: Found {len(csv_files)} CSV files")
        
        if not csv_files:
            return None

        # LIMIT: Only load the most recent files to prevent memory bloat
        recent_files = sorted(csv_files)[-3:]  # Only last 3 files (was 10)
        print(f"DEBUG: Loading only {len(recent_files)} most recent files")

        dfs = []
        total_rows = 0
        for file in recent_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    # LIMIT: Only keep recent data (last 2 hours)
                    df['date_time'] = pd.to_datetime(df['date_time'])
                    cutoff_time = datetime.now() - timedelta(hours=2)
                    df = df[df['date_time'] >= cutoff_time]
                    
                    if not df.empty:
                        dfs.append(df)
                        total_rows += len(df)
                        print(f"DEBUG: Loaded {len(df)} recent rows from {file}")
                        
                        # LIMIT: Stop if we have enough data
                        if total_rows > 1000:  # Max 1000 rows in memory
                            print(f"DEBUG: Reached row limit, stopping at {total_rows} rows")
                            break
            except Exception as e:
                print(f"DEBUG: Error reading {file}: {e}")

        if not dfs:
            return None

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('date_time').drop_duplicates(subset=['date_time'], keep='last')
        
        # FINAL LIMIT: Keep only last 500 rows
        if len(combined_df) > 500:
            combined_df = combined_df.tail(500)
            print(f"DEBUG: Trimmed to last 500 rows")
            
        print(f"DEBUG: Final dataframe has {len(combined_df)} rows")
        return combined_df

    except Exception as e:
        print(f"DEBUG: load_tracking_data error: {e}")
        return None

def calculate_metrics(df, current_time):
    """Fixed version with better debugging"""
    with data_lock:
        current_count = current_person_count

    print(f"DEBUG: calculate_metrics called with current_count={current_count}")

    if df is None or df.empty:
        print("DEBUG: No data available for metrics calculation")
        default_metrics = {
            'current': current_count,
            'avg_5m': 0, 'max_5m': 0,
            'avg_30m': 0, 'max_30m': 0,
            'avg_60m': 0, 'max_60m': 0,
            'daily_avg': 0, 'daily_max': 0,
            'daily_total_readings': 0
        }
        print(f"DEBUG: Returning default metrics: {default_metrics}")
        return default_metrics

    try:
        time_5m = current_time - timedelta(minutes=5)
        time_30m = current_time - timedelta(minutes=30)
        time_60m = current_time - timedelta(minutes=60)
        start_of_day = current_time.replace(hour=8, minute=0, second=0, microsecond=0)
        if current_time.hour < 8:
            start_of_day = start_of_day - timedelta(days=1)

        data_5m = df[df['date_time'] >= time_5m]
        data_30m = df[df['date_time'] >= time_30m]
        data_60m = df[df['date_time'] >= time_60m]
        data_daily = df[df['date_time'] >= start_of_day]

        print(f"DEBUG: Data counts - 5m: {len(data_5m)}, 30m: {len(data_30m)}, 60m: {len(data_60m)}, daily: {len(data_daily)}")

        metrics = {
            'current': current_count,
            'avg_5m': round(data_5m['persons'].mean(), 1) if not data_5m.empty else 0,
            'max_5m': int(data_5m['persons'].max()) if not data_5m.empty else 0,
            'avg_30m': round(data_30m['persons'].mean(), 1) if not data_30m.empty else 0,
            'max_30m': int(data_30m['persons'].max()) if not data_30m.empty else 0,
            'avg_60m': round(data_60m['persons'].mean(), 1) if not data_60m.empty else 0,
            'max_60m': int(data_60m['persons'].max()) if not data_60m.empty else 0,
            'daily_avg': round(data_daily['persons'].mean(), 1) if not data_daily.empty else 0,
            'daily_max': int(data_daily['persons'].max()) if not data_daily.empty else 0,
            'daily_total_readings': len(data_daily)
        }
        
        print(f"DEBUG: Calculated metrics: {metrics}")
        return metrics

    except Exception as e:
        print(f"DEBUG: Error in calculate_metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'current': current_count,
            'avg_5m': 0, 'max_5m': 0,
            'avg_30m': 0, 'max_30m': 0,
            'avg_60m': 0, 'max_60m': 0,
            'daily_avg': 0, 'daily_max': 0,
            'daily_total_readings': 0
        }

def update_metrics_data_thread():
    global historical_data, last_data_update, data_lock

    while not stop_event.is_set():
        try:
            df = load_tracking_data()
            with data_lock:
                historical_data = df
                last_data_update = datetime.now()

        except Exception as e:
            print(f"Metrics update error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        time.sleep(2)

def cleanup():
    print("\nShutting down application...")
    stop_event.set()

    if picam2:
        print("Stopping Picamera2...")
        try:
            picam2.stop()
            picam2.close()
        except Exception as e:
            print(f"Error closing camera: {e}")

    # Clean up HAILO using the module
    if HAS_HAILO_MODULE:
        print("Cleaning up HAILO...")
        try:
            cleanup_hailo()
            print("HAILO cleanup complete")
        except Exception as e:
            print(f"Error during HAILO cleanup: {e}")

    if csv_file:
        print("Closing CSV file...")
        csv_file.close()

    print("Cleanup complete.")
    
    
@app.route('/api/historical/<period>')
def get_historical_data(period):
    """Fixed with data limits"""
    df = load_tracking_data()

    if df is None or df.empty:
        return jsonify({'labels': [], 'avg_data': [], 'max_data': []})

    current_time = datetime.now()

    # REDUCED: Limit data points to prevent frontend memory issues
    if period == '1h':
        start_time = current_time - timedelta(hours=1)
        interval = '2T'  # 2-minute intervals (was 5T)
        max_points = 30
    elif period == '6h':
        start_time = current_time - timedelta(hours=6)
        interval = '10T'  # 10-minute intervals (was 15T)
        max_points = 36
    elif period == '24h':
        start_time = current_time - timedelta(hours=24)
        interval = '30T'  # 30-minute intervals (was 1H)
        max_points = 48
    else:
        return jsonify({'error': 'Invalid period'}), 400

    # Filter data
    filtered_df = df[df['date_time'] >= start_time]

    if filtered_df.empty:
        return jsonify({'labels': [], 'avg_data': [], 'max_data': []})

    # Resample data
    resampled = filtered_df.set_index('date_time').resample(interval)
    avg_data = resampled['persons'].mean().fillna(0)
    max_data = resampled['persons'].max().fillna(0)

    # LIMIT: Only return recent data points
    if len(avg_data) > max_points:
        avg_data = avg_data.tail(max_points)
        max_data = max_data.tail(max_points)

    # Format labels
    labels = [dt.strftime('%H:%M') for dt in avg_data.index]

    return jsonify({
        'labels': labels,
        'avg_data': avg_data.round(1).tolist(),
        'max_data': max_data.astype(int).tolist()
    })

def rotate_csv_files():
    """Keep only recent CSV files"""
    try:
        csv_files = glob.glob(os.path.join(TRACKINGLOG_PATH, "person_tracking_*.csv"))
        if len(csv_files) > 10:  # Keep only last 10 files
            old_files = sorted(csv_files)[:-10]
            for file in old_files:
                os.remove(file)
                print(f"DEBUG: Removed old CSV file: {file}")
    except Exception as e:
        print(f"DEBUG: CSV rotation error: {e}")

def update_metrics_data_thread():
    global historical_data, last_data_update, data_lock
    last_rotation = time.time()

    while not stop_event.is_set():
        try:
            df = load_tracking_data()
            with data_lock:
                historical_data = df
                last_data_update = datetime.now()

            # ADDED: Periodic file cleanup (every hour)
            if time.time() - last_rotation > 3600:
                rotate_csv_files()
                last_rotation = time.time()

        except Exception as e:
            print(f"Metrics update error: {e}", file=sys.stderr)
        time.sleep(5)  # Increased from 2 to 5 seconds to reduce CPU usage
        

# --- Main ---

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())

    os.makedirs(TRACKINGLOG_PATH, exist_ok=True)
    print(f"Logging to: {TRACKINGLOG_PATH}")

    try:
        # Enhanced CSV with more fields
        csv_filename = os.path.join(TRACKINGLOG_PATH, f"person_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['date_time', 'persons', 'unique_total', 'movement_patterns'])
        csv_file.flush()
        print(f"CSV file opened: {csv_filename}")
    except Exception as e:
        print(f"ERROR: Failed to open CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Start threads
    camera_inf_thread = threading.Thread(target=camera_and_inference_thread, daemon=True)
    camera_inf_thread.start()

    metrics_thread = threading.Thread(target=update_metrics_data_thread, daemon=True)
    metrics_thread.start()

    print("\nStarting Flask-SocketIO server...")
    print("Access dashboard at: http://<raspberry-pi-ip>:5000")
    print("\nAPI Endpoints:")
    print("  - /api/metrics - Current metrics")
    print("  - /api/analytics - Real-time analytics")
    print("  - /api/historical/<period> - Historical data (1h/6h/24h)")
    print("  - /api/llm_report - LLM-ready report")

    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected, initiating graceful shutdown...", file=sys.stderr)
    finally:
        cleanup()
