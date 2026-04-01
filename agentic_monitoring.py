#!/usr/bin/env python3
"""
Development Version of Traffic Monitor
Works with webcam and simulated inference for development
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
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
from openai import OpenAI
import requests

# Import development configuration
from config_dev import (
    DEV_MODE, SIMULATE_CAMERA, SIMULATE_HAILO, USE_WEBCAM,
    MONITOR_PATH, TRACKINGLOG_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    SIMULATE_PERSON_COUNT, SIMULATION_MOVEMENT_SPEED, SIMULATION_FRAME_SIZE,
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    check_hardware, get_camera_source, get_inference_mode
)

# Import HAILO detection module for real inference
try:
    from o4_hailo_detection_module import get_hailo_detector, cleanup_hailo, HailoDetector
    HAS_HAILO_MODULE = True
    print("✅ HAILO detection module imported successfully.")
except ImportError as e:
    print(f"❌ HAILO detection module import failed: {e}", file=sys.stderr)
    HAS_HAILO_MODULE = False

# DeGirum model configuration - use local model that exists
DEGIRUM_MODEL_NAME = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1"

# Check available hardware
HARDWARE_AVAILABLE = check_hardware()
CAMERA_SOURCE = get_camera_source()
INFERENCE_MODE = get_inference_mode()

print(f"🔧 Development Mode: {DEV_MODE}")
print(f"📹 Camera Source: {CAMERA_SOURCE}")
print(f"🧠 Inference Mode: {INFERENCE_MODE}")
print(f"🔌 Hardware Available: {HARDWARE_AVAILABLE}")

# Import OpenCV and NumPy
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV not available: {e}")
    HAS_CV2 = False
    sys.exit(1)

# Try to import camera modules
picam2 = None
if CAMERA_SOURCE == 'picamera2':
    try:
        from picamera2 import Picamera2
        HAS_PICAMERA = True
        print("✅ Picamera2 available")
    except ImportError:
        HAS_PICAMERA = False
        CAMERA_SOURCE = 'webcam'
        print("⚠️  Picamera2 not available, falling back to webcam")

# Global Variables
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load OpenAI API key and initialize client
try:
    with open('openai.txt', 'r') as f:
        openai_api_key = f.read().strip()
    openai_client = OpenAI(api_key=openai_api_key)
    print("✅ OpenAI client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize OpenAI client: {e}")
    openai_client = None
    openai_api_key = None

# Load Slack webhook URL
try:
    with open('slack.txt', 'r') as f:
        SLACK_WEBHOOK_URL = f.read().strip()
    print("✅ Slack webhook URL loaded successfully")
except Exception:
    SLACK_WEBHOOK_URL = None
    print("⚠️  slack.txt not found — Slack alerts disabled")

# Crowd alert cooldown state
last_alert_time = 0          # epoch seconds
ALERT_COOLDOWN = 60          # 1 minute cooldown
last_alert_message = ""      # shown in frontend popup

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

# Person Count History CSV Logging (persistent storage)
count_history_csv_filename = None
count_history_csv_file = None
count_history_csv_writer = None
last_logged_unique_visitors = -1  # Track last logged unique visitors count to only log changes

@dataclass
class Detection:
    """Represents a single person detection"""
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
    """Enhanced person tracking"""
    def __init__(self, track_id: int, initial_detection: Detection):
        self.track_id = track_id
        self.detections = deque(maxlen=10)
        self.detections.append(initial_detection)
        self.last_seen = initial_detection.timestamp
        self.first_seen = initial_detection.timestamp
        self.counted = False
        self.direction = None
        self.avg_velocity = (0, 0)

    def update(self, detection: Detection):
        """Update track with new detection"""
        self.detections.append(detection)
        self.last_seen = detection.timestamp
        self._calculate_direction()

    def _calculate_direction(self):
        """Calculate movement direction"""
        if len(self.detections) < 3:
            return

        recent_positions = [d.center for d in list(self.detections)[-3:]]
        dx = recent_positions[-1][0] - recent_positions[0][0]
        dy = recent_positions[-1][1] - recent_positions[0][1]

        self.avg_velocity = (dx / 3, dy / 3)

        if abs(dx) > abs(dy):
            self.direction = "right" if dx > 0 else "left"
        else:
            self.direction = "down" if dy > 0 else "up"

    @property
    def is_active(self) -> bool:
        """Check if track is still active"""
        return (time.time() - self.last_seen) < 2.0

    def predict_next_position(self) -> Tuple[float, float]:
        """Predict next position based on velocity"""
        if not self.detections:
            return (0, 0)
        last_center = self.detections[-1].center
        return (
            last_center[0] + self.avg_velocity[0],
            last_center[1] + self.avg_velocity[1]
        )

class PersonTracker:
    """Person tracking system"""
    def __init__(self):
        self.tracks: Dict[int, TrackedPerson] = {}
        self.next_track_id = 1
        self.person_count_history = deque(maxlen=3600)  # Store 1 hour of data (1 point per second)
        self.count_timestamps = deque(maxlen=3600)  # Store timestamps for each count
        self.unique_persons_counted = 0
        self.counted_track_ids = set()
        self.unique_person_timestamps = {}  # Track when each unique person was first detected
        self.daily_max_count = 0
        self.last_reset_date = datetime.now().date()
        self.last_history_update = 0  # Track when we last added to history (throttle to 1 per second)

    def update(self, detections: List[Detection]) -> List[TrackedPerson]:
        """Update tracker with new detections"""
        current_time = time.time()

        # Clean up old tracks
        self.tracks = {tid: track for tid, track in self.tracks.items()
                      if (current_time - track.last_seen) < 5.0}

        if not detections:
            current_count = len(self.tracks)
            # Only update history once per second to prevent rapid data overwrite
            if current_time - self.last_history_update >= 1.0:
                self.person_count_history.append(current_count)
                self.count_timestamps.append(current_time)
                self.last_history_update = current_time
            return list(self.tracks.values())

        # Associate detections with existing tracks
        matched_pairs, unmatched_detections = self._associate_detections(detections)

        # Update matched tracks
        for detection, track_id in matched_pairs:
            self.tracks[track_id].update(detection)

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = TrackedPerson(self.next_track_id, detection)
            self.tracks[self.next_track_id] = new_track

            # Count new person (using config threshold)
            if detection.confidence > CONFIDENCE_THRESHOLD:
                if self.next_track_id not in self.counted_track_ids:
                    self.unique_persons_counted += 1
                    self.counted_track_ids.add(self.next_track_id)
                    self.unique_person_timestamps[self.next_track_id] = current_time  # Record detection time
                    new_track.counted = True

            self.next_track_id += 1

        # Update count history with timestamp (throttled to once per second)
        current_count = len(self.tracks)
        if current_time - self.last_history_update >= 1.0:
            self.person_count_history.append(current_count)
            self.count_timestamps.append(current_time)
            self.last_history_update = current_time
        
        # Update daily maximum tracking
        self._update_daily_maximum(current_count)
        
        return list(self.tracks.values())

    def _associate_detections(self, detections: List[Detection]) -> Tuple[List[Tuple[Detection, int]], List[Detection]]:
        """Associate detections with existing tracks using IoU"""
        if not self.tracks:
            return [], detections

        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))

        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                iou = self._calculate_iou(detection, track.detections[-1])
                iou_matrix[i, j] = iou

        # Simple greedy matching
        matched_pairs = []
        used_detections = set()
        used_tracks = set()

        indices = np.argwhere(iou_matrix > IOU_THRESHOLD)
        indices = indices[np.argsort(iou_matrix[indices[:, 0], indices[:, 1]])[::-1]]

        for det_idx, track_idx in indices:
            if det_idx not in used_detections and track_idx not in used_tracks:
                matched_pairs.append((detections[det_idx], track_ids[track_idx]))
                used_detections.add(det_idx)
                used_tracks.add(track_idx)

        unmatched_detections = [d for i, d in enumerate(detections) if i not in used_detections]
        return matched_pairs, unmatched_detections

    def _calculate_iou(self, det1: Detection, det2: Detection) -> float:
        """Calculate IoU between two detections"""
        x1_1, y1_1, x2_1, y2_1 = det1.bbox
        x1_2, y1_2, x2_2, y2_2 = det2.bbox

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

    def get_current_count(self) -> int:
        """Get current person count"""
        if not self.person_count_history:
            return 0
        recent_counts = list(self.person_count_history)[-10:]
        return int(np.median(recent_counts)) if recent_counts else 0

    def get_time_based_tally(self, minutes: int) -> int:
        """Get count of unique people detected in the last N minutes"""
        if not self.unique_person_timestamps:
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - (minutes * 60)  # Convert minutes to seconds
        
        # Count unique people detected within the time window
        people_in_window = 0
        for track_id, detection_time in self.unique_person_timestamps.items():
            if detection_time >= cutoff_time:
                people_in_window += 1
        
        print(f"DEBUG: {minutes}min tally: {people_in_window} unique people detected")
        return people_in_window

    def get_time_based_maximum(self, minutes: int) -> int:
        """Get maximum occupancy count over the last N minutes"""
        if not self.person_count_history or not self.count_timestamps:
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - (minutes * 60)  # Convert minutes to seconds
        
        # Find counts within the time window
        recent_counts = []
        for i, timestamp in enumerate(self.count_timestamps):
            if timestamp >= cutoff_time:
                recent_counts.append(self.person_count_history[i])
        
        if not recent_counts:
            return 0
        
        return max(recent_counts)

    def _update_daily_maximum(self, current_count: int):
        """Update daily maximum count and reset all data at 08:00 daily"""
        now = datetime.now()
        current_date = now.date()
        current_hour = now.hour
        
        # Reset all tracking data at 08:00 each day (store opening time)
        if (current_date != self.last_reset_date and current_hour >= 8) or \
           (current_date == self.last_reset_date and current_hour == 8 and self.last_reset_date < current_date):
            self._perform_daily_reset(now)
        
        # Update daily maximum if current count is higher
        if current_count > self.daily_max_count:
            self.daily_max_count = current_count

    def _perform_daily_reset(self, reset_time: datetime):
        """Perform complete daily data reset at 08:00"""
        print(f"🔄 Performing daily data reset at {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Reset all counters
        self.unique_persons_counted = 0
        self.daily_max_count = 0
        self.last_reset_date = reset_time.date()
        
        # Clear all tracking data
        self.counted_track_ids.clear()
        self.unique_person_timestamps.clear()
        self.tracks.clear()
        self.next_track_id = 1
        self.last_history_update = 0  # Reset history update timestamp
        
        # Clear historical data (keep only data from 08:00 onwards)
        cutoff_time = reset_time.replace(hour=8, minute=0, second=0, microsecond=0).timestamp()
        filtered_history = deque(maxlen=3600)
        filtered_timestamps = deque(maxlen=3600)
        
        # If there's existing data from today after 08:00, keep it
        for i, timestamp in enumerate(self.count_timestamps):
            if timestamp >= cutoff_time:
                filtered_history.append(self.person_count_history[i])
                filtered_timestamps.append(timestamp)
        
        self.person_count_history = filtered_history
        self.count_timestamps = filtered_timestamps
        
        print(f"✅ Daily reset completed - all data cleared for new trading day")

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
            'avg_dwell_time': np.mean([t.last_seen - t.first_seen for t in self.tracks.values()]) if self.tracks else 0,
            'daily_max_count': self.daily_max_count
        }

def load_unique_visitors_history():
    """Load unique visitors history from CSV file for persistence and restore state"""
    global person_tracker, last_logged_unique_visitors
    
    if not count_history_csv_filename or not os.path.exists(count_history_csv_filename):
        print("📊 No existing unique visitors history file found")
        return
    
    try:
        with open(count_history_csv_filename, 'r') as f:
            csv_reader = csv.DictReader(f)
            loaded_count = 0
            latest_unique_visitors = 0
            
            # Get today's date for filtering
            today = datetime.now().date()
            store_opening_today = datetime.combine(today, datetime.min.time().replace(hour=8))
            
            # Temporary arrays to hold the unique visitor progression over time
            unique_visitor_timestamps = []
            unique_visitor_counts = []
            
            for row in csv_reader:
                try:
                    timestamp_str = row['timestamp']
                    # Try both column names for backward compatibility
                    if 'unique_visitors' in row:
                        unique_visitors = int(row['unique_visitors'])
                    else:
                        unique_visitors = int(row['person_count'])  # Legacy column name
                    
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamp_unix = timestamp.timestamp()
                    
                    # Only load data from today's store opening onwards
                    if timestamp >= store_opening_today:
                        unique_visitor_timestamps.append(timestamp_unix)
                        unique_visitor_counts.append(unique_visitors)
                        latest_unique_visitors = unique_visitors
                        loaded_count += 1
                
                except (ValueError, KeyError) as e:
                    print(f"⚠️  Skipping invalid row in unique visitors CSV: {e}")
                    continue
            
            # For now, we'll populate person_count_history with the unique visitor progression
            # This gives us the chart data showing unique visitor growth over time
            person_tracker.count_timestamps.extend(unique_visitor_timestamps)
            person_tracker.person_count_history.extend(unique_visitor_counts)
            
            # Restore the actual unique visitors count in the tracker
            if latest_unique_visitors > 0:
                person_tracker.unique_persons_counted = latest_unique_visitors
                last_logged_unique_visitors = latest_unique_visitors
                
                # Also need to set the next_track_id to be higher than the restored count
                # to prevent ID conflicts with new detections
                person_tracker.next_track_id = latest_unique_visitors + 1
                
                print(f"✅ Restored unique visitors count to {latest_unique_visitors}")
                print(f"✅ Set next track ID to {person_tracker.next_track_id}")
            
            print(f"✅ Loaded {loaded_count} unique visitors history records from today")
            
    except Exception as e:
        print(f"❌ Error loading unique visitors history: {e}")

def log_unique_visitors_change(unique_visitors: int):
    """Log unique visitors count to CSV only when it changes"""
    global last_logged_unique_visitors, count_history_csv_writer, count_history_csv_file
    
    if unique_visitors == last_logged_unique_visitors:
        return  # No change, don't log
    
    if not count_history_csv_writer:
        return  # CSV not initialized
    
    try:
        timestamp = datetime.now().isoformat()
        count_history_csv_writer.writerow([timestamp, unique_visitors])
        count_history_csv_file.flush()
        last_logged_unique_visitors = unique_visitors
        print(f"📊 Logged unique visitors change: {unique_visitors} at {timestamp}")
    except Exception as e:
        print(f"❌ Error logging unique visitors change: {e}")

# Initialize tracker
person_tracker = PersonTracker()

# Global HAILO detector
hailo_detector = None
hailo_initialized = False

def setup_hailo_inference():
    """Setup HAILO inference detector"""
    global hailo_detector

    if not HAS_HAILO_MODULE:
        print("DEBUG: HAILO detection module not available. Inference will be disabled.", file=sys.stderr)
        return False

    try:
        print(f"DEBUG: Setting up HAILO detector with DeGirum model: {DEGIRUM_MODEL_NAME}")
        
        # Get singleton detector instance from the imported module
        hailo_detector = get_hailo_detector(DEGIRUM_MODEL_NAME, cv2_ref_passed=cv2, np_ref_passed=np)

        if hailo_detector and hailo_detector.is_initialized:
            print("✅ HAILO detector initialized successfully (DeGirum PySDK)")
            return True
        else:
            print("❌ HAILO detector initialization failed (DeGirum PySDK)", file=sys.stderr)
            return False
    except Exception as e:
        print(f"❌ HAILO setup error (DeGirum PySDK): {e}", file=sys.stderr)
        return False

def run_hailo_inference(frame_rgb):
    """Run HAILO inference and return Detection objects"""
    current_time = time.time()

    # Check if detector is initialized, if not, use simulation
    if not hailo_detector or not hailo_detector.is_initialized:
        print("⚠️  HAILO detector not available, using simulation")
        return create_simulated_detections()

    try:
        # Run detection using the HailoDetector instance
        hailo_detections = hailo_detector.detect(frame_rgb)

        # Convert HailoDetection objects to Detection objects
        detections = []
        for det in hailo_detections:
            detection = Detection(
                bbox=det.bbox,
                confidence=det.confidence,
                timestamp=current_time
            )
            detections.append(detection)
        
        return detections

    except Exception as e:
        print(f"HAILO inference execution error: {e}", file=sys.stderr)
        return []

def create_simulated_detections() -> List[Detection]:
    """Create simulated person detections for development"""
    detections = []
    current_time = time.time()

    # Initialize simulation positions if not exists
    if not hasattr(create_simulated_detections, 'positions'):
        create_simulated_detections.positions = [
            [200, 200], [400, 300]
        ]

    # Move simulated people
    for i, pos in enumerate(create_simulated_detections.positions):
        pos[0] += np.random.randint(-SIMULATION_MOVEMENT_SPEED, SIMULATION_MOVEMENT_SPEED + 1)
        pos[1] += np.random.randint(-SIMULATION_MOVEMENT_SPEED, SIMULATION_MOVEMENT_SPEED + 1)

        # Keep within bounds
        pos[0] = max(50, min(SIMULATION_FRAME_SIZE[0] - 100, pos[0]))
        pos[1] = max(50, min(SIMULATION_FRAME_SIZE[1] - 120, pos[1]))

        detection = Detection(
            bbox=(pos[0], pos[1], pos[0] + 80, pos[1] + 120),
            confidence=0.85 + np.random.random() * 0.1,
            timestamp=current_time
        )
        detections.append(detection)

    return detections

def init_camera():
    """Initialize camera source"""
    global picam2

    if CAMERA_SOURCE == 'picamera2' and HARDWARE_AVAILABLE['picamera2']:
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
            return False

    elif CAMERA_SOURCE == 'webcam':
        # Try multiple camera indices
        for camera_index in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        # Set camera properties for better performance
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        print(f"✅ Webcam initialized on index {camera_index}")
                        return cap
                    else:
                        cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"⚠️  Camera index {camera_index} failed: {e}")
                continue
        
        print("❌ Could not open any webcam")
        return None

    return None

def camera_and_inference_thread():
    """Main camera and inference processing thread"""
    global current_person_count, hailo_detector, hailo_initialized

    print("🎥 Starting camera and inference thread")

    # Initialize camera
    camera = init_camera()
    
    # Initialize HAILO detector for live inference
    if INFERENCE_MODE == 'hailo' or not SIMULATE_HAILO:
        print("🧠 Initializing HAILO detector for live inference...")
        hailo_initialized = setup_hailo_inference()
        if hailo_initialized:
            print("✅ Live inference mode enabled")
        else:
            print("⚠️  Live inference initialization failed, falling back to simulation")
    
    frame_count = 0
    last_frame_time = time.time()

    while not stop_event.is_set():
        try:
            current_time = time.time()
            frame_count += 1

            # Capture frame
            if CAMERA_SOURCE == 'picamera2' and picam2:
                frame = picam2.capture_array()  # RGB888
            elif CAMERA_SOURCE == 'webcam' and camera:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print(f"⚠️  Webcam read failed, attempting to reinitialize...")
                    # Try to reinitialize camera
                    camera.release()
                    camera = init_camera()
                    if camera:
                        ret, frame = camera.read()
                        if ret and frame is not None:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = cv2.resize(frame, SIMULATION_FRAME_SIZE)
                        else:
                            frame = np.zeros((SIMULATION_FRAME_SIZE[1], SIMULATION_FRAME_SIZE[0], 3), dtype=np.uint8)
                            cv2.putText(frame, "Webcam Reinit Failed", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        frame = np.zeros((SIMULATION_FRAME_SIZE[1], SIMULATION_FRAME_SIZE[0], 3), dtype=np.uint8)
                        cv2.putText(frame, "No Webcam Available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, SIMULATION_FRAME_SIZE)
            else:
                # Create placeholder frame
                frame = np.zeros((SIMULATION_FRAME_SIZE[1], SIMULATION_FRAME_SIZE[0], 3), dtype=np.uint8)
                frame[:] = (50, 50, 50)
                cv2.putText(frame, f"Simulation Mode - {CAMERA_SOURCE}",
                          (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Run inference
            if hailo_initialized and hailo_detector and hailo_detector.is_initialized:
                # Real HAILO inference
                detections = run_hailo_inference(frame)
                # Add live mode overlay
                cv2.putText(frame, "LIVE MODE",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Fallback to simulation
                detections = create_simulated_detections()
                # Add simulation overlay
                cv2.putText(frame, "SIMULATION MODE",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Update tracker
            active_tracks = person_tracker.update(detections)
            analytics = person_tracker.get_analytics()

            # Update global count and log changes
            with data_lock:
                current_person_count = analytics['current_count']
                # Log unique visitors change to persistent CSV
                log_unique_visitors_change(analytics['unique_persons_total'])

            # Draw visualizations - only for persons with sufficient confidence
            for track in active_tracks:
                if track.detections:
                    latest_det = track.detections[-1]
                    
                    # Only show persons with confidence > 0.6 in yellow bounding boxes
                    if latest_det.confidence > CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = latest_det.bbox

                        # Use yellow (0, 255, 255) for all person detections as requested
                        color = (0, 255, 255)  # Yellow in BGR format
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        label = f"Person ID:{track.track_id}"
                        if track.direction:
                            label += f" {track.direction}"
                        label += f" ({latest_det.confidence:.2f})"

                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Draw trajectory only for valid detections
                        if len(track.detections) > 1:
                            points = [det.center for det in list(track.detections)[-5:]]
                            for i in range(1, len(points)):
                                pt1 = tuple(map(int, points[i-1]))
                                pt2 = tuple(map(int, points[i]))
                                cv2.line(frame, pt1, pt2, color, 1)

            # Add overlay information
            cv2.putText(frame, f"Count: {analytics['current_count']}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(frame, f"Total: {analytics['unique_persons_total']}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # FPS counter
            fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}",
                       (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Timestamp
            timestamp_str = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp_str,
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Encode and emit frame (Picamera2 RGB888 delivers BGR bytes natively)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': jpg_as_text})

            # Log to CSV
            if csv_writer and frame_count % 30 == 0:  # Every second
                try:
                    csv_writer.writerow([
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        analytics['current_count'],
                        analytics['unique_persons_total'],
                        json.dumps(analytics['direction_distribution'])
                    ])
                    csv_file.flush()
                except Exception as e:
                    print(f"CSV logging error: {e}")

            last_frame_time = current_time
            time.sleep(1/30)  # 30 FPS

        except Exception as e:
            print(f"Error in camera thread: {e}")
            traceback.print_exc()
            time.sleep(1)

    # Cleanup
    if CAMERA_SOURCE == 'webcam' and camera:
        camera.release()
    elif CAMERA_SOURCE == 'picamera2' and picam2:
        print("Stopping Picamera2...")
        try:
            picam2.stop()
            picam2.close()
        except Exception as e:
            print(f"Error closing camera: {e}")

# ── Crowd Alert Agent (OpenAI-powered) ───────────────────────────────────────

def _generate_alert_message(count: int, unique_today: int, daily_max: int, current_time: str) -> str:
    """Ask OpenAI to write a concise Slack alert message."""
    prompt = (
        f"You are a retail operations assistant. Write a concise 1-2 sentence Slack alert "
        f"for a supermarket manager. Current data: {count} persons in store at {current_time}, "
        f"{unique_today} unique visitors today, daily peak so far: {daily_max}. "
        f"Be specific and actionable."
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful retail operations assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=80,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

def _send_slack(message: str) -> bool:
    """Post message to Slack webhook. Returns True on success."""
    if not SLACK_WEBHOOK_URL:
        print("⚠️  Slack webhook not configured — skipping notification")
        return False
    resp = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
    return resp.status_code == 200

def crowd_alert_agent_thread():
    global last_alert_time, last_alert_message
    while not stop_event.is_set():
        time.sleep(10)  # check every 10 seconds
        now = time.time()
        if current_person_count >= 2 and (now - last_alert_time) >= ALERT_COOLDOWN:
            try:
                with data_lock:
                    analytics = person_tracker.get_analytics()
                current_time = datetime.now().strftime("%H:%M")
                message = _generate_alert_message(
                    count=current_person_count,
                    unique_today=analytics["unique_persons_total"],
                    daily_max=analytics["daily_max_count"],
                    current_time=current_time,
                )
                sent = _send_slack(message)
                last_alert_time = now
                last_alert_message = message
                socketio.emit("crowd_alert", {
                    "message": last_alert_message,
                    "count": current_person_count,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })
                print(f"✅ Crowd alert {'sent' if sent else 'generated (Slack skipped)'}: {message}")
            except Exception as e:
                print(f"❌ Crowd agent error: {e}", file=sys.stderr)

# Flask Routes
@app.route('/')
def index():
    return render_template('agentic_dashboard.html', has_camera=True)

@app.route('/api/last_alert')
def get_last_alert():
    return jsonify({
        "message": last_alert_message,
        "cooldown_remaining": max(0, int(ALERT_COOLDOWN - (time.time() - last_alert_time))),
        "last_alert_ts": datetime.fromtimestamp(last_alert_time).strftime("%H:%M:%S") if last_alert_time else None,
    })

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    with data_lock:
        analytics = person_tracker.get_analytics()

    current_time = datetime.now()

    # Calculate time-based tallies and maximums
    tally_1m = person_tracker.get_time_based_tally(1)
    max_1m = person_tracker.get_time_based_maximum(1)
    tally_5m = person_tracker.get_time_based_tally(5)
    max_5m = person_tracker.get_time_based_maximum(5)
    tally_30m = person_tracker.get_time_based_tally(30)
    max_30m = person_tracker.get_time_based_maximum(30)
    tally_60m = person_tracker.get_time_based_tally(60)
    max_60m = person_tracker.get_time_based_maximum(60)
    
    # Calculate daily tally (unique people detected since start of day)
    daily_tally = person_tracker.get_time_based_tally(24 * 60) if len(person_tracker.unique_person_timestamps) > 0 else 0
    
    metrics = {
        'current': analytics['current_count'],
        'unique_visitors': analytics['unique_persons_total'],
        'movement_patterns': analytics['direction_distribution'],
        'active_tracks': analytics['active_tracks'],
        'avg_1m': tally_1m,  # Now represents 1-min tally
        'max_1m': max_1m,
        'avg_5m': tally_5m,  # Now represents 5-min tally
        'max_5m': max_5m,
        'avg_30m': tally_30m,  # Now represents 30-min tally
        'max_30m': max_30m,
        'avg_60m': tally_60m,  # Now represents 60-min tally
        'max_60m': max_60m,
        'daily_avg': daily_tally,  # Now represents daily tally
        'daily_max': analytics['daily_max_count'],
        'daily_total_readings': len(person_tracker.person_count_history)
    }

    return jsonify({
        'metrics': metrics,
        'timestamp': current_time.isoformat(),
        'last_update': current_time.isoformat()
    })

@app.route('/api/analytics')
def get_analytics():
    """Real-time analytics endpoint"""
    with data_lock:
        analytics = person_tracker.get_analytics()
    return jsonify(analytics)

@app.route('/api/realtime/<time_window>')
def get_realtime_data(time_window):
    """Get real-time historical data for specified time window"""
    with data_lock:
        # Get current time
        now = time.time()
        
        # Calculate cutoff time based on window
        if time_window == 'trading':
            # Trading day: from 08:00 today till now
            today = datetime.now().date()
            trading_start = datetime.combine(today, datetime.min.time().replace(hour=8))
            cutoff_time = trading_start.timestamp()
        else:
            # Regular time window in minutes
            try:
                minutes = int(time_window)
                cutoff_time = now - (minutes * 60)
            except ValueError:
                return jsonify({'error': 'Invalid time window'}), 400
        
        # Filter historical data based on time window
        filtered_data = []
        filtered_labels = []
        filtered_timestamps = []
        
        for i, timestamp in enumerate(person_tracker.count_timestamps):
            if timestamp >= cutoff_time:
                # Create time label
                dt = datetime.fromtimestamp(timestamp)
                time_label = dt.strftime('%H:%M:%S')
                
                # Get corresponding count
                if i < len(person_tracker.person_count_history):
                    count = person_tracker.person_count_history[i]
                    
                    filtered_timestamps.append(timestamp)
                    filtered_labels.append(time_label)
                    filtered_data.append(count)
        
        # If no data found, return current count
        if not filtered_data:
            current_analytics = person_tracker.get_analytics()
            current_time = datetime.now()
            filtered_labels = [current_time.strftime('%H:%M:%S')]
            filtered_data = [current_analytics['current_count']]
            filtered_timestamps = [now]
        
        return jsonify({
            'labels': filtered_labels,
            'data': filtered_data,
            'timestamps': filtered_timestamps,
            'time_window': time_window,
            'data_points': len(filtered_data)
        })

@app.route('/api/reset_counter', methods=['POST'])
def reset_counter():
    """Reset all tracking data including CSV files"""
    with data_lock:
        # Reset in-memory tracking data
        person_tracker.unique_persons_counted = 0
        person_tracker.counted_track_ids.clear()
        person_tracker.unique_person_timestamps.clear()  # Clear detection timestamps
        person_tracker.tracks.clear()
        person_tracker.next_track_id = 1
        
        # Clear historical chart data (this was missing!)
        person_tracker.person_count_history.clear()
        person_tracker.count_timestamps.clear()
        person_tracker.daily_max_count = 0
        
        # Clear historical data
        global historical_data
        historical_data = None
        
        # Clear CSV files from trackinglog folder
        try:
            import glob
            csv_files = glob.glob(os.path.join(TRACKINGLOG_PATH, "*.csv"))
            json_files = glob.glob(os.path.join(TRACKINGLOG_PATH, "*.json"))
            
            files_removed = 0
            for file_path in csv_files + json_files:
                try:
                    os.remove(file_path)
                    files_removed += 1
                    print(f"Removed tracking file: {file_path}")
                except OSError as e:
                    print(f"Failed to remove {file_path}: {e}")
            
            print(f"Data reset complete. Removed {files_removed} tracking files.")
            
        except Exception as e:
            print(f"Error clearing tracking files: {e}")
    
    return jsonify({'status': 'success', 'message': 'All tracking data and files reset successfully'})


@app.route('/api/summarize', methods=['POST'])
def generate_summary():
    """Generate AI summary of shopper traffic"""
    if not openai_client:
        return jsonify({'error': 'OpenAI client not initialized'}), 500
    
    try:
        # Get current analytics data
        with data_lock:
            analytics = person_tracker.get_analytics()
        
        # Get current time
        current_time = datetime.now()
        store_opening = current_time.replace(hour=8, minute=0, second=0, microsecond=0)
        
        # Calculate time since store opening (08:00)
        hours_since_opening = (current_time - store_opening).total_seconds() / 3600
        
        # Create prompt for OpenAI
        prompt = f"""You are an AI assistant helping a supermarket manager understand shopper traffic patterns. 
        
Current shopper traffic data:
- Current people in store: {analytics['current_count']}
- Unique visitors today: {analytics['unique_persons_total']}
- Hours since store opening: {hours_since_opening:.1f}
- Current time: {current_time.strftime('%H:%M')}
- Active tracking zones: {analytics['active_tracks']} people being tracked

Please provide a concise 2-line summary (approximately 80 words total) for the supermarket manager covering:
1. Current traffic status and recent trends (last 5-30 minutes)  
2. Daily (from 0800 opening)performance compared to typical patterns

Format: Keep it professional, actionable, and focused on operational insights that help with staffing and customer service decisions."""

        # Make OpenAI API call using new client syntax
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in retail analytics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        
        return jsonify({
            'summary': summary,
            'timestamp': current_time.isoformat(),
            'data_used': {
                'current_count': analytics['current_count'],
                'unique_visitors': analytics['unique_persons_total'],
                'hours_since_opening': round(hours_since_opening, 1)
            }
        })
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return jsonify({'error': f'Failed to generate summary: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    print(f"Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected")

def cleanup():
    """Cleanup resources"""
    print("\n🛑 Shutting down development server...")
    stop_event.set()

    # Cleanup HAILO detector
    if hailo_detector:
        try:
            cleanup_hailo()
            print("✅ HAILO detector cleaned up")
        except Exception as e:
            print(f"⚠️  HAILO cleanup warning: {e}")

    if csv_file:
        csv_file.close()
    
    if count_history_csv_file:
        count_history_csv_file.close()

    print("✅ Cleanup complete")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())

    # Create directories
    os.makedirs(TRACKINGLOG_PATH, exist_ok=True)

    # Setup CSV logging
    try:
        csv_filename = os.path.join(TRACKINGLOG_PATH, f"dev_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['date_time', 'persons', 'unique_total', 'movement_patterns'])
        csv_file.flush()
        print(f"📊 CSV logging to: {csv_filename}")
    except Exception as e:
        print(f"❌ CSV setup failed: {e}")
    
    # Setup persistent unique visitors history CSV
    try:
        count_history_csv_filename = os.path.join(TRACKINGLOG_PATH, "unique_visitors_history.csv")
        
        # Check for old file with wrong data format and rename it
        old_file = os.path.join(TRACKINGLOG_PATH, "person_count_history.csv")
        if os.path.exists(old_file):
            backup_file = os.path.join(TRACKINGLOG_PATH, f"person_count_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.rename(old_file, backup_file)
            print(f"📊 Backed up old person_count_history.csv to {backup_file}")
        
        file_exists = os.path.exists(count_history_csv_filename)
        
        count_history_csv_file = open(count_history_csv_filename, 'a', newline='')
        count_history_csv_writer = csv.writer(count_history_csv_file)
        
        # Write header if file is new
        if not file_exists:
            count_history_csv_writer.writerow(['timestamp', 'unique_visitors'])
            count_history_csv_file.flush()
            print(f"📊 Created new unique visitors history CSV: {count_history_csv_filename}")
        else:
            print(f"📊 Using existing unique visitors history CSV: {count_history_csv_filename}")
            
    except Exception as e:
        print(f"❌ Unique visitors history CSV setup failed: {e}")
    
    # Load existing unique visitors history for persistence
    load_unique_visitors_history()

    # Start camera thread
    camera_thread = threading.Thread(target=camera_and_inference_thread, daemon=True)
    camera_thread.start()

    # Start crowd alert agent thread
    alert_thread = threading.Thread(target=crowd_alert_agent_thread, daemon=True)
    alert_thread.start()
    print("🤖 Crowd Alert Agent started (threshold: >2 persons, cooldown: 5 min)")

    print(f"\n🚀 Starting development server...")
    print(f"🌐 Dashboard: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"📊 API endpoints available at /api/")
    print(f"🎥 Camera mode: {CAMERA_SOURCE}")
    print(f"🧠 Inference mode: {INFERENCE_MODE}")

    try:
        socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt received")
    finally:
        cleanup()
