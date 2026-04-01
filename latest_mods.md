# G_Traffic - Camera-Based Shopper Traffic Monitoring

## Project Overview
AI-powered supermarket traffic monitoring system running on Raspberry Pi 5 using HAILO accelerator for real-time person detection and tracking. Features web-based dashboard with live video feed, analytics, and AI-generated operational summaries.

## Current Architecture
- **Main Script**: `dev_monitor.py` - Flask web application with SocketIO
- **Detection**: `o4_hailo_detection_module.py` - HAILO AI accelerator integration
- **Configuration**: `config_dev.py` - Development settings and hardware detection
- **Web Interface**: `templates/dashboard_final.html` - Real-time dashboard
- **AI Integration**: OpenAI GPT-4 Turbo for traffic summaries

## Key Features Implemented
1. **Person-Only Detection** (Confidence > 0.6)
   - Yellow bounding boxes for persons only
   - Filtered out all other object classes
   - Enhanced confidence threshold for accuracy

2. **AI Summary Feature** (Latest Addition)
   - Green "Summarize" button in dashboard (left of Refresh Data)
   - OpenAI GPT-4 Turbo integration with API key from `openai.txt`
   - Generates 2-line, ~80 word operational summaries
   - Shows current traffic, trends, and daily performance
   - Summary displayed in dedicated section below buttons

## Technical Details
- **Detection Model**: YOLOv8n with HAILO quantization
- **Confidence Threshold**: 0.6 (persons only)
- **Web Framework**: Flask + SocketIO for real-time updates
- **AI Model**: GPT-4-1106-preview via OpenAI API v1.0+
- **Hardware**: Raspberry Pi 5 with HAILO-8L accelerator

## File Structure
```
/home/icp/g_traffic/
├── dev_monitor.py              # Main Flask application
├── o4_hailo_detection_module.py # HAILO detection integration
├── config_dev.py              # Configuration settings
├── templates/dashboard_final.html # Web dashboard
├── openai.txt                  # OpenAI API key
├── models/                     # HAILO model files
└── trackinglog/               # CSV tracking data
```

## Current Status & Next Steps
- ✅ Person-only detection with confidence filtering
- ✅ AI-powered traffic summaries with GPT-4 integration
- ✅ Web dashboard with real-time updates
- ✅ OpenAI v1.0+ client implementation

## Recent Changes
1. **Person Detection Filter** - Only detects/displays persons with >0.6 confidence
2. **AI Summary Integration** - Added OpenAI GPT-4 client for operational insights
3. **UI Enhancement** - Green Summarize button and clean summary display
4. **API Modernization** - Updated to OpenAI v1.0+ client syntax

## Development Notes
- Uses development mode with simulated data when HAILO hardware unavailable
- CSV logging for all person tracking events
- Real-time video feed via SocketIO
- Responsive dashboard design with Chart.js integration
- Error handling for AI API failures

## Running the Application
```bash
cd /home/icp/g_traffic
python3 dev_monitor.py
```
Access dashboard at: http://localhost:5000

## API Endpoints
- `/api/metrics` - Current traffic metrics
- `/api/summarize` - Generate AI summary (POST)
- `/api/reset_counter` - Reset unique visitor counter (POST)
- `/api/historical/{period}` - Historical data (6h, 24h)

---

## Today's Modifications (June 25, 2025)

### Autostart System Implementation
Added complete autostart functionality to ensure the traffic monitoring application automatically starts after power cycles or system reboots.

#### Files Created:
1. **`traffic-monitor.service`** - Systemd service configuration
   - Service runs as user `icp` 
   - Uses virtual environment at `/home/icp/g_traffic/venv/bin/python3`
   - Automatic restart on failure with 10-second delay
   - Journal logging for system integration
   - Environment variables for HAILO/GPU access

2. **`install_autostart.sh`** - Installation script
   - Executable script to install and enable the service
   - Provides helpful command reference for service management

#### Key Features:
- **Automatic Startup**: Service starts automatically after boot
- **Auto-Restart**: Restarts application if it crashes
- **Virtual Environment**: Properly configured to use project venv
- **System Integration**: Uses systemd for reliable service management
- **Hardware Support**: Environment variables for HAILO accelerator access

#### Service Management Commands:
```bash
sudo systemctl start traffic-monitor    # Start service
sudo systemctl stop traffic-monitor     # Stop service  
sudo systemctl status traffic-monitor   # Check status
sudo journalctl -u traffic-monitor -f   # View logs
```

#### Installation Status:
- ✅ Service file created and configured
- ✅ Service installed to `/etc/systemd/system/`
- ✅ Service enabled for automatic startup
- ✅ Install script created for easy setup

This ensures the supermarket traffic monitoring system will automatically resume operation after any accidental power cycles, maintaining continuous shopper tracking and analytics.
