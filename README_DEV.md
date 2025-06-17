# Traffic Monitor - Development Setup

This document describes the development environment setup for the traffic monitoring system.

## Migration Summary

The system has been migrated to work in a Python virtual environment with the following improvements:

✅ **Development-friendly configuration**
✅ **Webcam support for testing**  
✅ **Simulation mode when hardware unavailable**
✅ **Cleaned up dependencies**
✅ **Easy startup scripts**

## Quick Start

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install/update dependencies:**
   ```bash
   python start_dev.py --install
   ```

3. **Start development server:**
   ```bash
   python start_dev.py
   ```

4. **Access dashboard:**
   Open http://127.0.0.1:5000 in your browser

## Available Modes

- **Development Mode** (`--mode dev`): Uses webcam + simulation
- **Original Mode** (`--mode original`): Uses hardware (Raspberry Pi + HAILO)

## File Structure

- `dev_monitor.py` - Development version with webcam support
- `config_dev.py` - Development configuration
- `start_dev.py` - Easy startup script
- `requirements.txt` - Updated dependencies
- `o4_dg_monitor.py` - Original hardware version

## Hardware Status

- ✅ **Webcam**: Available and working
- ✅ **Picamera2**: Available (optional)
- ⚠️  **HAILO**: Not available (optional for dev)
- ✅ **DeGirum**: Available (optional)

## Development Features

- **Webcam Integration**: Uses system camera for testing
- **Simulation Mode**: Generates fake detections for testing
- **Real-time Dashboard**: Web interface with live video feed
- **CSV Logging**: Automatic data logging
- **Fallback Modes**: Graceful degradation when hardware unavailable

## Testing

```bash
# Test environment
python test_env.py

# Check setup only
python start_dev.py --check-only

# Install requirements and start
python start_dev.py --install
```

## Troubleshooting

1. **No webcam**: System will use simulation mode
2. **Import errors**: Run `python start_dev.py --install`
3. **Port conflicts**: Check if port 5000 is in use
4. **Permission issues**: Ensure webcam permissions are granted

## API Endpoints

- `GET /api/metrics` - Current metrics
- `GET /api/analytics` - Real-time analytics  
- `POST /api/reset_counter` - Reset person counter
- `GET /api/historical/<period>` - Historical data

## Next Steps

The migration is complete! The system now works in your development environment with:

- Proper virtual environment isolation
- Webcam support for testing
- Fallback modes when hardware unavailable
- Clean development workflow