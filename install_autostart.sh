#!/bin/bash
# Autostart installation script for Traffic Monitor

echo "Installing Traffic Monitor autostart service..."

# Copy service file to systemd
sudo cp /home/icp/g_traffic/traffic-monitor.service /etc/systemd/system/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable traffic-monitor.service

echo "Service installed and enabled."
echo "Commands:"
echo "  Start:   sudo systemctl start traffic-monitor"
echo "  Stop:    sudo systemctl stop traffic-monitor"
echo "  Status:  sudo systemctl status traffic-monitor"
echo "  Logs:    sudo journalctl -u traffic-monitor -f"