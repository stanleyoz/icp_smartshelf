# Shopper Traffic Monitoring and Analysis System

An AI-powered traffic monitoring system built for supermarket environments, utilizing computer vision and edge AI inference on Raspberry Pi 5 hardware to track and analyze customer traffic patterns in real-time.

## Project Overview

This system provides comprehensive shopper traffic monitoring capabilities including:
- **Real-time person detection and tracking** using Pi Camera or webcam
- **Advanced AI inference** with HAILO-8 accelerator for edge computing
- **Live web dashboard** with real-time metrics and video feed
- **Persistent data logging** with CSV export and historical analysis
- **LLM-powered insights** integration with OpenAI API for business intelligence
- **Time-based analytics** including daily, hourly, and rolling window metrics

Built specifically for retail environments, the system tracks unique visitors, occupancy levels, movement patterns, and provides actionable insights for operational decision-making.

## Hardware Requirements

### Required Hardware
- **Raspberry Pi 5** (8GB recommended)
- **Pi Camera Module 3** or compatible USB webcam
- **HAILO-8 AI Accelerator** (optional, falls back to simulation)
- **MicroSD Card** (32GB+ Class 10)
- **Power Supply** (27W USB-C for Pi 5)

### Optional Hardware
- **PoE+ HAT** for network-powered deployment
- **Cooling fan/heatsink** for continuous operation
- **Enclosure** for retail environment protection

## Software Setup

### Prerequisites

1. **Install Raspberry Pi OS** (64-bit recommended)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ and pip
sudo apt install python3 python3-pip python3-venv git -y
```

2. **Clone the repository**
```bash
git clone <repository-url>
cd g_traffic
```

3. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Configuration

1. **Hardware Configuration**
   - Edit `config_dev.py` to match your hardware setup
   - Set camera source (picamera2/webcam)
   - Configure inference mode (hailo/simulation)

2. **OpenAI API Setup** (optional)
   - Create `openai.txt` with your API key for LLM insights
   - Required for AI-powered traffic summaries

3. **Network Configuration**
   - Default dashboard runs on `http://localhost:5000`
   - Edit `FLASK_HOST` in config for remote access

## System Operation

### Starting the System

1. **Development Mode**
```bash
python3 dev_monitor.py
```

2. **Background Execution** (for remote SSH deployments)
```bash
nohup python3 dev_monitor.py &
```

3. **Production Deployment**
```bash
# Install as systemd service
sudo cp traffic-monitor.service /etc/systemd/system/
sudo systemctl enable traffic-monitor
sudo systemctl start traffic-monitor
```

4. **Access Dashboard**
   - Open browser to `http://[pi-ip-address]:5000`
   - View live camera feed and real-time metrics

### Key Features

#### Real-Time Monitoring
- **Live video feed** with person detection overlays
- **Current occupancy** count with confidence indicators
- **Unique visitor tracking** with persistent daily totals
- **Movement pattern analysis** (directional flow)

#### Analytics Dashboard
- **Time-based metrics**: 1min, 5min, 30min, hourly windows
- **Daily performance**: Peak counts, visitor totals since 08:00
- **Historical charts**: Real-time data visualization
- **Movement patterns**: Traffic flow analysis

#### Data Management
- **Persistent storage**: CSV logging with automatic rotation
- **Daily reset**: Automatic data reset at store opening (08:00)
- **Export capabilities**: Historical data export
- **Backup system**: Automatic file management

#### AI Integration
- **Edge inference**: HAILO-8 accelerated person detection
- **LLM insights**: OpenAI-powered traffic summaries
- **Pattern recognition**: Advanced tracking algorithms
- **Confidence filtering**: Configurable detection thresholds

## Use Cases and Development Paths

### Immediate Supermarket Applications

#### Store Operations
- **Staff scheduling optimization**: Real-time data for shift planning
- **Queue management**: Monitor checkout area traffic
- **Security enhancement**: Unusual pattern detection
- **Customer service**: Peak time identification for support staffing

#### Business Intelligence
- **Peak hour analysis**: Identify busiest shopping periods
- **Seasonal patterns**: Track traffic variations over time
- **Space utilization**: Monitor department-specific traffic
- **Marketing effectiveness**: Measure promotional impact on foot traffic

### Supermarket Chain Integration

#### AWS Cloud Integration
```
Local Pi Devices → AWS IoT Core → Lambda Functions → DynamoDB/S3
                                      ↓
                           QuickSight Dashboards ← SNS Alerts
```

**Implementation Path:**
1. **AWS IoT Core**: Secure device communication and data ingestion
2. **AWS Lambda**: Real-time data processing and alerting
3. **Amazon DynamoDB**: High-performance traffic data storage
4. **Amazon S3**: Long-term data archival and analytics
5. **Amazon QuickSight**: Chain-wide dashboard and reporting
6. **Amazon SNS**: Alert system for anomaly detection

#### Multi-Store Deployment
- **Centralized monitoring**: Chain-wide traffic dashboard
- **Comparative analytics**: Store performance benchmarking
- **Predictive modeling**: ML-based traffic forecasting
- **Inventory optimization**: Traffic-based stock management

#### Advanced Analytics Integration
- **Amazon SageMaker**: Custom ML model development
- **AWS Kinesis**: Real-time data streaming
- **Amazon Athena**: SQL queries on historical data
- **AWS Glue**: ETL pipelines for data transformation

### Enterprise Features

#### Scalability Enhancements
- **Load balancing**: Multiple camera inputs per store
- **Edge computing**: Distributed inference processing
- **Data federation**: Multi-location data aggregation
- **API gateway**: Standardized data access

#### Compliance and Security
- **GDPR compliance**: Privacy-focused person tracking
- **Data encryption**: End-to-end security
- **Audit logging**: Complete system traceability
- **Role-based access**: Hierarchical user permissions

#### Integration Capabilities
- **POS system integration**: Correlate traffic with sales
- **ERP connectivity**: Link to inventory management
- **CRM integration**: Customer journey analysis
- **Weather API**: External factor correlation

### Future Development Roadmap

#### Phase 1: Enhanced Analytics
- **Heat mapping**: Store layout optimization
- **Dwell time analysis**: Customer engagement metrics
- **Conversion tracking**: Traffic to sales correlation
- **A/B testing**: Layout change impact measurement

#### Phase 2: Predictive Intelligence
- **Demand forecasting**: ML-based traffic prediction
- **Staffing optimization**: Automated schedule recommendations
- **Inventory alerts**: Traffic-based restocking triggers
- **Marketing automation**: Dynamic promotional content

#### Phase 3: Ecosystem Integration
- **Supply chain integration**: Vendor collaboration
- **Customer app integration**: Personalized shopping experiences
- **Smart building systems**: HVAC and lighting optimization
- **Competitive intelligence**: Market trend analysis

## Technical Architecture

### System Components
- **Detection Engine**: OpenCV + HAILO-8 inference
- **Tracking System**: Multi-object tracking with ID persistence
- **Web Framework**: Flask + SocketIO for real-time updates
- **Data Layer**: CSV logging + JSON configuration
- **AI Integration**: OpenAI API for insights generation

### Performance Specifications
- **Processing Speed**: 30 FPS real-time inference
- **Detection Accuracy**: >95% person detection confidence
- **Tracking Persistence**: Cross-frame identity maintenance
- **Memory Usage**: <2GB RAM typical operation
- **Storage Requirements**: ~100MB per day typical logging

### Network Requirements
- **Bandwidth**: 10Mbps minimum for remote dashboard access
- **Latency**: <100ms for real-time updates
- **Protocol Support**: HTTP/HTTPS, WebSocket, MQTT (future)

## Support and Maintenance

### Monitoring
- **Health checks**: Automatic system status monitoring
- **Log rotation**: Automated cleanup and archival
- **Performance metrics**: System resource utilization
- **Alert system**: Configurable notification thresholds

### Troubleshooting
- **Debug modes**: Verbose logging for issue diagnosis
- **Simulation modes**: Hardware-independent testing
- **Recovery procedures**: Automatic restart and fallback
- **Documentation**: Comprehensive operational guides

This system provides a robust foundation for supermarket traffic monitoring with clear paths for enterprise-scale deployment and AWS cloud integration.

