# ARM AI Optimization Challenge — Mobile LLM for Sensor Triage

**Track:** AI Model Optimization  
**Inference Engine:** llama.cpp + GGUF  
**Base Model:** Qwen2-0.5B (quantized to Q4_K_M)  
**Target Device:** Raspberry Pi 5 (ARM Cortex-A76)  
**Fine-tuning GPU:** RTX3000 8GB (local) / RTX4090 (SSH)

## Overview

This project optimizes a small language model (Qwen2-0.5B) for on-device ARM deployment
to triage real-time sensor data and trigger alerts within **1–2 seconds**.

**Key differentiator:** A hybrid approach combining fast rule-based threshold checks
(<10 ms) with LLM-powered context triage — the LLM only runs when thresholds are
breached, keeping average latency well under 2 seconds while retaining intelligence
for complex edge cases.

## Architecture

```
[Sensor Hardware: temp / humidity / people]
          |
          v (real-time CSV stream, 1 reading/s)
[Sensor Reader] --> SensorWindow (last N seconds)
          |
          v
[Threshold Rules] -- fast pass (<10 ms) --> [No alert] (~90 % of windows)
          |
          v triggered
[LLM Triage] (Qwen2-0.5B Q4_K_M via llama.cpp)
          |
          v structured JSON
[Alert Actions] --> BUZZER / log / network notification
```

## Directory Structure

```
optimize-arm-qc480b/
  src/
    model_optimization/
      download_model.py      # Pull Qwen2-0.5B from Hugging Face
      export_to_gguf.py      # HF --> GGUF conversion
      quantize_model.py      # Q4_K_M quantization + benchmarks
      benchmark.py           # tokens/s, TTFT, memory profiling
    data_processing/
      sensor_generator.py    # Synthetic CSV data with realistic patterns
      sensor_reader.py       # Unified CSV/JSON/SQLite3 --> SensorWindow
    alerting/
      threshold_rules.py     # Fast rule checks (temp, humidity, people)
      llm_triage.py          # Qwen inference via llama-cpp-python
      alert_engine.py        # Hybrid pipeline orchestrator
      alert_actions.py       # BUZZER / log / notify abstractions
      latency_monitor.py     # End-to-end timing, sub-2 s enforcement
  models/                   # Downloaded + quantized GGUF files
  docs/
    rpi5_deployment.md       # RPi 5 setup guide
    benchmark_results.md     # Performance measurements
  tests/                     # Unit tests for all modules
  requirements.txt
  setup.sh                   # One-command environment setup
  benchmark.sh               # Automated benchmark suite
  LICENSE                    # Apache 2.0
```

## Setup

### Prerequisites

- Python 3.10+
- ARM device (Raspberry Pi 5 recommended) or any Linux/macOS for development
- GPU (optional) for fine-tuning — RTX3000 8 GB is sufficient for Qwen2-0.5B LoRA

### Quick Start

```bash
git clone <repo> && cd optimize-arm-qc480b
./setup.sh                              # Install deps, build llama.cpp, download model
python -m src.model_optimization.quantize_model   # Quantize to Q4_K_M
python -m src.data_processing.sensor_generator    # Generate synthetic sensor data
python -m src.alerting.alert_engine               # Run hybrid alert pipeline
```

## Model Pipeline

1. **Download** `Qwen/Qwen2-0.5B` from Hugging Face
2. **Convert** to GGUF format via llama.cpp `convert.py`
3. **Quantize** to Q4_K_M (4.5 bits per weight — best quality/size ratio)
4. **Fine-tune** (optional) with LoRA on sensor domain data using RTX3000/RTX4090
5. **Benchmark** tokens/s, time-to-first-token, peak memory

## Alert Pipeline

1. **Sensor Ingestion** — reads CSV/JSON/SQLite3 streams, creates rolling windows
2. **Threshold check** — fast rules evaluate temperature (>40 °C), humidity (>90 %),
   people-count anomalies — completes in <10 ms
3. **LLM Triage** — if threshold triggered, Qwen2-0.5B analyzes the sensor window
   with a structured prompt and outputs `{"alert": bool, "reason": str, "severity": "low"|"medium"|"high"}`
4. **Alert Action** — triggers BUZZER (GPIO), writes to alert log,
   optionally notifies network endpoint
5. **Latency Monitoring** — measures every step, enforces p99 < 2000 ms

## Prompt Design

**System prompt for on-device LLM:**

```
You are a sensor data triage specialist. Analyze the recent sensor readings and
determine if an alert is needed. Respond ONLY with valid JSON:
{"alert": true/false, "reason": "brief explanation", "severity": "low/medium/high"}

Threshold rules have already triggered: {triggered_rules}
Recent sensor data (last {window_size} s):
{sensor_window_formatted}

Assess whether the current conditions truly warrant an alert, considering context
and trends. Consider: is this a transient spike? Is it part of a normal pattern?
```

**Example output:**

```json
{"alert": true, "reason": "Temperature 42°C exceeds threshold 40°C and rising trend over last 30 s. Room is occupied (3 people).", "severity": "high"}
```

## Benchmarking

```bash
./benchmark.sh              # Full benchmark suite
```

Outputs:

- Model size (MB) before / after quantization
- Inference speed (tokens/s) at fp16, Q8_0, Q4_K_M
- Time-to-first-token (ms)
- Peak memory usage (MB)
- End-to-end alert latency (p50 / p95 / p99)

## Latency Budget

| Path                  | Est. Time    | % of Windows | Contribution to p99 |
|-----------------------|-------------|--------------|---------------------|
| Threshold only        | ~10 ms      | ~90%         | Negligible          |
| Threshold + LLM       | 200–800 ms  | ~10%         | ~80 % of total      |
| **Effective p99**     | **~300 ms** | **All**      | Well under 2000 ms  |

## Raspberry Pi 5 Deployment

See `docs/rpi5_deployment.md` for:

- Cross-compiling llama.cpp for aarch64
- Transferring GGUF models
- GPIO wiring for BUZZER output
- Running as a systemd service

## License

Apache 2.0
