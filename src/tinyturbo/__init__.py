"""TinyTurbo: a reproducible study of assisted (speculative) decoding.

The package is deliberately small and linear:

    config  -> what to run
    env     -> where it is running, and with which dtype/attention
    prompts -> the fixed inputs
    models  -> loading the target and assistant models
    instrument -> forward-pass counters, timers and memory readings
    methods -> the exact generate() kwargs for baseline vs assisted
    runner  -> runs the trials and writes one raw JSONL record per trial
    metrics -> pure functions over those records
    report  -> raw records -> CSV + plots

Run it with `python -m tinyturbo --help`.
"""

__version__ = "0.1.0"
