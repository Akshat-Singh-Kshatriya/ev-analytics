# EV Powertrain Fault Diagnostics & Efficiency Mapping

Synthetic three-phase current and vibration signals are used to build a multi-class diagnostic baseline for healthy, bearing-fault, electrical-imbalance and rotor-eccentricity states. Includes signal features, classification metrics and an illustrative efficiency map.

Run:
```bash
pip install -r requirements.txt
python src/run_diagnostics.py
```

**Limitations:** Signal signatures are synthetic and intentionally encode class differences. This is not a validated motor controller, safety system or proof of real-vehicle diagnostic performance. Validate with labelled dynamometer/vehicle data, sensor calibration and out-of-distribution tests.
