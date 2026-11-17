import numpy as np
CLASSES=["healthy","bearing_fault","electrical_imbalance","rotor_eccentricity"]
def make_signal(label,speed_rpm,torque_nm,fs=2000,seconds=1,seed=42):
    rng=np.random.default_rng(seed); n=fs*seconds; t=np.arange(n)/fs
    f=speed_rpm/60; amp=2+torque_nm/100
    current=amp*np.sin(2*np.pi*f*t+rng.uniform(0,2*np.pi))+.12*rng.normal(size=n)
    vibration=.08*rng.normal(size=n)
    if label=="bearing_fault": vibration+=.35*np.sin(2*np.pi*4.2*f*t)+.18*np.sin(2*np.pi*8.4*f*t)
    elif label=="electrical_imbalance": current+=.45*np.sin(2*np.pi*2*f*t+.4)
    elif label=="rotor_eccentricity":
        current+=.3*np.sin(2*np.pi*.5*f*t); vibration+=.22*np.sin(2*np.pi*f*t)
    return current,vibration
