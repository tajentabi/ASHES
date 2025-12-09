#!/usr/bin/env python3
"""
RTL-SDR hydrogen line FFT integrator (windowed PSD averaging)
"""

from rtlsdr import RtlSdr
import numpy as np
import sys
from pathlib import Path
import tkinter as tk
from datetime import datetime as dt
import time

CENTER_HZ = 1_420_435_752 # 1420.405752 MHz
SAMP_RATE = 2_048_000
GAIN_DB = 49.6 # pick closest valid gain if exact not available
SAMPLES_PER_FRAME = 262_144 # power of 2 for fast FFT (~0.128 s @ 2.048 MS/s)
FRAMES = 1200 # ~2.5 min integration
WARMUP_FRAMES = 10 # discard to let AGC/LNA settle; AGC disabled anyway
APPLY_DC_NOTCH = True # remove a few bins around DC (zero IF)
DC_BINS = 3
OUTFILE_SUFFIX = "HAspectrum"
OBS_DUR_HRS = 6
INTERVAL_MINS = 10

def initialize_sdr():
    sdr = RtlSdr()
    sdr.sample_rate = SAMP_RATE
    sdr.center_freq = CENTER_HZ
    sdr.freq_correction = 60  # PPM
    sdr.set_agc_mode(False)
    try:
        valid = np.array(sdr.valid_gains_db, dtype=float)
        sdr.gain = float(valid[np.argmin(np.abs(valid - GAIN_DB))])
    except Exception:
        sdr.gain = "auto"
    try:
        sdr.bandwidth = 0
    except Exception:
        pass
    _ = sdr.read_samples(4096)
    return sdr

def frame_psd(x: np.ndarray, fs: float):
    x = np.asarray(x)
    N = x.size
    # Hann window
    w = np.hanning(N)
    X = np.fft.fftshift(np.fft.fft(x*w, n=N))
    # Power spectral density proportional to |X|^2, normalized by window power
    # (no absolute units without calibration chain; this keeps frames comparable)
    psd = (np.abs(X) ** 2) / (np.sum(w**2))
    frel = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/fs))  # relative Hz around DC
    return frel, psd

def main():
    sdr = None
    # Calculate num of observations and start integration loop
    obsnum = (OBS_DUR_HRS*60)/INTERVAL_MINS
    try:
        for obs in range(obsnum):
            sdr = initialize_sdr()
            print("Center: %.6f MHz | Fs: %.3f MS/s | Gain: %s" %
                (sdr.center_freq/1e6, sdr.sample_rate/1e6, str(sdr.gain)))
            # Collect digital sig from SDR
            for _ in range(WARMUP_FRAMES):
                _ = sdr.read_samples(SAMPLES_PER_FRAME)
            psd_sum = None
            frel = None
            good = 0
            # Process into frequency domain w/ psd
            for k in range(FRAMES):
                x = sdr.read_samples(SAMPLES_PER_FRAME)
                frel, psd = frame_psd(x, sdr.sample_rate)
                if APPLY_DC_NOTCH:
                    center = np.argmin(np.abs(frel))
                    lo = max(0, center - DC_BINS)
                    hi = min(psd.size, center + DC_BINS + 1)
                    left = psd[lo-1] if lo-1 >= 0 else psd[hi]
                    right = psd[hi] if hi < psd.size else psd[lo-1]
                    psd[lo:hi] = 0.5*(left + right)
                if psd_sum is None:
                    psd_sum = psd
                else:
                    psd_sum += psd
                good += 1
                if (k+1) % max(1, FRAMES//10) == 0:
                    print(f"Integrated {k+1}/{FRAMES} frames...")
            if good == 0:
                raise RuntimeError("No frames integrated.")
            psd_avg = psd_sum / good
            f_abs = CENTER_HZ + frel
            psd_db = 10*np.log10(psd_avg + 1e-20)
            Path(".").mkdir(parents=True, exist_ok=True)
            # Append timestamp from internal RTC and save frequency domain to npz
            now=dt.now()
            OUTFILE=f"{OUTFILE_SUFFIX}_{now.year}.{now.day}.{now.hour}.{now.minute}.{now.second}.npz"
            np.savez_compressed(OUTFILE, freq_Hz=f_abs.astype(np.float64), psd_dB=psd_db.astype(np.float32),
                    center_Hz=CENTER_HZ, fs_Hz=SAMP_RATE, frames=good, window="hann",
                    dc_notch=APPLY_DC_NOTCH, samples_per_frame=SAMPLES_PER_FRAME, time=now.isoformat())
            print(f"Saved: {OUTFILE} (frames={good})")
            time.sleep(INTERVAL_MINS*60)
    except Exception as e:
        sys.exit(f"Err: {e}")
    finally:
        try:
            if sdr is not None:
                sdr.close()
        except Exception:
            pass
        
if __name__ == "__main__":
    main()
