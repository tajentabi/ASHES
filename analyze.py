#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

FILE = "HAspectrum_Hz_dB.npz"
SMOOTH_WINDOW = 1
SMOOTH_POLY = 3
SHOW_BASELINE = False
EXPECTED_LINE_HZ = 1420.405752e6
TUNE_OFFSET_HZ = 30e3

data = np.load(FILE)
freq_Hz = data["freq_Hz"]
psd_dB = data["psd_dB"]

freq_Hz_corrected = freq_Hz - TUNE_OFFSET_HZ
f_rel_MHz = (freq_Hz_corrected - EXPECTED_LINE_HZ) / 1e6
psd_dB = np.array(psd_dB)
if SMOOTH_WINDOW > 3:
    psd_smooth = savgol_filter(psd_dB, SMOOTH_WINDOW, SMOOTH_POLY)
else:
    psd_smooth = psd_dB
if SHOW_BASELINE:
    z = np.polyfit(f_rel_MHz, psd_smooth, deg=3)
    baseline = np.polyval(z, f_rel_MHz)
    corrected = psd_smooth - baseline
else:
    baseline = np.zeros_like(psd_smooth)
    corrected = psd_smooth
plt.figure(figsize=(10, 5))
plt.plot(f_rel_MHz, psd_smooth, label="Smoothed Spectrum", color="tab:blue")
if SHOW_BASELINE:
    plt.plot(f_rel_MHz, baseline, "--", label="Fitted Baseline", color="gray")
    plt.plot(f_rel_MHz, corrected, label="Baseline-Removed", color="tab:red")

plt.title("Hydrogen Line Integration Spectrum")
plt.xlabel("Frequency Offset from 1420.405752 MHz (MHz)")
plt.ylabel("Relative Power (dB)")
plt.axvline(0, color="k", ls=":", label="Rest frequency")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

