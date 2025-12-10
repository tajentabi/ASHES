#!/usr/bin/env python3
"""
Basic waterfall spectrogram of RTL-SDR hydrogen-line integrations.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
from pathlib import Path
from scipy.signal import savgol_filter

FILE_PATTERN = "HAspectrum*.npz"
EXPECTED_LINE_HZ = 1420.405752e6
PLOT_RELATIVE_TO_LINE = True
SUBTRACT_PER_SPECTRUM_MEDIAN = True
SMOOTH_WINDOW = 41
SMOOTH_POLY = 3
SHOW_BASELINE = False
TUNE_OFFSET_HZ = 30e3

def load_all_spectra(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No files found matching pattern: {pattern}")

    print(f"Found {len(files)} files.")

    records = []
    for fname in files:
        d = np.load(fname)
        freq_Hz = np.asarray(d["freq_Hz"])
        psd_dB = np.asarray(d["psd_dB"])
        time_str = str(d["time"])
        t = datetime.fromisoformat(time_str)
        records.append((t, freq_Hz, psd_dB, Path(fname).name))

    records.sort(key=lambda r: r[0])
    return records

def build_waterfall(records):
    times = [r[0] for r in records]
    names = [r[3] for r in records]
    freq_ref = records[0][1]
    for t, freq_Hz, psd_dB, name in records[1:]:
        if freq_Hz.shape != freq_ref.shape or not np.allclose(freq_Hz, freq_ref, rtol=0, atol=1e-3):
            raise RuntimeError(
                f"Frequency axis mismatch in file {name}. "
                "Make sure all integrations use the same SDR settings."
            )
    waterfall = np.stack([r[2] for r in records], axis=0)
    if SUBTRACT_PER_SPECTRUM_MEDIAN:
        med = np.median(waterfall, axis=1, keepdims=True)
        waterfall = waterfall - med

    return times, freq_ref, waterfall, names

def plot_waterfall(ax, times, freq_Hz, waterfall):
    t_nums = mdates.date2num(times)
    if PLOT_RELATIVE_TO_LINE:
        x_axis = (freq_Hz - EXPECTED_LINE_HZ) / 1e6
        xlabel = "Frequency Offset from 1420.405752 MHz (MHz)"
    else:
        x_axis = freq_Hz / 1e6
        xlabel = "Frequency (MHz)"

    # Choose reasonable default steps
    # For time: 1/48 day = 30 minutes if only one row
    # For freq: use spacing between channels if >1, else ~0.01 MHz
    if x_axis.size > 1:
        dx_default = float(x_axis[1] - x_axis[0])
    else:
        dx_default = 0.01  # 10 kHz is a little arbitrary but whatevs
    def centers_to_edges(c, default_step=1.0):
        c = np.asarray(c)
        if c.size == 1:
            return np.array([c[0] - default_step / 2, c[0] + default_step / 2], dtype=c.dtype)
        edges = np.empty(c.size + 1, dtype=c.dtype)
        edges[1:-1] = 0.5 * (c[:-1] + c[1:])
        edges[0] = c[0] - (c[1] - c[0]) / 2
        edges[-1] = c[-1] + (c[-1] - c[-2]) / 2
        return edges
    x_edges = centers_to_edges(x_axis, default_step=dx_default)
    t_edges = centers_to_edges(np.array(t_nums), default_step=1/48)  # 30 min total height
    X, T = np.meshgrid(x_edges, t_edges)
    c = ax.pcolormesh(X, T, waterfall, shading="auto")
    plt.colorbar(c, ax=ax, label="Power (relative dB)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (UTC)")
    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter("%H:%M\n%m-%d")
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(formatter)
    if PLOT_RELATIVE_TO_LINE:
        ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_title("Hydrogen Line Waterfall Spectrogram")

def plot_line_spectrum(ax, freq_Hz, psd_dB):
    freq_Hz_corrected = freq_Hz - TUNE_OFFSET_HZ

    # Convert to MHz relative to 1420.405752 MHz
    f_rel_MHz = (freq_Hz_corrected - EXPECTED_LINE_HZ) / 1e6
    psd_dB = np.array(psd_dB)

    # Savitzky–Golay filter
    if SMOOTH_WINDOW > 3:
        psd_smooth = savgol_filter(psd_dB, SMOOTH_WINDOW, SMOOTH_POLY)
    else:
        psd_smooth = psd_dB

    # Optional baseline subtraction (simple polynomial fit)
    if SHOW_BASELINE:
        z = np.polyfit(f_rel_MHz, psd_smooth, deg=3)
        baseline = np.polyval(z, f_rel_MHz)
        corrected = psd_smooth - baseline
    else:
        baseline = np.zeros_like(psd_smooth)
        corrected = psd_smooth
    ax.plot(f_rel_MHz, psd_smooth, label="Smoothed Spectrum", color="tab:blue")
    if SHOW_BASELINE:
        ax.plot(f_rel_MHz, baseline, "--", label="Fitted Baseline", color="gray")
        ax.plot(f_rel_MHz, corrected, label="Baseline-Removed", color="tab:red")
    ax.set_title("Hydrogen Line Integration Spectrum")
    ax.set_xlabel("Frequency Offset from HA line (MHz)")
    ax.set_ylabel("Relative Power (dB)")
    ax.axvline(0, color="k", ls=":", label="Rest frequency")
    ax.legend()
    ax.grid(True, alpha=0.4)

def main():
    records = load_all_spectra(FILE_PATTERN)
    times, freq_Hz, waterfall, names = build_waterfall(records)

    print("Time span:")
    print(f"  Start: {times[0].isoformat()}")
    print(f"  End  : {times[-1].isoformat()}")
    print(f"Loaded {len(times)} spectra, {freq_Hz.size} channels each.")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
    plot_waterfall(ax1, times, freq_Hz, waterfall)
    plot_line_spectrum(ax2, freq_Hz, waterfall[0])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
