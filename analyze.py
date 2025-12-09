#!/usr/bin/env python3
"""
Waterfall spectrogram plotter for npz output
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
from pathlib import Path

FILE_PATTERN = "HAspectrum*.npz"
EXPECTED_LINE_HZ = 1420.405752e6
PLOT_RELATIVE_TO_LINE = True
SUBTRACT_PER_SPECTRUM_MEDIAN = True

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

def plot_waterfall(times, freq_Hz, waterfall):
    t_nums = mdates.date2num(times)

    if PLOT_RELATIVE_TO_LINE:
        x_axis = (freq_Hz - EXPECTED_LINE_HZ) / 1e6  # MHz offset
        xlabel = "Frequency Offset from 1420.405752 MHz (MHz)"
    else:
        x_axis = freq_Hz / 1e6
        xlabel = "Frequency (MHz)"
    def centers_to_edges(c):
        edges = np.empty(c.size + 1, dtype=c.dtype)
        edges[1:-1] = 0.5 * (c[:-1] + c[1:])
        edges[0] = c[0] - (c[1] - c[0]) / 2
        edges[-1] = c[-1] + (c[-1] - c[-2]) / 2
        return edges

    x_edges = centers_to_edges(x_axis)
    t_edges = centers_to_edges(np.array(t_nums))

    X, T = np.meshgrid(x_edges, t_edges)

    plt.figure(figsize=(11, 6))
    plt.pcolormesh(X, T, waterfall, shading="auto")
    cbar = plt.colorbar(label="Power (relative dB)")

    plt.xlabel(xlabel)
    plt.ylabel("Time (UTC)")

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter("%H:%M\n%m-%d")
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(formatter)

    if PLOT_RELATIVE_TO_LINE:
        plt.axvline(0.0, color="k", ls=":", lw=1)

    plt.title("Hydrogen Line Waterfall Spectrogram")
    plt.tight_layout()
    plt.show()

def main():
    records = load_all_spectra(FILE_PATTERN)
    times, freq_Hz, waterfall, names = build_waterfall(records)

    print("Time span:")
    print(f"  Start: {times[0].isoformat()}")
    print(f"  End  : {times[-1].isoformat()}")
    print(f"Loaded {len(times)} spectra, {freq_Hz.size} channels each.")
    plot_waterfall(times, freq_Hz, waterfall)

if __name__ == "__main__":
    main()
