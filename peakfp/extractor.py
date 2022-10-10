"""Extracts PeakFP fingerprints.
"""
import argparse
import pickle

import numpy as np
from librosa import load, stft
from skimage.feature import peak_local_max

import constants


def extract_peakfp(audio_path: str, peakfp_path: str,
                   peaks_distance: int, spectrogram_path: str=None):
    """Extracts PeakFP fingerprints.
    It uses skimage peak_local_max on an image spectrogram.
    Fingerprint is a numpy array of shape (N, 2).
        N: Number of peaks extracted
        2: Spectrogram coordinates of the peak.
            window_id --> Number of rows = n_ftt/2 + 1 (Nyquist)
            frequency -->
    To get the time, divide the hop sample size (128) by the sample rate (8000)
    and this makes every index in the fft represent 0.016s. When visualizing an
    spectrogram, we often transform the amplitude to dB using 20·log10(amp),
    but it doesn't mind if we find the local peaks on the dB spectrogram or the
    amplitude one.

    Args:
        audio_path (str): Origin audio path (any format accepted by librosa).
        peakfp_path (str): Destination fingerprint path.
        peaks_distance (int): The minimal allowed distance separating peaks.
            To find the maximum number of peaks, use min_distance=1.
        spectrogram_path (str): Optional destination spectrogram path
            for debugging purposes.
    Returns:
        None
    """

    audio_ts, _ = load(audio_path, sr=constants.SAMPLE_RATE,
                       mono=constants.MONO)
    spectrogram = np.abs(stft(audio_ts,
                              window=constants.WINDOW,
                              n_fft=constants.FFT_WINDOW_SIZE,
                              hop_length=constants.FFT_HOP_SIZE))
    peaks = peak_local_max(spectrogram, min_distance=peaks_distance)
    sorted_peaks = np.array(sorted(peaks, key=lambda x:x[1]))  # sort by time

    if spectrogram_path:
        with open(spectrogram_path, 'wb') as spec_file:
            pickle.dump(spectrogram, spec_file)

    with open(peakfp_path, 'wb') as peak_file:
        pickle.dump(sorted_peaks, peak_file)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path",
        help="Origin audio path (any format accepted by librosa).")
    parser.add_argument(
        "peakfp_path",
        help="Destination fingerprint path.")
    parser.add_argument(
        "--peaks_distance",
        default=5,
        type=int,
        help="Minimum score. Equivalent to number of peaks in common.")
    parser.add_argument(
        "--spectrogram_path",
        help="Optional destination spectrogram path (debug purposes).")
    args = parser.parse_args()
    extract_peakfp(args.audio_path,
                   args.peakfp_path,
                   args.peaks_distance,
                   args.spectrogram_path)
