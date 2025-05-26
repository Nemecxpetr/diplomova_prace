from asyncio import Handle
import os
import subprocess
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from librosa.sequence import dtw as constrained_dtw
from pathlib import Path
import csv
from scipy.signal import correlate
import pandas as pd


from Handler.audio_handler import sonify_midi, read_audio

def pad_to_equal_length(a, b):
    """
    Pads two 1D arrays to the same length by zero-padding the shorter one.

    Args:
        a (np.ndarray): First input array.
        b (np.ndarray): Second input array.

    Returns:
        tuple: Tuple of two arrays padded to equal length.
    """
    max_len = max(len(a), len(b))
    a_padded = np.pad(a, (0, max_len - len(a)), mode='constant')
    b_padded = np.pad(b, (0, max_len - len(b)), mode='constant')
    return a_padded, b_padded

def compute_novelty(audio_path, hop_length=512, feature_type='stft', threshold=0.4):
    """
    Computes a normalized spectral novelty function from an audio file.

    Args:
        audio_path (str): Path to the input audio file.
        hop_length (int): Hop length for STFT/CQT computation. Default is 512.
        feature_type (str): Type of spectral representation ('stft' or 'cqt_1').

    Returns:
        tuple: (novelty, sample_rate, hop_length), where novelty is a 1D np.ndarray.
    """
    y, sr = librosa.load(audio_path, sr=None)
    y, _ = librosa.effects.trim(y)

    if feature_type in ['cqt', 'cqt_1']:
        C = librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1'))
        S = np.abs(C)
    else:
        S = np.abs(librosa.stft(y, hop_length=hop_length))

    novelty = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop_length)
    novelty = (novelty - np.min(novelty)) / (np.max(novelty) - np.min(novelty) + 1e-8)

    novelty = np.where(novelty > threshold, novelty, 0)
    # QUICKER V.: novelty[novelty <= threshold] = 0

    return novelty.astype(np.float32), sr, hop_length

def binarize_novelty(novelty, threshold=0.1):
    """
    Converts a normalized novelty function to binary based on a threshold.

    Args:
        novelty (np.ndarray): Input novelty function.
        threshold (float): Threshold for binarization. Default is 0.1.

    Returns:
        np.ndarray: Binarized novelty (float32 array of 0s and 1s).
    """
    novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)
    return (novelty > threshold).astype(np.float32)

def dtw_score(nov1, nov2, band_width=100):
    """
    Computes the normalized DTW distance between two novelty curves.

    Args:
        nov1 (np.ndarray): First novelty function.
        nov2 (np.ndarray): Second novelty function.
        band_width (int): Sakoe-Chiba band radius. Default is 100.

    Returns:
        float: DTW distance normalized by warping path length.
    """
    cost = cdist(nov1[:, np.newaxis], nov2[:, np.newaxis], metric='euclidean')
    D, wp = constrained_dtw(C=cost, global_constraints=True, band_rad=band_width)
    return D[-1, -1] / len(wp)

def dtw_distance(nov1, nov2, hop_length, sr, band_width=100):
    """
    Computes the normalized DTW distance and returns warping path.
    """
    nov1, nov2 = pad_to_equal_length(nov1, nov2)
    cost = cdist(nov1[:, np.newaxis], nov2[:, np.newaxis], metric='euclidean')
    D, wp = constrained_dtw(C=cost, global_constraints=True, band_rad=band_width)
    
    dtw_score= D[-1, -1] / len(wp)

    return dtw_diagonal_deviation(wp, hop_length, sr)
    

def dtw_diagonal_deviation(wp, hop_length, sr):
    """
    Measures average deviation from diagonal of a warping path in seconds.

    Args:
        wp (np.ndarray): Warping path from DTW, shape (L, 2)
        hop_length (int): Hop size in samples
        sr (int): Sample rate

    Returns:
        float: Average deviation from diagonal in seconds
    """
    frame_diff = np.abs(wp[:, 0] - wp[:, 1])
    time_diff = frame_diff * hop_length / sr
    return np.mean(time_diff)


def compute_mean_error(nov1, nov2):
    """
    Computes the mean absolute error (MAE) between two novelty functions.

    Args:
        nov1 (np.ndarray): First novelty function.
        nov2 (np.ndarray): Second novelty function.

    Returns:
        float: Mean absolute error.
    """

    nov1, nov2 = pad_to_equal_length(nov1, nov2)
    return np.mean(np.abs(nov1 - nov2))

def novelty_cross_correlation(nov1, nov2):
    """
    Computes normalized cross-correlation and lag between two novelty functions.

    Args:
        nov1 (np.ndarray): First novelty function.
        nov2 (np.ndarray): Second novelty function.

    Returns:
        tuple: (max correlation value, lag in frames).
    """
    nov1, nov2 = pad_to_equal_length(nov1, nov2)
    x = nov1
    y = nov2
    xcorr = correlate(x, y, mode='full')
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm == 0:
        return 0.0, 0
    xcorr /= norm
    lags = np.arange(-len(x) + 1, len(x))
    peak_idx = np.argmax(xcorr)
    return xcorr[peak_idx], lags[peak_idx]

def detect_peaks(novelty, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=0.1, wait=10):
    """
    Detects peaks in a novelty function using Librosa's peak-picking algorithm.

    Args:
        novelty (np.ndarray): Input novelty function.
        pre_max (int): Lookback for local maximum. Default is 5.
        post_max (int): Lookahead for local maximum. Default is 5.
        pre_avg (int): Lookback for local average. Default is 5.
        post_avg (int): Lookahead for local average. Default is 5.
        delta (float): Minimum required difference to average. Default is 0.1.
        wait (int): Minimum frames between detected peaks. Default is 10.

    Returns:
        np.ndarray: Indices of detected peaks.
    """
    return librosa.util.peak_pick(x=novelty, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg,
                                  post_avg=post_avg, delta=delta, wait=wait)

def peak_alignment_error(nov1, nov2, sr, hop_length, band_width=100):
    """
    Computes the average time error between peak sequences in two novelty functions.

    Args:
        nov1 (np.ndarray): First novelty function.
        nov2 (np.ndarray): Second novelty function.
        sr (int): Sampling rate of the audio.
        hop_length (int): Hop length used during feature extraction.
        band_width (int): Bandwidth for DTW warping. Default is 100.

    Returns:
        float: Mean absolute difference in seconds between aligned peak times.
    """
    peaks1 = detect_peaks(nov1)
    peaks2 = detect_peaks(nov2)
    if len(peaks1) == 0 or len(peaks2) == 0:
        return np.nan
    cost = cdist(peaks1[:, np.newaxis], peaks2[:, np.newaxis], metric='euclidean')
    D, wp = constrained_dtw(C=cost, global_constraints=True, band_rad=band_width)
    times1 = librosa.frames_to_time(peaks1, sr=sr, hop_length=hop_length)
    times2 = librosa.frames_to_time(peaks2, sr=sr, hop_length=hop_length)
    diffs = [abs(times1[i] - times2[j]) for i, j in wp]
    return np.mean(diffs)


def beat_alignment_error(audio1, audio2, sr, hop_length):
    """
    Compares beat positions extracted from two audio signals and computes mean absolute error.

    Args:
        audio1 (np.ndarray): First audio signal (e.g., original interpretation).
        audio2 (np.ndarray): Second audio signal (e.g., sonified MIDI).
        sr (int): Sampling rate.
        hop_length (int): Hop length used for beat detection.
    
    Returns:
        float: Mean absolute time error between aligned beat positions, in seconds.
    """
    # Ensure mono signals
    if audio1.ndim > 1:
        audio1 = librosa.to_mono(audio1)
    if audio2.ndim > 1:
        audio2 = librosa.to_mono(audio2)

    # Beat tracking
    _, beats1 = librosa.beat.beat_track(y=audio1, sr=sr, hop_length=hop_length)
    _, beats2 = librosa.beat.beat_track(y=audio2, sr=sr, hop_length=hop_length)

    if len(beats1) < 8 or len(beats2) < 8:
        return np.nan

    # Convert to seconds
    times1 = librosa.frames_to_time(beats1, sr=sr, hop_length=hop_length)
    times2 = librosa.frames_to_time(beats2, sr=sr, hop_length=hop_length)

    # Align beat sequences using DTW
    cost = cdist(times1[:, np.newaxis], times2[:, np.newaxis], metric='euclidean')
    D, wp = constrained_dtw(C=cost, global_constraints=True, band_rad=100)

    errors = [abs(times1[i] - times2[j]) for i, j in wp]

    mean_error = np.mean(errors)
    return mean_error if mean_error < 5 else np.nan



def downsample(novelty, factor):
    return novelty[::factor]


def plot_novelties(nov_stft, nov_stft_midi, nov_cqt, nov_cqt_midi, hop, sr, out_path=None, show_plot=False):
    """
    Plots STFT and CQT novelty functions for original and MIDI-based signals.

    Args:
        nov_stft (np.ndarray): STFT novelty (original audio).
        nov_stft_midi (np.ndarray): STFT novelty (synthesized MIDI).
        nov_cqt (np.ndarray): CQT novelty (original audio).
        nov_cqt_midi (np.ndarray): CQT novelty (synthesized MIDI).
        hop (int): Hop length used during feature extraction.
        sr (int): Sampling rate.
        out_path (str, optional): Path to save figure as PNG or EPS. Default is None.
        show_plot (bool, optional): If True, show figure interactively. Default is False.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # STFT novelty
    nov_stft, nov_stft_midi = pad_to_equal_length(nov_stft, nov_stft_midi)
    times = librosa.frames_to_time(np.arange(len(nov_stft)), sr=sr, hop_length=hop)
    axs[0].plot(times, nov_stft, label="Original (STFT)", alpha=0.7)
    axs[0].plot(times, nov_stft_midi, label="MIDI (STFT)", alpha=0.7)
    axs[0].set_title("Spectral Novelty (STFT)")
    axs[0].legend()

    # CQT novelty
    nov_cqt, nov_cqt_midi = pad_to_equal_length(nov_cqt, nov_cqt_midi)
    times = librosa.frames_to_time(np.arange(len(nov_cqt)), sr=sr, hop_length=hop)
    axs[1].plot(times, nov_cqt, label="Original (CQT)", alpha=0.7)
    axs[1].plot(times, nov_cqt_midi, label="MIDI (CQT)", alpha=0.7)
    axs[1].set_title("Spectral Novelty (CQT)")
    axs[1].legend()
    axs[1].set_xlabel("Time (s)")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    if show_plot:
        plt.show()
    plt.close(fig)


def evaluate_dual_versions(original_audio_base,
                            midi_base_path,
                            preset_name,
                            soundfont,
                            visualize=False,
                            downsample_factor=1,
                            binarize=False):
    """ DEPRECATED:
    Evaluates and compares synchronization quality of STFT- and CQT-based MIDI alignments
    for a single piece.

    This function loads the original audio and two corresponding synchronized MIDI files
    (one aligned using STFT-based chroma, the other using CQT-based chroma), synthesizes
    audio for both, computes novelty functions, and evaluates alignment performance using
    various metrics.

    Metrics computed:
        - DTW Score (normalized DTW cost)
        - Mean Absolute Error (MAE) between novelty curves
        - Cross-Correlation score and lag
        - Peak Alignment Error
        - Beat-synchronous MAE

    Results are saved in:
        `data/eval/results/{preset_name}/{preset_name}_compare_stft_cqt.csv`

    Args:
        original_audio_base (str or Path): Path to the original reference audio (.wav file).
        midi_base_path (str or Path): Base folder path where MIDI files are located under subfolders 'stft/' and 'cqt_1/'.
        preset_name (str): Name of the evaluation preset (used for output folder).
        soundfont (str): Path to SoundFont (.sf2) file used for MIDI synthesis.
        visualize (bool): If True, generate and save a dual novelty comparison plot.
        downsample_factor (int): Factor by which novelty curves should be downsampled. Default is 1 (no downsampling).
        binarize (bool): If True, binarize the novelty functions before computing metrics. Default is False.

    Output:
        - CSV file comparing STFT vs. CQT results.
        - Optional dual novelty plot (if visualize=True).
        - Console log of progress and any exceptions.
    """
    versions = ["stft", "cqt_1"]
    results = []
    nov_pairs = []
    for version in versions:
        try:
            midi_path = Path(midi_base_path) / version / f"s_{Path(original_audio_base).stem}.mid"
            audio_path = Path(original_audio_base)
            filename = audio_path.stem
            result_dir = Path(f"../../data/eval/results/{preset_name}")
            temp_dir = Path(f"../../data/eval/temp_eval/{preset_name}")
            result_dir.mkdir(parents=True, exist_ok=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
            synth_path = temp_dir / f"{filename}_{version}_synth.wav"
            sonify_midi(str(midi_path), str(synth_path), soundfont=str(soundfont))
            nov_orig, sr, hop = compute_novelty(str(audio_path), feature_type=version)
            nov_synth, _, _ = compute_novelty(str(synth_path), feature_type=version)
            if downsample_factor > 1:
                nov_orig = downsample(nov_orig, downsample_factor)
                nov_synth = downsample(nov_synth, downsample_factor)
            if binarize:
                nov_orig = binarize_novelty(nov_orig)
                nov_synth = binarize_novelty(nov_synth)
            results.append([
                dtw_distance(nov_orig, nov_synth, hop, sr),
                compute_mean_error(nov_orig, nov_synth),
                *novelty_cross_correlation(nov_orig, nov_synth),
                peak_alignment_error(nov_orig, nov_synth, sr=sr, hop_length=hop)])
               
            nov_pairs.append((nov_orig, nov_synth))
        except Exception as e:
            print(f"Error evaluating {version} for {preset_name}: {e}")
            results.append(["error"] * 6)
            nov_pairs.append(([], []))

    result_path = Path(f"../../data/eval/results/csv/{preset_name}_compare_stft_cqt.csv")
    with open(result_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Version", "DTW Score", "Mean Absolute Error", "XCorr Score", "XCorr Lag", "Peak Alignment Error", "Beat Allignement Error"])
        writer.writerow(["stft"] + results[0])
        writer.writerow(["cqt"] + results[1])

    print(f"Comparison for {preset_name} saved to {result_path}")

def evaluate_all_versions_in_preset_folder(preset_name,
                                           soundfont,
                                           visualize=False,
                                           feature_types=["stft", "cqt_1"],
                                           downsample_factor=1,
                                           binarize=False,
                                           threshold=0.1,
                                           skip_sonification=False):
    """
    Evaluates synchronization quality between original audio recordings and MIDI-based
    syntheses for all takes in a given preset folder.

    For each `.wav` audio file in the specified preset, this function loads the corresponding
    MIDI files (e.g., aligned with STFT or CQT features), synthesizes audio from MIDI,
    computes novelty functions, and evaluates alignment quality using multiple metrics.

    Metrics computed:
        - DTW score (normalized)
        - Mean Absolute Error (MAE)
        - Cross-Correlation score and lag (in milliseconds)
        - Peak alignment error
        - Beat-synchronous MAE

    Optionally, visualization plots are saved, and results are written to a CSV file
    (`data/eval/results/csv/{preset}_full_evaluation.csv`).

    Args:
        preset_name (str): Name of the preset folder inside `data/input/audio`.
        soundfont (str): Path to the .sf2 soundfont used for MIDI synthesis.
        visualize (bool): Whether to save novelty comparison plots. Default is False.
        feature_types (list): List of feature types to evaluate (e.g., ["stft", "cqt_1"]).
        downsample_factor (int): Downsampling factor for novelty curves. Default is 1 (no downsampling).
        binarize (bool): Whether to binarize novelty functions before evaluation. Default is False.
        threshold (float): Threshold for binarization (if enabled). Default is 0.1.
        skip_sonification (bool): If True, skip MIDI-to-audio synthesis (assumes pre-rendered audio exists).

    Output:
        - CSV table saved in `eval/results/csv/{preset_name}_full_evaluation.csv`
        - EPS figures for novelty curves (if `visualize` is True)
        - Console printout of evaluation progress
    """

    base_path = Path("../../data/input/audio") / preset_name
    result_dir = Path(f"../../data/eval/results")
    temp_dir = Path(f"../../data/eval/temp_eval/{preset_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    csv_dir = result_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    result_file = csv_dir / f"{preset_name}_full_evaluation.csv"

    plot_dir = result_dir / f"novelty/{preset_name}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    take_index = 1
    sonified = set()

    for audio_file in sorted(base_path.glob("*.wav")):
        raw_name = audio_file.stem
        piece_name = f"Take {take_index}"
        take_index += 1

        novelties = {}
        sr = hop = None

        for feature_type in feature_types:
            try:
                midi_path = Path(f"../../data/output/MIDI/{preset_name}") / feature_type / f"s_{raw_name}.mid"
                if not midi_path.exists():
                    print(f"Missing MIDI: {midi_path}")
                    continue

                synth_path = temp_dir / f"{raw_name}_{feature_type}_synth.wav"
                key = (str(midi_path), str(synth_path))

                if not skip_sonification:
                    if key not in sonified:
                        os.system(f"fluidsynth -ni -F {synth_path} -T wav {soundfont} {midi_path}")
                        sonified.add(key)
                    else:
                        print(f"Skipping sonification, already done: {synth_path}")
                else:
                    print(f"Sonification skipped by parameter for: {synth_path}")
                if not synth_path.exists():
                    print(f"Missing synthesized audio file: {synth_path}")
                    continue

                y_audio, sr = read_audio(str(audio_file))
                y_midi, _ = read_audio(str(synth_path))

                nov1, _, hop = compute_novelty(str(audio_file), feature_type=feature_type)
                nov2, _, _ = compute_novelty(str(synth_path), feature_type=feature_type)

                if downsample_factor > 1:
                    nov1 = downsample(nov1, downsample_factor)
                    nov2 = downsample(nov2, downsample_factor)
                effective_hop = hop * downsample_factor
                if binarize:
                    nov1 = binarize_novelty(nov1, threshold)
                    nov2 = binarize_novelty(nov2, threshold)

                novelties[feature_type] = (nov1, nov2)

                dtw_distance_sec = dtw_distance(nov1, nov2, hop, sr)
                mae_score = compute_mean_error(nov1, nov2)
                xcorr_score, xcorr_lag = novelty_cross_correlation(nov1, nov2)
                xcorr_lag_ms = 1000 * (xcorr_lag * effective_hop / sr)
                peak_err = peak_alignment_error(nov1, nov2, sr=sr, hop_length=effective_hop)
                beat_mae = beat_alignment_error(y_audio, y_midi, sr=sr, hop_length=effective_hop)

                rows.append({
                    "Piece": piece_name,
                    "Version": "STFT" if feature_type == "stft" else "CQT",
                    "DTW Distance": round(dtw_distance_sec, 3),
                    "MAE": round(mae_score, 3),
                    "XCorr Score": round(xcorr_score, 3),
                    "XCorr Lag (ms)": round(xcorr_lag_ms, 3),
                    "Peak Alignment Error": round(peak_err, 3) if not np.isnan(peak_err) else "NaN",
                    "Beat Alignment Error" :round(beat_mae, 3) if not np.isnan(beat_mae) else "NaN"
                })
            except Exception as e:
                print(f"Error evaluating {piece_name} {feature_type}: {e}")
                rows.append({
                    "Piece": piece_name,
                    "Version": "STFT" if feature_type == "stft" else "CQT",
                    "DTW Distance": "error",
                    "MAE": "error",
                    "XCorr Score": "error",
                    "XCorr Lag (ms)": "error",
                    "Peak Alignment Error": "error",
                    "Beat Alignment Error" : "error" 
                })

        # Save plot if both novelty pairs exist
        if "stft" in novelties and "cqt_1" in novelties:
            fig_path = plot_dir / f"{piece_name.replace(' ', '_')}_dual_novelty_plot.pdf"
            plot_novelties(
                nov_stft=novelties["stft"][0],
                nov_stft_midi=novelties["stft"][1],
                nov_cqt=novelties["cqt_1"][0],
                nov_cqt_midi=novelties["cqt_1"][1],
                hop=effective_hop,
                sr=sr,
                out_path=str(fig_path),
                show_plot=visualize
            )

    df = pd.DataFrame(rows)
    df.to_csv(result_file, index=False)
    
    print(f"Evaluation completed. Results saved to {result_file}")