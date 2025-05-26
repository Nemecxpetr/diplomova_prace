"""
Module: DP
Author: Bc. Petr Nemec

This script is realizing the testing and accessing of the pipeline functions of my master's thesis [1]
(https://github.com/Nemecxpetr/diplomova_prace).

References:
[1] NĚMEC, Petr. Score-to-audio synchronization of music interpretations [online].
    Brno, 2024 [cit. 2024-03-01]. Available from: https://www.vut.cz/studenti/zav-prace/detail/159304.
    Master's Thesis. Vysok� u�en� technick� v Brn�, Fakulta elektrotechniky a komunika�n�ch technologi�,
    Department of Telecommunications. Supervisor Mat�j I�tv�nek.

[2] MUELLER, Meinard and ZALKOW, Frank: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing.
    Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
"""

# Imports
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import soundfile as sf
from scipy.io.wavfile import write
from IPython.display import Audio
import librosa

# Custom handlers
import Handler as handle
from Handler.MIDI_handler import midi_test
from SYNC.DTW import create_synced_object_from_MIDIfile, dtw_test
from SYNC.evaluator import evaluate_all_versions_in_preset_folder
from libfmp.b.b_sonification import sonify_chromagram_with_signal
from synctoolbox.feature.csv_tools import df_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma

def pipeline(filenames,
             folder,
             debug: bool = False,
             feature_type: str = 'stft',
             verbose: bool = False,
             figure_format: str = 'pdf'):
    """
    Synchronization pipeline for aligning MIDI to audio using chroma features and DTW.

    This function processes a list of audio interpretations, synchronizes them with a MIDI file,
    and evaluates the synchronization results. It performs the following main tasks:
        1. Load audio and MIDI input paths
        2. Convert input MIDI to CSV, then align with audio using DTW
        3. Compare chroma representations and visualize
        4. Save figures for evaluation
        5. Sonify aligned MIDI and merge stereo evaluation output


    Args:
        filenames (list of str): Names of audio interpretations to synchronize
        folder (str): Subdirectory where the data for this preset is stored
        debug (bool): If True, enables verbose MIDI parsing info
        feature_type (str): Type of chroma feature to use ('stft', 'cqt', 'cqt_1', 'cens')
        verbose (bool): If True, displays plots during synchronization
        figure_format (str): File format to save output plots (e.g. 'pdf', 'png')
    """
    for i, filename in enumerate(filenames):
        # STEP 1: Define paths
        input_midi_path = f'../../data/input/MIDI/{folder}/{filenames[0]}.mid'
        input_audio_path = f'../../data/input/audio/{folder}/{filename}.wav'
        output_midi_path = f'../../data/output/midi/{folder}/{feature_type}/s_{filename}.mid'
        csv_path = f'../../data/csv/{filename}.csv'
        output_audio_path = f'../../data/output/audio/{folder}/{feature_type}/{filename}.wav'
        sonified_midi_path = f'../../data/output/audio/{folder}/{feature_type}/s_{filename}_sonified.wav'

        for path in [input_midi_path, input_audio_path, output_midi_path, csv_path, output_audio_path, sonified_midi_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        # STEP 2: Synchronize symbolic score to audio interpretation
        sr = 48000
        x_audio, sr = librosa.load(input_audio_path, sr=sr)
        handle.midi_to_csv(midi=input_midi_path, csv_path=csv_path, debug=debug)
        synced_midi, audio_chroma, audio_hop = create_synced_object_from_MIDIfile(
            path_midi=input_midi_path,
            path_audio=input_audio_path,
            path_output=output_midi_path,
            path_csv=csv_path,
            feature_type=feature_type,
            verbose=verbose
        )

        # STEP 3: Visual comparison of chroma features
        feature_rate = sr / audio_hop
        df_in = handle.midi_to_csv(midi=input_midi_path)
        df_sync = handle.midi_to_csv(midi=output_midi_path)
        f_chroma_in = pitch_to_chroma(df_to_pitch_features(df_in, feature_rate=feature_rate))
        f_chroma_sync = pitch_to_chroma(df_to_pitch_features(df_sync, feature_rate=feature_rate))
        chromas = [f_chroma_in, f_chroma_sync, audio_chroma]
        chroma_names = ['Input MIDI', 'Synchronized MIDI', 'Audio']
        fig_chroma, _ = handle.compare_chroma(chromas, chroma_names, audio_hop=audio_hop, verbose=verbose)
        fig_piano, _ = handle.compare_midi(input_midi_path, output_midi_path, audio_chroma, audio_hop=audio_hop, verbose=verbose)

        # STEP 4: Save comparison figures
        figure_dir = Path(f'../../data/output/figures/{folder}/{feature_type}')
        figure_dir.mkdir(parents=True, exist_ok=True)
        if fig_chroma:
            fig_chroma.savefig(figure_dir / f'{filename}_chroma_comparison.{figure_format}')
            plt.close(fig_chroma)
        if fig_piano:
            fig_piano.savefig(figure_dir / f'{filename}_comparison.{figure_format}')
            plt.close(fig_piano)

        # STEP 5: Sonify synchronized MIDI and create stereo eval
        handle.audio_handler.sonify_midi(
            midi_file=output_midi_path,
            output_audio_file=sonified_midi_path,
            soundfont='../../data/soundfonts/FluidR3_GM.sf2',
            sample_rate=sr
        )
        x_synth, _ = librosa.load(sonified_midi_path, sr=sr)
        max_len = max(len(x_audio), len(x_synth))
        stereo = np.stack([
            np.pad(x_audio, (0, max_len - len(x_audio)), mode='constant'),
            np.pad(x_synth, (0, max_len - len(x_synth)), mode='constant')
        ], axis=1)
        handle.audio_handler.write_audio(output_audio_path, stereo, sr)

        print(f"SYNC {folder} | {feature_type} | {filename}: {round(((i+1)/len(filenames))*100, 2)}% complete")

def dataset_preset(preset: str, conv: bool = False):
    """
    Load dataset configuration for a given preset.

    This function defines the folder structure and filenames for different dataset presets.
    It can also convert audio files to WAV format if required.

    Args:
        preset (str): Name of the dataset preset (e.g., 'gymnopedie', 'summertime')
        conv (bool): If True, convert input files to WAV using audio_handler

    Returns:
        tuple[str, list[str]]: Folder path and list of filenames to process
    """
    presets = {
        'gymnopedie': {
            'folder': 'gymnopedie',
            'filenames': [
                'gymnopedie_no1', 'gymnopedie_no1_1', 'gymnopedie_no1_2',
                'gymnopedie_no1_khatia', 'gymnopedie_no1_3', 'gymnopedie_no1_4'
            ],
            'format': None
        },
        'unravel': {
            'folder': 'unravel',
            'filenames': ['unravel'],
            'format': None
        },
        'albeniz': {
            'folder': 'albeniz',
            'filenames': ['alb_se5', 'alb_se5_1', 'alb_se5_2', 'alb_se5_3'],
            'format': 'mp3'
        },
        'summertime': {
            'folder': 'summertime',
            'filenames': ['summertime', 'summertime_1', 'summertime_2'],
            'format': 'mp3'
        },
        'messiaen': {
            'folder': 'messiaen',
            'filenames': [
                'messiaen_le_banquet_celeste',
                'messiaen_le_banquet_celeste_1',
                'messiaen_le_banquet_celeste_2'
            ],
            'format': 'm4a'
        },
        'test_2': {
            'folder': 'tests/test_2',
            'filenames': ['test_2', 'test_2_1', 'test_2_2'],
            'format': None
        }
    }

    if preset not in presets:
        print(f"Unknown preset '{preset}'")
        return None, None

    entry = presets[preset]
    folder = entry['folder']
    filenames = entry['filenames']
    audio_format = entry['format']

    if conv and audio_format:
        for filename in filenames:
            src = f"../../data/input/audio/{folder}/{filename}.{audio_format}"
            dst = f"../../data/input/audio/{folder}/{filename}.wav"
            handle.audio_handler.convert_to_wav(src, dst, format=audio_format)

    return folder, filenames

def ultimate_test(feature_type: str = 'stft', figure_format : str = 'pdf'):
    """
    Run the full synchronization pipeline on all presets using a specified chroma feature type.

    Args:
        feature_type (str): Chroma feature type ('stft', 'cqt', 'cqt_1', 'cens')
    """
    all_presets = ['gymnopedie', 'unravel', 'albeniz', 'summertime', 'messiaen']

    for preset in all_presets:
        folder, filenames = dataset_preset(preset, conv=False)
        if folder and filenames:
            pipeline(filenames, folder, debug=False, feature_type=feature_type, verbose=False, figure_format = figure_format)
            print(f"Completed pipeline for '{preset}' using '{feature_type}' features.")

def plot_evaluation_results(presets, csv_dir="../../data/eval/results/csv", normalize=False, visualize=False):
    """
    Load evaluation CSVs and plot evaluation metrics for a list of presets.

    Args:
        presets (list[str]): Names of dataset presets to plot.
        csv_dir (str): Directory where evaluation CSVs are stored.
        normalize (bool): Normalize metric values across presets.
        visualize (bool): Whether to show the plots interactively.
    """
    csv_dir = Path(csv_dir)
    all_dfs = []

    for preset in presets:
        path = csv_dir / f"{preset}_full_evaluation.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["Preset"] = preset
            df = df.rename(columns={
                "DTW Distance": "DTW_Distance",
                "MAE": "Mean_Absolute_Error",
                "XCorr Score": "XCorr_Score",
                "XCorr Lag (ms)": "XCorr_Lag_(ms)",
                "Peak Alignment Error": "Peak_Alignment_Error",
                "Beat Alignment Error" : "Beat_Alignment_Error"
            })
            all_dfs.append(df)

    if not all_dfs:
        print("No evaluation data found.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # === FILTER OUTLIERS BASED ON METRIC THRESHOLDS ===
    metric_thresholds = {
        "DTW_Distance": 10.0,
        "Mean_Absolute_Error": 1.0,
        "XCorr_Score": 1.0,
        "XCorr_Lag_(ms)": 500.0,
        "Peak_Alignment_Error": 2.0,
        "Beat_Alignment_Error": 2.0
    }

    for metric, max_val in metric_thresholds.items():
        if metric in df_all.columns:
            df_all = df_all[(df_all[metric] <= max_val) | (df_all[metric].isna())]  

    df_melted = df_all.melt(
        id_vars=["Preset", "Version"],
        value_vars=[
            "DTW_Distance",
            "Mean_Absolute_Error",
            "XCorr_Score",
            "XCorr_Lag_(ms)",
            "Peak_Alignment_Error",
            "Beat_Alignment_Error"
        ],
        var_name="Metric",
        value_name="Score"
    )

    if normalize:
        df_melted["Score"] = df_melted.groupby("Metric")["Score"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
        )

    figure_dir = Path("../../data/eval/results/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
            "DTW_Distance",
            "Mean_Absolute_Error",
            "XCorr_Score",
            "XCorr_Lag_(ms)",
            "Peak_Alignment_Error",
            "Beat_Alignment_Error"
        ]

    for metric in metrics:
        plt.figure(figsize=(8, 4))
        subset = df_melted[df_melted["Metric"] == metric]

        sns.boxplot(
            data=subset,
            x="Preset",
            y="Score",
            hue="Version",
            palette="pastel",
            dodge=True,
            fliersize=0
        )
        sns.stripplot(
            data=subset,
            x="Preset",
            y="Score",
            hue="Version",
            dodge=True,
            marker="o",
            alpha=0.6,
            linewidth=0.5,
            edgecolor="gray"
        )

        title_suffix = " (Normalized)" if normalize else ""
        plt.title(f"{metric.replace('_', ' ')}{title_suffix}")
        plt.ylabel("Normalized Score" if normalize else "Score")

        # set correct ylabel based on metric
        if metric == "DTW_Distance":
            ylabel = "Mean deviation from diagonal (seconds)"
        elif metric == "Mean_Absolute_Error":
            ylabel = "Mean Absolute Error (unitless)"
        elif metric == "XCorr_Score":
            ylabel = "Cross-Correlation (normalized)"
        elif metric == "XCorr_Lag_(ms)":
            ylabel = "Lag (milliseconds)"
        elif metric == "Peak_Alignment_Error":
            ylabel = "Peak Alignment Error (seconds)"
        elif metric == "Beat_Alignment_Error":
            ylabel = "Beat Alignment Error (seconds)"
        elif normalize:
            ylabel = "Normalized Score"
        else:
            ylabel = "Score"
        plt.ylabel(ylabel)

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        figure_path = figure_dir / f"plot_points_line_{metric.lower()}.pdf"
        plt.savefig(figure_path, format="pdf")
        if visualize:
            plt.show()
        plt.close()

    print("Saved evaluation plots to:", figure_dir)

if __name__ == "__main__":
    # Choose dataset preset and settings
    folder, filenames = dataset_preset('summertime', conv=False)
    debug = False
    verbose = False

    # Run pipeline on all versions in selected preset
    # pipeline(filenames, folder, debug, feature_type='stft', verbose=verbose)
    # pipeline(filenames, folder, debug, feature_type='cqt_1', verbose=verbose)

    # Run ultimate test on all presets
    # 
    # ultimate_test(feature_type='stft')
    # ultimate_test(feature_type='cqt_1')
    #
    # for preset in ['gymnopedie', 'unravel', 'albeniz', 'summertime', 'messiaen', 'test_2']:
    #     evaluate_all_versions_in_preset_folder(preset_name=preset, 
    #                                            soundfont='../../data/soundfonts/FluidR3_GM.sf2',
    #                                            visualize=False,
    #                                            feature_types=['stft', 'cqt_1'],
    #                                            binarize=False,                                             
    #                                            downsample_factor=4,
    #                                            threshold=0.15,
    #                                            skip_sonification=True,
    #                                            debug=True)

    # Plot evaluation results
    plot_evaluation_results(
        presets=['unravel', 'albeniz', 'summertime', 'messiaen', 'gymnopedie'],
        csv_dir="../../data/eval/results/csv",
        normalize=False,
        visualize=False
    )

