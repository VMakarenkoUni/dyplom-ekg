#!/usr/bin/env python3
"""
Enhanced ECG Parser and Analyzer
Version 2.0 - Advanced Analysis Features
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch, hilbert, medfilt
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

from datetime import datetime
import json
from collections import Counter, defaultdict
from parser import convert_ecg_to_xml

# Optional imports for enhanced features
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("ІНФОРМАЦІЯ: matplotlib не встановлено. Візуалізація недоступна.")
    print("            Встановіть: pip install matplotlib")

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.chart import LineChart, Reference
    from openpyxl.chart.axis import DateAxis

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("ПОПЕРЕДЖЕННЯ: openpyxl не встановлено. Excel експорт недоступний.")
    print("              Встановіть: pip install openpyxl")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


DEFAULT_OUTPUT_DIR = os.environ.get(
    "EKG_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# ====================================================================================
# CONFIGURATION AND CONSTANTS
# ====================================================================================

class ECGConfig:
    """Enhanced configuration with more parameters"""

    # Filtering parameters
    NYQUIST_SAFETY_FACTOR = 0.95
    DRIFT_CUTOFF_HZ = 0.5
    QRS_LOWCUT_HZ = 5.0
    QRS_HIGHCUT_HZ = 25.0
    MUSCLE_NOISE_LOWCUT_HZ = 30.0
    MUSCLE_NOISE_HIGHCUT_HZ = 100.0
    FILTER_ORDER = 3

    # R-peak detection
    MIN_RR_INTERVAL_SEC = 0.2  # 300 bpm max
    MAX_RR_INTERVAL_SEC = 3.0  # 20 bpm min
    R_PEAK_HEIGHT_FACTOR = 0.5
    R_PEAK_PROMINENCE_FACTOR = 0.3
    REFRACTORY_PERIOD_MS = 200

    # Clinical thresholds - Heart Rate
    EXTREME_TACHYCARDIA_BPM = 150
    TACHYCARDIA_BPM = 100
    NORMAL_HR_HIGH_BPM = 90
    NORMAL_HR_LOW_BPM = 60
    BRADYCARDIA_BPM = 50
    SEVERE_BRADYCARDIA_BPM = 40
    EXTREME_BRADYCARDIA_BPM = 30

    # Clinical thresholds - Voltage
    VERY_LOW_VOLTAGE_MV = 0.3
    LOW_VOLTAGE_MV = 0.5
    NORMAL_VOLTAGE_MIN_MV = 0.5
    NORMAL_VOLTAGE_MAX_MV = 2.0
    HIGH_VOLTAGE_MV = 2.5
    VERY_HIGH_VOLTAGE_MV = 3.0

    # Clinical thresholds - Intervals (ms)
    PR_INTERVAL_MIN = 120
    PR_INTERVAL_MAX = 200
    PR_INTERVAL_PROLONGED = 220
    QRS_DURATION_NORMAL = 120
    QRS_DURATION_WIDE = 140
    QT_INTERVAL_MALE_MAX = 440
    QT_INTERVAL_FEMALE_MAX = 460

    # Heart Rate Variability thresholds
    VERY_LOW_SDNN_MS = 20
    LOW_SDNN_MS = 30
    NORMAL_SDNN_MIN_MS = 40
    NORMAL_SDNN_MAX_MS = 100
    HIGH_SDNN_MS = 150

    LOW_RMSSD_MS = 20
    NORMAL_RMSSD_MS = 40
    HIGH_RMSSD_MS = 60

    # Rhythm regularity
    REGULAR_RHYTHM_CV = 5
    SLIGHT_IRREGULAR_CV = 8
    MODERATE_IRREGULAR_CV = 12
    HIGHLY_IRREGULAR_CV = 15
    AF_THRESHOLD_CV = 20

    # Arrhythmia detection
    PVC_PREMATURITY_PERCENT = 25  # PVC if occurs 25% earlier than expected
    PAC_PREMATURITY_PERCENT = 20  # PAC if occurs 20% earlier
    PAUSE_THRESHOLD_SEC = 2.0
    SIGNIFICANT_PAUSE_SEC = 2.5
    CRITICAL_PAUSE_SEC = 3.0

    # Signal quality
    MIN_SIGNAL_LENGTH_SEC = 10.0
    OPTIMAL_SIGNAL_LENGTH_SEC = 30.0
    MIN_R_PEAKS_FOR_ANALYSIS = 5
    MIN_R_PEAKS_FOR_HRV = 20
    MIN_R_PEAKS_FOR_ADVANCED = 50
    SIGNAL_FLAT_THRESHOLD = 1e-6
    MIN_UNIQUE_VALUES = 10
    MAX_CLIPPING_PERCENT = 5
    MIN_SNR_DB = 10

    # Frequency domain
    VLF_BAND = (0.003, 0.04)  # Very Low Frequency
    LF_BAND = (0.04, 0.15)  # Low Frequency
    HF_BAND = (0.15, 0.4)  # High Frequency

    # Morphology analysis
    P_WAVE_DURATION_MAX = 120  # ms
    T_WAVE_NORMAL_RATIO = 0.2  # T/R amplitude ratio


# ====================================================================================
# SIGNAL PROCESSING UTILITIES
# ====================================================================================

class SignalProcessor:
    """Advanced signal processing utilities"""

    @staticmethod
    def remove_baseline_wander(signal, fs, cutoff=0.5):
        """Remove baseline wander using high-pass filter"""
        nyquist = 0.5 * fs
        if cutoff >= nyquist:
            return signal

        try:
            b, a = butter(ECGConfig.FILTER_ORDER, cutoff / nyquist, btype='high')
            return filtfilt(b, a, signal)
        except:
            return signal

    @staticmethod
    def remove_powerline_interference(signal, fs, freq=50, q=30):
        """Remove powerline interference using notch filter"""
        nyquist = 0.5 * fs
        if freq >= nyquist:
            return signal

        try:
            from scipy.signal import iirnotch
            b, a = iirnotch(freq, q, fs)
            return filtfilt(b, a, signal)
        except:
            return signal

    @staticmethod
    def adaptive_filter(signal, fs):
        """Adaptive filtering based on signal characteristics"""
        # Estimate noise level
        noise_estimate = np.std(np.diff(signal))
        signal_power = np.std(signal)

        if signal_power > 0:
            snr = 20 * np.log10(signal_power / noise_estimate)
        else:
            snr = 0

        # Apply stronger filtering for noisy signals
        if snr < 10:
            # Heavy filtering
            signal = SignalProcessor.remove_baseline_wander(signal, fs, 1.0)
            signal = SignalProcessor.bandpass_filter(signal, fs, 1, 40)
        elif snr < 20:
            # Moderate filtering
            signal = SignalProcessor.remove_baseline_wander(signal, fs, 0.5)
            signal = SignalProcessor.bandpass_filter(signal, fs, 0.5, 50)
        else:
            # Light filtering
            signal = SignalProcessor.remove_baseline_wander(signal, fs, 0.3)

        return signal, snr

    @staticmethod
    def bandpass_filter(signal, fs, lowcut, highcut, order=ECGConfig.FILTER_ORDER):
        """Bandpass filter with safety checks"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist * ECGConfig.NYQUIST_SAFETY_FACTOR

        if low >= high or low <= 0 or high >= 1:
            return signal

        try:
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, signal)
        except:
            return signal

    @staticmethod
    def compute_derivative(signal, fs):
        """Compute signal derivative for feature detection"""
        # Simple differentiation
        derivative = np.diff(signal) * fs
        # Pad to maintain length
        derivative = np.append(derivative, derivative[-1])
        return derivative

    @staticmethod
    def compute_signal_energy(signal, window_size):
        """Compute moving window energy"""
        squared = signal ** 2
        window = np.ones(window_size) / window_size
        energy = np.convolve(squared, window, mode='same')
        return energy


# ====================================================================================
# ADVANCED R-PEAK DETECTION
# ====================================================================================

class AdvancedRPeakDetector:
    """Multi-algorithm R-peak detection with voting"""

    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs
        self.length = len(signal)

    def detect_peaks_pan_tompkins(self):
        """Pan-Tompkins algorithm implementation for mV and uV data"""
        # 1. Bandpass filter 5-15 Hz
        filtered = SignalProcessor.bandpass_filter(self.signal, self.fs, 5, 15)

        # 2. Derivative
        derivative = SignalProcessor.compute_derivative(filtered, self.fs)

        # 3. Square
        squared = derivative ** 2

        # 4. Moving window integration
        window_size = int(0.150 * self.fs)  # 150ms window
        integrated = SignalProcessor.compute_signal_energy(squared, window_size)

        # 5. Find peaks
        min_distance = int(ECGConfig.MIN_RR_INTERVAL_SEC * self.fs)

        # ВИПРАВЛЕННЯ: Адаптивний поріг для різних масштабів
        signal_max = np.max(np.abs(self.signal))
        is_millivolts = signal_max < 10

        # Різні коефіцієнти для порогу
        if is_millivolts:
            threshold = 0.4 * np.max(integrated)  # Вищий поріг для мВ
        else:
            threshold = 0.35 * np.max(integrated)  # Оригінальний для мкВ

        peaks, _ = find_peaks(integrated,
                              height=threshold,
                              distance=min_distance)

        # Adjust peak locations to actual R-peaks
        adjusted_peaks = []
        window = int(0.05 * self.fs)  # 50ms window

        for peak in peaks:
            start = max(0, peak - window)
            end = min(self.length, peak + window)
            if end > start:
                # Знаходимо максимум в оригінальному сигналі
                local_max = np.argmax(np.abs(self.signal[start:end]))
                adjusted_peaks.append(start + local_max)

        return np.array(adjusted_peaks)

    def detect_peaks_wavelet(self):
        """Wavelet-based R-peak detection"""
        try:
            import pywt
        except ImportError:
            return np.array([])

        # Decompose signal
        coeffs = pywt.wavedec(self.signal, 'db4', level=4)

        # Reconstruct using detail coefficients
        coeffs[0] = np.zeros_like(coeffs[0])  # Remove approximation
        coeffs[-1] = np.zeros_like(coeffs[-1])  # Remove highest frequency
        reconstructed = pywt.waverec(coeffs, 'db4')

        # Ensure same length
        if len(reconstructed) > self.length:
            reconstructed = reconstructed[:self.length]
        elif len(reconstructed) < self.length:
            reconstructed = np.pad(reconstructed, (0, self.length - len(reconstructed)))

        # Find peaks
        min_distance = int(ECGConfig.MIN_RR_INTERVAL_SEC * self.fs)
        peaks, _ = find_peaks(reconstructed,
                              height=np.std(reconstructed),
                              distance=min_distance)

        return peaks

    def detect_peaks_adaptive_threshold(self):
        """Adaptive threshold method for mV and uV data"""
        # Preprocess
        filtered = SignalProcessor.bandpass_filter(self.signal, self.fs,
                                                   ECGConfig.QRS_LOWCUT_HZ,
                                                   ECGConfig.QRS_HIGHCUT_HZ)

        # ВИПРАВЛЕННЯ: Визначаємо масштаб даних
        signal_max = np.max(np.abs(filtered))
        is_millivolts = signal_max < 10  # Дані в мВ якщо макс < 10

        # Initialize thresholds
        signal_peak = np.max(filtered[:int(2 * self.fs)])  # First 2 seconds
        noise_peak = np.median(np.abs(filtered[:int(2 * self.fs)]))

        # Адаптуємо коефіцієнти для різних масштабів
        if is_millivolts:
            # Для мВ даних
            threshold1 = noise_peak + 0.4 * (signal_peak - noise_peak)
            threshold2 = 0.6 * threshold1

            # Мінімальний поріг для мВ
            min_threshold = 0.05  # 50 мкВ = 0.05 мВ
            threshold1 = max(threshold1, min_threshold)
            threshold2 = max(threshold2, min_threshold * 0.6)
        else:
            # Оригінальні коефіцієнти для мкВ
            threshold1 = noise_peak + 0.25 * (signal_peak - noise_peak)
            threshold2 = 0.5 * threshold1

        # Detect peaks with adaptive thresholds
        peaks = []
        refractory_samples = int(ECGConfig.REFRACTORY_PERIOD_MS * self.fs / 1000)

        i = 0
        while i < self.length:
            if filtered[i] > threshold1:
                # Find local maximum
                window_end = min(i + refractory_samples, self.length)
                local_max_idx = i + np.argmax(filtered[i:window_end])
                peaks.append(local_max_idx)

                # Update thresholds
                signal_peak = 0.875 * signal_peak + 0.125 * filtered[local_max_idx]

                if is_millivolts:
                    threshold1 = noise_peak + 0.4 * (signal_peak - noise_peak)
                    threshold1 = max(threshold1, min_threshold)
                else:
                    threshold1 = noise_peak + 0.25 * (signal_peak - noise_peak)

                threshold2 = 0.5 * threshold1

                i = local_max_idx + refractory_samples
            else:
                if filtered[i] > threshold2:
                    # Possible peak with lower threshold
                    noise_peak = 0.875 * noise_peak + 0.125 * filtered[i]
                i += 1

        return np.array(peaks)

    def detect_peaks_ensemble(self):
        """Ensemble method - для мВ даних використовуємо тільки основний алгоритм"""
        # Перевіряємо масштаб даних
        signal_max = np.max(np.abs(self.signal))
        is_millivolts = signal_max < 10

        if is_millivolts:
            # Для мВ даних використовуємо тільки основний метод
            # який працює з вашими даними
            return detect_r_peaks_refined(self.signal, self.fs)
        else:
            # Для мкВ даних - оригінальний ensemble
            peaks1 = self.detect_peaks_pan_tompkins()
            peaks2 = self.detect_peaks_adaptive_threshold()
            peaks3 = detect_r_peaks_refined(self.signal, self.fs)
            peaks4 = self.detect_peaks_wavelet()

            # Об'єднуємо всі піки
            all_peaks = []
            for p in [peaks1, peaks2, peaks3, peaks4]:
                if len(p) > 0:
                    all_peaks.extend(p)

            if len(all_peaks) == 0:
                return np.array([])

            all_peaks = np.array(all_peaks)
            all_peaks.sort()

            # Кластеризація близьких піків
            cluster_window = int(0.05 * self.fs)  # 50 мс
            final_peaks = []
            i = 0

            while i < len(all_peaks):
                cluster = [all_peaks[i]]
                j = i + 1

                while j < len(all_peaks) and all_peaks[j] - all_peaks[i] < cluster_window:
                    cluster.append(all_peaks[j])
                    j += 1

                # Потрібно хоча б 2 алгоритми
                if len(cluster) >= 2:
                    final_peaks.append(int(np.median(cluster)))

                i = j

            return np.array(final_peaks)


# ====================================================================================
# MORPHOLOGY ANALYSIS
# ====================================================================================

class MorphologyAnalyzer:
    """Analyze ECG waveform morphology"""

    def __init__(self, signal, r_peaks, fs):
        self.signal = signal
        self.r_peaks = r_peaks
        self.fs = fs
        self.rr_intervals = np.diff(r_peaks) / fs if len(r_peaks) > 1 else np.array([])

    def segment_beats(self, before_r=0.2, after_r=0.4):
        """Segment individual beats around R-peaks"""
        before_samples = int(before_r * self.fs)
        after_samples = int(after_r * self.fs)

        beats = []
        valid_indices = []

        for i, r_peak in enumerate(self.r_peaks):
            start = r_peak - before_samples
            end = r_peak + after_samples

            if start >= 0 and end < len(self.signal):
                beat = self.signal[start:end]
                beats.append(beat)
                valid_indices.append(i)

        return np.array(beats), np.array(valid_indices)

    def compute_average_beat(self, beats):
        """Compute average beat morphology"""
        if len(beats) == 0:
            return None

        # Align beats by correlation
        reference = beats[0]
        aligned_beats = [reference]

        for beat in beats[1:]:
            # Find best alignment
            correlation = np.correlate(reference, beat, mode='same')
            shift = np.argmax(correlation) - len(beat) // 2

            # Limit shift to reasonable range
            max_shift = int(0.05 * self.fs)  # 50ms
            shift = np.clip(shift, -max_shift, max_shift)

            # Align
            if shift > 0:
                aligned = np.pad(beat[shift:], (0, shift), mode='edge')
            elif shift < 0:
                aligned = np.pad(beat[:shift], (-shift, 0), mode='edge')
            else:
                aligned = beat

            aligned_beats.append(aligned[:len(reference)])

        # Compute median beat (robust to outliers)
        average_beat = np.median(aligned_beats, axis=0)
        return average_beat

    def detect_p_waves(self, beat, r_peak_relative):
        """Detect P-wave in a beat"""
        # Look for P-wave 200-50ms before R-peak
        p_window_start = int(max(0, r_peak_relative - 0.2 * self.fs))
        p_window_end = int(max(0, r_peak_relative - 0.05 * self.fs))

        if p_window_end <= p_window_start:
            return None

        p_segment = beat[p_window_start:p_window_end]
        if len(p_segment) == 0:
            return None

        # Find maximum in P-wave window
        p_peak = np.argmax(np.abs(p_segment))
        p_amplitude = p_segment[p_peak]

        # Validate P-wave
        if abs(p_amplitude) < 0.05 * abs(beat[r_peak_relative]):  # Too small
            return None

        return {
            'position': p_window_start + p_peak,
            'amplitude': p_amplitude,
            'duration': self.estimate_wave_duration(p_segment, p_peak)
        }

    def detect_t_waves(self, beat, r_peak_relative):
        """Detect T-wave in a beat"""
        # Look for T-wave 100-400ms after R-peak
        t_window_start = int(r_peak_relative + 0.1 * self.fs)
        t_window_end = int(min(len(beat), r_peak_relative + 0.4 * self.fs))

        if t_window_end <= t_window_start:
            return None

        t_segment = beat[t_window_start:t_window_end]
        if len(t_segment) == 0:
            return None

        # Find maximum in T-wave window
        t_peak = np.argmax(np.abs(t_segment))
        t_amplitude = t_segment[t_peak]

        # Validate T-wave
        if abs(t_amplitude) < 0.05 * abs(beat[r_peak_relative]):  # Too small
            return None

        return {
            'position': t_window_start + t_peak,
            'amplitude': t_amplitude,
            'duration': self.estimate_wave_duration(t_segment, t_peak),
            'polarity': 'positive' if t_amplitude > 0 else 'negative'
        }

    def estimate_wave_duration(self, wave_segment, peak_idx):
        """Estimate wave duration using half-maximum width"""
        if len(wave_segment) == 0 or peak_idx >= len(wave_segment):
            return 0

        peak_value = wave_segment[peak_idx]
        half_max = abs(peak_value) / 2

        # Find where signal crosses half-maximum
        start = peak_idx
        while start > 0 and abs(wave_segment[start]) > half_max:
            start -= 1

        end = peak_idx
        while end < len(wave_segment) - 1 and abs(wave_segment[end]) > half_max:
            end += 1

        duration_samples = end - start
        duration_ms = duration_samples * 1000 / self.fs

        return duration_ms

    def analyze_qrs_morphology(self, beats):
        """Analyze QRS complex characteristics"""
        qrs_features = []

        for beat in beats:
            r_peak_relative = len(beat) // 2  # R-peak should be at center

            # Find Q and S waves
            q_wave = self.find_q_wave(beat, r_peak_relative)
            s_wave = self.find_s_wave(beat, r_peak_relative)

            # QRS duration
            qrs_start = q_wave['position'] if q_wave else r_peak_relative - int(0.04 * self.fs)
            qrs_end = s_wave['position'] if s_wave else r_peak_relative + int(0.04 * self.fs)
            qrs_duration = (qrs_end - qrs_start) * 1000 / self.fs

            # QRS amplitude
            qrs_amplitude = beat[r_peak_relative]

            qrs_features.append({
                'duration': qrs_duration,
                'amplitude': qrs_amplitude,
                'q_wave': q_wave,
                's_wave': s_wave,
                'morphology': self.classify_qrs_morphology(beat, r_peak_relative)
            })

        return qrs_features

    def find_q_wave(self, beat, r_peak):
        """Find Q-wave (negative deflection before R)"""
        search_window = int(0.04 * self.fs)  # 40ms before R
        start = max(0, r_peak - search_window)

        segment = beat[start:r_peak]
        if len(segment) == 0:
            return None

        # Find minimum (most negative)
        q_idx = np.argmin(segment)
        q_value = segment[q_idx]

        if q_value < -0.05 * abs(beat[r_peak]):  # Significant Q-wave
            return {
                'position': start + q_idx,
                'amplitude': q_value,
                'depth': abs(q_value)
            }
        return None

    def find_s_wave(self, beat, r_peak):
        """Find S-wave (negative deflection after R)"""
        search_window = int(0.04 * self.fs)  # 40ms after R
        end = min(len(beat), r_peak + search_window)

        segment = beat[r_peak:end]
        if len(segment) == 0:
            return None

        # Find minimum (most negative)
        s_idx = np.argmin(segment)
        s_value = segment[s_idx]

        if s_value < -0.05 * abs(beat[r_peak]):  # Significant S-wave
            return {
                'position': r_peak + s_idx,
                'amplitude': s_value,
                'depth': abs(s_value)
            }
        return None

    def classify_qrs_morphology(self, beat, r_peak):
        """Classify QRS morphology pattern"""
        # Simple classification based on R-wave dominance
        pre_r = beat[max(0, r_peak - int(0.04 * self.fs)):r_peak]
        post_r = beat[r_peak:min(len(beat), r_peak + int(0.04 * self.fs))]

        if len(pre_r) == 0 or len(post_r) == 0:
            return "undefined"

        # Check for notching (possible bundle branch block)
        derivative = np.diff(beat[r_peak - int(0.02 * self.fs):r_peak + int(0.02 * self.fs)])
        zero_crossings = np.where(np.diff(np.sign(derivative)))[0]

        if len(zero_crossings) > 4:
            return "notched"
        elif np.mean(pre_r) < -0.2 * beat[r_peak]:
            return "Q-dominant"
        elif np.mean(post_r) < -0.2 * beat[r_peak]:
            return "S-dominant"
        else:
            return "normal"


def detect_r_peaks_refined(processed_signal, sampling_rate):
    """Покращена детекція R-піків для мВ даних"""
    if not isinstance(processed_signal, np.ndarray) or processed_signal.size == 0 or sampling_rate <= 0:
        return np.array([])

    from scipy.signal import butter, filtfilt, find_peaks, detrend

    # Визначаємо масштаб даних
    signal_max = np.max(np.abs(processed_signal))
    is_millivolts = signal_max < 10  # Дані в мВ якщо макс < 10

    print(f"DEBUG: Signal max = {signal_max:.3f}, is_millivolts = {is_millivolts}")
    print(f"DEBUG: Signal shape = {processed_signal.shape}, sampling_rate = {sampling_rate}")

    # КРОК 1: Видалення тренду
    signal = detrend(processed_signal, type='linear')

    # КРОК 2: Агресивніша фільтрація для виділення QRS
    nyquist = 0.5 * sampling_rate

    # Перший прохід - широкий діапазон
    if nyquist > 20:
        b1, a1 = butter(2, [8.0 / nyquist, min(20.0 / nyquist, 0.95)], btype='band')
        filtered1 = filtfilt(b1, a1, signal)
    else:
        filtered1 = signal

    # Другий прохід - вузький діапазон для QRS
    if nyquist > 15:
        b2, a2 = butter(2, [10.0 / nyquist, min(15.0 / nyquist, 0.95)], btype='band')
        filtered2 = filtfilt(b2, a2, signal)
    else:
        filtered2 = filtered1

    # КРОК 3: Підсилення QRS комплексів
    # Квадратування
    squared = filtered2 ** 2

    # Ковзне вікно
    window_size = int(0.08 * sampling_rate)  # 80 мс вікно
    if window_size < 1:
        window_size = 1

    window = np.ones(window_size) / window_size
    integrated = np.convolve(squared, window, mode='same')

    # КРОК 4: Знаходження всіх можливих піків
    all_peaks = []

    # Метод 1: Адаптивний поріг по сегментах (2 секунди)
    segment_size = int(2 * sampling_rate)
    for i in range(0, len(integrated), segment_size // 2):  # З перекриттям
        segment = integrated[i:i + segment_size]
        if len(segment) < int(0.5 * sampling_rate):
            continue

        # Локальна статистика
        seg_mean = np.mean(segment)
        seg_std = np.std(segment)
        seg_median = np.median(segment)

        # Пробуємо різні пороги
        for factor in [0.5, 0.8, 1.0, 1.2, 1.5]:
            threshold = seg_median + factor * seg_std

            # Мінімальна відстань між піками - 250 мс (240 bpm max)
            min_distance = int(0.25 * sampling_rate)

            peaks, properties = find_peaks(
                segment,
                height=threshold,
                distance=min_distance,
                prominence=threshold * 0.2  # Низька prominence для знаходження всіх піків
            )

            # Додаємо глобальні індекси
            global_peaks = peaks + i
            all_peaks.extend(global_peaks)

            # Якщо знайшли розумну кількість - виходимо
            if 10 <= len(peaks) <= 50:  # 2 секундний сегмент
                break

    # Метод 2: Прямий пошук в оригінальному сигналі
    if is_millivolts:
        # Для мВ даних - шукаємо піки > 0.3 мВ
        min_height = 0.3
    else:
        # Для мкВ даних
        min_height = 300

    # Нормалізуємо сигнал
    signal_normalized = processed_signal / np.max(np.abs(processed_signal))

    # Шукаємо локальні максимуми
    peaks2, _ = find_peaks(
        signal_normalized,
        height=0.3,  # 30% від максимуму
        distance=int(0.3 * sampling_rate),
        prominence=0.1
    )
    all_peaks.extend(peaks2)

    # Метод 3: Пошук по першій похідній
    diff_signal = np.diff(filtered1)
    diff_signal = np.append(diff_signal, diff_signal[-1])

    # Знаходимо точки переходу через нуль (максимуми)
    zero_crossings = np.where(np.diff(np.sign(diff_signal)))[0]

    for zc in zero_crossings:
        # Перевіряємо що це максимум (похідна змінюється з + на -)
        if zc > 0 and zc < len(diff_signal) - 1:
            if diff_signal[zc - 1] > 0 and diff_signal[zc + 1] < 0:
                # Перевіряємо висоту піка
                if processed_signal[zc] > min_height:
                    all_peaks.append(zc)

    # КРОК 5: Об'єднання та очищення
    if len(all_peaks) == 0:
        print("DEBUG: No peaks found!")
        return np.array([])

    # Видаляємо дублікати
    all_peaks = np.unique(all_peaks)
    all_peaks = np.sort(all_peaks)

    print(f"DEBUG: Found {len(all_peaks)} candidate peaks")

    # Кластеризація близьких піків
    final_peaks = []
    cluster_window = int(0.08 * sampling_rate)  # 80 мс

    i = 0
    while i < len(all_peaks):
        # Збираємо всі піки в межах вікна
        cluster = [all_peaks[i]]
        j = i + 1

        while j < len(all_peaks) and (all_peaks[j] - all_peaks[i]) < cluster_window:
            cluster.append(all_peaks[j])
            j += 1

        # Знаходимо найвищий пік в кластері
        cluster_values = [processed_signal[idx] for idx in cluster if idx < len(processed_signal)]
        if cluster_values:
            best_idx = cluster[np.argmax(cluster_values)]

            # Перевіряємо мінімальну відстань від попереднього піка
            min_rr = int(0.3 * sampling_rate)  # 300 мс = 200 bpm max

            if not final_peaks or (best_idx - final_peaks[-1]) >= min_rr:
                final_peaks.append(best_idx)

        i = j

    result = np.array(final_peaks)

    # КРОК 6: Валідація результатів
    if len(result) > 1:
        rr_intervals = np.diff(result) / sampling_rate
        mean_hr = 60.0 / np.mean(rr_intervals)

        print(f"DEBUG: Final peaks = {len(result)}")
        print(f"DEBUG: Mean HR = {mean_hr:.1f} bpm")
        print(f"DEBUG: RR range = {np.min(rr_intervals) * 1000:.0f}-{np.max(rr_intervals) * 1000:.0f} ms")

        # Якщо занадто мало піків - спробуємо знизити поріг
        if len(result) < 10 and len(processed_signal) > 5 * sampling_rate:
            print("DEBUG: Too few peaks, trying lower threshold...")

            # Простий метод з низьким порогом
            threshold = np.percentile(np.abs(processed_signal), 70)
            peaks, _ = find_peaks(
                processed_signal,
                height=threshold,
                distance=int(0.35 * sampling_rate)
            )

            if len(peaks) > len(result):
                result = peaks
                print(f"DEBUG: Found {len(peaks)} peaks with lower threshold")

    return result


def extract_r_peaks_from_xml(xml_root, sampling_rate):
    """Витягує R-піки з XML анотацій"""
    r_peaks = []

    # Шукаємо анотації
    annotations = xml_root.find('.//Annotations')
    if annotations is not None:
        for ann in annotations.findall('Annotation'):
            code = ann.findtext('Code', '')
            if code == 'R':  # R-peak annotation
                time_offset = ann.find('TimeOffset')
                if time_offset is not None:
                    time_sec = float(time_offset.text)
                    # Конвертуємо час в індекс семплу
                    sample_idx = int(time_sec * sampling_rate)
                    r_peaks.append(sample_idx)

    return np.array(r_peaks)
# ====================================================================================
# ARRHYTHMIA DETECTION
# ====================================================================================

class ArrhythmiaDetector:
    """Advanced arrhythmia detection algorithms"""

    def __init__(self, r_peaks, signal, fs):
        self.r_peaks = r_peaks
        self.signal = signal
        self.fs = fs
        self.rr_intervals = np.diff(r_peaks) / fs if len(r_peaks) > 1 else np.array([])
        self.hr_values = 60.0 / self.rr_intervals if len(self.rr_intervals) > 0 else np.array([])

    def detect_all_arrhythmias(self):
        """Run all arrhythmia detection algorithms"""
        results = {
            'summary': {},
            'beats': [],
            'episodes': [],
            'burden': {}
        }

        if len(self.r_peaks) < 3:
            results['summary']['status'] = 'Insufficient data'
            return results

        # Basic rhythm analysis
        results['summary']['mean_hr'] = np.mean(self.hr_values) if len(self.hr_values) > 0 else 0
        results['summary']['hr_std'] = np.std(self.hr_values) if len(self.hr_values) > 0 else 0
        results['summary']['total_beats'] = len(self.r_peaks)

        # Detect specific arrhythmias
        results['atrial_fibrillation'] = self.detect_atrial_fibrillation()
        results['premature_beats'] = self.detect_premature_beats()
        results['bradycardia'] = self.detect_bradycardia()
        results['tachycardia'] = self.detect_tachycardia()
        results['pauses'] = self.detect_pauses()
        results['bigeminy'] = self.detect_bigeminy_trigeminy()
        results['heart_blocks'] = self.detect_heart_blocks()

        # Calculate arrhythmia burden
        self._calculate_burden(results)

        return results

    def detect_atrial_fibrillation(self):
        """Detect atrial fibrillation using multiple criteria"""
        if len(self.rr_intervals) < 30:
            return {'detected': False, 'reason': 'Insufficient beats'}

        # 1. RR interval variability
        cv = np.std(self.rr_intervals) / np.mean(self.rr_intervals) * 100

        # 2. Sample entropy
        sample_entropy = self._calculate_sample_entropy(self.rr_intervals)

        # 3. Turning points ratio
        tpr = self._calculate_turning_points_ratio(self.rr_intervals)

        # 4. RMSSD
        rmssd = np.sqrt(np.mean(np.diff(self.rr_intervals) ** 2)) * 1000

        # 5. Absence of P-waves (simplified check)
        p_wave_absent = self._check_p_wave_absence()

        # AF detection criteria
        af_score = 0
        criteria = []

        if cv > ECGConfig.AF_THRESHOLD_CV:
            af_score += 2
            criteria.append(f"High CV: {cv:.1f}%")

        if sample_entropy > 1.5:
            af_score += 2
            criteria.append(f"High entropy: {sample_entropy:.2f}")

        if tpr > 0.6:
            af_score += 1
            criteria.append(f"High TPR: {tpr:.2f}")

        if rmssd > 100:
            af_score += 1
            criteria.append(f"High RMSSD: {rmssd:.0f}ms")

        if p_wave_absent:
            af_score += 2
            criteria.append("P-waves absent")

        # Decision
        af_detected = af_score >= 4
        confidence = min(100, af_score * 12.5)  # 0-100% confidence

        return {
            'detected': af_detected,
            'confidence': confidence,
            'criteria': criteria,
            'cv': cv,
            'sample_entropy': sample_entropy,
            'tpr': tpr,
            'rmssd': rmssd
        }

    def detect_premature_beats(self):
        """Detect PVCs and PACs"""
        premature_beats = []

        if len(self.rr_intervals) < 3:
            return {'pvcs': [], 'pacs': [], 'total': 0}

        # Calculate expected RR intervals using moving average
        window_size = min(10, len(self.rr_intervals) // 3)
        expected_rr = np.convolve(self.rr_intervals, np.ones(window_size) / window_size, mode='same')

        for i in range(1, len(self.rr_intervals) - 1):
            # Check if beat is premature
            prematurity = (expected_rr[i] - self.rr_intervals[i]) / expected_rr[i] * 100

            if prematurity > ECGConfig.PAC_PREMATURITY_PERCENT:
                # Check compensatory pause
                compensatory_pause = self.rr_intervals[i + 1] > expected_rr[i + 1] * 1.2

                # Analyze QRS morphology
                beat_morphology = self._analyze_beat_morphology(i + 1)

                if beat_morphology == 'wide' or compensatory_pause:
                    beat_type = 'PVC'
                    confidence = 80 if compensatory_pause else 60
                else:
                    beat_type = 'PAC'
                    confidence = 70

                premature_beats.append({
                    'index': i + 1,
                    'type': beat_type,
                    'prematurity': prematurity,
                    'compensatory_pause': compensatory_pause,
                    'confidence': confidence,
                    'coupling_interval': self.rr_intervals[i] * 1000  # ms
                })

        # Classify PVCs
        pvcs = [b for b in premature_beats if b['type'] == 'PVC']
        pacs = [b for b in premature_beats if b['type'] == 'PAC']

        # Check for patterns
        pvc_patterns = self._classify_pvc_patterns(pvcs)

        return {
            'pvcs': pvcs,
            'pacs': pacs,
            'total': len(premature_beats),
            'pvc_burden': len(pvcs) / len(self.r_peaks) * 100 if len(self.r_peaks) > 0 else 0,
            'pac_burden': len(pacs) / len(self.r_peaks) * 100 if len(self.r_peaks) > 0 else 0,
            'patterns': pvc_patterns
        }

    def detect_bradycardia(self):
        """Detect bradycardia episodes"""
        episodes = []

        if len(self.hr_values) == 0:
            return {'episodes': [], 'severity': 'none'}

        # Find consecutive beats with low HR
        brady_mask = self.hr_values < ECGConfig.BRADYCARDIA_BPM

        # Group consecutive bradycardic beats
        brady_groups = self._find_consecutive_groups(brady_mask)

        for group in brady_groups:
            start_idx, end_idx = group
            duration = (self.r_peaks[end_idx] - self.r_peaks[start_idx]) / self.fs
            min_hr = np.min(self.hr_values[start_idx:end_idx])
            mean_hr = np.mean(self.hr_values[start_idx:end_idx])

            severity = 'mild'
            if min_hr < ECGConfig.EXTREME_BRADYCARDIA_BPM:
                severity = 'extreme'
            elif min_hr < ECGConfig.SEVERE_BRADYCARDIA_BPM:
                severity = 'severe'

            episodes.append({
                'start_beat': start_idx,
                'end_beat': end_idx,
                'duration_sec': duration,
                'min_hr': min_hr,
                'mean_hr': mean_hr,
                'severity': severity
            })

        # Overall severity
        if len(episodes) == 0:
            overall_severity = 'none'
        else:
            severities = [e['severity'] for e in episodes]
            if 'extreme' in severities:
                overall_severity = 'extreme'
            elif 'severe' in severities:
                overall_severity = 'severe'
            else:
                overall_severity = 'mild'

        return {
            'episodes': episodes,
            'severity': overall_severity,
            'total_duration': sum(e['duration_sec'] for e in episodes),
            'percentage': len(np.where(brady_mask)[0]) / len(self.hr_values) * 100
        }

    def detect_tachycardia(self):
        """Detect tachycardia episodes with classification"""
        episodes = []

        if len(self.hr_values) == 0:
            return {'episodes': [], 'types': {}}

        # Find consecutive beats with high HR
        tachy_mask = self.hr_values > ECGConfig.TACHYCARDIA_BPM
        tachy_groups = self._find_consecutive_groups(tachy_mask)

        for group in tachy_groups:
            start_idx, end_idx = group
            duration = (self.r_peaks[end_idx] - self.r_peaks[start_idx]) / self.fs
            max_hr = np.max(self.hr_values[start_idx:end_idx])
            mean_hr = np.mean(self.hr_values[start_idx:end_idx])

            # Classify tachycardia type
            tachy_type = self._classify_tachycardia(start_idx, end_idx, mean_hr)

            episodes.append({
                'start_beat': start_idx,
                'end_beat': end_idx,
                'duration_sec': duration,
                'max_hr': max_hr,
                'mean_hr': mean_hr,
                'type': tachy_type['name'],
                'confidence': tachy_type['confidence']
            })

        # Summarize types
        type_summary = {}
        for episode in episodes:
            t = episode['type']
            if t not in type_summary:
                type_summary[t] = {'count': 0, 'total_duration': 0}
            type_summary[t]['count'] += 1
            type_summary[t]['total_duration'] += episode['duration_sec']

        return {
            'episodes': episodes,
            'types': type_summary,
            'total_duration': sum(e['duration_sec'] for e in episodes),
            'percentage': len(np.where(tachy_mask)[0]) / len(self.hr_values) * 100
        }

    def detect_pauses(self):
        """Detect significant pauses"""
        pauses = []

        for i, rr in enumerate(self.rr_intervals):
            if rr > ECGConfig.PAUSE_THRESHOLD_SEC:
                severity = 'mild'
                if rr > ECGConfig.CRITICAL_PAUSE_SEC:
                    severity = 'critical'
                elif rr > ECGConfig.SIGNIFICANT_PAUSE_SEC:
                    severity = 'significant'

                # Check if it's a sinus pause or AV block
                pause_type = self._classify_pause(i)

                pauses.append({
                    'beat_index': i,
                    'duration_sec': rr,
                    'severity': severity,
                    'type': pause_type,
                    'preceding_hr': 60.0 / self.rr_intervals[i - 1] if i > 0 else None
                })

        return {
            'pauses': pauses,
            'total': len(pauses),
            'max_pause': max([p['duration_sec'] for p in pauses]) if pauses else 0
        }

    def detect_bigeminy_trigeminy(self):
        """Detect bigeminy and trigeminy patterns"""
        patterns = []

        if len(self.rr_intervals) < 6:
            return {'patterns': [], 'bigeminy': False, 'trigeminy': False}

        # Look for repeating patterns
        for i in range(len(self.rr_intervals) - 5):
            # Check for bigeminy (short-long-short-long)
            if (self.rr_intervals[i] < 0.8 * np.mean(self.rr_intervals) and
                    self.rr_intervals[i + 1] > 1.2 * np.mean(self.rr_intervals) and
                    self.rr_intervals[i + 2] < 0.8 * np.mean(self.rr_intervals) and
                    self.rr_intervals[i + 3] > 1.2 * np.mean(self.rr_intervals)):

                patterns.append({
                    'type': 'bigeminy',
                    'start_beat': i,
                    'confidence': 80
                })

            # Check for trigeminy (normal-normal-short-long)
            elif i < len(self.rr_intervals) - 4:
                mean_rr = np.mean(self.rr_intervals)
                if (abs(self.rr_intervals[i] - mean_rr) < 0.2 * mean_rr and
                        abs(self.rr_intervals[i + 1] - mean_rr) < 0.2 * mean_rr and
                        self.rr_intervals[i + 2] < 0.8 * mean_rr and
                        self.rr_intervals[i + 3] > 1.2 * mean_rr):
                    patterns.append({
                        'type': 'trigeminy',
                        'start_beat': i,
                        'confidence': 75
                    })

        # Remove overlapping patterns
        patterns = self._remove_overlapping_patterns(patterns)

        has_bigeminy = any(p['type'] == 'bigeminy' for p in patterns)
        has_trigeminy = any(p['type'] == 'trigeminy' for p in patterns)

        return {
            'patterns': patterns,
            'bigeminy': has_bigeminy,
            'trigeminy': has_trigeminy,
            'total_patterns': len(patterns)
        }

    def detect_heart_blocks(self):
        """Detect various degrees of heart block"""
        blocks = []

        # This is simplified - real detection would need P-wave analysis
        # Here we detect based on RR patterns

        # 2:1 AV block pattern
        for i in range(len(self.rr_intervals) - 2):
            if (self.rr_intervals[i] > 1.8 * np.mean(self.rr_intervals) and
                    abs(self.rr_intervals[i + 1] - np.mean(self.rr_intervals)) < 0.2 * np.mean(self.rr_intervals)):
                blocks.append({
                    'type': '2:1 AV block',
                    'beat_index': i,
                    'confidence': 60
                })

        # Mobitz Type I (Wenckebach) - progressively longer RR then dropped beat
        for i in range(len(self.rr_intervals) - 4):
            if (self.rr_intervals[i] < self.rr_intervals[i + 1] < self.rr_intervals[i + 2] and
                    self.rr_intervals[i + 3] > 1.5 * self.rr_intervals[i + 2]):
                blocks.append({
                    'type': 'Mobitz Type I',
                    'beat_index': i,
                    'confidence': 65
                })

        return {
            'blocks': blocks,
            'detected': len(blocks) > 0
        }

    # Helper methods
    def _calculate_sample_entropy(self, data, m=2, r=0.2):
        """Calculate sample entropy of RR intervals"""
        N = len(data)
        if N < m + 1:
            return 0

        # Normalize
        std_data = np.std(data)
        if std_data == 0:
            return 0

        r = r * std_data

        def _maxdist(x_i, x_j, m):
            return max([abs(ua - va) for ua, va in zip(x_i[0:m], x_j[0:m])])

        def _phi(m):
            patterns = []
            for i in range(N - m):
                patterns.append(data[i:i + m])

            C = 0
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if _maxdist(patterns[i], patterns[j], m) <= r:
                        C += 1

            return C / (len(patterns) * (len(patterns) - 1) / 2) if len(patterns) > 1 else 0

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        if phi_m == 0 or phi_m1 == 0:
            return 0

        return -np.log(phi_m1 / phi_m)

    def _calculate_turning_points_ratio(self, data):
        """Calculate turning points ratio"""
        if len(data) < 3:
            return 0

        turning_points = 0
        for i in range(1, len(data) - 1):
            if ((data[i] > data[i - 1] and data[i] > data[i + 1]) or
                    (data[i] < data[i - 1] and data[i] < data[i + 1])):
                turning_points += 1

        return turning_points / (len(data) - 2)

    def _check_p_wave_absence(self):
        """Simplified check for P-wave absence"""
        # This would need actual P-wave detection
        # For now, return based on RR variability
        cv = np.std(self.rr_intervals) / np.mean(self.rr_intervals) * 100
        return cv > 25  # High variability suggests AF

    def _analyze_beat_morphology(self, beat_idx):
        """Analyze QRS morphology of a specific beat"""
        if beat_idx >= len(self.r_peaks):
            return 'normal'

        # Extract beat window
        r_peak = self.r_peaks[beat_idx]
        window = int(0.1 * self.fs)

        start = max(0, r_peak - window)
        end = min(len(self.signal), r_peak + window)

        if end <= start:
            return 'normal'

        beat_segment = self.signal[start:end]

        # Simple width estimation
        derivative = np.diff(beat_segment)
        threshold = 0.3 * np.max(np.abs(derivative))

        significant_changes = np.where(np.abs(derivative) > threshold)[0]

        if len(significant_changes) > 1:
            width_samples = significant_changes[-1] - significant_changes[0]
            width_ms = width_samples * 1000 / self.fs

            if width_ms > ECGConfig.QRS_DURATION_WIDE:
                return 'wide'
            elif width_ms > ECGConfig.QRS_DURATION_NORMAL:
                return 'borderline'

        return 'normal'

    def _classify_pvc_patterns(self, pvcs):
        """Classify PVC patterns"""
        if len(pvcs) < 2:
            return []

        patterns = []

        # Check for consecutive PVCs
        for i in range(len(pvcs) - 1):
            if pvcs[i + 1]['index'] - pvcs[i]['index'] == 1:
                patterns.append('couplet')
            elif i < len(pvcs) - 2 and pvcs[i + 2]['index'] - pvcs[i]['index'] == 2:
                patterns.append('triplet')

        # Check for R-on-T
        for pvc in pvcs:
            if pvc['coupling_interval'] < 300:  # Very short coupling interval
                patterns.append('R-on-T')

        return list(set(patterns))  # Unique patterns

    def _find_consecutive_groups(self, mask):
        """Find groups of consecutive True values in boolean mask"""
        groups = []
        start = None

        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                groups.append((start, i))
                start = None

        if start is not None:
            groups.append((start, len(mask)))

        return groups

    def _classify_tachycardia(self, start_idx, end_idx, mean_hr):
        """Classify type of tachycardia"""
        # Simplified classification
        segment_rr = self.rr_intervals[start_idx:end_idx]

        if len(segment_rr) < 3:
            return {'name': 'Unclassified tachycardia', 'confidence': 50}

        cv = np.std(segment_rr) / np.mean(segment_rr) * 100

        # Regular narrow complex tachycardia
        if cv < 10:
            if mean_hr > 150 and mean_hr < 220:
                return {'name': 'Possible SVT', 'confidence': 70}
            elif mean_hr >= 220:
                return {'name': 'Possible AVRT/AVNRT', 'confidence': 65}
            else:
                return {'name': 'Sinus tachycardia', 'confidence': 80}

        # Irregular
        elif cv > 20:
            return {'name': 'AF with RVR', 'confidence': 75}

        # Check for wide complex
        morphology = self._analyze_beat_morphology(start_idx + 1)
        if morphology == 'wide':
            return {'name': 'Wide complex tachycardia', 'confidence': 70}

        return {'name': 'Unclassified tachycardia', 'confidence': 50}

    def _classify_pause(self, pause_idx):
        """Classify type of pause"""
        # Simplified - would need P-wave analysis
        if pause_idx == 0 or pause_idx >= len(self.rr_intervals) - 1:
            return 'Unclassified'

        # Check if pause is exactly 2x normal RR (suggesting blocked P-wave)
        normal_rr = np.mean([self.rr_intervals[pause_idx - 1],
                             self.rr_intervals[pause_idx + 1] if pause_idx + 1 < len(self.rr_intervals) else
                             self.rr_intervals[pause_idx - 1]])

        ratio = self.rr_intervals[pause_idx] / normal_rr

        if 1.8 < ratio < 2.2:
            return 'Possible blocked PAC or 2:1 block'
        elif ratio > 2.5:
            return 'Sinus pause'
        else:
            return 'Unclassified pause'

    def _remove_overlapping_patterns(self, patterns):
        """Remove overlapping pattern detections"""
        if len(patterns) <= 1:
            return patterns

        # Sort by start beat
        patterns.sort(key=lambda x: x['start_beat'])

        filtered = [patterns[0]]
        for pattern in patterns[1:]:
            # Check if overlaps with last accepted pattern
            last = filtered[-1]
            pattern_length = 4 if pattern['type'] == 'trigeminy' else 3
            last_length = 4 if last['type'] == 'trigeminy' else 3

            if pattern['start_beat'] >= last['start_beat'] + last_length:
                filtered.append(pattern)

        return filtered

    def _calculate_burden(self, results):
        """Calculate overall arrhythmia burden"""
        total_beats = len(self.r_peaks)

        if total_beats == 0:
            return

        # Calculate burden for each type
        burden = {}

        # Premature beats
        if 'premature_beats' in results:
            burden['PVC'] = results['premature_beats']['pvc_burden']
            burden['PAC'] = results['premature_beats']['pac_burden']

        # AF burden
        if 'atrial_fibrillation' in results and results['atrial_fibrillation']['detected']:
            burden['AF'] = 100.0  # If AF detected, assume 100% during recording

        # Bradycardia burden
        if 'bradycardia' in results:
            burden['Bradycardia'] = results['bradycardia']['percentage']

        # Tachycardia burden
        if 'tachycardia' in results:
            burden['Tachycardia'] = results['tachycardia']['percentage']

        results['burden'] = burden


# ====================================================================================
# HEART RATE VARIABILITY ANALYSIS
# ====================================================================================

class HRVAnalyzer:
    """Comprehensive HRV analysis in time and frequency domains"""

    def __init__(self, rr_intervals, fs_original):
        self.rr_intervals = rr_intervals  # in seconds
        self.nn_intervals = self._filter_nn_intervals(rr_intervals)
        self.fs_original = fs_original

        # ДЕБАГ
        print(f"DEBUG HRV: Received {len(rr_intervals)} RR intervals")
        if len(rr_intervals) > 0:
            print(f"DEBUG HRV: RR range: {np.min(rr_intervals) * 1000:.0f}-{np.max(rr_intervals) * 1000:.0f} ms")
            print(f"DEBUG HRV: After filtering: {len(self.nn_intervals)} NN intervals")

    def analyze_all(self):
        """Perform complete HRV analysis"""
        results = {
            'time_domain': self.time_domain_analysis(),
            'frequency_domain': self.frequency_domain_analysis(),
            'nonlinear': self.nonlinear_analysis(),
            'quality': self.assess_data_quality()
        }

        # Add interpretation
        results['interpretation'] = self.interpret_hrv(results)

        return results

    def time_domain_analysis(self):
        """Time domain HRV parameters"""
        if len(self.nn_intervals) < 2:
            print("DEBUG HRV: Not enough NN intervals for time domain analysis")
            return self._empty_time_domain()

        nn_ms = self.nn_intervals * 1000  # Convert to milliseconds
        nn_diff = np.diff(nn_ms)

        print(
            f"DEBUG HRV: NN intervals in ms: min={np.min(nn_ms):.0f}, max={np.max(nn_ms):.0f}, mean={np.mean(nn_ms):.0f}")

        results = {
            # Basic statistics
            'mean_nn': np.mean(nn_ms),
            'median_nn': np.median(nn_ms),
            'mean_hr': 60000 / np.mean(nn_ms) if np.mean(nn_ms) > 0 else 0,

            # Standard HRV metrics
            'sdnn': np.std(nn_ms),
            'sdann': self._calculate_sdann(nn_ms),
            'rmssd': np.sqrt(np.mean(nn_diff ** 2)) if len(nn_diff) > 0 else 0,
            'sdsd': np.std(nn_diff) if len(nn_diff) > 0 else 0,

            # NN interval differences
            'nn50': np.sum(np.abs(nn_diff) > 50) if len(nn_diff) > 0 else 0,
            'pnn50': np.sum(np.abs(nn_diff) > 50) / len(nn_diff) * 100 if len(nn_diff) > 0 else 0,
            'nn20': np.sum(np.abs(nn_diff) > 20) if len(nn_diff) > 0 else 0,
            'pnn20': np.sum(np.abs(nn_diff) > 20) / len(nn_diff) * 100 if len(nn_diff) > 0 else 0,

            # Heart rate statistics
            'hr_mean': 60000 / np.mean(nn_ms) if np.mean(nn_ms) > 0 else 0,
            'hr_std': np.std(60000 / nn_ms) if len(nn_ms) > 1 else 0,
            'hr_min': np.min(60000 / nn_ms) if len(nn_ms) > 0 else 0,
            'hr_max': np.max(60000 / nn_ms) if len(nn_ms) > 0 else 0,

            # Geometric measures
            'triangular_index': self._calculate_triangular_index(nn_ms),
            'tinn': self._calculate_tinn(nn_ms)
        }
        # Додати валідацію:
        if results['sdnn'] < results['rmssd']:
            # Якщо SDNN < RMSSD, використовуємо оригінальні інтервали
            nn_ms = self.rr_intervals * 1000
            results['sdnn'] = np.std(nn_ms)

        print(f"DEBUG HRV: SDNN={results['sdnn']:.1f}, RMSSD={results['rmssd']:.1f}, pNN50={results['pnn50']:.1f}")

        return results

    def frequency_domain_analysis(self):
        """Frequency domain HRV analysis"""
        if len(self.nn_intervals) < 20:
            return self._empty_frequency_domain()

        # Resample to uniform sampling (4 Hz typical for HRV)
        target_fs = 4.0
        nn_interpolated = self._interpolate_nn_intervals(self.nn_intervals, target_fs)

        if len(nn_interpolated) < 64:  # Need enough samples for FFT
            return self._empty_frequency_domain()

        # Welch's method for PSD estimation
        freq, psd = welch(nn_interpolated, fs=target_fs, nperseg=min(256, len(nn_interpolated)))

        # Calculate power in different bands
        vlf_power = self._calculate_band_power(freq, psd, ECGConfig.VLF_BAND)
        lf_power = self._calculate_band_power(freq, psd, ECGConfig.LF_BAND)
        hf_power = self._calculate_band_power(freq, psd, ECGConfig.HF_BAND)
        total_power = vlf_power + lf_power + hf_power

        # Peak frequencies
        lf_peak_freq = self._find_peak_frequency(freq, psd, ECGConfig.LF_BAND)
        hf_peak_freq = self._find_peak_frequency(freq, psd, ECGConfig.HF_BAND)

        results = {
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'total_power': total_power,
            'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0,
            'lf_norm': lf_power / (lf_power + hf_power) * 100 if (lf_power + hf_power) > 0 else 0,
            'hf_norm': hf_power / (lf_power + hf_power) * 100 if (lf_power + hf_power) > 0 else 0,
            'lf_peak': lf_peak_freq,
            'hf_peak': hf_peak_freq
        }

        return results

    def nonlinear_analysis(self):
        """Nonlinear HRV analysis"""
        if len(self.nn_intervals) < 50:
            return self._empty_nonlinear()

        nn_ms = self.nn_intervals * 1000

        results = {
            # Poincaré plot parameters
            'sd1': self._calculate_sd1(nn_ms),
            'sd2': self._calculate_sd2(nn_ms),
            'sd_ratio': 0,

            # Sample entropy
            'sample_entropy': self._calculate_sample_entropy(nn_ms),

            # DFA (Detrended Fluctuation Analysis)
            'dfa_alpha1': self._calculate_dfa(nn_ms, scale_range=(4, 16)),
            'dfa_alpha2': self._calculate_dfa(nn_ms, scale_range=(16, 64))
        }

        # Calculate SD ratio
        if results['sd2'] > 0:
            results['sd_ratio'] = results['sd1'] / results['sd2']

        return results

    def assess_data_quality(self):
        """Assess quality of RR interval data"""
        quality_score = 100
        issues = []

        # Check for sufficient data
        if len(self.nn_intervals) < ECGConfig.MIN_R_PEAKS_FOR_HRV:
            quality_score -= 40
            issues.append(f"Insufficient beats ({len(self.nn_intervals)})")

        # Check for artifacts (removed beats)
        artifact_ratio = 1 - len(self.nn_intervals) / len(self.rr_intervals) if len(self.rr_intervals) > 0 else 0
        if artifact_ratio > 0.2:
            quality_score -= 30
            issues.append(f"High artifact rate ({artifact_ratio * 100:.1f}%)")

        # Check for extreme values
        if len(self.nn_intervals) > 0:
            cv = np.std(self.nn_intervals) / np.mean(self.nn_intervals) * 100
            if cv > 30:
                quality_score -= 20
                issues.append(f"Extreme variability (CV={cv:.1f}%)")

        return {
            'score': max(0, quality_score),
            'issues': issues,
            'usable': quality_score >= 60
        }

    def interpret_hrv(self, results):
        """Interpret HRV results"""
        interpretation = []

        time_domain = results['time_domain']
        freq_domain = results['frequency_domain']

        # SDNN interpretation
        sdnn = time_domain['sdnn']
        if sdnn < ECGConfig.VERY_LOW_SDNN_MS:
            interpretation.append("Very low HRV (SDNN < 20ms) - significantly reduced")
        elif sdnn < ECGConfig.LOW_SDNN_MS:
            interpretation.append("Low HRV (SDNN < 30ms) - reduced autonomic function")
        elif sdnn > ECGConfig.HIGH_SDNN_MS:
            interpretation.append("High HRV (SDNN > 150ms) - good autonomic function")
        else:
            interpretation.append("Normal HRV (SDNN 30-150ms)")

        # RMSSD interpretation (parasympathetic)
        rmssd = time_domain['rmssd']
        if rmssd < ECGConfig.LOW_RMSSD_MS:
            interpretation.append("Low parasympathetic activity (RMSSD < 20ms)")
        elif rmssd > ECGConfig.HIGH_RMSSD_MS:
            interpretation.append("High parasympathetic activity (RMSSD > 60ms)")

        # LF/HF ratio interpretation
        if freq_domain:
            lf_hf = freq_domain.get('lf_hf_ratio', 0)
            if lf_hf > 2:
                interpretation.append("Sympathetic dominance (LF/HF > 2)")
            elif lf_hf < 0.5:
                interpretation.append("Parasympathetic dominance (LF/HF < 0.5)")
            else:
                interpretation.append("Balanced autonomic activity")

        return interpretation

    # Helper methods
    def _filter_nn_intervals(self, rr_intervals):
        """Filter out artifacts and ectopic beats"""
        if len(rr_intervals) < 3:
            return rr_intervals

        # Remove physiologically impossible values
        valid_mask = (rr_intervals > ECGConfig.MIN_RR_INTERVAL_SEC) & \
                     (rr_intervals < ECGConfig.MAX_RR_INTERVAL_SEC)

        # Remove intervals that differ > 20% from neighbors
        for i in range(1, len(rr_intervals) - 1):
            if valid_mask[i]:
                neighbors_mean = (rr_intervals[i - 1] + rr_intervals[i + 1]) / 2
                if abs(rr_intervals[i] - neighbors_mean) / neighbors_mean > 0.3:
                    valid_mask[i] = False

        return rr_intervals[valid_mask]

    def _calculate_sdann(self, nn_ms, segment_length=300000):  # 5 min segments
        """Calculate SDANN - std of average NN in segments"""
        if len(nn_ms) < segment_length / 1000:
            return 0

        segment_averages = []
        for i in range(0, len(nn_ms), int(segment_length / np.mean(nn_ms))):
            segment = nn_ms[i:i + int(segment_length / np.mean(nn_ms))]
            if len(segment) > 0:
                segment_averages.append(np.mean(segment))

        return np.std(segment_averages) if len(segment_averages) > 1 else 0

    def _calculate_triangular_index(self, nn_ms):
        """HRV triangular index"""
        if len(nn_ms) < 20:
            return 0

        # Create histogram with 1/128 sec bins (7.8125 ms)
        bin_width = 7.8125
        hist, _ = np.histogram(nn_ms, bins=np.arange(min(nn_ms), max(nn_ms) + bin_width, bin_width))

        # Triangular index = total count / maximum bin count
        if np.max(hist) > 0:
            return len(nn_ms) / np.max(hist)
        return 0

    def _calculate_tinn(self, nn_ms):
        """TINN - baseline width of NN histogram"""
        if len(nn_ms) < 20:
            return 0

        # This is simplified - actual TINN requires triangular interpolation
        return np.percentile(nn_ms, 95) - np.percentile(nn_ms, 5)

    def _interpolate_nn_intervals(self, nn_intervals, target_fs):
        """Interpolate NN intervals to uniform sampling"""
        if len(nn_intervals) < 2:
            return nn_intervals

        # Create time vector
        time = np.cumsum(nn_intervals)
        time = np.insert(time, 0, 0)[:-1]

        # New uniform time vector
        uniform_time = np.arange(0, time[-1], 1 / target_fs)

        # Interpolate
        from scipy.interpolate import interp1d
        f = interp1d(time, nn_intervals, kind='linear', fill_value='extrapolate')

        return f(uniform_time)

    def _calculate_band_power(self, freq, psd, band):
        """Calculate power in frequency band"""
        band_mask = (freq >= band[0]) & (freq <= band[1])
        return np.trapz(psd[band_mask], freq[band_mask])

    def _find_peak_frequency(self, freq, psd, band):
        """Find peak frequency in band"""
        band_mask = (freq >= band[0]) & (freq <= band[1])
        if np.any(band_mask):
            band_psd = psd[band_mask]
            band_freq = freq[band_mask]
            peak_idx = np.argmax(band_psd)
            return band_freq[peak_idx]
        return 0

    def _calculate_sd1(self, nn_ms):
        if len(nn_ms) < 2:
            return 0
        diff = np.diff(nn_ms)
        return np.sqrt(np.mean(diff ** 2) / 2)  # або просто RMSSD/√2

    def _calculate_sd2(self, nn_ms):
        if len(nn_ms) < 2:
            return 0
        diff = np.diff(nn_ms)
        sdnn = np.std(nn_ms)
        sdsd = np.std(diff)
        return np.sqrt(2 * sdnn ** 2 - 0.5 * sdsd ** 2)

    def _calculate_sample_entropy(self, data, m=2, r=0.2):
        """Sample entropy calculation"""
        N = len(data)
        if N < m + 1:
            return 0

        # Normalize r to data SD
        r = r * np.std(data)

        # Count template matches
        def template_matches(m):
            templates = np.array([data[i:i + m] for i in range(N - m + 1)])
            matches = 0

            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) <= r:
                        matches += 1

            return matches

        B = template_matches(m)
        A = template_matches(m + 1)

        if B == 0:
            return 0

        return -np.log(A / B)

    def _calculate_dfa(self, nn_ms, scale_range=(4, 64)):
        """Detrended Fluctuation Analysis"""
        N = len(nn_ms)
        if N < scale_range[1]:
            return 0

        # Integrate signal
        y = np.cumsum(nn_ms - np.mean(nn_ms))

        scales = np.logspace(np.log10(scale_range[0]), np.log10(scale_range[1]), 20, dtype=int)
        fluct = []

        for scale in scales:
            if scale < N:
                # Divide into segments
                segments = N // scale
                rms = []

                for i in range(segments):
                    segment = y[i * scale:(i + 1) * scale]
                    x = np.arange(len(segment))

                    # Detrend
                    coeffs = np.polyfit(x, segment, 1)
                    fit = np.polyval(coeffs, x)

                    # RMS of residuals
                    rms.append(np.sqrt(np.mean((segment - fit) ** 2)))

                if rms:
                    fluct.append(np.mean(rms))

        if len(fluct) > 1:
            # Fit log-log
            coeffs = np.polyfit(np.log(scales[:len(fluct)]), np.log(fluct), 1)
            return coeffs[0]  # Scaling exponent (alpha)

        return 0

    def _empty_time_domain(self):
        """Return empty time domain results"""
        return {k: 0 for k in ['mean_nn', 'median_nn', 'mean_hr', 'sdnn', 'sdann',
                               'rmssd', 'sdsd', 'nn50', 'pnn50', 'nn20', 'pnn20',
                               'hr_mean', 'hr_std', 'hr_min', 'hr_max',
                               'triangular_index', 'tinn']}

    def _empty_frequency_domain(self):
        """Return empty frequency domain results"""
        return {k: 0 for k in ['vlf_power', 'lf_power', 'hf_power', 'total_power',
                               'lf_hf_ratio', 'lf_norm', 'hf_norm', 'lf_peak', 'hf_peak']}

    def _empty_nonlinear(self):
        """Return empty nonlinear results"""
        return {k: 0 for k in ['sd1', 'sd2', 'sd_ratio', 'sample_entropy',
                               'dfa_alpha1', 'dfa_alpha2']}


# ====================================================================================
# CLINICAL INTERPRETATION
# ====================================================================================

class ClinicalInterpreter:
    """Generate clinical interpretation and recommendations"""

    def __init__(self, analysis_results):
        self.results = analysis_results
        self.findings = []
        self.recommendations = []
        self.urgency = 'routine'  # routine, urgent, critical

    def generate_interpretation(self):
        """Generate complete clinical interpretation"""
        interpretation = {
            'summary': self._generate_summary(),
            'findings': [],
            'recommendations': [],
            'urgency': 'routine',
            'risk_factors': [],
            'follow_up': []
        }

        # Analyze each component
        self._interpret_rhythm()
        self._interpret_rate()
        self._interpret_morphology()
        self._interpret_hrv()
        self._interpret_arrhythmias()

        # Compile results
        interpretation['findings'] = self.findings
        interpretation['recommendations'] = self.recommendations
        interpretation['urgency'] = self.urgency
        interpretation['risk_factors'] = self._identify_risk_factors()
        interpretation['follow_up'] = self._suggest_follow_up()

        return interpretation

    def _generate_summary(self):
        """Generate executive summary"""
        summary_parts = []

        # Basic rhythm
        if 'arrhythmias' in self.results:
            arr = self.results['arrhythmias']
            if arr.get('atrial_fibrillation', {}).get('detected'):
                summary_parts.append("Фібриляція передсердь")
            elif 'summary' in arr and arr['summary'].get('mean_hr'):
                hr = arr['summary']['mean_hr']
                if hr < ECGConfig.BRADYCARDIA_BPM:
                    summary_parts.append("Брадикардія")
                elif hr > ECGConfig.TACHYCARDIA_BPM:
                    summary_parts.append("Тахікардія")
                else:
                    summary_parts.append("Нормальний синусовий ритм")

        # Major findings
        if 'morphology' in self.results:
            morph = self.results['morphology']
            if morph.get('qrs_duration_mean', 0) > ECGConfig.QRS_DURATION_WIDE:
                summary_parts.append("Розширений QRS")

        if len(summary_parts) == 0:
            summary_parts.append("ЕКГ в межах норми")

        return ". ".join(summary_parts)

    def _interpret_rhythm(self):
        """Interpret cardiac rhythm"""
        if 'arrhythmias' not in self.results:
            return

        arr = self.results['arrhythmias']

        # Atrial fibrillation
        af = arr.get('atrial_fibrillation', {})
        if af.get('detected'):
            self.findings.append({
                'category': 'Rhythm',
                'finding': 'Фібриляція передсердь',
                'severity': 'significant',
                'confidence': af.get('confidence', 0),
                'details': f"CV: {af.get('cv', 0):.1f}%, Ентропія: {af.get('sample_entropy', 0):.2f}"
            })
            self.urgency = 'urgent'
            self.recommendations.append("Консультація кардіолога для оцінки ризику інсульту (CHA2DS2-VASc)")
            self.recommendations.append("Розглянути антикоагулянтну терапію")

        # Regular rhythm assessment
        elif 'summary' in arr:
            hr_std = arr['summary'].get('hr_std', 0)
            if hr_std < 5:
                self.findings.append({
                    'category': 'Rhythm',
                    'finding': 'Регулярний ритм',
                    'severity': 'normal',
                    'confidence': 90
                })
            elif hr_std < 10:
                self.findings.append({
                    'category': 'Rhythm',
                    'finding': 'Незначна нерегулярність ритму',
                    'severity': 'mild',
                    'confidence': 80
                })

    def _interpret_rate(self):
        """Interpret heart rate"""
        if 'arrhythmias' not in self.results:
            return

        arr = self.results['arrhythmias']
        if 'summary' not in arr:
            return

        mean_hr = arr['summary'].get('mean_hr', 0)

        # Bradycardia
        brady = arr.get('bradycardia', {})
        if brady.get('episodes'):
            severity = brady.get('severity', 'mild')
            self.findings.append({
                'category': 'Rate',
                'finding': f'Брадикардія ({severity})',
                'severity': severity,
                'confidence': 85,
                'details': f"Середня ЧСС: {mean_hr:.0f} уд/хв, Епізодів: {len(brady['episodes'])}"
            })

            if severity in ['severe', 'extreme']:
                self.urgency = 'urgent' if self.urgency == 'routine' else self.urgency
                self.recommendations.append("Оцінити необхідність кардіостимулятора")
                self.recommendations.append("Перевірити медикаменти (бета-блокатори, дигоксин)")

        # Tachycardia
        tachy = arr.get('tachycardia', {})
        if tachy.get('episodes'):
            for episode in tachy['episodes']:
                if episode['type'] == 'Possible SVT':
                    self.findings.append({
                        'category': 'Rate',
                        'finding': 'Можлива суправентрикулярна тахікардія',
                        'severity': 'significant',
                        'confidence': episode['confidence'],
                        'details': f"Макс ЧСС: {episode['max_hr']:.0f} уд/хв"
                    })
                    self.urgency = 'urgent'
                    self.recommendations.append("Електрофізіологічне дослідження")

    def _interpret_morphology(self):
        """Interpret ECG morphology"""
        if 'morphology' not in self.results:
            return

        morph = self.results['morphology']

        # QRS duration
        qrs_mean = morph.get('qrs_duration_mean', 0)
        if qrs_mean > ECGConfig.QRS_DURATION_WIDE:
            self.findings.append({
                'category': 'Morphology',
                'finding': 'Розширений QRS комплекс',
                'severity': 'significant',
                'confidence': 90,
                'details': f"Тривалість: {qrs_mean:.0f} мс"
            })
            self.recommendations.append("Виключити блокаду ніжки пучка Гіса")

        # P-wave abnormalities
        p_wave_stats = morph.get('p_wave_stats', {})
        if p_wave_stats.get('absent_ratio', 0) > 0.5:
            self.findings.append({
                'category': 'Morphology',
                'finding': 'Відсутні P-хвилі',
                'severity': 'significant',
                'confidence': 80,
                'details': "Можлива фібриляція передсердь"
            })

        # T-wave abnormalities
        t_wave_stats = morph.get('t_wave_stats', {})
        if t_wave_stats.get('inverted_ratio', 0) > 0.3:
            self.findings.append({
                'category': 'Morphology',
                'finding': 'Інверсія T-хвиль',
                'severity': 'moderate',
                'confidence': 75,
                'details': f"В {t_wave_stats['inverted_ratio'] * 100:.0f}% комплексів"
            })
            self.recommendations.append("Виключити ішемію міокарда")

    def _interpret_hrv(self):
        """Interpret HRV results"""
        if 'hrv' not in self.results:
            return

        hrv = self.results['hrv']
        quality = hrv.get('quality', {})

        # Check data quality first
        if not quality.get('usable', True):
            self.findings.append({
                'category': 'HRV',
                'finding': 'Низька якість даних для HRV аналізу',
                'severity': 'mild',
                'confidence': 90,
                'details': '; '.join(quality.get('issues', []))
            })
            return

        time_domain = hrv.get('time_domain', {})
        freq_domain = hrv.get('frequency_domain', {})

        # SDNN interpretation
        sdnn = time_domain.get('sdnn', 0)
        if sdnn < ECGConfig.VERY_LOW_SDNN_MS:
            self.findings.append({
                'category': 'HRV',
                'finding': 'Критично низька варіабельність серцевого ритму',
                'severity': 'severe',
                'confidence': 85,
                'details': f"SDNN: {sdnn:.1f} мс"
            })
            self.recommendations.append("Оцінити ризик серцево-судинних подій")
            self.recommendations.append("Розглянути холтерівське моніторування")
        elif sdnn < ECGConfig.LOW_SDNN_MS:
            self.findings.append({
                'category': 'HRV',
                'finding': 'Знижена варіабельність серцевого ритму',
                'severity': 'moderate',
                'confidence': 85,
                'details': f"SDNN: {sdnn:.1f} мс"
            })

        # Autonomic balance
        if freq_domain:
            lf_hf = freq_domain.get('lf_hf_ratio', 1)
            if lf_hf > 2.5:
                self.findings.append({
                    'category': 'HRV',
                    'finding': 'Симпатична домінантність',
                    'severity': 'mild',
                    'confidence': 75,
                    'details': f"LF/HF: {lf_hf:.2f}"
                })
                self.recommendations.append("Оцінити рівень стресу")
                self.recommendations.append("Розглянути методи релаксації")

    def _interpret_arrhythmias(self):
        """Interpret specific arrhythmias"""
        if 'arrhythmias' not in self.results:
            return

        arr = self.results['arrhythmias']

        # Premature beats
        premature = arr.get('premature_beats', {})
        if premature:
            pvc_burden = premature.get('pvc_burden', 0)
            if pvc_burden > 10:
                self.findings.append({
                    'category': 'Arrhythmia',
                    'finding': 'Часті шлуночкові екстрасистоли',
                    'severity': 'significant',
                    'confidence': 85,
                    'details': f"Навантаження: {pvc_burden:.1f}%"
                })
                self.urgency = 'urgent'
                self.recommendations.append("Ехокардіографія для оцінки функції ЛШ")
                self.recommendations.append("Розглянути 24-год холтер")

            # Check for dangerous patterns
            patterns = premature.get('patterns', [])
            if 'R-on-T' in patterns:
                self.findings.append({
                    'category': 'Arrhythmia',
                    'finding': 'R-on-T феномен',
                    'severity': 'critical',
                    'confidence': 80,
                    'details': "Ризик шлуночкової тахікардії"
                })
                self.urgency = 'critical'
                self.recommendations.append("ТЕРМІНОВА консультація кардіолога")

        # Pauses
        pauses = arr.get('pauses', {})
        if pauses.get('pauses'):
            max_pause = pauses.get('max_pause', 0)
            if max_pause > ECGConfig.CRITICAL_PAUSE_SEC:
                self.findings.append({
                    'category': 'Arrhythmia',
                    'finding': 'Критичні паузи',
                    'severity': 'critical',
                    'confidence': 90,
                    'details': f"Макс пауза: {max_pause:.1f} сек"
                })
                self.urgency = 'critical'
                self.recommendations.append("Розглянути імплантацію кардіостимулятора")

        # Bigeminy/Trigeminy
        patterns = arr.get('bigeminy', {})
        if patterns.get('bigeminy'):
            self.findings.append({
                'category': 'Arrhythmia',
                'finding': 'Бігемінія',
                'severity': 'moderate',
                'confidence': 80,
                'details': "Чергування нормальних та передчасних скорочень"
            })

    def _identify_risk_factors(self):
        """Identify cardiovascular risk factors from ECG"""
        risk_factors = []

        # AF → Stroke risk
        if any(f['finding'] == 'Фібриляція передсердь' for f in self.findings):
            risk_factors.append({
                'factor': 'Ризик інсульту',
                'reason': 'Фібриляція передсердь',
                'recommendation': 'Оцінка CHA2DS2-VASc score'
            })

        # Low HRV → Sudden cardiac death risk
        if any('Критично низька варіабельність' in f['finding'] for f in self.findings):
            risk_factors.append({
                'factor': 'Підвищений серцево-судинний ризик',
                'reason': 'Низька HRV',
                'recommendation': 'Комплексна оцінка ризику'
            })

        # Frequent PVCs → Cardiomyopathy risk
        if any('Часті шлуночкові екстрасистоли' in f['finding'] for f in self.findings):
            risk_factors.append({
                'factor': 'Ризик кардіоміопатії',
                'reason': 'Високе навантаження PVC',
                'recommendation': 'Ехокардіографія та МРТ серця'
            })

        return risk_factors

    def _suggest_follow_up(self):
        """Suggest follow-up actions"""
        follow_up = []

        # Based on urgency
        if self.urgency == 'critical':
            follow_up.append("ТЕРМІНОВА госпіталізація або консультація кардіолога")
        elif self.urgency == 'urgent':
            follow_up.append("Консультація кардіолога протягом 1-2 днів")
        else:
            follow_up.append("Планова консультація кардіолога")

        # Additional tests
        if any('холтер' in r.lower() for r in self.recommendations):
            follow_up.append("24-годинне холтерівське моніторування")

        if any('ехокардіо' in r.lower() for r in self.recommendations):
            follow_up.append("Трансторакальна ехокардіографія")

        # Lifestyle modifications
        if any(f['category'] == 'HRV' for f in self.findings):
            follow_up.append("Модифікація способу життя (фізична активність, управління стресом)")

        return follow_up


# ====================================================================================
# VISUALIZATION
# ====================================================================================

class ECGVisualizer:
    """Create various ECG visualizations"""

    def __init__(self, signal, r_peaks, fs, lead_name="ECG"):
        self.signal = signal
        self.r_peaks = r_peaks
        self.fs = fs
        self.lead_name = lead_name
        self.time = np.arange(len(signal)) / fs

    def plot_full_ecg(self, save_path=None, show_peaks=True):
        """Plot full ECG with R-peaks marked"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        # Plot signal
        ax.plot(self.time, self.signal, 'b-', linewidth=0.5, label=self.lead_name)

        # Mark R-peaks
        if show_peaks and len(self.r_peaks) > 0:
            ax.plot(self.time[self.r_peaks], self.signal[self.r_peaks],
                    'ro', markersize=4, label='R-peaks')

        # Formatting
        ax.set_xlabel('Час (с)', fontsize=12)
        ax.set_ylabel('Амплітуда (мВ)', fontsize=12)
        ax.set_title(f'ЕКГ запис - {self.lead_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Add heart rate annotation
        if len(self.r_peaks) > 1:
            hr = 60 * self.fs / np.mean(np.diff(self.r_peaks))
            ax.text(0.02, 0.98, f'Середня ЧСС: {hr:.0f} уд/хв',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_rhythm_strip(self, duration=10, save_path=None):
        """Plot rhythm strip (10 seconds typical)"""
        samples = int(duration * self.fs)
        num_strips = len(self.signal) // samples + 1

        fig, axes = plt.subplots(num_strips, 1, figsize=(15, 3 * num_strips))
        if num_strips == 1:
            axes = [axes]

        for i in range(num_strips):
            start = i * samples
            end = min((i + 1) * samples, len(self.signal))

            if start >= len(self.signal):
                axes[i].set_visible(False)
                continue

            # Time for this strip
            strip_time = self.time[start:end] - self.time[start]

            # Plot signal
            axes[i].plot(strip_time, self.signal[start:end], 'b-', linewidth=0.8)

            # Mark R-peaks in this strip
            strip_peaks = self.r_peaks[(self.r_peaks >= start) & (self.r_peaks < end)] - start
            if len(strip_peaks) > 0:
                axes[i].plot(strip_time[strip_peaks], self.signal[start:end][strip_peaks],
                             'ro', markersize=4)

            # Grid and labels
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylabel('мВ')
            axes[i].set_ylim([np.min(self.signal) * 1.1, np.max(self.signal) * 1.1])

            # Time markers every second
            for sec in range(int(duration) + 1):
                axes[i].axvline(x=sec, color='gray', linestyle='--', alpha=0.3)

            if i == num_strips - 1:
                axes[i].set_xlabel('Час (с)')

        plt.suptitle(f'Ритм-стрічка - {self.lead_name}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_heart_rate(self, rr_intervals, save_path=None):
        """Plot heart rate over time"""
        if len(rr_intervals) < 2:
            return

        # Calculate instantaneous HR
        hr_instant = 60.0 / rr_intervals
        time_hr = np.cumsum(rr_intervals)[:-1]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Heart rate plot
        ax1.plot(time_hr, hr_instant[:-1], 'b-', linewidth=1.5)
        ax1.axhline(y=ECGConfig.TACHYCARDIA_BPM, color='r', linestyle='--', alpha=0.5, label='Тахікардія')
        ax1.axhline(y=ECGConfig.BRADYCARDIA_BPM, color='r', linestyle='--', alpha=0.5, label='Брадикардія')
        ax1.fill_between(time_hr, ECGConfig.NORMAL_HR_LOW_BPM, ECGConfig.NORMAL_HR_HIGH_BPM,
                         alpha=0.2, color='green', label='Норма')

        ax1.set_ylabel('ЧСС (уд/хв)', fontsize=12)
        ax1.set_title('Динаміка частоти серцевих скорочень', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # RR intervals plot
        ax2.plot(time_hr, rr_intervals[:-1] * 1000, 'g-', linewidth=1.5)
        ax2.set_xlabel('Час (с)', fontsize=12)
        ax2.set_ylabel('RR інтервал (мс)', fontsize=12)
        ax2.set_title('RR інтервали', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_poincare(self, rr_intervals, save_path=None):
        """Create Poincaré plot for HRV analysis"""
        if len(rr_intervals) < 2:
            return

        rr_ms = rr_intervals * 1000

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Scatter plot
        ax.scatter(rr_ms[:-1], rr_ms[1:], alpha=0.5, s=20)

        # Identity line
        min_rr = np.min(rr_ms)
        max_rr = np.max(rr_ms)
        ax.plot([min_rr, max_rr], [min_rr, max_rr], 'k--', alpha=0.5)

        # Calculate and plot SD1/SD2 ellipse
        sd1 = np.std(np.diff(rr_ms)) / np.sqrt(2)
        sd2 = np.sqrt(2 * np.var(rr_ms) - 0.5 * np.var(np.diff(rr_ms)))

        mean_rr = np.mean(rr_ms)
        ellipse = plt.matplotlib.patches.Ellipse((mean_rr, mean_rr),
                                                 2 * sd2, 2 * sd1,
                                                 angle=45,
                                                 fill=False,
                                                 edgecolor='red',
                                                 linewidth=2)
        ax.add_patch(ellipse)

        # Labels and formatting
        ax.set_xlabel('RR(n) (мс)', fontsize=12)
        ax.set_ylabel('RR(n+1) (мс)', fontsize=12)
        ax.set_title(f'Графік Пуанкаре\nSD1={sd1:.1f}мс, SD2={sd2:.1f}мс', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_frequency_spectrum(self, rr_intervals, save_path=None):
        """Plot HRV frequency spectrum"""
        if len(rr_intervals) < 20:
            return

        # Resample to uniform 4Hz
        from scipy.interpolate import interp1d
        time_rr = np.cumsum(rr_intervals)
        time_rr = np.insert(time_rr, 0, 0)[:-1]

        fs_resample = 4.0
        time_uniform = np.arange(0, time_rr[-1], 1 / fs_resample)

        f_interp = interp1d(time_rr, rr_intervals, kind='linear', fill_value='extrapolate')
        rr_uniform = f_interp(time_uniform)

        # Calculate PSD
        freq, psd = welch(rr_uniform, fs=fs_resample, nperseg=min(256, len(rr_uniform)))

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot PSD
        ax.semilogy(freq, psd, 'b-', linewidth=1.5)

        # Mark frequency bands
        ax.axvspan(ECGConfig.VLF_BAND[0], ECGConfig.VLF_BAND[1], alpha=0.2, color='gray', label='VLF')
        ax.axvspan(ECGConfig.LF_BAND[0], ECGConfig.LF_BAND[1], alpha=0.2, color='red', label='LF')
        ax.axvspan(ECGConfig.HF_BAND[0], ECGConfig.HF_BAND[1], alpha=0.2, color='blue', label='HF')

        ax.set_xlabel('Частота (Гц)', fontsize=12)
        ax.set_ylabel('Потужність (с²/Гц)', fontsize=12)
        ax.set_title('Спектральний аналіз HRV', fontsize=14)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def create_report_figure(self, analysis_results, save_path):
        """Create comprehensive figure for report"""
        fig = plt.figure(figsize=(16, 20))

        # 1. ECG strip (top)
        ax1 = plt.subplot(5, 1, 1)
        duration_to_show = min(10, len(self.signal) / self.fs)
        samples_to_show = int(duration_to_show * self.fs)

        ax1.plot(self.time[:samples_to_show], self.signal[:samples_to_show], 'b-', linewidth=0.8)
        peaks_to_show = self.r_peaks[self.r_peaks < samples_to_show]
        if len(peaks_to_show) > 0:
            ax1.plot(self.time[peaks_to_show], self.signal[peaks_to_show], 'ro', markersize=4)

        ax1.set_title(f'ЕКГ запис - {self.lead_name}', fontsize=14)
        ax1.set_xlabel('Час (с)')
        ax1.set_ylabel('Амплітуда (мВ)')
        ax1.grid(True, alpha=0.3)

        # 2. Heart rate plot
        if 'arrhythmias' in analysis_results:
            ax2 = plt.subplot(5, 1, 2)
            rr_intervals = np.diff(self.r_peaks) / self.fs
            if len(rr_intervals) > 0:
                hr_instant = 60.0 / rr_intervals
                time_hr = self.time[self.r_peaks[:-1]]

                ax2.plot(time_hr, hr_instant, 'b-', linewidth=1)
                ax2.axhline(y=ECGConfig.TACHYCARDIA_BPM, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=ECGConfig.BRADYCARDIA_BPM, color='r', linestyle='--', alpha=0.5)
                ax2.fill_between(time_hr, ECGConfig.NORMAL_HR_LOW_BPM, ECGConfig.NORMAL_HR_HIGH_BPM,
                                 alpha=0.2, color='green')

                ax2.set_title('Динаміка ЧСС', fontsize=14)
                ax2.set_xlabel('Час (с)')
                ax2.set_ylabel('ЧСС (уд/хв)')
                ax2.grid(True, alpha=0.3)

        # 3. Poincaré plot
        if 'hrv' in analysis_results and len(rr_intervals) > 2:
            ax3 = plt.subplot(5, 2, 5)
            rr_ms = rr_intervals * 1000

            ax3.scatter(rr_ms[:-1], rr_ms[1:], alpha=0.5, s=20)
            ax3.plot([np.min(rr_ms), np.max(rr_ms)], [np.min(rr_ms), np.max(rr_ms)], 'k--', alpha=0.5)
            ax3.set_xlabel('RR(n) (мс)')
            ax3.set_ylabel('RR(n+1) (мс)')
            ax3.set_title('Графік Пуанкаре')
            ax3.grid(True, alpha=0.3)
            ax3.set_aspect('equal')

        # 4. HRV histogram
        if len(rr_intervals) > 5:
            ax4 = plt.subplot(5, 2, 6)
            rr_ms = rr_intervals * 1000

            ax4.hist(rr_ms, bins=30, alpha=0.7, edgecolor='black')
            ax4.axvline(x=np.mean(rr_ms), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rr_ms):.0f}ms')
            ax4.set_xlabel('RR інтервал (мс)')
            ax4.set_ylabel('Частота')
            ax4.set_title('Розподіл RR інтервалів')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Average beat morphology
        if 'morphology' in analysis_results and 'average_beat' in analysis_results['morphology']:
            ax5 = plt.subplot(5, 2, 7)
            avg_beat = analysis_results['morphology']['average_beat']
            beat_time = np.arange(len(avg_beat)) / self.fs * 1000  # in ms

            ax5.plot(beat_time, avg_beat, 'b-', linewidth=2)
            ax5.set_xlabel('Час (мс)')
            ax5.set_ylabel('Амплітуда (мВ)')
            ax5.set_title('Усереднений комплекс')
            ax5.grid(True, alpha=0.3)

            # Mark waves if detected
            if 'p_wave' in analysis_results['morphology']:
                p_pos = analysis_results['morphology']['p_wave']['position']
                ax5.plot(beat_time[p_pos], avg_beat[p_pos], 'go', markersize=8, label='P')
            if 't_wave' in analysis_results['morphology']:
                t_pos = analysis_results['morphology']['t_wave']['position']
                ax5.plot(beat_time[t_pos], avg_beat[t_pos], 'mo', markersize=8, label='T')

            ax5.legend()

        # 6. Clinical findings summary
        if 'clinical_interpretation' in analysis_results:
            ax6 = plt.subplot(5, 1, 5)
            ax6.axis('off')

            findings_text = "КЛІНІЧНА ІНТЕРПРЕТАЦІЯ:\n\n"
            findings = analysis_results['clinical_interpretation'].get('findings', [])

            for i, finding in enumerate(findings[:5]):  # Show top 5 findings
                findings_text += f"{i + 1}. {finding['finding']}\n"
                if 'details' in finding:
                    findings_text += f"   {finding['details']}\n"

            ax6.text(0.05, 0.95, findings_text, transform=ax6.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ====================================================================================
# SIGNAL QUALITY ASSESSMENT
# ====================================================================================

def validate_ecg_signal(signal_data, sampling_rate, lead_name):
    """Enhanced signal quality validation"""
    warnings = []
    quality_score = 100

    # Basic checks
    if not isinstance(signal_data, (list, np.ndarray)):
        return False, [f"Некоректний тип сигналу для {lead_name}: {type(signal_data)}"], 0

    signal = np.array(signal_data, dtype=float)

    if signal.size == 0:
        return False, [f"Порожній сигнал для {lead_name}"], 0

    if sampling_rate <= 0:
        return False, [f"Некоректна частота дискретизації: {sampling_rate} Гц"], 0

    # Length check
    duration_sec = len(signal) / sampling_rate
    if duration_sec < ECGConfig.MIN_SIGNAL_LENGTH_SEC:
        quality_score -= 30
        warnings.append(f"Короткий сигнал: {duration_sec:.1f}с (мін. {ECGConfig.MIN_SIGNAL_LENGTH_SEC}с)")

    # Flat signal check
    signal_range = np.ptp(signal)
    if signal_range < ECGConfig.SIGNAL_FLAT_THRESHOLD:
        return False, [f"Плоский сигнал в {lead_name} (амплітуда < {ECGConfig.SIGNAL_FLAT_THRESHOLD})"], 0

    # Clipping detection
    unique_values = np.unique(signal)
    if len(unique_values) < ECGConfig.MIN_UNIQUE_VALUES:
        quality_score -= 20
        warnings.append(f"Можливе обрізання: лише {len(unique_values)} унікальних значень")

    # Check for clipping at extremes
    if len(signal) > 100:
        max_val = np.max(signal)
        min_val = np.min(signal)
        clipping_ratio = np.sum((signal == max_val) | (signal == min_val)) / len(signal)

        if clipping_ratio > ECGConfig.MAX_CLIPPING_PERCENT / 100:
            quality_score -= 25
            warnings.append(f"Обрізання сигналу: {clipping_ratio * 100:.1f}% відліків")

    # Noise estimation
    if len(signal) > 100:
        # High-frequency noise
        derivative = np.diff(signal)
        noise_estimate = np.std(derivative)
        signal_power = np.std(signal)

        if signal_power > 0:
            snr_db = 20 * np.log10(signal_power / noise_estimate)
            if snr_db < ECGConfig.MIN_SNR_DB:
                quality_score -= 15
                warnings.append(f"Високий рівень шуму (SNR: {snr_db:.1f} dB)")

    # Outlier detection
    if len(signal) > 20:
        q1, q3 = np.percentile(signal, [25, 75])
        iqr = q3 - q1

        if iqr > 0:
            outliers = np.sum((signal < q1 - 3 * iqr) | (signal > q3 + 3 * iqr))
            outlier_percent = outliers / len(signal) * 100

            if outlier_percent > 5:
                quality_score -= 10
                warnings.append(f"Багато викидів: {outlier_percent:.1f}%")

    # Baseline drift check
    if len(signal) > sampling_rate * 5:  # At least 5 seconds
        # Fit polynomial to detect drift
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, 1)
        drift_slope = coeffs[0]

        # Significant drift if slope > 10% of signal range per second
        drift_per_second = abs(drift_slope) * sampling_rate
        if drift_per_second > 0.1 * signal_range:
            quality_score -= 10
            warnings.append("Значний дрейф базової лінії")

    # Final quality assessment
    is_valid = quality_score >= 40  # Minimum acceptable quality

    return is_valid, warnings, quality_score


# ====================================================================================
# MAIN ANALYSIS FUNCTION
# ====================================================================================

def analyze_ecg_comprehensive(sampling_rate, lead_name, signal_data_str, units="N/A", r_peaks_from_xml=None):
    """
    Comprehensive ECG analysis with all advanced features

    Args:
        sampling_rate: Sampling rate in Hz
        lead_name: Name of the lead
        signal_data_str: Signal data as string
        units: Signal units
        r_peaks_from_xml: Pre-detected R-peaks from XML annotations
    """
    analysis_results = {
        "basic_info": {
            "lead_name": lead_name,
            "units": units,
            "sampling_rate": sampling_rate
        },
        "signal_quality": {},
        "r_peaks": {},
        "morphology": {},
        "arrhythmias": {},
        "hrv": {},
        "clinical_interpretation": {},
        "warnings": [],
        "processing_log": []
    }

    # Convert string data to array
    if not signal_data_str:
        analysis_results["warnings"].append(f"Відведення {lead_name}: Немає даних для аналізу")
        return analysis_results

    try:
        signal_data = np.array([float(s) for s in signal_data_str.split()])
    except ValueError:
        analysis_results["warnings"].append(f"Відведення {lead_name}: Некоректні дані сигналу")
        return analysis_results

    if signal_data.size == 0:
        analysis_results["warnings"].append(f"Відведення {lead_name}: Порожні дані сигналу")
        return analysis_results

    # Step 1: Signal Quality Assessment
    print(f"  1. Оцінка якості сигналу...")
    is_valid, quality_warnings, quality_score = validate_ecg_signal(signal_data, sampling_rate, lead_name)

    analysis_results["signal_quality"] = {
        "is_valid": is_valid,
        "score": quality_score,
        "warnings": quality_warnings
    }

    if not is_valid:
        analysis_results["warnings"].append(f"Сигнал не пройшов валідацію: {'; '.join(quality_warnings)}")
        return analysis_results

    # Step 2: Signal Preprocessing
    print(f"  2. Попередня обробка сигналу...")
    processor = SignalProcessor()

    # Adaptive filtering
    filtered_signal, snr = processor.adaptive_filter(signal_data, sampling_rate)
    analysis_results["signal_quality"]["snr_db"] = snr

    # Remove powerline interference if needed
    if snr < 20:
        # Detect powerline frequency (50 or 60 Hz)
        freq, psd = welch(filtered_signal, fs=sampling_rate, nperseg=min(1024, len(filtered_signal)))

        # Check for peaks at 50 and 60 Hz
        idx_50 = np.argmin(np.abs(freq - 50))
        idx_60 = np.argmin(np.abs(freq - 60))

        if idx_50 < len(psd) and psd[idx_50] > np.mean(psd) * 5:
            filtered_signal = processor.remove_powerline_interference(filtered_signal, sampling_rate, 50)
            analysis_results["processing_log"].append("Видалено інтерференцію 50 Гц")
        elif idx_60 < len(psd) and psd[idx_60] > np.mean(psd) * 5:
            filtered_signal = processor.remove_powerline_interference(filtered_signal, sampling_rate, 60)
            analysis_results["processing_log"].append("Видалено інтерференцію 60 Гц")

    # Step 3: R-peak Detection
    print(f"  3. Детекція R-піків...")

    # Check if we have R-peaks from XML
    if r_peaks_from_xml is not None and len(r_peaks_from_xml) > 0:
        r_peaks = np.array(r_peaks_from_xml)
        print(f"  Використовую {len(r_peaks)} R-піків з XML анотацій")
        analysis_results["processing_log"].append(f"R-піки з XML: {len(r_peaks)}")
    else:
        # Detect R-peaks
        detector = AdvancedRPeakDetector(filtered_signal, sampling_rate)
        r_peaks = detector.detect_peaks_ensemble()
        print(f"  Детектовано {len(r_peaks)} R-піків")
        analysis_results["processing_log"].append(f"R-піки детектовані: {len(r_peaks)}")

    analysis_results["r_peaks"] = {
        "count": len(r_peaks),
        "positions": r_peaks.tolist() if len(r_peaks) > 0 else []
    }

    if len(r_peaks) < ECGConfig.MIN_R_PEAKS_FOR_ANALYSIS:
        analysis_results["warnings"].append(
            f"Недостатньо R-піків для аналізу ({len(r_peaks)} < {ECGConfig.MIN_R_PEAKS_FOR_ANALYSIS})"
        )
        return analysis_results

    # Calculate basic metrics
    rr_intervals = np.diff(r_peaks) / sampling_rate
    hr_instant = 60.0 / rr_intervals

    analysis_results["basic_metrics"] = {
        "mean_hr": np.mean(hr_instant),
        "std_hr": np.std(hr_instant),
        "min_hr": np.min(hr_instant),
        "max_hr": np.max(hr_instant),
        "mean_rr": np.mean(rr_intervals) * 1000,  # in ms
        "total_beats": len(r_peaks)
    }

    # Step 4: Morphology Analysis
    print(f"  4. Аналіз морфології...")
    if len(r_peaks) >= 10:
        morph_analyzer = MorphologyAnalyzer(signal_data, r_peaks, sampling_rate)

        # Segment beats
        beats, valid_indices = morph_analyzer.segment_beats()

        if len(beats) > 5:
            # Average beat
            average_beat = morph_analyzer.compute_average_beat(beats)
            if average_beat is not None:
                analysis_results["morphology"]["average_beat"] = average_beat.tolist()

            # Analyze average beat morphology
            r_peak_in_beat = len(average_beat) // 2

            # P-wave detection
            p_wave = morph_analyzer.detect_p_waves(average_beat, r_peak_in_beat)
            if p_wave:
                analysis_results["morphology"]["p_wave"] = p_wave

            # T-wave detection
            t_wave = morph_analyzer.detect_t_waves(average_beat, r_peak_in_beat)
            if t_wave:
                analysis_results["morphology"]["t_wave"] = t_wave

            # QRS analysis
            qrs_features = morph_analyzer.analyze_qrs_morphology(beats[:20])  # Analyze first 20 beats
            if qrs_features:
                analysis_results["morphology"]["qrs_duration_mean"] = np.mean([f['duration'] for f in qrs_features])
                analysis_results["morphology"]["qrs_amplitude_mean"] = np.mean([f['amplitude'] for f in qrs_features])

                # Count morphology types
                morphology_counts = Counter([f['morphology'] for f in qrs_features])
                analysis_results["morphology"]["qrs_morphology_types"] = dict(morphology_counts)

    # Step 5: Arrhythmia Detection
    print(f"  5. Детекція аритмій...")
    if len(r_peaks) >= ECGConfig.MIN_R_PEAKS_FOR_ANALYSIS:
        arrhythmia_detector = ArrhythmiaDetector(r_peaks, signal_data, sampling_rate)
        analysis_results["arrhythmias"] = arrhythmia_detector.detect_all_arrhythmias()

        # Step 6: HRV Analysis (оновлена частина функції analyze_ecg_comprehensive)
        print(f"  6. Аналіз варіабельності серцевого ритму...")

        # ВАЖЛИВО: Перевіряємо що маємо R-піки
        if len(r_peaks) < 2:
            analysis_results["warnings"].append(
                f"Недостатньо R-піків для HRV аналізу ({len(r_peaks)} < 2)"
            )
            print(f"  ⚠️  Недостатньо R-піків: {len(r_peaks)}")
        else:
            # Розраховуємо RR інтервали
            rr_intervals = np.diff(r_peaks) / sampling_rate  # в секундах

            print(f"  📊 R-піків: {len(r_peaks)}, RR інтервалів: {len(rr_intervals)}")

            if len(rr_intervals) >= ECGConfig.MIN_R_PEAKS_FOR_HRV:
                try:
                    # Створюємо HRV аналізатор
                    hrv_analyzer = HRVAnalyzer(rr_intervals, sampling_rate)

                    # Виконуємо аналіз
                    hrv_results = hrv_analyzer.analyze_all()

                    # Зберігаємо результати
                    analysis_results["hrv"] = hrv_results

                    # Виводимо основні метрики для дебагу
                    if 'time_domain' in hrv_results:
                        td = hrv_results['time_domain']
                        print(f"  ✅ HRV розраховано:")
                        print(f"     SDNN: {td.get('sdnn', 0):.1f} мс")
                        print(f"     RMSSD: {td.get('rmssd', 0):.1f} мс")
                        print(f"     pNN50: {td.get('pnn50', 0):.1f}%")
                        print(f"     Mean RR: {td.get('mean_nn', 0):.0f} мс")

                    # Додаємо в базові метрики теж
                    if 'basic_metrics' not in analysis_results:
                        analysis_results['basic_metrics'] = {}

                    analysis_results['basic_metrics']['rr_intervals_count'] = len(rr_intervals)
                    analysis_results['basic_metrics']['hrv_calculated'] = True

                except Exception as e:
                    print(f"  ❌ Помилка HRV аналізу: {e}")
                    import traceback
                    traceback.print_exc()
                    analysis_results["warnings"].append(f"Помилка HRV аналізу: {str(e)}")
            else:
                analysis_results["warnings"].append(
                    f"Недостатньо R-піків для HRV аналізу ({len(rr_intervals) + 1} < {ECGConfig.MIN_R_PEAKS_FOR_HRV})"
                )
                print(f"  ⚠️  Недостатньо RR інтервалів: {len(rr_intervals)} < {ECGConfig.MIN_R_PEAKS_FOR_HRV - 1}")

    # Step 7: Clinical Interpretation
    print(f"  7. Клінічна інтерпретація...")
    interpreter = ClinicalInterpreter(analysis_results)
    analysis_results["clinical_interpretation"] = interpreter.generate_interpretation()

    # Step 8: Generate visualizations if matplotlib available
    if MATPLOTLIB_AVAILABLE and len(r_peaks) > 0:
        print(f"  8. Створення візуалізацій...")
        visualizer = ECGVisualizer(signal_data, r_peaks, sampling_rate, lead_name)

        # Create temp directory for plots
        import tempfile
        temp_dir = tempfile.mkdtemp()

        plot_paths = {}
        try:
            # Generate plots
            plot_paths['full_ecg'] = os.path.join(temp_dir, 'full_ecg.png')
            visualizer.plot_full_ecg(plot_paths['full_ecg'])

            plot_paths['rhythm_strip'] = os.path.join(temp_dir, 'rhythm_strip.png')
            visualizer.plot_rhythm_strip(save_path=plot_paths['rhythm_strip'])

            if len(rr_intervals) > 5:
                plot_paths['heart_rate'] = os.path.join(temp_dir, 'heart_rate.png')
                visualizer.plot_heart_rate(rr_intervals, plot_paths['heart_rate'])

                plot_paths['poincare'] = os.path.join(temp_dir, 'poincare.png')
                visualizer.plot_poincare(rr_intervals, plot_paths['poincare'])

            if len(rr_intervals) > 20:
                plot_paths['frequency'] = os.path.join(temp_dir, 'frequency.png')
                visualizer.plot_frequency_spectrum(rr_intervals, plot_paths['frequency'])

            analysis_results["visualizations"] = plot_paths

        except Exception as e:
            analysis_results["warnings"].append(f"Помилка створення візуалізацій: {str(e)}")

    return analysis_results


# ====================================================================================
# EXCEL REPORT GENERATION
# ====================================================================================

def create_enhanced_excel_report(all_results, excel_output_path):
    """Create comprehensive Excel report with multiple sheets and charts"""
    if not OPENPYXL_AVAILABLE:
        print("Excel export недоступний - openpyxl не встановлено")
        return False

    wb = openpyxl.Workbook()

    # Remove default sheet
    wb.remove(wb.active)

    # Define styles
    header_font = Font(bold=True, size=14)
    subheader_font = Font(bold=True, size=12)
    normal_font = Font(size=11)

    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font_white = Font(bold=True, color="FFFFFF", size=12)

    good_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    warning_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    bad_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Sheet 1: Summary
    ws_summary = wb.create_sheet("Загальний звіт")

    # Title
    ws_summary.merge_cells('A1:F1')
    ws_summary['A1'] = "Звіт аналізу ЕКГ"
    ws_summary['A1'].font = Font(bold=True, size=16)
    ws_summary['A1'].alignment = Alignment(horizontal="center")

    # Patient info
    row = 3
    ws_summary[f'A{row}'] = "Інформація про пацієнта"
    ws_summary[f'A{row}'].font = subheader_font
    row += 1

    patient_info = [
        ("ID пацієнта:", all_results.get("patient_id", "N/A")),
        ("Ім'я:", all_results.get("patient_name", "N/A")),
        ("Дата запису:", all_results.get("recording_date", datetime.now().strftime("%Y-%m-%d"))),
        ("Час аналізу:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ]

    for label, value in patient_info:
        ws_summary[f'A{row}'] = label
        ws_summary[f'B{row}'] = value
        ws_summary[f'A{row}'].font = Font(bold=True)
        row += 1

    # Technical parameters
    row += 1
    ws_summary[f'A{row}'] = "Технічні параметри"
    ws_summary[f'A{row}'].font = subheader_font
    row += 1

    if 'lead_analysis_data' in all_results:
        analysis = all_results['lead_analysis_data']
        basic_info = analysis.get('basic_info', {})

        tech_params = [
            ("Відведення:", all_results.get("main_lead_name", "N/A")),
            ("Частота дискретизації:", f"{all_results.get('sampling_rate_hz', 'N/A')} Гц"),
            ("Тривалість запису:",
             f"{analysis.get('basic_metrics', {}).get('total_beats', 0) / analysis.get('basic_metrics', {}).get('mean_hr', 60):.1f} хв"),
            ("Якість сигналу:", f"{analysis.get('signal_quality', {}).get('score', 0)}%")
        ]

        for label, value in tech_params:
            ws_summary[f'A{row}'] = label
            ws_summary[f'B{row}'] = value
            ws_summary[f'A{row}'].font = Font(bold=True)
            row += 1

    # Main findings
    row += 1
    ws_summary[f'A{row}'] = "Основні знахідки"
    ws_summary[f'A{row}'].font = subheader_font
    ws_summary.merge_cells(f'A{row}:F{row}')
    row += 1

    if 'lead_analysis_data' in all_results:
        clinical = all_results['lead_analysis_data'].get('clinical_interpretation', {})

        # Urgency
        urgency = clinical.get('urgency', 'routine')
        urgency_text = {
            'routine': 'Планове обстеження',
            'urgent': 'Потребує термінової консультації',
            'critical': 'КРИТИЧНИЙ СТАН - термінова допомога!'
        }.get(urgency, urgency)

        ws_summary[f'A{row}'] = "Терміновість:"
        ws_summary[f'B{row}'] = urgency_text
        ws_summary[f'A{row}'].font = Font(bold=True)

        if urgency == 'critical':
            ws_summary[f'B{row}'].fill = bad_fill
        elif urgency == 'urgent':
            ws_summary[f'B{row}'].fill = warning_fill
        else:
            ws_summary[f'B{row}'].fill = good_fill

        row += 2

        # Clinical findings
        findings = clinical.get('findings', [])
        if findings:
            # Headers
            headers = ['№', 'Категорія', 'Знахідка', 'Важливість', 'Впевненість', 'Деталі']
            for col, header in enumerate(headers, 1):
                cell = ws_summary.cell(row=row, column=col, value=header)
                cell.font = header_font_white
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")

            row += 1

            # Data
            for i, finding in enumerate(findings, 1):
                ws_summary.cell(row=row, column=1, value=i)
                ws_summary.cell(row=row, column=2, value=finding.get('category', ''))
                ws_summary.cell(row=row, column=3, value=finding.get('finding', ''))
                ws_summary.cell(row=row, column=4, value=finding.get('severity', ''))
                ws_summary.cell(row=row, column=5, value=f"{finding.get('confidence', 0)}%")
                ws_summary.cell(row=row, column=6, value=finding.get('details', ''))

                # Color code by severity
                severity = finding.get('severity', 'normal')
                if severity in ['critical', 'severe']:
                    for col in range(1, 7):
                        ws_summary.cell(row=row, column=col).fill = bad_fill
                elif severity in ['significant', 'moderate']:
                    for col in range(1, 7):
                        ws_summary.cell(row=row, column=col).fill = warning_fill

                row += 1

    # Adjust column widths
    column_widths = {'A': 25, 'B': 30, 'C': 40, 'D': 15, 'E': 15, 'F': 40}
    for col, width in column_widths.items():
        ws_summary.column_dimensions[col].width = width

    # Sheet 2: Detailed Metrics
    ws_metrics = wb.create_sheet("Детальні метрики")

    row = 1
    ws_metrics[f'A{row}'] = "Детальний аналіз параметрів ЕКГ"
    ws_metrics[f'A{row}'].font = Font(bold=True, size=14)
    ws_metrics.merge_cells(f'A{row}:E{row}')
    row += 2

    if 'lead_analysis_data' in all_results:
        analysis = all_results['lead_analysis_data']

        # Basic metrics
        ws_metrics[f'A{row}'] = "Базові метрики"
        ws_metrics[f'A{row}'].font = subheader_font
        ws_metrics.merge_cells(f'A{row}:E{row}')
        row += 1

        # Headers
        headers = ['Параметр', 'Значення', 'Норма', 'Статус', 'Коментар']
        for col, header in enumerate(headers, 1):
            cell = ws_metrics.cell(row=row, column=col, value=header)
            cell.font = header_font_white
            cell.fill = header_fill
        row += 1

        # Heart rate metrics
        basic = analysis.get('basic_metrics', {})
        if basic:
            metrics_data = [
                ('Середня ЧСС', f"{basic.get('mean_hr', 0):.1f} уд/хв",
                 f"{ECGConfig.NORMAL_HR_LOW_BPM}-{ECGConfig.NORMAL_HR_HIGH_BPM}",
                 _get_hr_status(basic.get('mean_hr', 0))),

                ('Мін. ЧСС', f"{basic.get('min_hr', 0):.1f} уд/хв",
                 f">{ECGConfig.BRADYCARDIA_BPM}",
                 _get_hr_status(basic.get('min_hr', 0))),

                ('Макс. ЧСС', f"{basic.get('max_hr', 0):.1f} уд/хв",
                 f"<{ECGConfig.TACHYCARDIA_BPM}",
                 _get_hr_status(basic.get('max_hr', 0))),

                ('Варіабельність ЧСС', f"{basic.get('std_hr', 0):.1f} уд/хв",
                 "5-15", ""),

                ('Середній RR', f"{basic.get('mean_rr', 0):.0f} мс",
                 "600-1000 мс", "")
            ]

            for metric, value, norm, status in metrics_data:
                ws_metrics.cell(row=row, column=1, value=metric)
                ws_metrics.cell(row=row, column=2, value=value)
                ws_metrics.cell(row=row, column=3, value=norm)
                ws_metrics.cell(row=row, column=4, value=status)

                # Color coding
                if 'Норма' in status:
                    ws_metrics.cell(row=row, column=4).fill = good_fill
                elif any(word in status for word in ['Тахікардія', 'Брадикардія', 'Критично']):
                    ws_metrics.cell(row=row, column=4).fill = bad_fill

                row += 1

        # HRV metrics
        if 'hrv' in analysis:
            row += 1
            ws_metrics[f'A{row}'] = "Варіабельність серцевого ритму (HRV)"
            ws_metrics[f'A{row}'].font = subheader_font
            ws_metrics.merge_cells(f'A{row}:E{row}')
            row += 1

            hrv = analysis['hrv']
            time_domain = hrv.get('time_domain', {})
            freq_domain = hrv.get('frequency_domain', {})

            hrv_metrics = [
                ('SDNN', f"{time_domain.get('sdnn', 0):.1f} мс",
                 f"{ECGConfig.NORMAL_SDNN_MIN_MS}-{ECGConfig.NORMAL_SDNN_MAX_MS} мс",
                 _get_hrv_status(time_domain.get('sdnn', 0), 'sdnn')),

                ('RMSSD', f"{time_domain.get('rmssd', 0):.1f} мс",
                 f">{ECGConfig.LOW_RMSSD_MS} мс",
                 _get_hrv_status(time_domain.get('rmssd', 0), 'rmssd')),

                ('pNN50', f"{time_domain.get('pnn50', 0):.1f}%",
                 ">5%", ""),

                ('LF/HF співвідношення', f"{freq_domain.get('lf_hf_ratio', 0):.2f}",
                 "0.5-2.0", _get_autonomic_balance(freq_domain.get('lf_hf_ratio', 0))),

                ('Загальна потужність', f"{freq_domain.get('total_power', 0):.0f} мс²",
                 ">1000 мс²", "")
            ]

            for metric, value, norm, status in hrv_metrics:
                ws_metrics.cell(row=row, column=1, value=metric)
                ws_metrics.cell(row=row, column=2, value=value)
                ws_metrics.cell(row=row, column=3, value=norm)
                ws_metrics.cell(row=row, column=4, value=status)
                row += 1

    # Adjust column widths
    for col in ['A', 'B', 'C', 'D', 'E']:
        ws_metrics.column_dimensions[col].width = 20

    # Sheet 3: Arrhythmias
    if 'lead_analysis_data' in all_results and 'arrhythmias' in all_results['lead_analysis_data']:
        ws_arrhythmias = wb.create_sheet("Аритмії")

        row = 1
        ws_arrhythmias[f'A{row}'] = "Детекція аритмій"
        ws_arrhythmias[f'A{row}'].font = Font(bold=True, size=14)
        ws_arrhythmias.merge_cells(f'A{row}:F{row}')
        row += 2

        arrhythmias = all_results['lead_analysis_data']['arrhythmias']

        # Summary
        summary_data = []

        # AF
        af = arrhythmias.get('atrial_fibrillation', {})
        if af.get('detected'):
            summary_data.append(('Фібриляція передсердь', 'Виявлено',
                                 f"{af.get('confidence', 0)}%", 'critical'))

        # Premature beats
        premature = arrhythmias.get('premature_beats', {})
        if premature.get('total', 0) > 0:
            summary_data.append((
                'Передчасні скорочення',
                f"PVC: {len(premature.get('pvcs', []))}, PAC: {len(premature.get('pacs', []))}",
                f"Навантаження: {premature.get('pvc_burden', 0):.1f}%",
                'warning' if premature.get('pvc_burden', 0) > 5 else 'info'
            ))

        # Pauses
        pauses = arrhythmias.get('pauses', {})
        if pauses.get('total', 0) > 0:
            summary_data.append((
                'Паузи',
                f"Кількість: {pauses['total']}",
                f"Макс: {pauses.get('max_pause', 0):.1f}с",
                'critical' if pauses.get('max_pause', 0) > 2.5 else 'warning'
            ))

        if summary_data:
            # Headers
            headers = ['Тип аритмії', 'Статус', 'Деталі', 'Важливість']
            for col, header in enumerate(headers, 1):
                cell = ws_arrhythmias.cell(row=row, column=col, value=header)
                cell.font = header_font_white
                cell.fill = header_fill
            row += 1

            # Data
            for arr_type, status, details, severity in summary_data:
                ws_arrhythmias.cell(row=row, column=1, value=arr_type)
                ws_arrhythmias.cell(row=row, column=2, value=status)
                ws_arrhythmias.cell(row=row, column=3, value=details)
                ws_arrhythmias.cell(row=row, column=4, value=severity)

                if severity == 'critical':
                    for col in range(1, 5):
                        ws_arrhythmias.cell(row=row, column=col).fill = bad_fill
                elif severity == 'warning':
                    for col in range(1, 5):
                        ws_arrhythmias.cell(row=row, column=col).fill = warning_fill

                row += 1

        # Burden chart
        if 'burden' in arrhythmias and arrhythmias['burden']:
            row += 2
            ws_arrhythmias[f'A{row}'] = "Навантаження аритміями (%)"
            ws_arrhythmias[f'A{row}'].font = subheader_font
            row += 1

            # Create data for chart
            chart_row_start = row + 1
            ws_arrhythmias[f'A{row}'] = "Тип"
            ws_arrhythmias[f'B{row}'] = "Відсоток"
            row += 1

            for arr_type, percentage in arrhythmias['burden'].items():
                ws_arrhythmias[f'A{row}'] = arr_type
                ws_arrhythmias[f'B{row}'] = percentage
                row += 1

            # Create pie chart
            if row > chart_row_start + 1:
                pie = openpyxl.chart.PieChart()
                labels = openpyxl.chart.Reference(ws_arrhythmias, min_col=1,
                                                  min_row=chart_row_start, max_row=row - 1)
                data = openpyxl.chart.Reference(ws_arrhythmias, min_col=2,
                                                min_row=chart_row_start - 1, max_row=row - 1)
                pie.add_data(data, titles_from_data=True)
                pie.set_categories(labels)
                pie.title = "Розподіл аритмій"
                ws_arrhythmias.add_chart(pie, f"D{chart_row_start}")

    # Sheet 4: Recommendations
    ws_recommendations = wb.create_sheet("Рекомендації")

    row = 1
    ws_recommendations[f'A{row}'] = "Клінічні рекомендації"
    ws_recommendations[f'A{row}'].font = Font(bold=True, size=14)
    ws_recommendations.merge_cells(f'A{row}:D{row}')
    row += 2

    if 'lead_analysis_data' in all_results:
        clinical = all_results['lead_analysis_data'].get('clinical_interpretation', {})

        # Recommendations
        recommendations = clinical.get('recommendations', [])
        if recommendations:
            ws_recommendations[f'A{row}'] = "Рекомендації:"
            ws_recommendations[f'A{row}'].font = subheader_font
            row += 1

            for i, rec in enumerate(recommendations, 1):
                ws_recommendations[f'A{row}'] = f"{i}."
                ws_recommendations[f'B{row}'] = rec
                ws_recommendations.merge_cells(f'B{row}:D{row}')
                row += 1

        # Risk factors
        row += 1
        risk_factors = clinical.get('risk_factors', [])
        if risk_factors:
            ws_recommendations[f'A{row}'] = "Фактори ризику:"
            ws_recommendations[f'A{row}'].font = subheader_font
            row += 1

            for risk in risk_factors:
                ws_recommendations[f'A{row}'] = "•"
                ws_recommendations[f'B{row}'] = risk.get('factor', '')
                ws_recommendations[f'C{row}'] = risk.get('reason', '')
                ws_recommendations[f'D{row}'] = risk.get('recommendation', '')
                row += 1

        # Follow-up
        row += 1
        follow_up = clinical.get('follow_up', [])
        if follow_up:
            ws_recommendations[f'A{row}'] = "План подальшого спостереження:"
            ws_recommendations[f'A{row}'].font = subheader_font
            row += 1

            for i, action in enumerate(follow_up, 1):
                ws_recommendations[f'A{row}'] = f"{i}."
                ws_recommendations[f'B{row}'] = action
                ws_recommendations.merge_cells(f'B{row}:D{row}')
                row += 1

    # Add images if available
    if 'lead_analysis_data' in all_results and 'visualizations' in all_results['lead_analysis_data']:
        ws_plots = wb.create_sheet("Графіки")

        from openpyxl.drawing.image import Image

        plots = all_results['lead_analysis_data']['visualizations']
        row = 1

        for plot_name, plot_path in plots.items():
            if os.path.exists(plot_path):
                try:
                    img = Image(plot_path)
                    # Scale image
                    img.width = 600
                    img.height = 400

                    ws_plots.add_image(img, f'A{row}')
                    row += 25  # Space for next image
                except Exception as e:
                    print(f"Не вдалося додати зображення {plot_name}: {e}")

    # Save workbook
    try:
        wb.save(excel_output_path)
        print(f"✅ Excel звіт збережено: {excel_output_path}")
        return True
    except Exception as e:
        print(f"❌ Помилка збереження Excel: {e}")
        return False


# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

def _get_hr_status(hr):
    """Get heart rate status description"""
    if hr < ECGConfig.EXTREME_BRADYCARDIA_BPM:
        return "Критична брадикардія"
    elif hr < ECGConfig.SEVERE_BRADYCARDIA_BPM:
        return "Виражена брадикардія"
    elif hr < ECGConfig.BRADYCARDIA_BPM:
        return "Брадикардія"
    elif hr < ECGConfig.NORMAL_HR_LOW_BPM:
        return "Нижня межа норми"
    elif hr <= ECGConfig.NORMAL_HR_HIGH_BPM:
        return "Норма"
    elif hr <= ECGConfig.TACHYCARDIA_BPM:
        return "Верхня межа норми"
    elif hr < ECGConfig.EXTREME_TACHYCARDIA_BPM:
        return "Тахікардія"
    else:
        return "Виражена тахікардія"


def _get_hrv_status(value, metric):
    """Get HRV metric status"""
    if metric == 'sdnn':
        if value < ECGConfig.VERY_LOW_SDNN_MS:
            return "Критично низька"
        elif value < ECGConfig.LOW_SDNN_MS:
            return "Знижена"
        elif value < ECGConfig.NORMAL_SDNN_MIN_MS:
            return "Нижня межа норми"
        elif value <= ECGConfig.NORMAL_SDNN_MAX_MS:
            return "Норма"
        elif value <= ECGConfig.HIGH_SDNN_MS:
            return "Підвищена"
        else:
            return "Значно підвищена"

    elif metric == 'rmssd':
        if value < ECGConfig.LOW_RMSSD_MS:
            return "Знижена парасимпатична активність"
        elif value < ECGConfig.NORMAL_RMSSD_MS:
            return "Норма"
        elif value < ECGConfig.HIGH_RMSSD_MS:
            return "Підвищена парасимпатична активність"
        else:
            return "Значно підвищена парасимпатична активність"

    return ""


def _get_autonomic_balance(lf_hf_ratio):
    """Get autonomic balance interpretation"""
    if lf_hf_ratio < 0.5:
        return "Парасимпатична домінантність"
    elif lf_hf_ratio < 1.0:
        return "Переважання парасимпатичної активності"
    elif lf_hf_ratio < 2.0:
        return "Збалансована активність"
    elif lf_hf_ratio < 3.0:
        return "Переважання симпатичної активності"
    else:
        return "Симпатична домінантність"


def analyze_ecg_xml_enhanced(xml_filepath):
    """
    Enhanced ECG XML analysis with all new features
    """
    print(f"\n{'=' * 60}")
    print(f"Розширений аналіз ЕКГ")
    print(f"Файл: {xml_filepath}")
    print(f"{'=' * 60}\n")

    overall_analysis_data = {
        "patient_id": "N/A",
        "patient_name": "N/A",
        "recording_date": datetime.now().strftime("%Y-%m-%d"),
        "sampling_rate_hz": None,
        "main_lead_name": "N/A",
        "main_lead_units": "N/A",
        "lead_analysis_data": None,
        "all_text_findings": [],
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "software_version": "2.0"
    }

    # Parse XML
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        overall_analysis_data["all_text_findings"].append(
            f"Помилка: Не вдалося розпарсити XML файл: {xml_filepath}. Деталі: {str(e)}"
        )
        return overall_analysis_data
    except FileNotFoundError:
        overall_analysis_data["all_text_findings"].append(
            f"Помилка: XML файл не знайдено: {xml_filepath}"
        )
        return overall_analysis_data

    # Extract patient info
    patient_info_el = root.find("PatientInfo")
    if patient_info_el is not None:
        overall_analysis_data["patient_id"] = patient_info_el.findtext("ID", "N/A")
        overall_analysis_data["patient_name"] = patient_info_el.findtext("Name", "N/A")

    # Extract acquisition details
    acq_details = root.find("AcquisitionDetails")
    sampling_freq_str = acq_details.findtext("SamplingFrequency", "N/A") if acq_details is not None else "N/A"

    try:
        sampling_rate = float(sampling_freq_str)
        if sampling_rate <= 0:
            raise ValueError("Частота дискретизації повинна бути додатною")
        overall_analysis_data["sampling_rate_hz"] = sampling_rate
    except (ValueError, TypeError) as e:
        overall_analysis_data["all_text_findings"].append(
            f"Критична помилка: Некоректна частота дискретизації ({sampling_freq_str})"
        )
        return overall_analysis_data

    # Get recording date/time if available
    if acq_details is not None:
        acq_datetime = acq_details.findtext("AcquisitionDateTime", "N/A")
        if acq_datetime != "N/A":
            try:
                # Try to parse date
                if "T" in acq_datetime:
                    date_part = acq_datetime.split("T")[0]
                    overall_analysis_data["recording_date"] = date_part
            except:
                pass

    # Find leads
    waveforms_data = root.find("WaveformData")
    if waveforms_data is None:
        overall_analysis_data["all_text_findings"].append("Попередження: Секція WaveformData відсутня в XML.")
        return overall_analysis_data

    leads = waveforms_data.findall("Lead")
    if not leads:
        overall_analysis_data["all_text_findings"].append("Попередження: Не знайдено відведень у WaveformData.")
        return overall_analysis_data

    # Extract R-peaks from annotations first
    r_peaks_from_xml = []
    annotations_elem = root.find("Annotations")
    if annotations_elem is not None:
        for ann in annotations_elem.findall("Annotation"):
            code = ann.findtext("Code", "")
            if code in ['R', 'N']:  # R-peak or Normal beat
                time_elem = ann.find("TimeOffset")
                if time_elem is not None:
                    try:
                        time_sec = float(time_elem.text)
                        sample_idx = int(time_sec * sampling_rate)
                        r_peaks_from_xml.append(sample_idx)
                    except:
                        pass

    if len(r_peaks_from_xml) > 0:
        print(f"Знайдено {len(r_peaks_from_xml)} R-піків в XML анотаціях")

    # Select lead for analysis
    lead_to_analyze = None
    preferred_leads = ["II", "MLII", "V5", "V2", "V1", "I", "III", "AVF", "AVL", "AVR", "V3", "V4", "V6"]

    for pref_name in preferred_leads:
        for lead_obj in leads:
            name_elem = lead_obj.find("Name")
            data_elem = lead_obj.find("Data")
            if name_elem is not None and name_elem.text and data_elem is not None and data_elem.text:
                if name_elem.text.strip().upper() == pref_name.upper():
                    lead_to_analyze = lead_obj
                    break
        if lead_to_analyze:
            break

    # If no preferred lead found, take first with data
    if not lead_to_analyze:
        for lead_obj in leads:
            data_elem = lead_obj.find("Data")
            if data_elem is not None and data_elem.text and data_elem.text.strip():
                lead_to_analyze = lead_obj
                break

    if lead_to_analyze:
        lead_name = lead_to_analyze.findtext("Name", "UnknownLead")
        signal_data_str = lead_to_analyze.findtext("Data", "")
        units = lead_to_analyze.findtext("SignalUnits", "N/A")

        overall_analysis_data["main_lead_name"] = lead_name
        overall_analysis_data["main_lead_units"] = units

        print(f"Аналіз відведення: {lead_name}")
        print(f"Одиниці виміру: {units}")
        print(f"Частота дискретизації: {sampling_rate} Гц\n")

        # Pass R-peaks from XML to analysis
        lead_analysis_result = analyze_ecg_comprehensive(
            sampling_rate, lead_name, signal_data_str, units,
            r_peaks_from_xml=r_peaks_from_xml if r_peaks_from_xml else None
        )

        overall_analysis_data["lead_analysis_data"] = lead_analysis_result

        # Compile text findings
        if lead_analysis_result.get("warnings"):
            overall_analysis_data["all_text_findings"].extend(lead_analysis_result["warnings"])

        # Add clinical interpretation summary
        if "clinical_interpretation" in lead_analysis_result:
            clinical = lead_analysis_result["clinical_interpretation"]
            overall_analysis_data["all_text_findings"].append(
                f"\nКЛІНІЧНИЙ ВИСНОВОК: {clinical.get('summary', 'N/A')}"
            )
            overall_analysis_data["all_text_findings"].append(
                f"Терміновість: {clinical.get('urgency', 'routine')}"
            )
    else:
        overall_analysis_data["all_text_findings"].append("Не знайдено жодного відведення з даними для аналізу.")

    print("\n" + "=" * 60)
    print("Аналіз завершено")
    print("=" * 60)

    return overall_analysis_data


def analyze_ecg_file_any_format(input_filepath, output_xml=None, output_excel=None, output_dir=None):
    """
    Аналізує ЕКГ файл будь-якого підтримуваного формату

    Args:
        input_filepath: шлях до вхідного файлу
        output_xml: шлях для збереження XML (опціонально)
        output_excel: шлях для збереження Excel звіту (опціонально)
        output_dir: директорія для збереження результатів (опціонально)

    Returns:
        dict з результатами аналізу
    """
    from parser import detect_file_format, convert_ecg_to_xml

    # Використовуємо глобальну директорію якщо не вказано іншу
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR

    # Створюємо директорію якщо не існує
    os.makedirs(output_dir, exist_ok=True)

    # Автоматичне визначення формату
    file_format = detect_file_format(input_filepath)

    if not file_format:
        # Спробуємо визначити за розширенням
        ext = os.path.splitext(input_filepath)[1].lower()
        format_map = {
            '.xml': 'xml',
            '.csv': 'csv',
            '.txt': 'csv',
            '.dcm': 'dicom',
            '.dicom': 'dicom',
            '.edf': 'edf',
            '.rec': 'edf',
            '.bdf': 'edf',
            '.dat': 'wfdb',
            '.hea': 'wfdb',
            '.scp': 'scp',
            '.ecg': 'scp'
        }
        file_format = format_map.get(ext)

        # Для XML потрібна додаткова перевірка
        if file_format == 'xml':
            try:
                with open(input_filepath, 'r', encoding='utf-8') as f:
                    content = f.read(1000)
                    if 'hl7' in content.lower():
                        file_format = 'hl7aecg'
                    elif 'philips' in content.lower():
                        file_format = 'philips_xml'
                    elif 'muse' in content.lower() or 'restingecg' in content.lower():
                        file_format = 'ge_muse'
                    else:
                        file_format = 'hl7aecg'
            except:
                pass

    if not file_format:
        print(f"❌ Не вдалося визначити формат файлу: {input_filepath}")
        return None

    print(f"✅ Визначено формат: {file_format}")

    # Генеруємо назви файлів в output директорії
    base_name = os.path.splitext(os.path.basename(input_filepath))[0]

    if not output_xml:
        output_xml = os.path.join(output_dir, f"{base_name}_converted.xml")
    if not output_excel:
        output_excel = os.path.join(output_dir, f"{base_name}_analysis.xlsx")

    # Спеціальні параметри для різних форматів
    kwargs = {}

    if file_format == 'csv':
        # Автовизначення параметрів CSV
        kwargs = detect_csv_parameters(input_filepath)
        print(f"   CSV параметри: роздільник='{kwargs.get('delimiter', ',')}', "
              f"частота={kwargs.get('sampling_rate_hz', 'auto')} Hz")

    # Крок 1: Конвертація в XML
    print(f"\n📄 Конвертація {file_format.upper()} → XML...")
    success = convert_ecg_to_xml(input_filepath, output_xml, file_format, **kwargs)

    if not success:
        print(f"❌ Помилка конвертації")
        return None

    print(f"✅ Конвертовано в: {output_xml}")

    # Крок 2: Аналіз
    print(f"\n🔬 Аналіз ЕКГ...")
    analysis_results = analyze_ecg_xml_enhanced(output_xml)

    # Крок 3: Створення Excel звіту
    if OPENPYXL_AVAILABLE and analysis_results.get('lead_analysis_data'):
        print(f"\n📊 Створення Excel звіту...")
        if create_enhanced_excel_report(analysis_results, output_excel):
            print(f"✅ Excel звіт: {output_excel}")

    # Крок 4: Створюємо текстовий звіт теж
    summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Аналіз ЕКГ: {os.path.basename(input_filepath)}\n")
        f.write(f"{'=' * 50}\n\n")

        if 'lead_analysis_data' in analysis_results:
            analysis = analysis_results['lead_analysis_data']

            # R-піки
            if 'r_peaks' in analysis:
                f.write(f"R-піків знайдено: {analysis['r_peaks'].get('count', 0)}\n")

            # Базові метрики
            if 'basic_metrics' in analysis:
                metrics = analysis['basic_metrics']
                f.write(f"Середня ЧСС: {metrics.get('mean_hr', 0):.0f} уд/хв\n")
                f.write(f"Кількість комплексів: {metrics.get('total_beats', 0)}\n\n")

            # HRV
            if 'hrv' in analysis and 'time_domain' in analysis['hrv']:
                hrv = analysis['hrv']['time_domain']
                f.write("HRV метрики:\n")
                f.write(f"  SDNN: {hrv.get('sdnn', 0):.1f} мс\n")
                f.write(f"  RMSSD: {hrv.get('rmssd', 0):.1f} мс\n")
                f.write(f"  pNN50: {hrv.get('pnn50', 0):.1f}%\n\n")

            # Клінічна інтерпретація
            if 'clinical_interpretation' in analysis:
                clinical = analysis['clinical_interpretation']
                f.write(f"Висновок: {clinical.get('summary', 'N/A')}\n")
                f.write(f"Терміновість: {clinical.get('urgency', 'routine')}\n")

    print(f"✅ Текстовий звіт: {summary_path}")

    return analysis_results


def detect_csv_parameters(csv_file):
    """Автоматичне визначення параметрів CSV файлу"""
    import csv

    params = {
        'delimiter': ',',
        'num_header_rows': 0,
        'lead_names_row': None,
        'data_start_row': 1,
        'sampling_rate_hz': None
    }

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Читаємо перші рядки
            sample_lines = []
            for i in range(20):
                line = f.readline()
                if not line:
                    break
                sample_lines.append(line)

            # Визначаємо роздільник
            sample = ''.join(sample_lines[:5])
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
                params['delimiter'] = dialect.delimiter
            except:
                # Рахуємо роздільники
                delimiters = [',', '\t', ';', '|']
                delimiter_counts = {d: sample.count(d) for d in delimiters}
                params['delimiter'] = max(delimiter_counts.items(), key=lambda x: x[1])[0]

            # Шукаємо рядок з назвами відведень
            ecg_terms = ['lead', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
                         'avr', 'avl', 'avf', 'time', 'ecg']

            for i, line in enumerate(sample_lines):
                line_lower = line.lower()
                if any(term in line_lower for term in ecg_terms):
                    params['lead_names_row'] = i
                    params['num_header_rows'] = i
                    params['data_start_row'] = i + 1
                    break

            # Шукаємо частоту дискретизації
            import re
            for line in sample_lines:
                # Шукаємо патерни типу "500 Hz", "500Hz", "sampling: 500"
                freq_match = re.search(r'(\d+)\s*[Hh][Zz]', line)
                if not freq_match:
                    freq_match = re.search(r'[Ss]ampling.*?(\d+)', line)
                if not freq_match:
                    freq_match = re.search(r'[Ff]requency.*?(\d+)', line)

                if freq_match:
                    params['sampling_rate_hz'] = float(freq_match.group(1))
                    break

            # Якщо не знайшли - спробуємо з назви файлу
            if not params['sampling_rate_hz']:
                filename = os.path.basename(csv_file)
                freq_match = re.search(r'(\d+)\s*[Hh][Zz]', filename)
                if not freq_match:
                    freq_match = re.search(r'_(\d+)_', filename)

                if freq_match:
                    freq = int(freq_match.group(1))
                    if 50 <= freq <= 2000:  # Розумний діапазон для ЕКГ
                        params['sampling_rate_hz'] = float(freq)

            # За замовчуванням 500 Hz якщо не знайшли
            if not params['sampling_rate_hz']:
                params['sampling_rate_hz'] = 500.0

    except Exception as e:
        print(f"Помилка при автовизначенні CSV параметрів: {e}")

    return params


def batch_analyze_directory(input_dir, output_dir=None, file_patterns=None):
    """
    Аналізує всі ЕКГ файли в директорії

    Args:
        input_dir: вхідна директорія
        output_dir: директорія для результатів (створить якщо не існує)
        file_patterns: список патернів файлів (наприклад ['*.csv', '*.dcm'])
    """
    import glob

    if not output_dir:
        output_dir = os.path.join(input_dir, 'analysis_results')

    os.makedirs(output_dir, exist_ok=True)

    # Знаходимо всі файли
    if not file_patterns:
        # Всі підтримувані розширення
        file_patterns = ['*.csv', '*.txt', '*.dcm', '*.dicom', '*.edf', '*.rec',
                         '*.bdf', '*.dat', '*.hea', '*.scp', '*.ecg', '*.xml']

    all_files = []
    for pattern in file_patterns:
        files = glob.glob(os.path.join(input_dir, '**', pattern), recursive=True)
        all_files.extend(files)

    # Видаляємо дублікати
    all_files = list(set(all_files))

    print(f"\nЗнайдено {len(all_files)} файлів для аналізу")

    results = {}
    successful = 0
    failed = 0

    for i, filepath in enumerate(all_files, 1):
        print(f"\n{'=' * 60}")
        print(f"Обробка {i}/{len(all_files)}: {os.path.basename(filepath)}")
        print(f"{'=' * 60}")

        try:
            # Створюємо підпапку для кожного файлу
            file_base = os.path.splitext(os.path.basename(filepath))[0]
            file_output_dir = os.path.join(output_dir, file_base)
            os.makedirs(file_output_dir, exist_ok=True)

            # Аналізуємо
            xml_path = os.path.join(file_output_dir, f"{file_base}.xml")
            excel_path = os.path.join(file_output_dir, f"{file_base}_analysis.xlsx")

            result = analyze_ecg_file_any_format(filepath, xml_path, excel_path, DEFAULT_OUTPUT_DIR)

            if result:
                results[filepath] = {'status': 'success', 'result': result}
                successful += 1

                # Зберігаємо резюме
                summary_path = os.path.join(file_output_dir, f"{file_base}_summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"Аналіз ЕКГ: {os.path.basename(filepath)}\n")
                    f.write(f"{'=' * 50}\n\n")

                    if 'lead_analysis_data' in result:
                        analysis = result['lead_analysis_data']

                        # Базова інформація
                        if 'basic_metrics' in analysis:
                            metrics = analysis['basic_metrics']
                            f.write(f"Середня ЧСС: {metrics.get('mean_hr', 0):.0f} уд/хв\n")
                            f.write(f"Кількість комплексів: {metrics.get('total_beats', 0)}\n\n")

                        # Клінічна інтерпретація
                        if 'clinical_interpretation' in analysis:
                            clinical = analysis['clinical_interpretation']
                            f.write(f"Висновок: {clinical.get('summary', 'N/A')}\n")
                            f.write(f"Терміновість: {clinical.get('urgency', 'routine')}\n\n")

                            findings = clinical.get('findings', [])
                            if findings:
                                f.write("Основні знахідки:\n")
                                for finding in findings[:5]:
                                    f.write(f"- {finding.get('finding', 'N/A')}\n")

            else:
                results[filepath] = {'status': 'failed', 'error': 'Analysis failed'}
                failed += 1

        except Exception as e:
            results[filepath] = {'status': 'failed', 'error': str(e)}
            failed += 1
            print(f"❌ Помилка: {e}")

    # Фінальний звіт
    print(f"\n{'=' * 60}")
    print(f"ПІДСУМОК АНАЛІЗУ")
    print(f"{'=' * 60}")
    print(f"Всього файлів: {len(all_files)}")
    print(f"Успішно: {successful}")
    print(f"Помилки: {failed}")
    print(f"Результати збережено в: {output_dir}")

    # Зберігаємо загальний звіт
    report_path = os.path.join(output_dir, 'batch_analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ====================================================================================
# MAIN FUNCTION
# ====================================================================================

# Замініть функцію main() в main.py на цю версію:

# Замініть функцію main() на цю версію:

def main():
    """Enhanced main function with support for all formats"""
    parser = argparse.ArgumentParser(
        description="Розширений конвертер та аналізатор ЕКГ файлів v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Підтримувані формати:
  - CSV/TXT - текстові файли з даними
  - DICOM - медичні зображення (.dcm, .dicom)  
  - EDF/BDF - European Data Format (.edf, .rec, .bdf)
  - WFDB - PhysioNet формат (.dat + .hea)
  - SCP-ECG - стандарт EN1064 (.scp, .ecg)
  - HL7 aECG - XML стандарт HL7
  - Philips XML - формат Philips 
  - GE MUSE XML - формат General Electric

Приклади:
  %(prog)s ecg_file.csv
  %(prog)s patient.dcm -o result.xml --excel_output report.xlsx
  %(prog)s -b /path/to/ecg/files/ --analyze-all
  %(prog)s -i  # інтерактивний режим
        """
    )

    parser.add_argument("input_filepath", nargs='?', default=None,
                        help="Шлях до вхідного ЕКГ файлу")

    parser.add_argument("-o", "--output_filepath", default=None,
                        help="Шлях для збереження XML файлу")

    parser.add_argument("-f", "--format", default=None,
                        choices=['dicom', 'wfdb', 'edf', 'hl7aecg', 'csv', 'scp',
                                 'philips_xml', 'ge_muse', 'auto'],
                        help="Формат вхідного файлу (auto = автовизначення)")

    parser.add_argument("--excel_output", default=None,
                        help="Шлях для збереження звіту в Excel")

    parser.add_argument("--skip_analysis", action="store_true",
                        help="Пропустити аналіз і лише конвертувати файл в XML")

    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Інтерактивний режим")

    parser.add_argument("-b", "--batch", action="store_true",
                        help="Пакетна обробка всіх файлів в директорії")

    parser.add_argument("--analyze-all", action="store_true",
                        help="Проаналізувати всі знайдені файли (для пакетного режиму)")

    # CSV specific arguments
    csv_group = parser.add_argument_group('CSV параметри')
    csv_group.add_argument("--csv_delimiter", default=',',
                           help="Роздільник для CSV файлів")
    csv_group.add_argument("--csv_num_header_rows", type=int, default=0,
                           help="Кількість рядків заголовку у CSV")
    csv_group.add_argument("--csv_lead_names_row", type=int, default=None,
                           help="Номер рядка з назвами відведень у CSV")
    csv_group.add_argument("--csv_data_start_row", type=int, default=1,
                           help="Номер рядка, з якого починаються дані у CSV")
    csv_group.add_argument("--csv_sampling_rate", type=float, default=None,
                           help="Частота дискретизації для CSV")

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 60)
    print("ECG Parser & Analyzer v2.0")
    print("Enhanced with AI-powered analysis")
    print("Підтримка всіх основних ЕКГ форматів")
    print("=" * 60 + "\n")

    # ПАКЕТНИЙ РЕЖИМ
    if args.batch or (args.input_filepath and os.path.isdir(args.input_filepath)):
        input_dir = args.input_filepath if args.input_filepath else '.'

        if not os.path.isdir(input_dir):
            print(f"❌ Директорія не існує: {input_dir}")
            return 1

        print(f"📁 Пакетна обробка директорії: {input_dir}")

        if args.analyze_all:
            # Аналізуємо всі підтримувані файли
            results = batch_analyze_directory(input_dir)
        else:
            # Просто показуємо що знайшли
            from parser import get_converter_info
            info = get_converter_info()

            all_extensions = []
            for fmt_info in info.values():
                all_extensions.extend(fmt_info['extensions'])

            import glob
            found_files = {}
            for ext in set(all_extensions):
                pattern = os.path.join(input_dir, '**', f'*{ext}')
                files = glob.glob(pattern, recursive=True)
                if files:
                    found_files[ext] = files

            if found_files:
                print("\nЗнайдені ЕКГ файли:")
                for ext, files in found_files.items():
                    print(f"\n{ext}: {len(files)} файлів")
                    for f in files[:3]:  # Показуємо перші 3
                        print(f"  - {os.path.relpath(f, input_dir)}")
                    if len(files) > 3:
                        print(f"  ... та ще {len(files) - 3}")

                print(f"\nВсього знайдено: {sum(len(f) for f in found_files.values())} файлів")
                print("\nДля аналізу всіх файлів використайте: --analyze-all")
            else:
                print("ЕКГ файли не знайдені")

        return 0

    # ІНТЕРАКТИВНИЙ РЕЖИМ
    if args.interactive or not args.input_filepath:
        print("🎯 ІНТЕРАКТИВНИЙ РЕЖИМ\n")

        # Показуємо доступні формати
        from parser import get_converter_info
        info = get_converter_info()

        print("Підтримувані формати:")
        for fmt_id, details in info.items():
            status = "✓" if details['available'] else "✗"
            print(f"  {status} {details['name']:<15} ({', '.join(details['extensions'])})")
            if not details['available'] and details['required_package']:
                print(f"     → Встановіть: pip install {details['required_package']}")
        print()

        # Отримати файл
        input_file_path = input("📁 Введіть шлях до ЕКГ файлу (або перетягніть файл сюди): ").strip().strip('"')
        if not input_file_path:
            print("❌ Шлях до файлу не введено.")
            return

        if not os.path.exists(input_file_path):
            print(f"❌ Файл не знайдено: {input_file_path}")
            return

        # Автоматичне визначення формату
        result = analyze_ecg_file_any_format(input_file_path)

        if result:
            print("\n✅ Аналіз завершено успішно!")
        else:
            print("\n❌ Аналіз не вдався")

        return 0

        # ЗВИЧАЙНИЙ РЕЖИМ
        if not args.input_filepath:
            parser.error("Потрібен вхідний файл (або використайте -i для інтерактивного режиму)")

        # Перевіряємо чи це директорія
        if os.path.isdir(args.input_filepath):
            print(f"📁 Це директорія. Використайте -b для пакетної обробки")
            return 1

        # Використовуємо універсальну функцію для всіх форматів
        output_xml = args.output_filepath
        output_excel = args.excel_output

        if args.skip_analysis:
            # Тільки конвертація - потрібно імпортувати
            from parser import detect_file_format, convert_ecg_to_xml

            file_format = args.format or detect_file_format(args.input_filepath)
            if not file_format:
                print("❌ Не вдалося визначити формат файлу")
                return 1

            if not output_xml:
                base_name = os.path.splitext(os.path.basename(args.input_filepath))[0]
                output_xml = f"{base_name}_converted.xml"

            print(f"Конвертація {file_format.upper()} → XML...")

            # CSV параметри
            csv_kwargs = {}
            if file_format == 'csv':
                csv_kwargs = {
                    'delimiter': args.csv_delimiter,
                    'num_header_rows': args.csv_num_header_rows,
                    'lead_names_row': args.csv_lead_names_row,
                    'data_start_row': args.csv_data_start_row,
                    'sampling_rate_hz': args.csv_sampling_rate
                }

            success = convert_ecg_to_xml(args.input_filepath, output_xml, file_format, **csv_kwargs)

            if success:
                print(f"✅ Конвертовано в: {output_xml}")
                return 0
            else:
                print("❌ Конвертація не вдалася")
                return 1
        else:
            # Повний аналіз - використовуємо універсальну функцію
            result = analyze_ecg_file_any_format(args.input_filepath, output_xml, output_excel)

            if result:
                print(f"\n✅ Аналіз завершено успішно!")
                return 0
            else:
                print(f"\n❌ Аналіз не вдався")
                return 1


if __name__ == "__main__":
    main()