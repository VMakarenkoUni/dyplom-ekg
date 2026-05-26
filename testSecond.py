"""
Альтернативна реалізація детекції R-піків для тестування
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, detrend
import matplotlib.pyplot as plt


def alternative_r_peak_detection(signal, fs, debug=True):
    """
    Альтернативний метод детекції R-піків для мВ даних
    """
    if debug:
        print("\nАЛЬТЕРНАТИВНА ДЕТЕКЦІЯ R-ПІКІВ")
        print("-" * 40)

    # 1. Видаляємо тренд
    signal_detrended = detrend(signal)

    # 2. Фільтрація 5-20 Hz
    nyquist = fs / 2
    low = 5 / nyquist
    high = 20 / nyquist

    if low < 1 and high < 1:
        b, a = butter(2, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_detrended)
    else:
        filtered = signal_detrended

    # 3. Визначаємо поріг
    # Для мВ даних використовуємо фіксований поріг
    threshold = 0.1  # 100 мкВ

    # Альтернативний поріг - на основі статистики
    signal_std = np.std(filtered)
    signal_mean = np.mean(np.abs(filtered))
    adaptive_threshold = signal_mean + signal_std

    # Використовуємо більший поріг
    final_threshold = max(threshold, adaptive_threshold)

    if debug:
        print(f"Фіксований поріг: {threshold:.3f} мВ")
        print(f"Адаптивний поріг: {adaptive_threshold:.3f} мВ")
        print(f"Фінальний поріг: {final_threshold:.3f} мВ")

    # 4. Знаходимо піки
    # Мінімальна відстань 0.5 сек (макс 120 уд/хв)
    min_distance = int(0.5 * fs)

    peaks, properties = find_peaks(
        filtered,
        height=final_threshold,
        distance=min_distance,
        prominence=final_threshold * 0.5
    )

    if debug:
        print(f"\nЗнайдено {len(peaks)} піків після фільтрації")

    # 5. Якщо знайдено мало піків, пробуємо інший підхід
    if len(peaks) < 5:
        if debug:
            print("Мало піків, пробуємо альтернативний підхід...")

        # Шукаємо піки в оригінальному сигналі
        peaks2, _ = find_peaks(
            signal,
            height=0.05,  # 50 мкВ для оригінального сигналу
            distance=min_distance
        )

        # Беремо найвищі піки
        if len(peaks2) > 0:
            peak_heights = signal[peaks2]
            # Сортуємо за висотою
            sorted_indices = np.argsort(peak_heights)[::-1]

            # Беремо топ N піків
            expected_peaks = int(len(signal) / fs * 1.2)  # ~72 уд/хв
            n_peaks = min(expected_peaks, len(peaks2))

            top_peak_indices = sorted_indices[:n_peaks]
            peaks = peaks2[top_peak_indices]
            peaks = np.sort(peaks)  # Сортуємо за часом

            if debug:
                print(f"Знайдено {len(peaks)} піків в оригінальному сигналі")

    # 6. Фінальна перевірка
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        avg_hr = 60 / np.mean(rr_intervals)

        if debug:
            print(f"\nСередня ЧСС: {avg_hr:.1f} уд/хв")

        # Якщо ЧСС нереалістична
        if avg_hr < 40 or avg_hr > 150:
            if debug:
                print("⚠️ Нереалістична ЧСС!")

    return peaks


def test_alternative_detection():
    """Тестування альтернативної детекції"""
    import csv

    print("ТЕСТ АЛЬТЕРНАТИВНОЇ ДЕТЕКЦІЇ")
    print("=" * 50)

    with open('C:/Users/Admin/Desktop/Курсова/Макаренко/TestDownloader/ecg_samples_new/csv/12lead_500hz.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    lead_ii = np.array([float(row['II']) for row in data])
    fs = 500

    print(f"Завантажено {len(lead_ii)} точок ({len(lead_ii) / fs:.1f} сек)")
    print(f"Діапазон: {np.min(lead_ii):.3f} до {np.max(lead_ii):.3f} мВ")

    # Альтернативна детекція
    peaks = alternative_r_peak_detection(lead_ii, fs, debug=True)

    print(f"\nРЕЗУЛЬТАТ: Знайдено {len(peaks)} R-піків")

    if len(peaks) > 1:
        # Розраховуємо метрики
        rr_intervals = np.diff(peaks) / fs * 1000  # в мс

        print(f"\nHRV метрики:")
        print(f"- SDNN: {np.std(rr_intervals):.1f} мс")
        print(f"- RMSSD: {np.sqrt(np.mean(np.diff(rr_intervals) ** 2)):.1f} мс")
        print(f"- pNN50: {np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100:.1f}%")

    # Візуалізація
    plt.figure(figsize=(15, 10))

    # Графік 1: Повний сигнал
    plt.subplot(3, 1, 1)
    time = np.arange(len(lead_ii)) / fs
    plt.plot(time, lead_ii, 'b-', linewidth=0.5)
    if len(peaks) > 0:
        plt.plot(time[peaks], lead_ii[peaks], 'ro', markersize=4)
    plt.title(f'Повний сигнал ({len(peaks)} R-піків)')
    plt.ylabel('мВ')
    plt.grid(True, alpha=0.3)

    # Графік 2: Перші 10 секунд
    plt.subplot(3, 1, 2)
    end_idx = int(10 * fs)
    plt.plot(time[:end_idx], lead_ii[:end_idx], 'b-', linewidth=1)
    segment_peaks = peaks[peaks < end_idx]
    if len(segment_peaks) > 0:
        plt.plot(time[segment_peaks], lead_ii[segment_peaks], 'ro', markersize=6)
        for i, peak in enumerate(segment_peaks):
            plt.text(time[peak], lead_ii[peak] + 0.02, f'{i + 1}', ha='center', fontsize=8)
    plt.title('Перші 10 секунд (детально)')
    plt.ylabel('мВ')
    plt.grid(True, alpha=0.3)

    # Графік 3: Гістограма RR інтервалів
    if len(peaks) > 1:
        plt.subplot(3, 1, 3)
        rr_intervals = np.diff(peaks) / fs * 1000  # в мс
        plt.hist(rr_intervals, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(rr_intervals), color='r', linestyle='--',
                    label=f'Середнє: {np.mean(rr_intervals):.0f} мс')
        plt.xlabel('RR інтервал (мс)')
        plt.ylabel('Кількість')
        plt.title('Розподіл RR інтервалів')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('alternative_detection_result.png', dpi=150)
    print(f"\nГрафік збережено: alternative_detection_result.png")
    plt.show()


if __name__ == "__main__":
    test_alternative_detection()