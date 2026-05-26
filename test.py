#!/usr/bin/env python3
"""
Тест ECG з простими змінними для шляхів
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, detrend
import xml.etree.ElementTree as ET

INPUT_FILE = "C:/Users/Admin/Desktop/Курсова/Макаренко/TestDownloader/ecg_samples_new/csv/12lead_500hz.csv"
OUTPUT_DIR = "C:/Users/Admin/Desktop/Курсова/Макаренко/Project/TestsOutput"
SAMPLING_RATE = 500.0  # Hz
SHOW_PLOTS = True  # True = показувати графіки, False = тільки зберегти


# ==================================


def enhanced_r_peak_detection(signal, fs, visualize=False):
    """Покращена детекція R-піків"""
    print("\n=== ДЕТЕКЦІЯ R-ПІКІВ ===")

    # Видаляємо тренд
    signal_detrended = detrend(signal)

    # Фільтрація
    nyquist = 0.5 * fs
    b1, a1 = butter(2, [5.0 / nyquist, min(15.0 / nyquist, 0.95)], btype='band')
    filtered = filtfilt(b1, a1, signal_detrended)

    # Pan-Tompkins
    diff = np.diff(filtered)
    diff = np.append(diff, diff[-1])
    squared = diff ** 2

    window_size = int(0.15 * fs)
    window = np.ones(window_size) / window_size
    integrated = np.convolve(squared, window, mode='same')

    # Знаходимо піки
    all_peaks = []

    # Пробуємо різні пороги
    mean_val = np.mean(integrated)
    std_val = np.std(integrated)

    for factor in [2.0, 1.5, 1.0, 0.8, 0.6]:
        threshold = mean_val + factor * std_val
        peaks, _ = find_peaks(
            integrated,
            height=threshold,
            distance=int(0.3 * fs),
            prominence=threshold * 0.3
        )

        if 10 <= len(peaks) <= 200:
            print(f"  Знайдено {len(peaks)} піків (фактор {factor})")
            all_peaks = peaks
            break

    # Уточнюємо позиції
    final_peaks = []
    search_window = int(0.04 * fs)

    for peak in all_peaks:
        start = max(0, peak - search_window)
        end = min(len(signal), peak + search_window)

        if end > start:
            local_seg = signal[start:end]
            local_max = np.argmax(local_seg)
            refined_peak = start + local_max

            if not final_peaks or (refined_peak - final_peaks[-1]) >= int(0.3 * fs):
                final_peaks.append(refined_peak)

    r_peaks = np.array(final_peaks)

    print(f"\n  РЕЗУЛЬТАТ: {len(r_peaks)} R-піків")

    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs
        mean_hr = 60.0 / np.mean(rr_intervals)
        print(f"  Середня ЧСС: {mean_hr:.1f} уд/хв")

    # Візуалізація
    if visualize and len(r_peaks) > 0:
        time = np.arange(len(signal)) / fs

        plt.figure(figsize=(15, 10))

        # Верхній графік - повний сигнал
        plt.subplot(2, 1, 1)
        plt.plot(time, signal, 'b-', linewidth=0.5)
        plt.plot(time[r_peaks], signal[r_peaks], 'ro', markersize=8)
        plt.title(f'ЕКГ сигнал ({len(r_peaks)} R-піків)')
        plt.xlabel('Час (с)')
        plt.ylabel('Амплітуда (мВ)')
        plt.grid(True, alpha=0.3)

        # Нижній графік - перші 10 секунд
        plt.subplot(2, 1, 2)
        zoom_end = min(int(10 * fs), len(signal))
        plt.plot(time[:zoom_end], signal[:zoom_end], 'b-', linewidth=1)

        for peak in r_peaks:
            if peak < zoom_end:
                plt.plot(time[peak], signal[peak], 'ro', markersize=10)
                plt.axvline(x=time[peak], color='r', linestyle='--', alpha=0.3)

        plt.title('Перші 10 секунд (детально)')
        plt.xlabel('Час (с)')
        plt.ylabel('Амплітуда (мВ)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Зберігаємо
        plot_file = os.path.join(OUTPUT_DIR, 'r_peaks_detection.png')
        plt.savefig(plot_file, dpi=150)
        print(f"\n  Графік збережено: {plot_file}")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()

    return r_peaks


def main():
    """Головна функція"""

    print(f"\n{'=' * 70}")
    print(f"ТЕСТ ECG PIPELINE")
    print(f"{'=' * 70}")
    print(f"Вхідний файл: {INPUT_FILE}")
    print(f"Вихідна папка: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    # Перевіряємо чи існує файл
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ПОМИЛКА: Файл не знайдено!")
        print(f"   Шлях: {INPUT_FILE}")
        print(f"\n   Вкажіть правильний шлях у змінній INPUT_FILE")
        return

    # Створюємо вихідну папку якщо не існує
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Визначаємо назви файлів
    base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    xml_file = os.path.join(OUTPUT_DIR, f"{base_name}_output.xml")
    excel_file = os.path.join(OUTPUT_DIR, f"{base_name}_report.xlsx")

    # 1. Читаємо CSV
    print("\n1. ЧИТАННЯ CSV")
    try:
        df = pd.read_csv(INPUT_FILE)
        lead_cols = [col for col in df.columns if col.lower() not in ['time', 'час', 'seconds']]

        if not lead_cols:
            print("❌ Не знайдено колонок з даними ЕКГ")
            return

        signal = df[lead_cols[0]].values
        print(f"✓ Прочитано {len(signal)} семплів з відведення {lead_cols[0]}")
        print(f"  Діапазон: [{signal.min():.3f}, {signal.max():.3f}]")

    except Exception as e:
        print(f"❌ Помилка читання CSV: {e}")
        return

    # 2. Детекція R-піків
    print("\n2. ДЕТЕКЦІЯ R-ПІКІВ")
    r_peaks = enhanced_r_peak_detection(signal, SAMPLING_RATE, visualize=True)

    if len(r_peaks) == 0:
        print("❌ R-піки не знайдено!")
        return

    # 3. Розрахунок HRV
    print("\n3. РОЗРАХУНОК HRV")
    if len(r_peaks) > 5:
        rr_intervals = np.diff(r_peaks) / SAMPLING_RATE

        # Фільтруємо артефакти
        valid_rr = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]

        if len(valid_rr) > 2:
            sdnn = np.std(valid_rr) * 1000
            rmssd = np.sqrt(np.mean(np.diff(valid_rr) ** 2)) * 1000
            nn50 = np.sum(np.abs(np.diff(valid_rr) * 1000) > 50)
            pnn50 = nn50 / len(np.diff(valid_rr)) * 100 if len(valid_rr) > 1 else 0

            print(f"✓ HRV метрики:")
            print(f"  SDNN:  {sdnn:.1f} мс")
            print(f"  RMSSD: {rmssd:.1f} мс")
            print(f"  pNN50: {pnn50:.1f}%")

            # Зберігаємо результати в текстовий файл
            results_file = os.path.join(OUTPUT_DIR, f"{base_name}_results.txt")
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write(f"РЕЗУЛЬТАТИ АНАЛІЗУ ЕКГ\n")
                f.write(f"=====================\n\n")
                f.write(f"Файл: {INPUT_FILE}\n")
                f.write(f"Дата аналізу: {pd.Timestamp.now()}\n\n")
                f.write(f"R-ПІКИ:\n")
                f.write(f"  Кількість: {len(r_peaks)}\n")
                f.write(f"  Середня ЧСС: {60 / np.mean(rr_intervals):.1f} уд/хв\n\n")
                f.write(f"HRV МЕТРИКИ:\n")
                f.write(f"  SDNN:  {sdnn:.1f} мс\n")
                f.write(f"  RMSSD: {rmssd:.1f} мс\n")
                f.write(f"  pNN50: {pnn50:.1f}%\n")

            print(f"\n✓ Результати збережено: {results_file}")

    # 4. Конвертація в XML
    print("\n4. КОНВЕРТАЦІЯ В XML")

    # Додаємо шлях до parser.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    try:
        from parser import convert_ecg_to_xml

        success = convert_ecg_to_xml(
            INPUT_FILE,
            xml_file,
            'csv',
            sampling_rate_hz=SAMPLING_RATE
        )

        if success:
            print(f"✓ XML створено: {xml_file}")

            # Перевіряємо R-піки в XML
            tree = ET.parse(xml_file)
            root = tree.getroot()

            annotations = root.find('.//Annotations')
            xml_r_peaks = 0

            if annotations is not None:
                for ann in annotations.findall('Annotation'):
                    if ann.findtext('Code') in ['R', 'N']:
                        xml_r_peaks += 1

            print(f"  R-піків в XML: {xml_r_peaks}")

            if xml_r_peaks == 0:
                print("  ⚠️ УВАГА: R-піки не записались в XML!")
        else:
            print("❌ Конвертація не вдалась")

    except Exception as e:
        print(f"❌ Помилка конвертації: {e}")

    # 5. Створення Excel звіту
    print("\n5. СТВОРЕННЯ EXCEL ЗВІТУ")
    try:
        from main import analyze_ecg_xml_enhanced, create_enhanced_excel_report

        if os.path.exists(xml_file):
            results = analyze_ecg_xml_enhanced(xml_file)

            if results and 'lead_analysis_data' in results:
                if create_enhanced_excel_report(results, excel_file):
                    print(f"✓ Excel звіт створено: {excel_file}")
                else:
                    print("❌ Не вдалось створити Excel")
    except Exception as e:
        print(f"❌ Помилка створення Excel: {e}")

    print("\n" + "=" * 70)
    print("ТЕСТ ЗАВЕРШЕНО")
    print(f"Всі результати збережено в: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()