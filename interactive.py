#!/usr/bin/env python3
"""
Enhanced ECG Parser and Analyzer - Interactive Mode
Version 2.0
"""

import os
import sys
from datetime import datetime
from parser import convert_ecg_to_xml, detect_file_format
from pathlib import Path


def clear_screen():
    """Очистити екран"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Вивести заголовок"""
    print("=" * 60)
    print("🫀 ECG Parser & Analyzer v2.0 - Інтерактивний режим")
    print("=" * 60)
    print()


def print_menu():
    """Головне меню"""
    print("\n📋 ГОЛОВНЕ МЕНЮ:")
    print("1. 📄 Конвертувати один файл")
    print("2. 📁 Конвертувати всі файли в папці")
    print("3. 🔍 Конвертувати + Аналіз")
    print("4. ℹ️  Інформація про формати")
    print("5. ❌ Вихід")
    print()


def get_file_format(filepath):
    """Визначити формат файлу"""
    # Спробувати автовизначення
    detected = detect_file_format(filepath)
    if detected:
        return detected

    # Якщо не вдалося, запитати користувача
    print("\n❓ Не вдалося автоматично визначити формат.")
    print("Виберіть формат:")
    print("1. CSV")
    print("2. DICOM")
    print("3. EDF/EDF+")
    print("4. WFDB")
    print("5. HL7 XML")
    print("6. SCP-ECG")

    choice = input("\nВаш вибір (1-6): ").strip()

    format_map = {
        '1': 'csv',
        '2': 'dicom',
        '3': 'edf',
        '4': 'wfdb',
        '5': 'hl7aecg',
        '6': 'scp'
    }

    return format_map.get(choice, 'csv')


def get_csv_parameters():
    """Отримати параметри для CSV файлу"""
    print("\n📊 Параметри CSV файлу:")

    # Роздільник
    print("\nРоздільник:")
    print("1. Кома (,)")
    print("2. Табуляція (\\t)")
    print("3. Крапка з комою (;)")
    print("4. Пробіл")
    print("5. Інший")

    delim_choice = input("Виберіть роздільник (1-5) [1]: ").strip() or '1'

    delimiters = {
        '1': ',',
        '2': '\t',
        '3': ';',
        '4': ' ',
    }

    if delim_choice == '5':
        delimiter = input("Введіть роздільник: ")
    else:
        delimiter = delimiters.get(delim_choice, ',')

    # Частота дискретизації
    print("\n💓 Частота дискретизації (Hz):")
    print("1. 250 Hz")
    print("2. 500 Hz")
    print("3. 1000 Hz")
    print("4. Інша")

    freq_choice = input("Виберіть частоту (1-4) [2]: ").strip() or '2'

    frequencies = {
        '1': 250,
        '2': 500,
        '3': 1000
    }

    if freq_choice == '4':
        try:
            sampling_rate = float(input("Введіть частоту (Hz): "))
        except:
            sampling_rate = 500
    else:
        sampling_rate = frequencies.get(freq_choice, 500)

    # Додаткові параметри
    print("\n📝 Додаткові параметри (Enter для пропуску):")

    header_rows = input("Кількість рядків заголовку [0]: ").strip()
    header_rows = int(header_rows) if header_rows else 0

    lead_names_row = input("Номер рядка з назвами відведень (або Enter): ").strip()
    lead_names_row = int(lead_names_row) if lead_names_row else None

    data_start_row = input(f"З якого рядка починаються дані [{header_rows + 1}]: ").strip()
    data_start_row = int(data_start_row) if data_start_row else header_rows + 1

    return {
        'delimiter': delimiter,
        'sampling_rate_hz': sampling_rate,
        'num_header_rows': header_rows,
        'lead_names_row': lead_names_row,
        'data_start_row': data_start_row
    }


def convert_single_file():
    """Конвертувати один файл"""
    clear_screen()
    print_header()
    print("🔄 КОНВЕРТАЦІЯ ОДНОГО ФАЙЛУ\n")

    # Вибір файлу
    filepath = input("Введіть шлях до файлу (або перетягніть файл): ").strip().strip('"')

    if not os.path.exists(filepath):
        print(f"\n❌ Файл не знайдено: {filepath}")
        input("\nНатисніть Enter для продовження...")
        return

    # Визначити формат
    file_format = get_file_format(filepath)
    print(f"\n✅ Формат: {file_format.upper()}")

    # Отримати параметри для CSV
    kwargs = {}
    if file_format == 'csv':
        kwargs = get_csv_parameters()

    # Вихідний файл
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_file = f"{base_name}_converted.xml"

    custom_output = input(f"\nВихідний файл [{output_file}]: ").strip()
    if custom_output:
        output_file = custom_output

    # Конвертація
    print(f"\n🔄 Конвертую {filepath} -> {output_file}...")

    try:
        success = convert_ecg_to_xml(filepath, output_file, file_format, **kwargs)

        if success:
            print("✅ Конвертація успішна!")

            # Запитати про аналіз
            if input("\nВиконати аналіз? (y/n) [n]: ").lower() == 'y':
                analyze_file(output_file)
        else:
            print("❌ Конвертація не вдалася!")

    except Exception as e:
        print(f"❌ Помилка: {e}")

    input("\nНатисніть Enter для продовження...")


def convert_batch():
    """Конвертувати всі файли в папці"""
    clear_screen()
    print_header()
    print("📁 ПАКЕТНА КОНВЕРТАЦІЯ\n")

    # Вибір папки
    folder = input("Введіть шлях до папки: ").strip().strip('"')

    if not os.path.isdir(folder):
        print(f"\n❌ Папка не знайдена: {folder}")
        input("\nНатисніть Enter для продовження...")
        return

    # Вибір формату
    print("\nВиберіть формат файлів:")
    print("1. CSV файли")
    print("2. DICOM файли")
    print("3. EDF файли")
    print("4. WFDB файли")
    print("5. Всі підтримувані формати")

    format_choice = input("\nВаш вибір (1-5) [5]: ").strip() or '5'

    patterns = {
        '1': ('*.csv', 'csv'),
        '2': ('*.dcm', 'dicom'),
        '3': ('*.edf', 'edf'),
        '4': ('*.dat', 'wfdb'),
        '5': ('*', None)
    }

    pattern, file_format = patterns.get(format_choice, ('*', None))

    # Знайти файли
    files = list(Path(folder).glob(pattern))

    if not files:
        print(f"\n❌ Не знайдено файлів за шаблоном {pattern}")
        input("\nНатисніть Enter для продовження...")
        return

    print(f"\n📊 Знайдено файлів: {len(files)}")

    # Параметри для CSV
    kwargs = {}
    if file_format == 'csv':
        print("\nПараметри будуть застосовані до всіх CSV файлів:")
        kwargs = get_csv_parameters()

    # Конвертація
    output_folder = input("\nПапка для результатів [converted]: ").strip() or "converted"
    os.makedirs(output_folder, exist_ok=True)

    success_count = 0

    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Конвертую {file.name}...")

        output_file = os.path.join(output_folder, f"{file.stem}_converted.xml")

        try:
            # Визначити формат якщо не вказано
            current_format = file_format or get_file_format(str(file))

            success = convert_ecg_to_xml(str(file), output_file, current_format, **kwargs)

            if success:
                print(f"  ✅ Успішно")
                success_count += 1
            else:
                print(f"  ❌ Помилка")

        except Exception as e:
            print(f"  ❌ Помилка: {e}")

    print(f"\n📊 Результат: {success_count}/{len(files)} файлів конвертовано")
    input("\nНатисніть Enter для продовження...")


def convert_and_analyze():
    """Конвертувати та проаналізувати"""
    clear_screen()
    print_header()
    print("🔍 КОНВЕРТАЦІЯ + АНАЛІЗ\n")

    # Вибір файлу
    filepath = input("Введіть шлях до файлу: ").strip().strip('"')

    if not os.path.exists(filepath):
        print(f"\n❌ Файл не знайдено: {filepath}")
        input("\nНатисніть Enter для продовження...")
        return

    # Визначити формат
    file_format = get_file_format(filepath)
    print(f"\n✅ Формат: {file_format.upper()}")

    # Отримати параметри для CSV
    kwargs = {}
    if file_format == 'csv':
        kwargs = get_csv_parameters()

    # Конвертація
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    xml_file = f"{base_name}_converted.xml"

    print(f"\n🔄 Конвертую {filepath} -> {xml_file}...")

    try:
        success = convert_ecg_to_xml(filepath, xml_file, file_format, **kwargs)

        if success:
            print("✅ Конвертація успішна!")

            # Аналіз
            print("\n🔬 Виконую аналіз...")

            from main import analyze_ecg_xml_enhanced, create_enhanced_excel_report

            results = analyze_ecg_xml_enhanced(xml_file)

            # Зберегти звіт
            excel_file = f"{base_name}_analysis.xlsx"
            if create_enhanced_excel_report(results, excel_file):
                print(f"✅ Звіт збережено: {excel_file}")

            # Показати основні результати
            if 'lead_analysis_data' in results:
                clinical = results['lead_analysis_data'].get('clinical_interpretation', {})
                if clinical:
                    print(f"\n📋 КЛІНІЧНИЙ ВИСНОВОК:")
                    print(f"   {clinical.get('summary', 'N/A')}")
                    print(f"   Терміновість: {clinical.get('urgency', 'routine')}")

                    findings = clinical.get('findings', [])[:3]
                    if findings:
                        print("\n   Основні знахідки:")
                        for i, finding in enumerate(findings, 1):
                            print(f"   {i}. {finding.get('finding', 'N/A')}")
        else:
            print("❌ Конвертація не вдалася!")

    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()

    input("\nНатисніть Enter для продовження...")


def analyze_file(xml_file):
    """Проаналізувати XML файл"""
    print("\n🔬 Виконую аналіз...")

    try:
        from main import analyze_ecg_xml_enhanced, create_enhanced_excel_report

        results = analyze_ecg_xml_enhanced(xml_file)

        # Зберегти звіт
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        excel_file = f"{base_name}_analysis.xlsx"

        if create_enhanced_excel_report(results, excel_file):
            print(f"✅ Звіт збережено: {excel_file}")

    except Exception as e:
        print(f"❌ Помилка аналізу: {e}")


def show_formats_info():
    """Показати інформацію про формати"""
    clear_screen()
    print_header()
    print("ℹ️ ПІДТРИМУВАНІ ФОРМАТИ\n")

    from parser import get_converter_info

    info = get_converter_info()

    for format_id, details in info.items():
        status = "✅" if details['available'] else "❌"
        print(f"{status} {details['name']:<15} {', '.join(details['extensions']):<20}")
        print(f"   {details['description']}")
        if not details['available'] and details['required_package']:
            print(f"   ⚠️  Потрібен пакет: pip install {details['required_package']}")
        print()

    input("\nНатисніть Enter для продовження...")


def interactive_mode():
    """Головний інтерактивний режим"""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("Ваш вибір (1-5): ").strip()

        if choice == '1':
            convert_single_file()
        elif choice == '2':
            convert_batch()
        elif choice == '3':
            convert_and_analyze()
        elif choice == '4':
            show_formats_info()
        elif choice == '5':
            print("\n👋 До побачення!")
            break
        else:
            print("\n❌ Невірний вибір!")
            input("\nНатисніть Enter для продовження...")


# Запуск інтерактивного режиму
if __name__ == "__main__":
    # Якщо запущено без параметрів - інтерактивний режим
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        # Інакше - звичайний режим з параметрами
        from main import main

        main()