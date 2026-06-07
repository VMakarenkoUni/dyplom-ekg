#!/usr/bin/env python3
"""
Enhanced ECG Parser v2.0
Supports multiple ECG formats with advanced features
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import sys
import struct
import csv
import json
import gzip
import zipfile
from datetime import datetime
import traceback
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
import concurrent.futures
from dataclasses import dataclass
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports with availability flags
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy не встановлено. Деякі функції будуть обмежені.")

try:
    import pydicom
    from pydicom.errors import InvalidDicomError

    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.info("pydicom не встановлено. DICOM підтримка недоступна.")

try:
    import wfdb

    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    logger.info("wfdb не встановлено. WFDB підтримка недоступна.")

try:
    from pyedflib import EdfReader

    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False
    logger.info("pyedflib не встановлено. EDF підтримка недоступна.")

try:
    from lxml import etree as lxml_etree

    LXML_AVAILABLE = True
except ImportError:
    lxml_etree = None
    LXML_AVAILABLE = False
    logger.info("lxml не встановлено. HL7 aECG підтримка може бути обмежена.")

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    logger.info("MNE не встановлено. Розширена EDF/BDF підтримка недоступна.")


# ====================================================================================
# DATA CLASSES
# ====================================================================================

@dataclass
class ECGAnnotation:
    """ECG annotation/event"""
    time_sec: float
    code: str
    description: str
    confidence: float = 1.0
    lead: Optional[str] = None


@dataclass
class ECGLead:
    """Single ECG lead data"""
    name: str
    data: List[float]
    units: str = "uV"
    gain: float = 1.0
    baseline: float = 0.0
    sampling_rate: float = 0.0

    def get_duration(self) -> float:
        """Get lead duration in seconds"""
        if self.sampling_rate > 0:
            return len(self.data) / self.sampling_rate
        return 0.0

    def get_amplitude_range(self) -> Tuple[float, float]:
        """Get min and max amplitude"""
        if not self.data:
            return (0.0, 0.0)
        return (min(self.data), max(self.data))


@dataclass
class ECGMetadata:
    """Metadata for ECG recording"""
    patient_id: str = "N/A"
    patient_name: str = "N/A"
    birth_date: str = "N/A"
    sex: str = "N/A"
    age: str = "N/A"

    recording_date: str = "N/A"
    recording_time: str = "N/A"
    device: str = "N/A"
    institution: str = "N/A"
    technician: str = "N/A"

    sampling_rate: float = 0.0
    num_leads: int = 0
    duration_sec: float = 0.0

    filters: Dict[str, Any] = None
    medications: List[str] = None
    diagnoses: List[str] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.medications is None:
            self.medications = []
        if self.diagnoses is None:
            self.diagnoses = []
@dataclass
class ECGMetadata:
    """Metadata for ECG recording"""
    patient_id: str = "N/A"
    patient_name: str = "N/A"
    birth_date: str = "N/A"
    sex: str = "N/A"
    age: str = "N/A"

    recording_date: str = "N/A"
    recording_time: str = "N/A"
    device: str = "N/A"
    institution: str = "N/A"
    technician: str = "N/A"

    sampling_rate: float = 0.0
    num_leads: int = 0
    duration_sec: float = 0.0

    filters: Dict[str, Any] = None
    medications: List[str] = None
    diagnoses: List[str] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.medications is None:
            self.medications = []
        if self.diagnoses is None:
            self.diagnoses = []


@dataclass
class ECGLead:
    """Single ECG lead data"""
    name: str
    data: List[float]
    units: str = "uV"
    gain: float = 1.0
    baseline: float = 0.0
    sampling_rate: float = 0.0

    def get_duration(self) -> float:
        """Get lead duration in seconds"""
        if self.sampling_rate > 0:
            return len(self.data) / self.sampling_rate
        return 0.0

    def get_amplitude_range(self) -> Tuple[float, float]:
        """Get min and max amplitude"""
        if not self.data:
            return (0.0, 0.0)
        return (min(self.data), max(self.data))


@dataclass
class ECGAnnotation:
    """ECG annotation/event"""
    time_sec: float
    code: str
    description: str
    confidence: float = 1.0
    lead: Optional[str] = None


# ====================================================================================
# COMPRESSION UTILITIES
# ====================================================================================

class CompressionHandler:
    """Handle compression/decompression of ECG files"""

    @staticmethod
    def detect_compression(filepath: str) -> Optional[str]:
        """Detect if file is compressed"""
        # Check by extension
        if filepath.endswith('.gz'):
            return 'gzip'
        elif filepath.endswith('.zip'):
            return 'zip'

        # Check by magic bytes
        try:
            with open(filepath, 'rb') as f:
                magic = f.read(4)
                if magic[:2] == b'\x1f\x8b':  # gzip
                    return 'gzip'
                elif magic == b'PK\x03\x04':  # zip
                    return 'zip'
        except:
            pass

        return None

    @staticmethod
    def decompress(filepath: str, compression_type: str) -> str:
        """Decompress file and return path to decompressed file"""
        import tempfile

        if compression_type == 'gzip':
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp()
            try:
                with gzip.open(filepath, 'rb') as f_in:
                    with os.fdopen(temp_fd, 'wb') as f_out:
                        f_out.write(f_in.read())
                return temp_path
            except Exception as e:
                os.close(temp_fd)
                os.unlink(temp_path)
                raise e

        elif compression_type == 'zip':
            # Extract first ECG file from zip
            with zipfile.ZipFile(filepath, 'r') as zf:
                # Look for ECG files
                ecg_files = [f for f in zf.namelist()
                             if any(f.lower().endswith(ext)
                                    for ext in ['.dcm', '.dat', '.edf', '.xml', '.csv', '.scp'])]

                if not ecg_files:
                    raise ValueError("No ECG files found in ZIP archive")

                # Extract first ECG file
                temp_dir = tempfile.mkdtemp()
                extracted_path = zf.extract(ecg_files[0], temp_dir)
                return extracted_path

        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")

    @staticmethod
    def compress_xml(xml_filepath: str, compression: str = 'gzip') -> str:
        """Compress XML output"""
        if compression == 'gzip':
            output_path = xml_filepath + '.gz'
            with open(xml_filepath, 'rb') as f_in:
                with gzip.open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            return output_path
        else:
            raise ValueError(f"Unsupported compression: {compression}")


# ====================================================================================
# XML UTILITIES
# ====================================================================================

def prettify_xml(elem):
    """Format XML for better readability with error handling"""
    try:
        if lxml_etree:
            rough_string = ET.tostring(elem, 'utf-8')
            reparsed = lxml_etree.fromstring(rough_string)
            return lxml_etree.tostring(reparsed, pretty_print=True,
                                       encoding='utf-8', xml_declaration=True).decode('utf-8')
    except Exception as e:
        logger.warning(f"lxml prettify failed: {e}, falling back to minidom")

    # Fallback to minidom
    try:
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')
    except Exception as e:
        logger.error(f"XML prettify failed: {e}")
        # Return unprettified
        return ET.tostring(elem, encoding='utf-8').decode('utf-8')


# ====================================================================================
# BASE CONVERTER CLASS
# ====================================================================================

class BaseECGConverter(ABC):
    """Enhanced base class for ECG converters"""

    def __init__(self, input_filepath: str):
        self.input_filepath = input_filepath
        self.original_filepath = input_filepath  # Store original in case of decompression
        self.temp_files = []  # Track temporary files for cleanup

        # Check compression
        compression = CompressionHandler.detect_compression(input_filepath)
        if compression:
            logger.info(f"Detected {compression} compression, decompressing...")
            self.input_filepath = CompressionHandler.decompress(input_filepath, compression)
            self.temp_files.append(self.input_filepath)

        # Validate file exists
        if not os.path.exists(self.input_filepath):
            raise FileNotFoundError(f"File not found: {self.input_filepath}")

        # Initialize data structures
        self.metadata = ECGMetadata()
        self.leads: List[ECGLead] = []
        self.annotations: List[ECGAnnotation] = []
        self.raw_header: Dict[str, str] = {}

        # Initialize XML structure
        self.root = ET.Element("ECGRecord")
        self.root.set("Version", "2.0")
        self.root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

        # Add creation timestamp
        creation_time = datetime.now().isoformat()
        self.root.set("CreationDateTime", creation_time)

        # Create main sections
        self.patient_info = ET.SubElement(self.root, "PatientInfo")
        self.acquisition_details = ET.SubElement(self.root, "AcquisitionDetails")
        self.waveforms_data = ET.SubElement(self.root, "WaveformData")
        self.annotations_elem = ET.SubElement(self.root, "Annotations")
        self.raw_header_info = ET.SubElement(self.root, "RawHeaderInfo")
        self.conversion_info = ET.SubElement(self.root, "ConversionInfo")

        # Add conversion metadata
        ET.SubElement(self.conversion_info, "ConverterVersion").text = "2.0"
        ET.SubElement(self.conversion_info, "ConversionDate").text = creation_time
        ET.SubElement(self.conversion_info, "SourceFile").text = os.path.basename(self.original_filepath)

        # Calculate file hash for integrity
        self._add_file_hash()

    def _add_file_hash(self):
        """Add source file hash for integrity checking"""
        try:
            sha256_hash = hashlib.sha256()
            with open(self.input_filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            ET.SubElement(self.conversion_info, "SourceFileHash").text = sha256_hash.hexdigest()
            ET.SubElement(self.conversion_info, "HashAlgorithm").text = "SHA-256"
        except Exception as e:
            logger.warning(f"Could not calculate file hash: {e}")

    def _safe_str(self, value: Any) -> str:
        """Safely convert any value to string"""
        if value is None:
            return "N/A"
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode('utf-8', errors='ignore').strip()
            except:
                return "N/A"
        try:
            str_val = str(value).strip()
            return str_val if str_val else "N/A"
        except:
            return "N/A"

    def _populate_patient_info(self):
        """Populate patient info from metadata"""
        elements = [
            ("ID", self.metadata.patient_id),
            ("Name", self.metadata.patient_name),
            ("BirthDate", self.metadata.birth_date),
            ("Sex", self.metadata.sex),
            ("Age", self.metadata.age)
        ]

        for tag, value in elements:
            if value and value != "N/A":
                ET.SubElement(self.patient_info, tag).text = self._safe_str(value)

        # Add medications if any
        if self.metadata.medications:
            meds_elem = ET.SubElement(self.patient_info, "Medications")
            for med in self.metadata.medications:
                ET.SubElement(meds_elem, "Medication").text = self._safe_str(med)

        # Add diagnoses if any
        if self.metadata.diagnoses:
            diag_elem = ET.SubElement(self.patient_info, "Diagnoses")
            for diag in self.metadata.diagnoses:
                ET.SubElement(diag_elem, "Diagnosis").text = self._safe_str(diag)

    def _populate_acquisition_details(self):
        """Populate acquisition details from metadata"""
        # Combine date and time if both available
        if self.metadata.recording_date != "N/A" and self.metadata.recording_time != "N/A":
            datetime_str = f"{self.metadata.recording_date}T{self.metadata.recording_time}"
        else:
            datetime_str = self.metadata.recording_date

        ET.SubElement(self.acquisition_details, "AcquisitionDateTime").text = datetime_str

        # Sampling frequency
        sf_elem = ET.SubElement(self.acquisition_details, "SamplingFrequency", Units="Hz")
        sf_elem.text = str(self.metadata.sampling_rate)

        # Other details
        ET.SubElement(self.acquisition_details, "NumberOfChannels").text = str(self.metadata.num_leads)

        dur_elem = ET.SubElement(self.acquisition_details, "RecordDuration", Units="seconds")
        dur_elem.text = f"{self.metadata.duration_sec:.3f}"

        ET.SubElement(self.acquisition_details, "Device").text = self._safe_str(self.metadata.device)

        if self.metadata.institution != "N/A":
            ET.SubElement(self.acquisition_details, "Institution").text = self._safe_str(self.metadata.institution)

        if self.metadata.technician != "N/A":
            ET.SubElement(self.acquisition_details, "Technician").text = self._safe_str(self.metadata.technician)

        # Add filter settings
        if self.metadata.filters:
            filters_elem = ET.SubElement(self.acquisition_details, "FilterSettings")
            for filter_type, value in self.metadata.filters.items():
                filter_elem = ET.SubElement(filters_elem, "Filter")
                filter_elem.set("Type", filter_type)
                filter_elem.text = str(value)

    def _populate_leads(self):
        """Populate lead data"""
        for lead in self.leads:
            lead_elem = ET.SubElement(self.waveforms_data, "Lead")

            ET.SubElement(lead_elem, "Name").text = self._safe_str(lead.name)
            ET.SubElement(lead_elem, "SignalUnits").text = self._safe_str(lead.units)
            ET.SubElement(lead_elem, "Gain").text = str(lead.gain)
            ET.SubElement(lead_elem, "Baseline").text = str(lead.baseline)
            ET.SubElement(lead_elem, "SamplingRate").text = str(lead.sampling_rate)
            ET.SubElement(lead_elem, "NumberOfSamples").text = str(len(lead.data))

            # Add statistics
            if lead.data:
                stats_elem = ET.SubElement(lead_elem, "Statistics")
                if NUMPY_AVAILABLE:
                    data_array = np.array(lead.data)
                    ET.SubElement(stats_elem, "Min").text = f"{np.min(data_array):.3f}"
                    ET.SubElement(stats_elem, "Max").text = f"{np.max(data_array):.3f}"
                    ET.SubElement(stats_elem, "Mean").text = f"{np.mean(data_array):.3f}"
                    ET.SubElement(stats_elem, "StdDev").text = f"{np.std(data_array):.3f}"
                else:
                    ET.SubElement(stats_elem, "Min").text = f"{min(lead.data):.3f}"
                    ET.SubElement(stats_elem, "Max").text = f"{max(lead.data):.3f}"
                    ET.SubElement(stats_elem, "Mean").text = f"{sum(lead.data) / len(lead.data):.3f}"

            # Data - convert to string efficiently
            if NUMPY_AVAILABLE:
                data_str = ' '.join(np.array(lead.data).astype(str))
            else:
                data_str = ' '.join(map(str, lead.data))

            ET.SubElement(lead_elem, "Data").text = data_str

    def _populate_annotations(self):
        """Populate annotations"""
        for ann in self.annotations:
            ann_elem = ET.SubElement(self.annotations_elem, "Annotation")

            ET.SubElement(ann_elem, "TimeOffset", Units="seconds").text = f"{ann.time_sec:.3f}"
            ET.SubElement(ann_elem, "Code").text = self._safe_str(ann.code)
            ET.SubElement(ann_elem, "Description").text = self._safe_str(ann.description)

            if ann.confidence < 1.0:
                ET.SubElement(ann_elem, "Confidence").text = f"{ann.confidence:.2f}"

            if ann.lead:
                ET.SubElement(ann_elem, "Lead").text = self._safe_str(ann.lead)

    def _populate_raw_header(self):
        """Populate raw header info"""
        for key, value in self.raw_header.items():
            if value and len(str(value)) < 10000:  # Limit size
                item = ET.SubElement(self.raw_header_info, "Item")
                item.set("Key", self._safe_str(key))
                item.text = self._safe_str(value)

    def _auto_detect_r_peaks(self):
        """Автоматично детектує R-піки для всіх форматів якщо їх немає"""
        # Якщо вже є анотації R-піків - нічого не робимо
        r_peak_count = sum(1 for ann in self.annotations if ann.code == 'R')
        if r_peak_count > 0:
            logger.info(f"Вже є {r_peak_count} R-піків в анотаціях")
            return

        # Якщо немає відведень - виходимо
        if not self.leads:
            return

        # Вибираємо найкраще відведення для детекції
        best_lead = None
        preferred_leads = ['II', 'MLII', 'I', 'V5', 'V2']

        # Спочатку шукаємо за назвою
        for pref_name in preferred_leads:
            for lead in self.leads:
                if lead.name.upper() == pref_name:
                    best_lead = lead
                    break
            if best_lead:
                break

        # Якщо не знайшли - беремо перше
        if not best_lead:
            best_lead = self.leads[0]

        # Перевіряємо чи достатньо даних
        if len(best_lead.data) < 100:
            logger.warning(f"Недостатньо даних для детекції R-піків: {len(best_lead.data)} семплів")
            return

        try:
            # Конвертуємо в numpy якщо доступно
            if NUMPY_AVAILABLE:
                signal = np.array(best_lead.data)
            else:
                signal = best_lead.data

            # Імпортуємо функцію детекції
            import sys
            import os

            # Додаємо шлях до main.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            try:
                from main import detect_r_peaks_refined
            except ImportError:
                # Спробуємо альтернативний імпорт
                try:
                    import main
                    detect_r_peaks_refined = main.detect_r_peaks_refined
                except:
                    logger.error("Не можу імпортувати detect_r_peaks_refined")
                    return

            # Детектуємо R-піки
            r_peaks = detect_r_peaks_refined(signal, best_lead.sampling_rate)

            if len(r_peaks) > 0:
                logger.info(f"Автоматично знайдено {len(r_peaks)} R-піків у відведенні {best_lead.name}")

                # Додаємо як анотації
                for i, peak_idx in enumerate(r_peaks):
                    time_sec = float(peak_idx) / best_lead.sampling_rate

                    annotation = ECGAnnotation(
                        time_sec=time_sec,
                        code='R',
                        description='R-peak (auto-detected)',
                        confidence=0.9,
                        lead=best_lead.name
                    )
                    self.annotations.append(annotation)

                # Додаємо статистику в raw_header
                if len(r_peaks) > 1:
                    rr_intervals = np.diff(r_peaks) / best_lead.sampling_rate
                    mean_hr = 60.0 / np.mean(rr_intervals)
                    self.raw_header['Auto_Detected_R_Peaks'] = str(len(r_peaks))
                    self.raw_header['Auto_Detected_Mean_HR'] = f"{mean_hr:.1f}"
                    self.raw_header['Auto_Detection_Lead'] = best_lead.name

        except Exception as e:
            logger.error(f"Помилка при автоматичній детекції R-піків: {e}")
            import traceback
            traceback.print_exc()

    # Замініть метод _auto_detect_and_add_r_peaks в BaseECGConverter на цей:

    def _auto_detect_and_add_r_peaks(self):
        """Покращений метод автоматичної детекції R-піків"""
        # Перевіряємо чи вже є R-піки
        r_peak_count = sum(1 for ann in self.annotations if ann.code in ['R', 'N'])
        if r_peak_count > 0:
            logger.info(f"Вже є {r_peak_count} R-піків/beats в анотаціях")
            return

        if not self.leads:
            return

        # Вибираємо найкраще відведення
        best_lead = None
        preferred_leads = ['II', 'MLII', 'I', 'V5', 'V2', 'Lead_II', 'Lead_I']

        for pref_name in preferred_leads:
            for lead in self.leads:
                if pref_name.upper() in lead.name.upper():
                    best_lead = lead
                    break
            if best_lead:
                break

        if not best_lead:
            best_lead = self.leads[0]

        if len(best_lead.data) < 500:
            logger.warning(f"Недостатньо даних: {len(best_lead.data)} семплів")
            return

        try:
            signal = best_lead.data
            fs = best_lead.sampling_rate or self.metadata.sampling_rate

            if fs <= 0:
                logger.error("Невідома частота дискретизації")
                return

            logger.info(f"Детекція R-піків у {best_lead.name}: {len(signal)} семплів @ {fs} Hz")

            if NUMPY_AVAILABLE:
                signal_array = np.array(signal)

                # 1. Видаляємо тренд
                from scipy.signal import detrend
                signal_detrended = detrend(signal_array)

                # 2. Паралельна фільтрація двома фільтрами
                from scipy.signal import butter, filtfilt
                nyquist = 0.5 * fs

                # Фільтр 1: 5-15 Hz (Pan-Tompkins)
                b1, a1 = butter(2, [5.0 / nyquist, min(15.0 / nyquist, 0.95)], btype='band')
                filtered1 = filtfilt(b1, a1, signal_detrended)

                # Фільтр 2: 8-20 Hz (чіткіші піки)
                b2, a2 = butter(2, [8.0 / nyquist, min(20.0 / nyquist, 0.95)], btype='band')
                filtered2 = filtfilt(b2, a2, signal_detrended)

                # 3. Обробка за Pan-Tompkins
                diff = np.diff(filtered1)
                diff = np.append(diff, diff[-1])
                squared = diff ** 2

                window_size = int(0.15 * fs)
                window = np.ones(window_size) / window_size
                integrated = np.convolve(squared, window, mode='same')

                # 4. Комбінована детекція піків
                from scipy.signal import find_peaks
                all_candidate_peaks = []

                # Метод 1: Адаптивний поріг на інтегрованому сигналі
                mean_val = np.mean(integrated)
                std_val = np.std(integrated)

                for factor in [2.0, 1.5, 1.2, 1.0, 0.8]:
                    threshold = mean_val + factor * std_val
                    peaks, _ = find_peaks(
                        integrated,
                        height=threshold,
                        distance=int(0.3 * fs),
                        prominence=threshold * 0.3
                    )

                    expected_peaks = int(len(signal) / fs * 1.2)  # ~72 bpm
                    if expected_peaks * 0.5 <= len(peaks) <= expected_peaks * 2:
                        logger.info(f"  Метод 1: {len(peaks)} піків (фактор {factor})")
                        all_candidate_peaks.extend(peaks)
                        break

                # Метод 2: Процентильний підхід
                for percentile in [85, 80, 75, 70, 65]:
                    threshold = np.percentile(integrated, percentile)
                    peaks, _ = find_peaks(
                        integrated,
                        height=threshold,
                        distance=int(0.3 * fs)
                    )

                    if 5 <= len(peaks) <= 300:
                        logger.info(f"  Метод 2: {len(peaks)} піків ({percentile}й процентиль)")
                        all_candidate_peaks.extend(peaks)
                        break

                # Метод 3: Прямий пошук у фільтрованому сигналі
                signal_norm = filtered2 / np.max(np.abs(filtered2))

                for min_height in [0.5, 0.4, 0.35, 0.3, 0.25]:
                    peaks, _ = find_peaks(
                        signal_norm,
                        height=min_height,
                        distance=int(0.35 * fs),
                        prominence=min_height * 0.4
                    )

                    if 5 <= len(peaks) <= 300:
                        logger.info(f"  Метод 3: {len(peaks)} піків (висота {min_height})")
                        all_candidate_peaks.extend(peaks)
                        break

                # 5. Об'єднання та очищення
                if len(all_candidate_peaks) > 0:
                    all_candidate_peaks = np.unique(all_candidate_peaks)
                    all_candidate_peaks = np.sort(all_candidate_peaks)

                    # Кластеризація та уточнення
                    final_peaks = []
                    i = 0

                    while i < len(all_candidate_peaks):
                        # Збираємо піки в межах 50 мс
                        cluster = [all_candidate_peaks[i]]
                        j = i + 1

                        while j < len(all_candidate_peaks) and (all_candidate_peaks[j] - all_candidate_peaks[i]) < int(
                                0.05 * fs):
                            cluster.append(all_candidate_peaks[j])
                            j += 1

                        # Медіана кластера
                        peak_pos = int(np.median(cluster))

                        # Уточнюємо в оригінальному сигналі
                        search_start = max(0, peak_pos - int(0.04 * fs))
                        search_end = min(len(signal_array), peak_pos + int(0.04 * fs))

                        if search_end > search_start:
                            local_seg = signal_array[search_start:search_end]
                            local_max = np.argmax(local_seg)
                            refined_peak = search_start + local_max

                            # Перевіряємо мінімальну відстань
                            min_distance = int(0.3 * fs)  # 300 мс = 200 bpm max

                            if not final_peaks or (refined_peak - final_peaks[-1]) >= min_distance:
                                # Перевіряємо що це локальний максимум
                                if (refined_peak > 0 and refined_peak < len(signal_array) - 1 and
                                        signal_array[refined_peak] >= signal_array[refined_peak - 1] and
                                        signal_array[refined_peak] >= signal_array[refined_peak + 1]):
                                    final_peaks.append(refined_peak)

                        i = j

                    r_peaks = final_peaks
                else:
                    r_peaks = []

                # Якщо знайдено мало піків - останній шанс
                if len(r_peaks) < 5:
                    logger.warning(f"Знайдено тільки {len(r_peaks)} піків, спроба аварійного методу")

                    # Простий пошук високих піків
                    threshold = np.percentile(np.abs(signal_array), 90)
                    peaks, _ = find_peaks(
                        signal_array,
                        height=threshold,
                        distance=int(0.4 * fs)
                    )

                    if len(peaks) > len(r_peaks):
                        r_peaks = peaks.tolist()
                        logger.info(f"  Аварійний метод знайшов {len(peaks)} піків")

            else:
                # Без NumPy
                logger.warning("NumPy недоступний, базова детекція")

                signal_abs = [abs(x) for x in signal]
                threshold = sorted(signal_abs, reverse=True)[int(len(signal_abs) * 0.1)]

                r_peaks = []
                min_distance = int(0.35 * fs)

                for i in range(1, len(signal) - 1):
                    if (signal[i] > signal[i - 1] and
                            signal[i] > signal[i + 1] and
                            abs(signal[i]) > threshold and
                            (not r_peaks or i - r_peaks[-1] >= min_distance)):
                        r_peaks.append(i)

            # Додаємо знайдені піки
            if len(r_peaks) > 0:
                logger.info(f"✓ Знайдено {len(r_peaks)} R-піків")

                # Статистика
                if len(r_peaks) > 1:
                    rr_intervals = np.diff(r_peaks) / fs
                    mean_hr = 60.0 / np.mean(rr_intervals)
                    logger.info(f"  Середня ЧСС: {mean_hr:.1f} уд/хв")
                    logger.info(
                        f"  Діапазон RR: {np.min(rr_intervals) * 1000:.0f}-{np.max(rr_intervals) * 1000:.0f} мс")

                # Додаємо анотації
                for peak_idx in r_peaks:
                    annotation = ECGAnnotation(
                        time_sec=float(peak_idx) / fs,
                        code='R',
                        description='R-peak (auto-detected)',
                        confidence=0.9,
                        lead=best_lead.name
                    )
                    self.annotations.append(annotation)

                # Метадані
                self.raw_header['Auto_R_Peaks'] = str(len(r_peaks))
                if len(r_peaks) > 1:
                    self.raw_header['Auto_Mean_HR'] = f"{mean_hr:.1f}"
                    self.raw_header['Auto_Lead'] = best_lead.name
            else:
                logger.error("❌ Не вдалось знайти R-піки!")

        except Exception as e:
            logger.error(f"Помилка детекції R-піків: {e}")
            import traceback
            traceback.print_exc()



    @abstractmethod
    def _parse_file(self) -> bool:
        """Parse the specific file format - must be implemented by subclasses"""
        pass

    def convert(self) -> bool:
        """Main conversion method"""
        self.root.set("SourceFormat", self.__class__.__name__.replace("ECGConverter", ""))

        try:
            # Parse file
            logger.info(f"Parsing {self.root.get('SourceFormat')} file...")
            success = self._parse_file()

            if not success:
                logger.error("File parsing failed")
                return False

            # Populate XML sections
            logger.info("Populating XML structure...")
            self._populate_patient_info()
            self._populate_acquisition_details()
            self._populate_leads()
            self._populate_annotations()

            # ВАЖЛИВО: Автоматична детекція R-піків якщо їх немає
            self._auto_detect_and_add_r_peaks()

            # Перезаписуємо анотації після додавання R-піків
            if len(self.annotations) > 0:
                # Очищаємо старі анотації
                self.annotations_elem.clear()
                # Додаємо оновлені
                self._populate_annotations()

            self._populate_raw_header()

            # Add statistics
            stats_elem = ET.SubElement(self.conversion_info, "Statistics")
            ET.SubElement(stats_elem, "LeadsConverted").text = str(len(self.leads))
            ET.SubElement(stats_elem, "AnnotationsConverted").text = str(len(self.annotations))
            ET.SubElement(stats_elem, "TotalSamples").text = str(sum(len(lead.data) for lead in self.leads))

            logger.info(f"Conversion successful: {len(self.leads)} leads, {len(self.annotations)} annotations")
            return True

        except Exception as e:
            logger.error(f"Conversion error: {e}")
            traceback.print_exc()
            return False
        finally:
            # Cleanup temporary files
            for temp_file in self.temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def save_xml(self, output_filepath: str, compress: bool = False) -> bool:
        """Save XML with optional compression"""
        try:
            xml_string = prettify_xml(self.root)

            # Save XML
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(xml_string)

            logger.info(f"XML saved: {output_filepath}")

            # Compress if requested
            if compress:
                compressed_path = CompressionHandler.compress_xml(output_filepath, 'gzip')
                logger.info(f"Compressed XML saved: {compressed_path}")
                # Remove uncompressed file
                os.unlink(output_filepath)
                return True

            return True

        except Exception as e:
            logger.error(f"Error saving XML: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Get conversion summary"""
        return {
            'source_file': self.original_filepath,
            'source_format': self.root.get('SourceFormat'),
            'patient_id': self.metadata.patient_id,
            'patient_name': self.metadata.patient_name,
            'recording_date': self.metadata.recording_date,
            'duration_sec': self.metadata.duration_sec,
            'sampling_rate': self.metadata.sampling_rate,
            'num_leads': len(self.leads),
            'num_annotations': len(self.annotations),
            'lead_names': [lead.name for lead in self.leads]
        }


# ====================================================================================
# PARALLEL PROCESSING UTILITIES
# ====================================================================================

class ParallelProcessor:
    """Handle parallel processing of multi-channel data"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or os.cpu_count()

    def process_leads_parallel(self, leads_data: List[Tuple[str, List[float]]],
                               process_func, **kwargs) -> List[Any]:
        """Process multiple leads in parallel"""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = []
            for lead_name, lead_data in leads_data:
                future = executor.submit(process_func, lead_name, lead_data, **kwargs)
                futures.append((lead_name, future))

            # Collect results
            for lead_name, future in futures:
                try:
                    result = future.result()
                    results.append((lead_name, result))
                except Exception as e:
                    logger.error(f"Error processing lead {lead_name}: {e}")
                    results.append((lead_name, None))

        return results

    def batch_process_files(self, file_list: List[str], converter_class,
                            output_dir: str, **kwargs) -> Dict[str, bool]:
        """Process multiple files in parallel"""
        results = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all conversions
            futures = {}
            for filepath in file_list:
                output_file = os.path.join(output_dir,
                                           os.path.splitext(os.path.basename(filepath))[0] + ".xml")
                future = executor.submit(self._convert_single_file,
                                         filepath, output_file, converter_class, **kwargs)
                futures[future] = filepath

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                filepath = futures[future]
                try:
                    success = future.result()
                    results[filepath] = success
                except Exception as e:
                    logger.error(f"Error converting {filepath}: {e}")
                    results[filepath] = False

        return results

    @staticmethod
    def _convert_single_file(input_file: str, output_file: str,
                             converter_class, **kwargs) -> bool:
        """Convert a single file (used for parallel processing)"""
        try:
            converter = converter_class(input_file, **kwargs)
            if converter.convert():
                return converter.save_xml(output_file)
            return False
        except Exception as e:
            logger.error(f"Conversion failed for {input_file}: {e}")
            return False


# ====================================================================================
# DICOM ECG CONVERTER
# ====================================================================================

class DicomECGConverter(BaseECGConverter):
    """Enhanced DICOM ECG converter with better error handling"""

    def _parse_file(self) -> bool:
        if not PYDICOM_AVAILABLE:
            logger.error("pydicom not installed. Install with: pip install pydicom")
            return False

        try:
            ds = pydicom.dcmread(self.input_filepath)

            # Extract patient information
            self._extract_patient_info(ds)

            # Extract acquisition info
            self._extract_acquisition_info(ds)

            # Check for waveform data
            if not hasattr(ds, 'WaveformSequence') or not ds.WaveformSequence:
                logger.warning("No WaveformSequence found, checking for alternative storage...")
                # Try to find waveform in private tags or overlay
                return self._extract_alternative_waveform(ds)

            # Process each waveform multiplex group
            for waveform_idx, waveform in enumerate(ds.WaveformSequence):
                if not self._process_waveform_multiplex(waveform, waveform_idx):
                    logger.warning(f"Failed to process waveform multiplex {waveform_idx}")

            # Extract annotations if available
            self._extract_annotations(ds)

            # Add raw DICOM tags to header
            self._add_dicom_header_info(ds)

            return len(self.leads) > 0

        except InvalidDicomError as e:
            logger.error(f"Invalid DICOM file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error reading DICOM file: {e}")
            traceback.print_exc()
            return False

    def _extract_patient_info(self, ds):
        """Extract patient information from DICOM dataset"""
        # Basic demographics
        self.metadata.patient_id = getattr(ds, 'PatientID', 'N/A')

        # Handle patient name properly
        patient_name = getattr(ds, 'PatientName', None)
        if patient_name:
            self.metadata.patient_name = str(patient_name).replace('^', ' ')

        # Birth date formatting
        birth_date = getattr(ds, 'PatientBirthDate', None)
        if birth_date:
            try:
                self.metadata.birth_date = f"{birth_date[:4]}-{birth_date[4:6]}-{birth_date[6:8]}"
            except:
                self.metadata.birth_date = str(birth_date)

        self.metadata.sex = getattr(ds, 'PatientSex', 'N/A')

        # Calculate age if birth date available
        if hasattr(ds, 'PatientAge'):
            self.metadata.age = str(ds.PatientAge)
        elif birth_date and hasattr(ds, 'StudyDate'):
            try:
                birth = datetime.strptime(birth_date, '%Y%m%d')
                study = datetime.strptime(ds.StudyDate, '%Y%m%d')
                age_years = (study - birth).days // 365
                self.metadata.age = f"{age_years}Y"
            except:
                pass

        # Additional patient info
        if hasattr(ds, 'PatientWeight'):
            self.raw_header['PatientWeight'] = f"{ds.PatientWeight} kg"
        if hasattr(ds, 'PatientSize'):
            self.raw_header['PatientHeight'] = f"{ds.PatientSize} m"

        # Clinical info
        if hasattr(ds, 'AdmittingDiagnosesDescription'):
            self.metadata.diagnoses.append(str(ds.AdmittingDiagnosesDescription))

    def _extract_acquisition_info(self, ds):
        """Extract acquisition information"""
        # Date and time
        acq_date = getattr(ds, 'AcquisitionDate', getattr(ds, 'StudyDate', None))
        acq_time = getattr(ds, 'AcquisitionTime', getattr(ds, 'StudyTime', None))

        if acq_date:
            try:
                self.metadata.recording_date = f"{acq_date[:4]}-{acq_date[4:6]}-{acq_date[6:8]}"
            except:
                self.metadata.recording_date = str(acq_date)

        if acq_time:
            try:
                # Handle fractional seconds
                time_str = acq_time.split('.')[0]
                if len(time_str) >= 6:
                    self.metadata.recording_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                else:
                    self.metadata.recording_time = str(acq_time)
            except:
                self.metadata.recording_time = str(acq_time)

        # Device info
        self.metadata.device = ' '.join(filter(None, [
            getattr(ds, 'Manufacturer', ''),
            getattr(ds, 'ManufacturerModelName', ''),
            getattr(ds, 'SoftwareVersions', '') if hasattr(ds, 'SoftwareVersions') else ''
        ]))

        self.metadata.institution = getattr(ds, 'InstitutionName', 'N/A')

        # Operator/Technician
        if hasattr(ds, 'OperatorsName'):
            self.metadata.technician = str(ds.OperatorsName).replace('^', ' ')
        elif hasattr(ds, 'PerformingPhysicianName'):
            self.metadata.technician = str(ds.PerformingPhysicianName).replace('^', ' ')

    def _process_waveform_multiplex(self, waveform, multiplex_idx: int) -> bool:
        """Process a single waveform multiplex group"""
        try:
            # Get basic parameters
            sampling_frequency = float(waveform.SamplingFrequency)
            num_channels = int(waveform.NumberOfWaveformChannels)
            num_samples = int(waveform.NumberOfWaveformSamples)

            if num_channels == 0 or num_samples == 0:
                return False

            # Update metadata
            if self.metadata.sampling_rate == 0:
                self.metadata.sampling_rate = sampling_frequency

            # Get waveform data
            if not hasattr(waveform, 'WaveformData'):
                logger.warning(f"No WaveformData in multiplex {multiplex_idx}")
                return False

            raw_data = waveform.WaveformData
            bits_allocated = int(waveform.WaveformBitsAllocated)
            sample_interpretation = waveform.get('WaveformSampleInterpretation', 'SS')

            # Determine data format
            fmt_char = self._get_format_string(bits_allocated, sample_interpretation)
            if not fmt_char:
                logger.error(f"Unsupported waveform format: {bits_allocated} bits, {sample_interpretation}")
                return False

            bytes_per_sample = bits_allocated // 8

            # Process each channel
            channel_definitions = waveform.get('ChannelDefinitionSequence', [])

            for channel_idx in range(num_channels):
                # Extract channel info
                channel_info = self._get_channel_info(channel_definitions, channel_idx)

                # Extract samples for this channel
                samples = []
                for sample_idx in range(num_samples):
                    offset = (sample_idx * num_channels + channel_idx) * bytes_per_sample

                    if offset + bytes_per_sample <= len(raw_data):
                        sample_bytes = raw_data[offset:offset + bytes_per_sample]
                        try:
                            adc_value = struct.unpack(f'<{fmt_char}', sample_bytes)[0]
                            # Convert to physical units
                            physical_value = (adc_value - channel_info['baseline']) * channel_info['gain']
                            samples.append(physical_value)
                        except struct.error:
                            samples.append(0.0)
                    else:
                        samples.append(0.0)

                # Create lead object
                lead = ECGLead(
                    name=channel_info['name'],
                    data=samples,
                    units=channel_info['units'],
                    gain=channel_info['gain'],
                    baseline=channel_info['baseline'],
                    sampling_rate=sampling_frequency
                )
                self.leads.append(lead)

                logger.info(f"Processed lead {lead.name}: {len(samples)} samples at {sampling_frequency} Hz")

            # Update duration
            duration = num_samples / sampling_frequency
            if duration > self.metadata.duration_sec:
                self.metadata.duration_sec = duration

            self.metadata.num_leads = len(self.leads)

            return True

        except Exception as e:
            logger.error(f"Error processing waveform multiplex {multiplex_idx}: {e}")
            return False

    def _get_format_string(self, bits_allocated: int, sample_interpretation: str) -> Optional[str]:
        """Get struct format string for waveform data"""
        formats = {
            (8, 'SB'): 'b', (8, 'UB'): 'B', (8, 'SS'): 'b', (8, 'US'): 'B',
            (16, 'SS'): 'h', (16, 'US'): 'H',
            (32, 'SS'): 'i', (32, 'US'): 'I', (32, 'SL'): 'i', (32, 'UL'): 'I'
        }
        return formats.get((bits_allocated, sample_interpretation))

    def _get_channel_info(self, channel_definitions: list, channel_idx: int) -> dict:
        """Extract channel information"""
        info = {
            'name': f'Channel_{channel_idx + 1}',
            'units': 'uV',
            'gain': 1.0,
            'baseline': 0.0
        }

        if channel_idx < len(channel_definitions):
            ch_def = channel_definitions[channel_idx]

            # Channel label/name
            if hasattr(ch_def, 'ChannelLabel'):
                info['name'] = str(ch_def.ChannelLabel)
            elif hasattr(ch_def, 'ChannelSourceSequence') and ch_def.ChannelSourceSequence:
                source = ch_def.ChannelSourceSequence[0]
                if hasattr(source, 'CodeMeaning'):
                    info['name'] = str(source.CodeMeaning)
                elif hasattr(source, 'CodeValue'):
                    # Map common codes to lead names
                    code_map = {
                        '5.6.3-9-1': 'I', '5.6.3-9-2': 'II', '5.6.3-9-61': 'III',
                        '5.6.3-9-62': 'aVR', '5.6.3-9-63': 'aVL', '5.6.3-9-64': 'aVF',
                        '5.6.3-9-3': 'V1', '5.6.3-9-4': 'V2', '5.6.3-9-5': 'V3',
                        '5.6.3-9-6': 'V4', '5.6.3-9-7': 'V5', '5.6.3-9-8': 'V6'
                    }
                    info['name'] = code_map.get(str(source.CodeValue), f'Lead_{channel_idx + 1}')

            # Channel sensitivity (gain)
            if hasattr(ch_def, 'ChannelSensitivity'):
                sensitivity = float(ch_def.ChannelSensitivity)
                correction = float(ch_def.get('ChannelSensitivityCorrectionFactor', 1.0))
                info['gain'] = sensitivity * correction

            # Units
            if hasattr(ch_def, 'ChannelSensitivityUnitsSequence') and ch_def.ChannelSensitivityUnitsSequence:
                units_item = ch_def.ChannelSensitivityUnitsSequence[0]
                if hasattr(units_item, 'CodeMeaning'):
                    info['units'] = str(units_item.CodeMeaning)
                elif hasattr(units_item, 'CodeValue'):
                    # Map common unit codes
                    unit_map = {
                        'uV': 'uV', 'mV': 'mV', 'V': 'V',
                        'UCUM:uV': 'uV', 'UCUM:mV': 'mV'
                    }
                    info['units'] = unit_map.get(str(units_item.CodeValue), 'uV')

            # Baseline
            if hasattr(ch_def, 'ChannelBaseline'):
                info['baseline'] = float(ch_def.ChannelBaseline)

        return info

    def _extract_alternative_waveform(self, ds) -> bool:
        """Try to extract waveform from alternative storage methods"""
        # Check for Curve Data (older DICOM)
        if hasattr(ds, 'CurveData'):
            logger.info("Found CurveData - attempting extraction...")
            # Implementation would go here
            return False

        # Check for private tags that might contain ECG
        for elem in ds:
            if elem.tag.is_private and 'ECG' in str(elem.name).upper():
                logger.info(f"Found private ECG tag: {elem.tag}")
                # Implementation would go here

        return False

    def _extract_annotations(self, ds):
        """Extract annotations/measurements from DICOM"""
        # Check for WaveformAnnotationSequence
        if hasattr(ds, 'WaveformAnnotationSequence'):
            for ann in ds.WaveformAnnotationSequence:
                if hasattr(ann, 'TemporalRangeType') and ann.TemporalRangeType == 'POINT':
                    time_point = float(ann.ReferencedSamplePositions[0]) / self.metadata.sampling_rate

                    annotation = ECGAnnotation(
                        time_sec=time_point,
                        code=getattr(ann, 'ConceptNameCodeSequence', [{}])[0].get('CodeValue', 'N/A'),
                        description=getattr(ann, 'UnformattedTextValue', 'N/A')
                    )
                    self.annotations.append(annotation)

        # Check for measurements
        if hasattr(ds, 'WaveformPaddingValue'):
            self.raw_header['WaveformPaddingValue'] = str(ds.WaveformPaddingValue)

    def _add_dicom_header_info(self, ds):
        """Add relevant DICOM header information"""
        important_tags = [
            'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
            'StudyDescription', 'SeriesDescription',
            'Modality', 'BodyPartExamined',
            'StudyDate', 'StudyTime',
            'AcquisitionDate', 'AcquisitionTime',
            'ContentDate', 'ContentTime',
            'ReferringPhysicianName', 'RequestingPhysician',
            'PerformingPhysicianName', 'OperatorsName'
        ]

        for tag in important_tags:
            if hasattr(ds, tag):
                value = getattr(ds, tag)
                if value is not None:
                    self.raw_header[tag] = str(value)


# ====================================================================================
# WFDB ECG CONVERTER
# ====================================================================================

class WfdbECGConverter(BaseECGConverter):
    """Enhanced WFDB converter with annotation support"""

    def __init__(self, input_filepath: str):
        # WFDB needs base name without extension
        if input_filepath.lower().endswith(('.dat', '.hea')):
            input_filepath = os.path.splitext(input_filepath)[0]
        super().__init__(input_filepath)

    def _parse_file(self) -> bool:
        if not WFDB_AVAILABLE:
            logger.error("wfdb-python not installed. Install with: pip install wfdb")
            return False

        record_name = os.path.basename(self.input_filepath)
        record_dir = os.path.dirname(self.input_filepath) or None

        try:
            # Read header first for metadata
            header = wfdb.rdheader(record_name, pb_dir=None, pn_dir=record_dir)
            self._extract_header_info(header)

            # Read full record with signals
            record, fields = wfdb.rdsamp(record_name, pb_dir=None, pn_dir=record_dir)

            if record is None or fields is None:
                logger.error("Failed to read WFDB record")
                return False

            # Process signals
            self._process_signals(record, fields)

            # Try to read annotations
            self._read_annotations(record_name, record_dir, fields['fs'])

            # Read additional files if present
            self._read_additional_files(record_name, record_dir)

            return len(self.leads) > 0

        except Exception as e:
            logger.error(f"Error reading WFDB record: {e}")
            traceback.print_exc()
            return False

    def _extract_header_info(self, header):
        """Extract metadata from WFDB header"""
        # Basic info
        self.metadata.sampling_rate = float(header.fs)
        self.metadata.num_leads = int(header.n_sig)

        # Duration
        if header.sig_len and header.fs:
            self.metadata.duration_sec = header.sig_len / header.fs

        # Date/time
        if header.base_date:
            self.metadata.recording_date = header.base_date.replace('/', '-')
        if header.base_time:
            self.metadata.recording_time = str(header.base_time)

        # Comments contain patient info
        if header.comments:
            self._parse_comments(header.comments)

    def _parse_comments(self, comments: list):
        """Parse WFDB comments for patient info"""
        for comment in comments:
            comment_lower = comment.lower()

            # Patient ID
            if 'patient id:' in comment_lower or 'id:' in comment_lower:
                self.metadata.patient_id = comment.split(':', 1)[1].strip()

            # Name
            elif 'name:' in comment_lower:
                self.metadata.patient_name = comment.split(':', 1)[1].strip()

            # Age
            elif 'age:' in comment_lower:
                age_str = comment.split(':', 1)[1].strip()
                # Handle different age formats
                if age_str.isdigit():
                    self.metadata.age = f"{age_str}Y"
                else:
                    self.metadata.age = age_str

            # Sex/Gender
            elif 'sex:' in comment_lower or 'gender:' in comment_lower:
                sex_str = comment.split(':', 1)[1].strip().upper()
                # Standardize sex values
                if sex_str in ['M', 'MALE']:
                    self.metadata.sex = 'M'
                elif sex_str in ['F', 'FEMALE']:
                    self.metadata.sex = 'F'
                else:
                    self.metadata.sex = sex_str

            # Diagnosis
            elif 'diagnosis:' in comment_lower or 'dx:' in comment_lower:
                self.metadata.diagnoses.append(comment.split(':', 1)[1].strip())

            # Medications
            elif 'medication:' in comment_lower or 'meds:' in comment_lower:
                self.metadata.medications.append(comment.split(':', 1)[1].strip())

            # Device
            elif 'device:' in comment_lower or 'machine:' in comment_lower:
                self.metadata.device = comment.split(':', 1)[1].strip()

            # Add all comments to raw header
            self.raw_header[f'Comment_{len(self.raw_header)}'] = comment

    def _process_signals(self, signals, fields):
        """Process WFDB signals"""
        n_sig = fields['n_sig']
        fs = fields['fs']

        # Get signal metadata
        sig_names = fields.get('sig_name', [f'sig{i}' for i in range(n_sig)])
        units = fields.get('units', ['mV'] * n_sig)
        adc_gains = fields.get('adc_gain', [1.0] * n_sig)
        baselines = fields.get('baseline', [0] * n_sig)
        adc_zeros = fields.get('adc_zero', [0] * n_sig)

        # Process each signal
        for i in range(n_sig):
            # Get samples
            if signals.ndim == 1:
                samples = signals.tolist()
            else:
                samples = signals[:, i].tolist()

            # Create lead
            lead = ECGLead(
                name=sig_names[i] if i < len(sig_names) else f'Signal_{i + 1}',
                data=samples,
                units=units[i] if i < len(units) else 'mV',
                gain=adc_gains[i] if i < len(adc_gains) else 1.0,
                baseline=baselines[i] if i < len(baselines) else 0.0,
                sampling_rate=fs
            )

            self.leads.append(lead)

            # Add signal-specific info to raw header
            self.raw_header[f'Signal_{i}_Name'] = lead.name
            self.raw_header[f'Signal_{i}_Units'] = lead.units
            self.raw_header[f'Signal_{i}_Gain'] = str(lead.gain)
            self.raw_header[f'Signal_{i}_Baseline'] = str(lead.baseline)
            if i < len(adc_zeros):
                self.raw_header[f'Signal_{i}_ADCZero'] = str(adc_zeros[i])

        # Add general fields to raw header
        for key, value in fields.items():
            if key not in ['sig_name', 'units', 'adc_gain', 'baseline', 'adc_zero', 'p_signal', 'd_signal']:
                if not isinstance(value, (list, np.ndarray)):
                    self.raw_header[key] = str(value)

    def _read_annotations(self, record_name: str, record_dir: Optional[str], fs: float):
        """Read all available annotation files"""
        # Common annotation extensions
        ann_extensions = ['atr', 'ann', 'ecg', 'ari', 'st', 'vf', 'af', 'qrs']

        for ext in ann_extensions:
            try:
                ann = wfdb.rdann(record_name, ext, pb_dir=None, pn_dir=record_dir)

                logger.info(f"Found {ext} annotations: {len(ann.sample)} annotations")

                # Convert to our annotation format
                for i in range(len(ann.sample)):
                    time_sec = ann.sample[i] / fs

                    # Get symbol/code
                    symbol = ann.symbol[i] if hasattr(ann, 'symbol') and i < len(ann.symbol) else 'N'

                    # Get description
                    if hasattr(ann, 'aux_note') and i < len(ann.aux_note) and ann.aux_note[i]:
                        description = ann.aux_note[i]
                    else:
                        # Map common symbols to descriptions
                        symbol_map = {
                            'N': 'Normal beat',
                            'V': 'Premature ventricular contraction',
                            'A': 'Atrial premature beat',
                            'F': 'Fusion of ventricular and normal beat',
                            'J': 'Nodal (junctional) premature beat',
                            '/': 'Paced beat',
                            'Q': 'Unclassifiable beat',
                            '[': 'Start of ventricular flutter/fibrillation',
                            ']': 'End of ventricular flutter/fibrillation',
                            '(': 'Waveform onset',
                            ')': 'Waveform end'
                        }
                        description = symbol_map.get(symbol, f'Annotation {symbol}')

                    # Get subtype if available
                    subtype = ann.subtype[i] if hasattr(ann, 'subtype') and i < len(ann.subtype) else 0

                    # Get channel if available
                    chan = ann.chan[i] if hasattr(ann, 'chan') and i < len(ann.chan) else None
                    lead_name = self.leads[chan].name if chan is not None and chan < len(self.leads) else None

                    annotation = ECGAnnotation(
                        time_sec=time_sec,
                        code=f"{symbol}:{subtype}" if subtype else symbol,
                        description=description,
                        lead=lead_name
                    )

                    self.annotations.append(annotation)

                # Add annotation statistics to raw header
                if hasattr(ann, 'contained_symbols'):
                    self.raw_header[f'{ext}_symbols'] = ','.join(ann.contained_symbols)

            except FileNotFoundError:
                # No annotation file with this extension
                continue
            except Exception as e:
                logger.warning(f"Could not read {ext} annotations: {e}")

    def _read_additional_files(self, record_name: str, record_dir: Optional[str]):
        """Read additional WFDB files if present"""
        # ВИПРАВЛЕННЯ: Файли вже в поточній директорії
        base_path = record_name

        # Check for info file
        info_path = base_path + '.info'
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info_content = f.read()
                    self.raw_header['InfoFile'] = info_content[:1000]  # Limit size
            except:
                pass

        # Check for patient file
        patient_path = base_path + '.patient'
        if os.path.exists(patient_path):
            try:
                with open(patient_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()

                            if key.lower() == 'id':
                                self.metadata.patient_id = value
                            elif key.lower() == 'name':
                                self.metadata.patient_name = value
                            elif key.lower() == 'dob':
                                self.metadata.birth_date = value
            except:
                pass


# ====================================================================================
# EDF/EDF+ ECG CONVERTER
# ====================================================================================

class EdfECGConverter(BaseECGConverter):
    """Enhanced EDF/EDF+ converter with better annotation support"""

    def _parse_file(self) -> bool:
        # Try MNE first if available (better EDF+ support)
        if MNE_AVAILABLE and self._try_mne_parser():
            return True

        # Fall back to pyedflib
        if PYEDFLIB_AVAILABLE:
            return self._parse_with_pyedflib()

        logger.error("No EDF parser available. Install pyedflib or mne")
        return False

    def _try_mne_parser(self) -> bool:
        """Try parsing with MNE (better for complex EDF+)"""
        try:
            import mne

            # Read raw EDF
            raw = mne.io.read_raw_edf(self.input_filepath, preload=True, verbose=False)

            # Extract metadata
            info = raw.info
            self.metadata.sampling_rate = info['sfreq']
            self.metadata.num_leads = len(info['ch_names'])

            # Get recording date
            if info['meas_date'] is not None:
                meas_date = info['meas_date']
                if hasattr(meas_date, 'strftime'):
                    self.metadata.recording_date = meas_date.strftime('%Y-%m-%d')
                    self.metadata.recording_time = meas_date.strftime('%H:%M:%S')

            # Duration
            self.metadata.duration_sec = raw.n_times / info['sfreq']

            # Extract patient info from description
            if 'description' in info and info['description']:
                self._parse_edf_patient_info(info['description'])

            # Get data
            data, times = raw.get_data(return_times=True)

            # Process each channel
            for i, ch_name in enumerate(info['ch_names']):
                # Skip non-EEG/ECG channels
                ch_type = raw.get_channel_types([ch_name])[0]
                if ch_type not in ['eeg', 'ecg', 'bio', 'misc']:
                    continue

                # Get channel data
                samples = data[i, :].tolist()

                # Get units (MNE uses SI units internally)
                unit = 'V'  # MNE converts to volts

                # Create lead
                lead = ECGLead(
                    name=ch_name,
                    data=samples,
                    units=unit,
                    gain=1.0,
                    baseline=0.0,
                    sampling_rate=info['sfreq']
                )
                self.leads.append(lead)

            # Extract annotations
            annotations = raw.annotations
            for ann in annotations:
                annotation = ECGAnnotation(
                    time_sec=ann['onset'],
                    code='EDF+',
                    description=ann['description']
                )
                self.annotations.append(annotation)

            return len(self.leads) > 0

        except Exception as e:
            logger.debug(f"MNE parsing failed: {e}")
            return False

    def _parse_with_pyedflib(self) -> bool:
        """Parse with pyedflib"""
        f = None
        try:
            f = EdfReader(self.input_filepath)

            # Extract patient info
            self._extract_edf_patient_info(f)

            # Extract recording info
            self._extract_edf_recording_info(f)

            # Process signals
            self._process_edf_signals(f)

            # Read annotations (EDF+)
            self._read_edf_annotations(f)

            # Add header info
            self._add_edf_header_info(f)

            return len(self.leads) > 0

        except Exception as e:
            logger.error(f"Error reading EDF file: {e}")
            traceback.print_exc()
            return False
        finally:
            if f is not None:
                try:
                    f._close()
                except:
                    pass

    def _extract_edf_patient_info(self, f):
        """Extract patient information from EDF"""
        # Patient ID
        patient_code = f.getPatientCode()
        if patient_code:
            self.metadata.patient_id = patient_code

        # Patient name
        patient_name = f.getPatientName()
        if patient_name:
            self.metadata.patient_name = patient_name

        # Birth date
        birthdate = f.getBirthdate()
        if birthdate:
            self.metadata.birth_date = birthdate.strftime('%Y-%m-%d')

        # Gender
        gender = f.getGender()
        if gender:
            self.metadata.sex = gender

        # Additional patient info
        patient_additional = f.getPatientAdditional()
        if patient_additional:
            self._parse_edf_patient_info(patient_additional)

    def _parse_edf_patient_info(self, info_string: str):
        """Parse EDF patient info string"""
        # EDF+ patient info format: "code sex birthdate name"
        # or key=value pairs

        if '=' in info_string:
            # Key-value format
            pairs = info_string.split()
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.lower()

                    if key in ['id', 'code']:
                        self.metadata.patient_id = value
                    elif key == 'name':
                        self.metadata.patient_name = value
                    elif key in ['sex', 'gender']:
                        self.metadata.sex = value
                    elif key in ['age']:
                        self.metadata.age = value
                    elif key in ['dob', 'birthdate']:
                        self.metadata.birth_date = value
        else:
            # Space-separated format
            parts = info_string.split()
            if len(parts) >= 1 and self.metadata.patient_id == 'N/A':
                self.metadata.patient_id = parts[0]
            if len(parts) >= 2 and self.metadata.sex == 'N/A':
                self.metadata.sex = parts[1]
            if len(parts) >= 3 and self.metadata.birth_date == 'N/A':
                self.metadata.birth_date = parts[2]
            if len(parts) >= 4 and self.metadata.patient_name == 'N/A':
                self.metadata.patient_name = ' '.join(parts[3:])

    def _extract_edf_recording_info(self, f):
        """Extract recording information from EDF"""
        # Start date/time
        startdate = f.getStartdatetime()
        if startdate:
            self.metadata.recording_date = startdate.strftime('%Y-%m-%d')
            self.metadata.recording_time = startdate.strftime('%H:%M:%S')

        # Duration
        self.metadata.duration_sec = f.getFileDuration()

        # Equipment
        equipment = f.getEquipment()
        if equipment:
            self.metadata.device = equipment

        # Technician
        technician = f.getTechnician()
        if technician:
            self.metadata.technician = technician

        # Admin code (institution)
        admincode = f.getAdmincode()
        if admincode:
            self.metadata.institution = admincode

    def _process_edf_signals(self, f):
        """Process EDF signals"""
        n_signals = f.signals_in_file
        # pyedflib renamed getSignalLabel(i) → getLabel(i) and added the bulk
        # getSignalLabels() helper. Use whichever the installed version exposes.
        if hasattr(f, "getSignalLabel"):
            _get_label = f.getSignalLabel
        elif hasattr(f, "getLabel"):
            _get_label = f.getLabel
        else:
            _labels = list(f.getSignalLabels())
            _get_label = lambda i: _labels[i]

        for i in range(n_signals):
            # Get signal info
            label = _get_label(i)
            dimension = f.getPhysicalDimension(i)
            sample_rate = f.getSampleFrequency(i)

            # Skip if not ECG-like
            label_lower = label.lower()
            if not any(ecg_term in label_lower for ecg_term in
                       ['ecg', 'ekg', 'lead', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
                        'avr', 'avl', 'avf', 'cardiac', 'heart']):
                # Check dimension as well
                if dimension.lower() not in ['uv', 'mv', 'v', 'micro-volt', 'milli-volt', 'volt']:
                    logger.info(f"Skipping non-ECG signal: {label} ({dimension})")
                    continue

            # Read signal data
            try:
                signal_data = f.readSignal(i)

                # Get physical/digital ranges for gain calculation
                phys_min = f.getPhysicalMinimum(i)
                phys_max = f.getPhysicalMaximum(i)
                dig_min = f.getDigitalMinimum(i)
                dig_max = f.getDigitalMaximum(i)

                # Calculate gain (digital units per physical unit)
                phys_range = phys_max - phys_min
                dig_range = dig_max - dig_min

                if phys_range != 0 and dig_range != 0:
                    gain = dig_range / phys_range
                else:
                    gain = 1.0

                # Create lead
                lead = ECGLead(
                    name=label,
                    data=signal_data.tolist(),
                    units=dimension,
                    gain=gain,
                    baseline=dig_min,
                    sampling_rate=sample_rate
                )
                self.leads.append(lead)

                # Update metadata
                if self.metadata.sampling_rate == 0:
                    self.metadata.sampling_rate = sample_rate

                # Add signal-specific info
                self.raw_header[f'Signal_{i}_Label'] = label
                self.raw_header[f'Signal_{i}_Transducer'] = f.getTransducer(i)
                self.raw_header[f'Signal_{i}_Prefilter'] = f.getPrefilter(i)

            except Exception as e:
                logger.warning(f"Could not read signal {i} ({label}): {e}")

    def _read_edf_annotations(self, f):
        """Read EDF+ annotations"""
        try:
            annotations = f.readAnnotations()

            if annotations and len(annotations) == 3:
                onsets, durations, texts = annotations

                for onset, duration, text in zip(onsets, durations, texts):
                    # Parse annotation text
                    # Format can be: "event_type" or "event_type:details"
                    if ':' in text:
                        code, description = text.split(':', 1)
                    else:
                        code = text
                        description = text

                    annotation = ECGAnnotation(
                        time_sec=onset,
                        code=code.strip(),
                        description=description.strip()
                    )
                    self.annotations.append(annotation)

        except Exception as e:
            logger.debug(f"No annotations found or error reading: {e}")

    def _add_edf_header_info(self, f):
        """Add EDF header information"""
        try:
            header = f.getHeader()

            # Add relevant header fields
            important_fields = [
                'version', 'patient', 'recording', 'startdate', 'starttime',
                'duration', 'reserved', 'equipment', 'admincode', 'technician'
            ]

            for field in important_fields:
                if field in header:
                    value = header[field]
                    if value and str(value).strip():
                        self.raw_header[f'EDF_{field}'] = str(value)

        except Exception as e:
            logger.warning(f"Could not extract full EDF header: {e}")


# ====================================================================================
# HL7 aECG CONVERTER
# ====================================================================================

class Hl7aECGConverter(BaseECGConverter):
    """HL7 aECG XML format converter"""

    def _parse_file(self) -> bool:
        if not LXML_AVAILABLE:
            logger.error("lxml not installed. HL7 aECG support requires lxml. Install with: pip install lxml")
            return False

        try:
            parser = lxml_etree.XMLParser(remove_blank_text=True, recover=True)
            hl7_tree = lxml_etree.parse(self.input_filepath, parser)
            hl7_root = hl7_tree.getroot()

            # Get namespaces
            ns = self._extract_namespaces(hl7_root)

            # Extract patient information
            self._extract_hl7_patient_info(hl7_root, ns)

            # Extract series/acquisition info
            self._extract_hl7_acquisition_info(hl7_root, ns)

            # Extract waveform data
            self._extract_hl7_waveforms(hl7_root, ns)

            # Extract annotations
            self._extract_hl7_annotations(hl7_root, ns)

            # Extract clinical info
            self._extract_hl7_clinical_info(hl7_root, ns)

            return len(self.leads) > 0

        except Exception as e:
            logger.error(f"Error parsing HL7 aECG file: {e}")
            traceback.print_exc()
            return False

    def _extract_namespaces(self, root) -> dict:
        """Extract XML namespaces"""
        ns = {'hl7': 'urn:hl7-org:v3'}  # Default

        # Get from root element
        if root.nsmap:
            # Find the main HL7 namespace
            for prefix, uri in root.nsmap.items():
                if 'hl7' in uri.lower() or 'v3' in uri:
                    ns = {prefix if prefix else 'hl7': uri}
                    break

        logger.info(f"Using namespace: {ns}")
        return ns

    def _extract_hl7_patient_info(self, root, ns):
        """Extract patient information from HL7"""
        # Find patient element
        patient_el = root.find('.//hl7:patient/hl7:patientPerson', namespaces=ns)
        if patient_el is None:
            # Try alternate paths
            patient_el = root.find('.//patientPerson', namespaces=ns)

        if patient_el is not None:
            # Patient ID
            id_el = patient_el.find('.//hl7:id', namespaces=ns)
            if id_el is not None:
                self.metadata.patient_id = id_el.get('extension', 'N/A')

            # Patient name
            name_el = patient_el.find('.//hl7:name', namespaces=ns)
            if name_el is not None:
                name_parts = []
                for part in ['given', 'family']:
                    part_els = name_el.findall(f'.//hl7:{part}', namespaces=ns)
                    for el in part_els:
                        if el.text:
                            name_parts.append(el.text.strip())

                if name_parts:
                    self.metadata.patient_name = ' '.join(name_parts)

            # Birth date
            birthtime_el = patient_el.find('.//hl7:birthTime', namespaces=ns)
            if birthtime_el is not None:
                birth_value = birthtime_el.get('value', '')
                if len(birth_value) >= 8:
                    try:
                        self.metadata.birth_date = f"{birth_value[:4]}-{birth_value[4:6]}-{birth_value[6:8]}"
                    except:
                        self.metadata.birth_date = birth_value

            # Gender
            gender_el = patient_el.find('.//hl7:administrativeGenderCode', namespaces=ns)
            if gender_el is not None:
                self.metadata.sex = gender_el.get('code', 'N/A')

    def _extract_hl7_acquisition_info(self, root, ns):
        """Extract acquisition information"""
        # Find series element
        series_el = root.find('.//hl7:component/hl7:series', namespaces=ns)
        if series_el is None:
            series_el = root.find('.//series', namespaces=ns)

        if series_el is not None:
            # Effective time
            time_el = series_el.find('.//hl7:effectiveTime/hl7:low', namespaces=ns)
            if time_el is not None:
                time_value = time_el.get('value', '')
                if len(time_value) >= 14:
                    try:
                        self.metadata.recording_date = f"{time_value[:4]}-{time_value[4:6]}-{time_value[6:8]}"
                        self.metadata.recording_time = f"{time_value[8:10]}:{time_value[10:12]}:{time_value[12:14]}"
                    except:
                        pass

            # Device info
            device_el = series_el.find('.//hl7:author/hl7:assignedAuthoringDevice', namespaces=ns)
            if device_el is not None:
                # Manufacturer and model
                manuf_el = device_el.find('.//hl7:manufacturerModelName', namespaces=ns)
                if manuf_el is not None and manuf_el.text:
                    self.metadata.device = manuf_el.text.strip()

                # Software version
                software_el = device_el.find('.//hl7:softwareName', namespaces=ns)
                if software_el is not None and software_el.text:
                    if self.metadata.device != "N/A":
                        self.metadata.device += f" ({software_el.text.strip()})"
                    else:
                        self.metadata.device = software_el.text.strip()

    def _extract_hl7_waveforms(self, root, ns):
        """Extract waveform data from HL7"""
        # Find all sequence elements (each contains one lead)
        sequence_els = root.findall('.//hl7:component/hl7:sequenceSet/hl7:component/hl7:sequence', namespaces=ns)

        if not sequence_els:
            # Try without namespace prefix
            sequence_els = root.findall('.//sequenceSet/component/sequence')

        # First pass - get common sampling rate
        common_sampling_rate = None
        for seq_el in sequence_els:
            value_el = seq_el.find('.//hl7:value', namespaces=ns)
            if value_el is None:
                value_el = seq_el.find('.//value')

            if value_el is not None:
                increment_el = value_el.find('.//hl7:increment', namespaces=ns)
                if increment_el is None:
                    increment_el = value_el.find('.//increment')

                if increment_el is not None:
                    inc_value = increment_el.get('value')
                    inc_unit = increment_el.get('unit', 's')

                    if inc_value and inc_unit == 's':
                        try:
                            sampling_interval = float(inc_value)
                            if sampling_interval > 0:
                                common_sampling_rate = 1.0 / sampling_interval
                                break
                        except:
                            pass

        if common_sampling_rate:
            self.metadata.sampling_rate = common_sampling_rate

        # Process each sequence/lead
        for seq_idx, seq_el in enumerate(sequence_els):
            # Get lead name
            lead_name = f"Lead_{seq_idx + 1}"
            code_el = seq_el.find('.//hl7:code', namespaces=ns)
            if code_el is None:
                code_el = seq_el.find('.//code')

            if code_el is not None:
                # Try display name first
                display_name = code_el.get('displayName', '') or code_el.get('codeSystemName', '')
                if display_name:
                    lead_name = display_name
                else:
                    # Map code values to standard lead names
                    code_value = code_el.get('code', '')
                    code_map = {
                        'MDC_ECG_LEAD_I': 'I',
                        'MDC_ECG_LEAD_II': 'II',
                        'MDC_ECG_LEAD_III': 'III',
                        'MDC_ECG_LEAD_AVR': 'aVR',
                        'MDC_ECG_LEAD_AVL': 'aVL',
                        'MDC_ECG_LEAD_AVF': 'aVF',
                        'MDC_ECG_LEAD_V1': 'V1',
                        'MDC_ECG_LEAD_V2': 'V2',
                        'MDC_ECG_LEAD_V3': 'V3',
                        'MDC_ECG_LEAD_V4': 'V4',
                        'MDC_ECG_LEAD_V5': 'V5',
                        'MDC_ECG_LEAD_V6': 'V6'
                    }
                    lead_name = code_map.get(code_value, lead_name)

            # Get value element
            value_el = seq_el.find('.//hl7:value', namespaces=ns)
            if value_el is None:
                value_el = seq_el.find('.//value')

            if value_el is not None:
                # Get scale/units
                units = 'uV'
                gain = 1.0
                baseline = 0.0

                scale_el = value_el.find('.//hl7:scale', namespaces=ns)
                if scale_el is None:
                    scale_el = value_el.find('.//scale')

                if scale_el is not None:
                    units = scale_el.get('unit', 'uV')
                    try:
                        gain = float(scale_el.get('value', '1'))
                    except:
                        gain = 1.0

                # Get origin/baseline
                origin_el = value_el.find('.//hl7:origin', namespaces=ns)
                if origin_el is None:
                    origin_el = value_el.find('.//origin')

                if origin_el is not None:
                    try:
                        baseline = float(origin_el.get('value', '0'))
                    except:
                        baseline = 0.0

                # Get digits (actual waveform data)
                digits_el = value_el.find('.//hl7:digits', namespaces=ns)
                if digits_el is None:
                    digits_el = value_el.find('.//digits')

                if digits_el is not None and digits_el.text:
                    # Parse sample values
                    samples_str = digits_el.text.strip()
                    samples = []

                    for sample_str in samples_str.split():
                        try:
                            samples.append(float(sample_str))
                        except:
                            continue

                    if samples:
                        # Create lead
                        lead = ECGLead(
                            name=lead_name,
                            data=samples,
                            units=units,
                            gain=gain,
                            baseline=baseline,
                            sampling_rate=common_sampling_rate or 500.0
                        )
                        self.leads.append(lead)

                        # Update duration
                        duration = len(samples) / lead.sampling_rate
                        if duration > self.metadata.duration_sec:
                            self.metadata.duration_sec = duration

        self.metadata.num_leads = len(self.leads)

    def _extract_hl7_annotations(self, root, ns):
        """Extract annotations from HL7"""
        # Find annotation set
        ann_els = root.findall('.//hl7:component/hl7:annotationSet/hl7:component/hl7:annotation', namespaces=ns)

        if not ann_els:
            ann_els = root.findall('.//annotationSet/component/annotation')

        for ann_el in ann_els:
            # Get time
            time_sec = 0.0
            time_el = ann_el.find('.//hl7:effectiveTime/hl7:low', namespaces=ns)
            if time_el is None:
                time_el = ann_el.find('.//effectiveTime/low')

            if time_el is not None:
                time_value = time_el.get('value', '')
                try:
                    # Could be absolute time or offset in seconds
                    if '.' in time_value:
                        time_sec = float(time_value)
                    elif len(time_value) >= 14:
                        # Parse as timestamp - would need reference time
                        pass
                except:
                    pass

            # Get code
            code = "N/A"
            code_el = ann_el.find('.//hl7:code', namespaces=ns)
            if code_el is None:
                code_el = ann_el.find('.//code')

            if code_el is not None:
                code = code_el.get('displayName', code_el.get('code', 'N/A'))

            # Get text/description
            description = ""
            text_el = ann_el.find('.//hl7:text', namespaces=ns)
            if text_el is None:
                text_el = ann_el.find('.//text')

            if text_el is not None and text_el.text:
                description = text_el.text.strip()

            # Create annotation
            if code != "N/A" or description:
                annotation = ECGAnnotation(
                    time_sec=time_sec,
                    code=code,
                    description=description
                )
                self.annotations.append(annotation)

    def _extract_hl7_clinical_info(self, root, ns):
        """Extract clinical information"""
        # Look for clinical statements
        statements = root.findall('.//hl7:component/hl7:clinicalStatement', namespaces=ns)

        for statement in statements:
            # Extract diagnoses
            code_el = statement.find('.//hl7:code', namespaces=ns)
            if code_el is not None:
                diagnosis = code_el.get('displayName', '')
                if diagnosis:
                    self.metadata.diagnoses.append(diagnosis)

        # Look for medications
        med_els = root.findall('.//hl7:substanceAdministration', namespaces=ns)
        for med_el in med_els:
            material_el = med_el.find('.//hl7:consumable/hl7:manufacturedProduct/hl7:manufacturedMaterial/hl7:code',
                                      namespaces=ns)
            if material_el is not None:
                med_name = material_el.get('displayName', '')
                if med_name:
                    self.metadata.medications.append(med_name)


# ====================================================================================
# CSV ECG CONVERTER
# ====================================================================================

class CsvECGConverter(BaseECGConverter):
    """CSV file converter with flexible parsing"""

    def __init__(self, input_filepath: str, delimiter: str = ',',
                 num_header_rows: int = 0, lead_names_row: Optional[int] = None,
                 data_start_row: int = 1, sampling_rate_hz: Optional[float] = None,
                 time_column: Optional[int] = None, encoding: str = 'utf-8'):
        super().__init__(input_filepath)
        self.delimiter = delimiter
        self.num_header_rows = num_header_rows
        self.lead_names_row = lead_names_row
        self.data_start_row = data_start_row
        self.sampling_rate_hz = sampling_rate_hz
        self.time_column = time_column
        self.encoding = encoding

    def _parse_file(self) -> bool:
        all_rows = []

        try:
            # Try to detect delimiter if auto
            if self.delimiter == 'auto':
                self.delimiter = self._detect_delimiter()

            # Read all rows
            with open(self.input_filepath, 'r', newline='', encoding=self.encoding) as csvfile:
                reader = csv.reader(csvfile, delimiter=self.delimiter)
                for row in reader:
                    all_rows.append(row)

        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return False

        if not all_rows:
            logger.error("CSV file is empty")
            return False

        # Parse header rows
        self._parse_header_rows(all_rows)

        # Get lead names
        lead_names = self._extract_lead_names(all_rows)

        # Parse data rows
        data_rows = all_rows[self.data_start_row:]
        if not data_rows:
            logger.warning("No data rows found")
            return False

        # Determine if we have time column
        has_time_column = self._detect_time_column(data_rows)

        # Extract signals
        return self._extract_signals(data_rows, lead_names, has_time_column)

    def _detect_delimiter(self) -> str:
        """Auto-detect CSV delimiter"""
        with open(self.input_filepath, 'r', encoding=self.encoding) as f:
            sample = f.read(4096)

        # Count occurrences of common delimiters
        delimiters = [',', '\t', ';', '|', ' ']
        delimiter_counts = {}

        for delim in delimiters:
            delimiter_counts[delim] = sample.count(delim)

        # Return most common
        return max(delimiter_counts.items(), key=lambda x: x[1])[0]

    def _parse_header_rows(self, all_rows: list):
        """Parse header rows for metadata"""
        for i in range(min(self.num_header_rows, len(all_rows))):
            row = all_rows[i]
            if not row:
                continue

            # Store in raw header
            self.raw_header[f'Header_Row_{i}'] = self.delimiter.join(row)

            # Try to extract metadata
            row_str = ' '.join(row).lower()

            # Look for patient info
            if 'patient' in row_str or 'id' in row_str:
                for cell in row:
                    if ':' in cell:
                        key, value = cell.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()

                        if 'id' in key:
                            self.metadata.patient_id = value
                        elif 'name' in key:
                            self.metadata.patient_name = value
                        elif 'age' in key:
                            self.metadata.age = value
                        elif 'sex' in key or 'gender' in key:
                            self.metadata.sex = value

            # Look for sampling rate
            if 'sampling' in row_str or 'frequency' in row_str or 'rate' in row_str or 'hz' in row_str:
                for cell in row:
                    # Try to extract number
                    import re
                    numbers = re.findall(r'[\d.]+', cell)
                    if numbers and any(term in cell.lower() for term in ['hz', 'sampling', 'frequency']):
                        try:
                            self.metadata.sampling_rate = float(numbers[0])
                            logger.info(f"Found sampling rate in header: {self.metadata.sampling_rate} Hz")
                        except:
                            pass

            # Look for date/time
            if 'date' in row_str or 'time' in row_str:
                for cell in row:
                    if ':' in cell and any(c.isdigit() for c in cell):
                        if 'date' in cell.lower():
                            parts = cell.split(':', 1)
                            if len(parts) > 1:
                                self.metadata.recording_date = parts[1].strip()
                        elif 'time' in cell.lower():
                            parts = cell.split(':', 1)
                            if len(parts) > 1:
                                self.metadata.recording_time = parts[1].strip()

    def _extract_lead_names(self, all_rows: list) -> List[str]:
        """Extract lead names from CSV"""
        lead_names = []

        if self.lead_names_row is not None and self.lead_names_row < len(all_rows):
            # Use specified row
            name_row = all_rows[self.lead_names_row]

            # Skip time column if present
            start_col = 1 if self.time_column == 0 else 0

            for i, name in enumerate(name_row[start_col:]):
                if name.strip():
                    lead_names.append(name.strip())
                else:
                    lead_names.append(f"Lead_{i + 1}")
        else:
            # Auto-detect lead names row
            for i in range(min(5, len(all_rows))):  # Check first 5 rows
                row = all_rows[i]
                if row and any('lead' in cell.lower() or
                               any(ecg_name in cell.upper() for ecg_name in
                                   ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                                    'AVR', 'AVL', 'AVF'])
                               for cell in row):
                    # Found lead names row
                    start_col = 1 if self._looks_like_time(row[0]) else 0
                    lead_names = [cell.strip() for cell in row[start_col:] if cell.strip()]
                    self.lead_names_row = i
                    logger.info(f"Auto-detected lead names at row {i}")
                    break

        return lead_names

    def _detect_time_column(self, data_rows: list) -> bool:
        """Detect if first column is time"""
        if not data_rows or not data_rows[0]:
            return False

        # НОВИЙ КОД: Перевірка заголовка якщо він був знайдений
        if hasattr(self, 'lead_names_row') and self.lead_names_row is not None:
            # Якщо ми знаємо де заголовок - перевіримо першу колонку
            if self.lead_names_row < len(data_rows):
                header_row = data_rows[self.lead_names_row]
                if header_row and len(header_row) > 0:
                    first_col_name = header_row[0].strip().lower()
                    # Якщо перша колонка має назву пов'язану з часом
                    if any(time_word in first_col_name for time_word in
                           ['time', 'час', 'sec', 'ms', 'sample', 'index', 't']):
                        logger.info(f"Знайдено колонку часу за назвою: '{header_row[0]}'")
                        return True

        # Оригінальна перевірка за вмістом даних
        time_like_count = 0
        for i in range(min(10, len(data_rows))):
            if data_rows[i]:
                if self._looks_like_time(data_rows[i][0]):
                    time_like_count += 1

        result = time_like_count > 5
        if result:
            logger.info("Знайдено колонку часу за вмістом даних")

        return result

    def _looks_like_time(self, value: str) -> bool:
        """Check if value looks like time/timestamp"""
        value = value.strip()

        # Check for common time patterns
        if ':' in value:  # HH:MM:SS format
            return True

        # Check if it's a sequential number (time index)
        try:
            float_val = float(value)
            # Time values are usually sequential and start near 0
            if 0 <= float_val < 1000000:  # Reasonable time range
                return True
        except:
            pass

        return False

    # В parser.py, клас CsvECGConverter, замініть метод _extract_signals на цей:

    def _extract_signals(self, data_rows: list, lead_names: List[str], has_time_column: bool) -> bool:
        """Extract signal data from CSV"""
        if not data_rows:
            return False

        # Determine number of channels
        first_data_row = data_rows[0]
        start_col = 1 if has_time_column else 0
        num_channels = len(first_data_row) - start_col

        if num_channels <= 0:
            logger.error("No data channels found")
            return False

        # ВИПРАВЛЕННЯ: Якщо є колонка часу і вона в lead_names - видалити її
        if has_time_column and lead_names:
            # Перевірка чи перша назва - це час
            if lead_names[0].lower() in ['time', 'час', 'seconds', 'ms', 'sample', 'index']:
                logger.info(f"Видаляю колонку часу '{lead_names[0]}' з назв відведень")
                lead_names = lead_names[1:]  # Видаляємо першу назву

        # Extend lead names if needed
        while len(lead_names) < num_channels:
            lead_names.append(f"Lead_{len(lead_names) + 1}")

        # Initialize data lists for each channel
        channel_data = [[] for _ in range(num_channels)]
        time_data = []

        # Parse data rows
        for row_idx, row in enumerate(data_rows):
            if len(row) < start_col + 1:
                continue

            # Extract time if present
            if has_time_column:
                try:
                    time_data.append(float(row[0]))
                except:
                    time_data.append(row_idx)

            # Extract channel data
            for ch_idx in range(num_channels):
                col_idx = start_col + ch_idx
                if col_idx < len(row):
                    try:
                        value = float(row[col_idx].strip())
                        channel_data[ch_idx].append(value)
                    except:
                        # Handle missing/invalid data
                        if channel_data[ch_idx]:
                            # Use last valid value
                            channel_data[ch_idx].append(channel_data[ch_idx][-1])
                        else:
                            channel_data[ch_idx].append(0.0)

        # Determine sampling rate if not provided
        if not self.metadata.sampling_rate and self.sampling_rate_hz:
            self.metadata.sampling_rate = self.sampling_rate_hz
        elif not self.metadata.sampling_rate and time_data and len(time_data) > 1:
            # Calculate from time data
            time_diffs = [time_data[i + 1] - time_data[i] for i in range(len(time_data) - 1)]
            avg_diff = sum(time_diffs) / len(time_diffs)
            if avg_diff > 0:
                self.metadata.sampling_rate = 1.0 / avg_diff
                logger.info(f"Calculated sampling rate: {self.metadata.sampling_rate:.2f} Hz")

        # Default sampling rate if still not determined
        if not self.metadata.sampling_rate:
            self.metadata.sampling_rate = 500.0  # Common ECG sampling rate
            logger.warning("Could not determine sampling rate, using default 500 Hz")

        # Create leads
        for ch_idx, data in enumerate(channel_data):
            if data:  # Only create lead if has data
                lead = ECGLead(
                    name=lead_names[ch_idx] if ch_idx < len(lead_names) else f"Lead_{ch_idx + 1}",
                    data=data,
                    units="mV",  # Assume mV for CSV
                    gain=1.0,
                    baseline=0.0,
                    sampling_rate=self.metadata.sampling_rate
                )
                self.leads.append(lead)

        # Update metadata
        self.metadata.num_leads = len(self.leads)
        if self.leads:
            self.metadata.duration_sec = len(self.leads[0].data) / self.metadata.sampling_rate

        logger.info(
            f"Extracted {len(self.leads)} leads with {len(channel_data[0]) if channel_data else 0} samples each")

        return len(self.leads) > 0

    def _add_r_peaks_annotations(self):
        """Додає R-піки як анотації після конвертації"""
        # Перевіряємо чи є дані для детекції
        if not self.leads or len(self.leads[0].data) < 100:
            return

        # Беремо перше відведення для детекції R-піків
        lead = self.leads[0]
        signal = np.array(lead.data) if NUMPY_AVAILABLE else lead.data

        # Імпортуємо функцію детекції з main.py
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from main import detect_r_peaks_refined

            # Детектуємо R-піки
            r_peaks = detect_r_peaks_refined(signal, lead.sampling_rate)

            if len(r_peaks) > 0:
                logger.info(f"Знайдено {len(r_peaks)} R-піків, додаю до анотацій")

                # Конвертуємо в анотації
                for i, peak_idx in enumerate(r_peaks):
                    time_sec = peak_idx / lead.sampling_rate

                    annotation = ECGAnnotation(
                        time_sec=time_sec,
                        code='R',
                        description='R-peak',
                        confidence=0.95,
                        lead=lead.name
                    )
                    self.annotations.append(annotation)

                # Додаємо також RR інтервали
                if len(r_peaks) > 1:
                    rr_intervals = np.diff(r_peaks) / lead.sampling_rate
                    mean_hr = 60.0 / np.mean(rr_intervals)
                    logger.info(f"Середня ЧСС: {mean_hr:.1f} уд/хв")

        except ImportError:
            logger.warning("Не можу імпортувати detect_r_peaks_refined")

# ====================================================================================
# SCP-ECG CONVERTER
# ====================================================================================

class ScpECGConverter(BaseECGConverter):
    """SCP-ECG (EN1064) format converter"""

    # SCP-ECG section IDs
    SECTION_IDS = {
        0: 'Pointers',
        1: 'Patient/ECG Data',
        2: 'Huffman Tables',
        3: 'ECG Lead Definition',
        4: 'QRS Locations',
        5: 'Reference Beat',
        6: 'Rhythm Data',
        7: 'Global Measurements',
        8: 'Textual Diagnosis',
        9: 'Manufacturer Specific',
        10: 'Lead Measurements',
        11: 'Universal Statement Codes'
    }

    def _parse_file(self) -> bool:
        """Parse SCP-ECG file"""
        try:
            with open(self.input_filepath, 'rb') as f:
                # Read and validate header
                if not self._read_header(f):
                    return False

                # Read section pointers (Section 0)
                sections = self._read_section_pointers(f)

                # Read each section
                for section_id, (offset, length) in sections.items():
                    if length > 0:
                        f.seek(offset)
                        self._read_section(f, section_id, length)

                return len(self.leads) > 0

        except Exception as e:
            logger.error(f"Error parsing SCP-ECG file: {e}")
            traceback.print_exc()
            return False

    def _read_header(self, f) -> bool:
        """Read and validate SCP-ECG header"""
        # Check CRC (first 2 bytes)
        crc = struct.unpack('<H', f.read(2))[0]

        # File size
        file_size = struct.unpack('<I', f.read(4))[0]

        # Validate file size
        f.seek(0, 2)  # Seek to end
        actual_size = f.tell()
        f.seek(6)  # Back to after header

        if actual_size != file_size:
            logger.warning(f"File size mismatch: header says {file_size}, actual is {actual_size}")

        return True

    def _read_section_pointers(self, f) -> Dict[int, Tuple[int, int]]:
        """Read Section 0 - pointers to other sections"""
        sections = {}

        # Section 0 header
        section_id = struct.unpack('<H', f.read(2))[0]
        section_length = struct.unpack('<I', f.read(4))[0]

        if section_id != 0:
            logger.error(f"Expected Section 0, got {section_id}")
            return sections

        # Read pointers
        num_sections = (section_length - 8) // 10  # Each pointer is 10 bytes

        for _ in range(int(num_sections)):
            sect_id = struct.unpack('<H', f.read(2))[0]
            sect_length = struct.unpack('<I', f.read(4))[0]
            sect_offset = struct.unpack('<I', f.read(4))[0]

            sections[sect_id] = (sect_offset, sect_length)

        return sections

    def _read_section(self, f, section_id: int, length: int):
        """Read a specific section"""
        section_name = self.SECTION_IDS.get(section_id, f'Unknown ({section_id})')
        logger.info(f"Reading Section {section_id}: {section_name}")

        # ВИПРАВЛЕННЯ: Перевіряємо чи достатньо даних
        current_pos = f.tell()

        # Перевіряємо розмір секції
        if length < 16:  # Мінімальний розмір заголовка секції
            logger.warning(f"Section {section_id} too small: {length} bytes")
            return

        # Read section header
        header_data = f.read(2)
        if len(header_data) < 2:
            logger.error(f"Insufficient data for section {section_id} header")
            return

        actual_id = struct.unpack('<H', header_data)[0]

        length_data = f.read(4)
        if len(length_data) < 4:
            logger.error(f"Insufficient data for section {section_id} length")
            return

        actual_length = struct.unpack('<I', length_data)[0]

        if actual_id != section_id:
            logger.warning(f"Section ID mismatch: expected {section_id}, got {actual_id}")
            # Повертаємось назад і пропускаємо секцію
            f.seek(current_pos + length)
            return

        # Skip section version and protocol version
        version_data = f.read(4)  # 2 bytes section version + 2 bytes protocol version
        if len(version_data) < 4:
            logger.error(f"Insufficient data for section {section_id} version")
            return

        # Reserved
        reserved_data = f.read(6)
        if len(reserved_data) < 6:
            logger.error(f"Insufficient data for section {section_id} reserved")
            return

        # Calculate actual data length
        data_length = actual_length - 16  # Subtract header size

        # Перевіряємо чи не виходимо за межі
        if data_length < 0 or data_length > length - 16:
            logger.warning(f"Invalid data length for section {section_id}: {data_length}")
            # Переходимо до наступної секції
            f.seek(current_pos + length)
            return

        # Read section data based on type
        try:
            if section_id == 1:
                self._read_patient_data(f, data_length)
            elif section_id == 3:
                self._read_lead_definition(f, data_length)
            elif section_id == 6:
                self._read_rhythm_data(f, data_length)
            elif section_id == 7:
                self._read_global_measurements(f, data_length)
            elif section_id == 8:
                self._read_textual_diagnosis(f, data_length)
            else:
                # Store as raw data
                raw_data = f.read(data_length)
                self.raw_header[f'Section_{section_id}_Data'] = f"Binary data ({len(raw_data)} bytes)"
        except Exception as e:
            logger.error(f"Error reading section {section_id} data: {e}")
            # Переходимо до кінця секції
            f.seek(current_pos + length)

    def _read_patient_data(self, f, length: int):
        """Read Section 1 - Patient and ECG acquisition data"""
        start_pos = f.tell()

        # Patient ID
        patient_id = self._read_string(f, 21)
        self.metadata.patient_id = patient_id

        # Patient name (last, first)
        last_name = self._read_string(f, 21)
        first_name = self._read_string(f, 21)
        if last_name or first_name:
            self.metadata.patient_name = f"{first_name} {last_name}".strip()

        # Patient ID 2
        f.read(21)  # Skip

        # Age
        age = struct.unpack('<H', f.read(2))[0]
        if age > 0 and age < 200:
            self.metadata.age = f"{age}Y"

        # Date of birth
        birth_date = self._read_date(f)
        if birth_date:
            self.metadata.birth_date = birth_date

        # Sex
        sex = struct.unpack('B', f.read(1))[0]
        sex_map = {1: 'M', 2: 'F', 0: 'N/A'}
        self.metadata.sex = sex_map.get(sex, 'N/A')

        # Race
        f.read(1)  # Skip

        # Drugs
        num_drugs = struct.unpack('<H', f.read(2))[0]
        for _ in range(num_drugs):
            drug_class = f.read(1)
            drug_length = struct.unpack('B', f.read(1))[0]
            drug_name = self._read_string(f, drug_length)
            if drug_name:
                self.metadata.medications.append(drug_name)

        # Skip to acquisition data
        f.seek(start_pos + length)

    def _read_lead_definition(self, f, length: int):
        """Read Section 3 - Lead definition"""
        start_pos = f.tell()

        # Number of leads
        num_leads = struct.unpack('B', f.read(1))[0]

        # Flags
        flags = struct.unpack('B', f.read(1))[0]

        # Lead definitions
        for i in range(num_leads):
            # Lead ID
            lead_id = struct.unpack('B', f.read(1))[0]

            # Lead specification length
            spec_length = struct.unpack('B', f.read(1))[0]

            # Lead specification data
            lead_spec = f.read(spec_length - 1)

            # Map lead ID to name
            lead_names = {
                0: 'I', 1: 'II', 2: 'V1', 3: 'V2', 4: 'V3',
                5: 'V4', 6: 'V5', 7: 'V6', 8: 'V7', 9: 'V2R',
                10: 'V3R', 11: 'V4R', 12: 'V5R', 13: 'V6R', 14: 'V7R',
                15: 'X', 16: 'Y', 17: 'Z', 18: 'CC5', 19: 'CM5',
                20: 'LA', 21: 'RA', 22: 'LL', 23: 'fI', 24: 'fE',
                25: 'fC', 26: 'fA', 27: 'fM', 28: 'fF', 29: 'fH',
                30: 'dI', 31: 'dII', 32: 'dV1', 33: 'dV2', 34: 'dV3',
                35: 'dV4', 36: 'dV5', 37: 'dV6', 38: 'dV7', 39: 'dV2R',
                40: 'dV3R', 41: 'dV4R', 42: 'dV5R', 43: 'dV6R', 44: 'dV7R',
                45: 'dX', 46: 'dY', 47: 'dZ', 48: 'dCC5', 49: 'dCM5',
                50: 'dLA', 51: 'dRA', 52: 'dLL', 53: 'dfI', 54: 'dfE',
                55: 'dfC', 56: 'dfA', 57: 'dfM', 58: 'dfF', 59: 'dfH',
                60: 'III', 61: 'aVR', 62: 'aVL', 63: 'aVF', 64: 'aVRneg',
                65: 'V8', 66: 'V9', 67: 'V8R', 68: 'V9R', 69: 'D',
                70: 'A', 71: 'J', 72: 'Defib', 73: 'Extern', 74: 'A1',
                75: 'A2', 76: 'A3', 77: 'A4', 78: 'dV8', 79: 'dV9',
                80: 'dV8R', 81: 'dV9R', 82: 'dD', 83: 'dA', 84: 'dJ',
                85: 'Chest', 86: 'V', 87: 'VR', 88: 'VL', 89: 'VF',
                90: 'MCL', 91: 'MCL1', 92: 'MCL2', 93: 'MCL3', 94: 'MCL4',
                95: 'MCL5', 96: 'MCL6', 97: 'CC', 98: 'CC1', 99: 'CC2',
                100: 'CC3', 101: 'CC4', 102: 'CC6', 103: 'CC7', 104: 'CM',
                105: 'CM1', 106: 'CM2', 107: 'CM3', 108: 'CM4', 109: 'CM6',
                110: 'dIII', 111: 'daVR', 112: 'daVL', 113: 'daVF', 114: 'daVRneg',
                115: 'dChest', 116: 'dV', 117: 'dVR', 118: 'dVL', 119: 'dVF',
                120: 'CM7', 121: 'CH5', 122: 'CS5', 123: 'CB5', 124: 'CR5',
                125: 'ML', 126: 'AB1', 127: 'AB2', 128: 'AB3', 129: 'AB4',
                130: 'ES', 131: 'AS', 132: 'AI', 133: 'S', 134: 'dDefib',
                135: 'dExtern', 136: 'dA1', 137: 'dA2', 138: 'dA3', 139: 'dA4',
                140: 'dMCL', 141: 'dMCL1', 142: 'dMCL2', 143: 'dMCL3', 144: 'dMCL4',
                145: 'dMCL5', 146: 'dMCL6', 147: 'dCC', 148: 'dCC1', 149: 'dCC2',
                150: 'dCC3', 151: 'dCC4', 152: 'dCC6', 153: 'dCC7', 154: 'dCM',
                155: 'dCM1', 156: 'dCM2', 157: 'dCM3', 158: 'dCM4', 159: 'dCM6',
                160: 'dCM7', 161: 'dCH5', 162: 'dCS5', 163: 'dCB5', 164: 'dCR5',
                165: 'dML', 166: 'dAB1', 167: 'dAB2', 168: 'dAB3', 169: 'dAB4'
            }

            lead_name = lead_names.get(lead_id, f'Lead_{lead_id}')

            # Create placeholder lead (actual data comes from Section 6)
            self.raw_header[f'Lead_{i}_Name'] = lead_name
            self.raw_header[f'Lead_{i}_ID'] = str(lead_id)

    def _read_rhythm_data(self, f, length: int):
        """Read Section 6 - Rhythm data"""
        start_pos = f.tell()

        # Amplitude scaling
        amplitude_value = struct.unpack('<H', f.read(2))[0]
        amplitude_units = struct.unpack('B', f.read(1))[0]

        # Time scaling
        sample_time_interval = struct.unpack('<H', f.read(2))[0]
        sample_time_units = struct.unpack('B', f.read(1))[0]

        # Calculate sampling rate
        if sample_time_interval > 0:
            # Convert to Hz based on units
            if sample_time_units == 2:  # microseconds
                self.metadata.sampling_rate = 1000000.0 / sample_time_interval
            elif sample_time_units == 3:  # milliseconds
                self.metadata.sampling_rate = 1000.0 / sample_time_interval
            else:
                self.metadata.sampling_rate = 500.0  # Default

        # Encoding type
        encoding_type = struct.unpack('B', f.read(1))[0]

        # Compression type
        compression_type = struct.unpack('B', f.read(1))[0]

        # Reserved
        f.read(3)

        # Read compressed/encoded data
        data_length = length - 12
        raw_data = f.read(data_length)

        # Note: Actual decompression would require implementing
        # Huffman decoding and difference decoding as per SCP-ECG spec
        logger.warning("SCP-ECG rhythm data decompression not fully implemented")

        # Create placeholder leads
        # In a full implementation, this would decompress and decode the actual waveform data
        lead = ECGLead(
            name="Rhythm Data",
            data=[0.0] * 1000,  # Placeholder
            units="uV" if amplitude_units == 2 else "mV",
            gain=float(amplitude_value),
            baseline=0.0,
            sampling_rate=self.metadata.sampling_rate
        )
        self.leads.append(lead)

        self.metadata.num_leads = 1
        self.metadata.duration_sec = 1000 / self.metadata.sampling_rate

    def _read_global_measurements(self, f, length: int):
        """Read Section 7 - Global measurements"""
        # ВИПРАВЛЕННЯ: Перевіряємо мінімальну довжину
        if length < 20:  # Мінімум для базових вимірів
            logger.warning(f"Section 7 too short: {length} bytes")
            # Читаємо що є
            data = f.read(length)
            return

        try:
            # Skip reference beat info
            ref_beat_offset = struct.unpack('<H', f.read(2))[0]
            ref_beat_length = struct.unpack('<H', f.read(2))[0]

            # P onset, offset
            p_onset = struct.unpack('<h', f.read(2))[0]
            p_offset = struct.unpack('<h', f.read(2))[0]

            # QRS onset, offset
            qrs_onset = struct.unpack('<h', f.read(2))[0]
            qrs_offset = struct.unpack('<h', f.read(2))[0]

            # T offset
            t_offset = struct.unpack('<h', f.read(2))[0]

            # P axis
            p_axis = struct.unpack('<h', f.read(2))[0]

            # QRS axis
            qrs_axis = struct.unpack('<h', f.read(2))[0]

            # T axis
            t_axis = struct.unpack('<h', f.read(2))[0]

            # Store measurements
            self.raw_header['P_Duration'] = str(p_offset - p_onset) if p_onset != -1 and p_offset != -1 else 'N/A'
            self.raw_header['QRS_Duration'] = str(
                qrs_offset - qrs_onset) if qrs_onset != -1 and qrs_offset != -1 else 'N/A'
            self.raw_header['P_Axis'] = str(p_axis) if p_axis != -1 else 'N/A'
            self.raw_header['QRS_Axis'] = str(qrs_axis) if qrs_axis != -1 else 'N/A'
            self.raw_header['T_Axis'] = str(t_axis) if t_axis != -1 else 'N/A'

            # Читаємо залишок секції якщо є
            remaining = length - 20
            if remaining > 0:
                f.read(remaining)

        except struct.error as e:
            logger.error(f"Error unpacking global measurements: {e}")
            # Читаємо залишок
            try:
                current = f.tell()
                f.seek(0, 2)  # End of file
                file_size = f.tell()
                f.seek(current)

                remaining = min(length - (current - f.tell()), file_size - current)
                if remaining > 0:
                    f.read(remaining)
            except:
                pass

    def _read_textual_diagnosis(self, f, length: int):
        """Read Section 8 - Textual diagnosis"""
        # Confirmed
        confirmed = struct.unpack('B', f.read(1))[0]

        # Date/time
        f.read(1)  # Year
        f.read(1)  # Month
        f.read(1)  # Day
        f.read(1)  # Hour
        f.read(1)  # Minute
        f.read(1)  # Second

        # Number of statements
        num_statements = struct.unpack('B', f.read(1))[0]

        # Read diagnosis statements
        for _ in range(num_statements):
            # Sequence number
            f.read(1)

            # Statement length
            stmt_length = struct.unpack('B', f.read(1))[0]

            # Statement text
            statement = self._read_string(f, stmt_length - 1)
            if statement:
                self.metadata.diagnoses.append(statement)

    def _read_string(self, f, length: int) -> str:
        """Read null-terminated string"""
        data = f.read(length)
        try:
            # Find null terminator
            null_idx = data.index(b'\x00')
            return data[:null_idx].decode('latin-1')
        except ValueError:
            # No null terminator
            return data.decode('latin-1', errors='ignore').strip()
        except Exception:
            return ""

    def _read_date(self, f) -> Optional[str]:
        """Read SCP date format"""
        year = struct.unpack('<H', f.read(2))[0]
        month = struct.unpack('B', f.read(1))[0]
        day = struct.unpack('B', f.read(1))[0]

        if year > 1900 and year < 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None


# ====================================================================================
# ADDITIONAL FORMAT CONVERTERS
# ====================================================================================

class MitFormatECGConverter(BaseECGConverter):
    """MIT Format (MIT-BIH) converter - essentially an alias for WFDB"""

    def __init__(self, input_filepath: str):
        # MIT format is essentially WFDB format
        super().__init__(input_filepath)

    def _parse_file(self) -> bool:
        # Use WFDB converter
        wfdb_converter = WfdbECGConverter(self.input_filepath)
        if wfdb_converter.convert():
            # Copy results
            self.metadata = wfdb_converter.metadata
            self.leads = wfdb_converter.leads
            self.annotations = wfdb_converter.annotations
            self.raw_header = wfdb_converter.raw_header
            return True
        return False


class PhilipsXMLECGConverter(BaseECGConverter):
    """Philips XML ECG format converter"""

    def _parse_file(self) -> bool:
        try:
            tree = ET.parse(self.input_filepath)
            root = tree.getroot()

            # Philips XML has a specific structure
            # Look for patient info
            patient = root.find('.//patient')
            if patient is not None:
                self.metadata.patient_id = patient.findtext('patientid', 'N/A')

                # Parse name
                name_elem = patient.find('name')
                if name_elem is not None:
                    given = name_elem.findtext('given', '')
                    family = name_elem.findtext('family', '')
                    self.metadata.patient_name = f"{given} {family}".strip()

                # Demographics
                self.metadata.birth_date = patient.findtext('birthdate', 'N/A')
                self.metadata.sex = patient.findtext('gender', 'N/A')

            # Look for waveform data
            waveforms = root.findall('.//waveform')

            for waveform in waveforms:
                # Get lead info
                lead_name = waveform.findtext('leadname', 'Unknown')

                # Get sampling info
                sampling_rate = float(waveform.findtext('samplerate', '500'))

                # Get waveform data
                data_elem = waveform.find('data')
                if data_elem is not None and data_elem.text:
                    # Parse samples
                    samples = []
                    for sample in data_elem.text.split():
                        try:
                            samples.append(float(sample))
                        except:
                            continue

                    if samples:
                        lead = ECGLead(
                            name=lead_name,
                            data=samples,
                            units='mV',
                            gain=1.0,
                            baseline=0.0,
                            sampling_rate=sampling_rate
                        )
                        self.leads.append(lead)

            # Update metadata
            if self.leads:
                self.metadata.sampling_rate = self.leads[0].sampling_rate
                self.metadata.num_leads = len(self.leads)
                self.metadata.duration_sec = len(self.leads[0].data) / self.metadata.sampling_rate

            return len(self.leads) > 0

        except Exception as e:
            logger.error(f"Error parsing Philips XML: {e}")
            return False


class GEMuseXMLConverter(BaseECGConverter):
    """GE MUSE XML format converter"""

    def _parse_file(self) -> bool:
        try:
            tree = ET.parse(self.input_filepath)
            root = tree.getroot()

            # GE MUSE has RestingECG as root usually
            if 'RestingECG' not in root.tag:
                logger.warning("May not be a GE MUSE XML file")

            # Patient demographics
            patient = root.find('.//PatientDemographics')
            if patient is not None:
                self.metadata.patient_id = patient.findtext('PatientID', 'N/A')
                self.metadata.patient_name = ' '.join(filter(None, [
                    patient.findtext('PatientFirstName', ''),
                    patient.findtext('PatientLastName', '')
                ]))
                self.metadata.birth_date = patient.findtext('DateofBirth', 'N/A')
                self.metadata.sex = patient.findtext('Gender', 'N/A')

            # Test info
            test_info = root.find('.//TestDemographics')
            if test_info is not None:
                # Parse acquisition time
                acq_date = test_info.findtext('AcquisitionDate', '')
                acq_time = test_info.findtext('AcquisitionTime', '')

                if acq_date:
                    self.metadata.recording_date = acq_date
                if acq_time:
                    self.metadata.recording_time = acq_time

            # Waveform data
            waveform = root.find('.//Waveform')
            if waveform is not None:
                # Get rhythm leads
                rhythm_leads = waveform.findall('.//LeadData')

                for lead_data in rhythm_leads:
                    lead_name = lead_data.findtext('LeadID', 'Unknown')

                    # Get waveform samples
                    waveform_data = lead_data.findtext('WaveFormData', '')
                    if waveform_data:
                        # GE MUSE uses base64 encoding
                        import base64
                        try:
                            decoded = base64.b64decode(waveform_data)
                            # Interpret as 16-bit signed integers
                            samples = []
                            for i in range(0, len(decoded), 2):
                                if i + 1 < len(decoded):
                                    sample = struct.unpack('<h', decoded[i:i + 2])[0]
                                    samples.append(float(sample))

                            if samples:
                                lead = ECGLead(
                                    name=lead_name,
                                    data=samples,
                                    units='uV',
                                    gain=1.0,
                                    baseline=0.0,
                                    sampling_rate=500.0  # GE MUSE default
                                )
                                self.leads.append(lead)

                        except Exception as e:
                            logger.error(f"Error decoding waveform data: {e}")

            # Update metadata
            if self.leads:
                self.metadata.num_leads = len(self.leads)
                self.metadata.sampling_rate = 500.0  # Standard for GE MUSE
                self.metadata.duration_sec = len(self.leads[0].data) / self.metadata.sampling_rate

            return len(self.leads) > 0

        except Exception as e:
            logger.error(f"Error parsing GE MUSE XML: {e}")
            return False


# ====================================================================================
# FORMAT DETECTION
# ====================================================================================

def detect_file_format(filepath: str) -> Optional[str]:
    """Auto-detect ECG file format"""

    # Check by extension first
    ext = os.path.splitext(filepath)[1].lower()

    extension_map = {
        '.dcm': 'dicom',
        '.dicom': 'dicom',
        '.dat': 'wfdb',
        '.hea': 'wfdb',
        '.edf': 'edf',
        '.rec': 'edf',
        '.bdf': 'edf',
        '.csv': 'csv',
        '.txt': 'csv',  # Often CSV files have .txt extension
        '.xml': None,  # Need to check XML type
        '.scp': 'scp',
        '.ecg': None  # Could be various formats
    }

    format_hint = extension_map.get(ext)

    if format_hint and format_hint != 'xml':
        return format_hint

    # For XML files or unknown extensions, check content
    try:
        # Check if it's a text file
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read first few lines
            lines = []
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                lines.append(line)

            content = '\n'.join(lines)

            # Check for XML
            if '<?xml' in content or '<' in content:
                # Determine XML type
                if 'hl7' in content.lower() or 'urn:hl7' in content:
                    return 'hl7aecg'
                elif 'philips' in content.lower():
                    return 'philips_xml'
                elif 'restingecg' in content.lower() or 'muse' in content.lower():
                    return 'ge_muse'
                else:
                    # Generic HL7 assumption for unknown XML
                    return 'hl7aecg'

            # Check for CSV patterns
            elif any(delimiter in content for delimiter in [',', '\t', ';', '|']):
                # Likely CSV
                return 'csv'

    except UnicodeDecodeError:
        # Binary file
        pass

    # Check binary file signatures
    try:
        with open(filepath, 'rb') as f:
            # Read file signature
            signature = f.read(128)

            # DICOM
            if b'DICM' in signature[128:132] if len(signature) > 132 else False:
                return 'dicom'

            # EDF
            if signature.startswith(b'0       '):
                return 'edf'

            # SCP-ECG has specific structure
            if len(signature) >= 6:
                # Check for valid CRC and file size fields
                try:
                    crc = struct.unpack('<H', signature[0:2])[0]
                    file_size = struct.unpack('<I', signature[2:6])[0]

                    # Sanity check file size
                    actual_size = os.path.getsize(filepath)
                    if abs(file_size - actual_size) < 100:  # Allow small difference
                        return 'scp'
                except:
                    pass

    except Exception as e:
        logger.debug(f"Error checking file signature: {e}")

    # Check for WFDB by looking for .hea file
    base_name = os.path.splitext(filepath)[0]
    if os.path.exists(base_name + '.hea'):
        return 'wfdb'

    return None


# ====================================================================================
# HIGH-LEVEL CONVERSION FUNCTIONS
# ====================================================================================

def convert_ecg_to_xml(input_filepath: str, output_filepath: str,
                       file_format: Optional[str] = None, **kwargs) -> bool:
    """
    Convert ECG file to XML format

    Args:
        input_filepath: Path to input ECG file
        output_filepath: Path to output XML file
        file_format: Format of input file (auto-detect if None)
        **kwargs: Additional format-specific parameters

    Returns:
        bool: True if successful, False otherwise
    """

    # Auto-detect format if not specified
    if not file_format:
        file_format = detect_file_format(input_filepath)

        if not file_format:
            logger.error(f"Could not detect file format for {input_filepath}")
            return False

        logger.info(f"Auto-detected format: {file_format}")

    # Map format to converter class
    converter_map = {
        'dicom': DicomECGConverter,
        'wfdb': WfdbECGConverter,
        'edf': EdfECGConverter,
        'hl7aecg': Hl7aECGConverter,
        'csv': CsvECGConverter,
        'scp': ScpECGConverter,
        'mit': MitFormatECGConverter,
        'philips_xml': PhilipsXMLECGConverter,
        'ge_muse': GEMuseXMLConverter
    }

    converter_class = converter_map.get(file_format.lower())

    if not converter_class:
        logger.error(f"Unsupported format: {file_format}")
        return False

    try:
        # Create converter instance
        if file_format == 'csv':
            # CSV converter accepts additional parameters
            converter = converter_class(input_filepath, **kwargs)
        else:
            converter = converter_class(input_filepath)

        # Perform conversion
        if converter.convert():
            # Save XML
            return converter.save_xml(output_filepath)
        else:
            logger.error("Conversion failed")
            return False

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        traceback.print_exc()
        return False


def batch_convert_directory(input_dir: str, output_dir: str,
                            file_pattern: str = "*",
                            format_map: Optional[Dict[str, str]] = None,
                            parallel: bool = True,
                            max_workers: Optional[int] = None) -> Dict[str, bool]:
    """
    Convert all ECG files in a directory

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        file_pattern: File pattern to match (e.g., "*.dcm")
        format_map: Optional dict mapping file extensions to formats
        parallel: Whether to use parallel processing
        max_workers: Number of parallel workers

    Returns:
        Dict mapping input files to conversion success status
    """

    import glob

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all matching files
    pattern_path = os.path.join(input_dir, file_pattern)
    input_files = glob.glob(pattern_path)

    if not input_files:
        logger.warning(f"No files found matching {pattern_path}")
        return {}

    logger.info(f"Found {len(input_files)} files to convert")

    if parallel and len(input_files) > 1:
        # Use parallel processing
        processor = ParallelProcessor(max_workers)

        # Prepare format map
        if not format_map:
            format_map = {}

        # Group files by format for efficient processing
        format_groups = {}
        for filepath in input_files:
            ext = os.path.splitext(filepath)[1].lower()
            format_name = format_map.get(ext) or detect_file_format(filepath)

            if format_name:
                if format_name not in format_groups:
                    format_groups[format_name] = []
                format_groups[format_name].append(filepath)

        # Process each format group
        all_results = {}

        for format_name, files in format_groups.items():
            converter_class = {
                'dicom': DicomECGConverter,
                'wfdb': WfdbECGConverter,
                'edf': EdfECGConverter,
                'hl7aecg': Hl7aECGConverter,
                'csv': CsvECGConverter,
                'scp': ScpECGConverter
            }.get(format_name)

            if converter_class:
                results = processor.batch_process_files(files, converter_class, output_dir)
                all_results.update(results)

        return all_results

    else:
        # Sequential processing
        results = {}

        for i, filepath in enumerate(input_files):
            logger.info(f"Converting {i + 1}/{len(input_files)}: {os.path.basename(filepath)}")

            # Determine format
            ext = os.path.splitext(filepath)[1].lower()
            format_name = (format_map.get(ext) if format_map else None) or detect_file_format(filepath)

            if not format_name:
                logger.error(f"Could not determine format for {filepath}")
                results[filepath] = False
                continue

            # Output file
            output_name = os.path.splitext(os.path.basename(filepath))[0] + '.xml'
            output_path = os.path.join(output_dir, output_name)

            # Convert
            success = convert_ecg_to_xml(filepath, output_path, format_name)
            results[filepath] = success

        return results


def get_converter_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available converters"""

    info = {
        'dicom': {
            'name': 'DICOM',
            'extensions': ['.dcm', '.dicom'],
            'available': PYDICOM_AVAILABLE,
            'required_package': 'pydicom',
            'description': 'Digital Imaging and Communications in Medicine'
        },
        'wfdb': {
            'name': 'WFDB/MIT',
            'extensions': ['.dat', '.hea'],
            'available': WFDB_AVAILABLE,
            'required_package': 'wfdb',
            'description': 'Waveform Database format (PhysioNet)'
        },
        'edf': {
            'name': 'EDF/EDF+',
            'extensions': ['.edf', '.rec', '.bdf'],
            'available': PYEDFLIB_AVAILABLE or MNE_AVAILABLE,
            'required_package': 'pyedflib or mne',
            'description': 'European Data Format'
        },
        'hl7aecg': {
            'name': 'HL7 aECG',
            'extensions': ['.xml'],
            'available': LXML_AVAILABLE,
            'required_package': 'lxml',
            'description': 'HL7 Annotated ECG XML standard'
        },
        'csv': {
            'name': 'CSV',
            'extensions': ['.csv', '.txt'],
            'available': True,
            'required_package': None,
            'description': 'Comma-separated values'
        },
        'scp': {
            'name': 'SCP-ECG',
            'extensions': ['.scp'],
            'available': True,
            'required_package': None,
            'description': 'Standard Communications Protocol for ECG (EN1064)'
        }
    }

    return info


# ====================================================================================
# COMMAND LINE INTERFACE
# ====================================================================================

def main():
    """Enhanced command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ECG File Converter v2.0 - Convert various ECG formats to XML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.dcm output.xml
  %(prog)s input.edf -f edf
  %(prog)s -b input_dir/ output_dir/ -p "*.dcm"
  %(prog)s --list-formats
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('input_file', nargs='?', help='Input ECG file')
    mode_group.add_argument('-b', '--batch', nargs=2, metavar=('INPUT_DIR', 'OUTPUT_DIR'),
                            help='Batch convert directory')
    mode_group.add_argument('--list-formats', action='store_true',
                            help='List available formats and their status')

    parser.add_argument('output_file', nargs='?', help='Output XML file')

    # Conversion options
    parser.add_argument('-f', '--format',
                        choices=['dicom', 'wfdb', 'edf', 'hl7aecg', 'csv', 'scp'],
                        help='Input file format (auto-detect if not specified)')
    parser.add_argument('-c', '--compress', action='store_true',
                        help='Compress output XML with gzip')
    parser.add_argument('-p', '--pattern', default='*',
                        help='File pattern for batch mode (default: *)')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Use parallel processing for batch mode')
    parser.add_argument('-w', '--workers', type=int,
                        help='Number of parallel workers')

    # CSV options
    csv_group = parser.add_argument_group('CSV options')
    csv_group.add_argument('--csv-delimiter', default=',',
                           help='CSV delimiter (default: ,)')
    csv_group.add_argument('--csv-header-rows', type=int, default=0,
                           help='Number of header rows')
    csv_group.add_argument('--csv-lead-names-row', type=int,
                           help='Row containing lead names')
    csv_group.add_argument('--csv-data-start-row', type=int, default=1,
                           help='First data row')
    csv_group.add_argument('--csv-sampling-rate', type=float,
                           help='Sampling rate in Hz')

    # Logging
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode')

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # List formats mode
    if args.list_formats:
        print("\nAvailable ECG Formats:")
        print("-" * 60)

        info = get_converter_info()
        for format_id, details in info.items():
            status = "✓" if details['available'] else "✗"
            print(f"{status} {details['name']:<15} {', '.join(details['extensions']):<20} {details['description']}")
            if not details['available'] and details['required_package']:
                print(f"  → Install with: pip install {details['required_package']}")

        print()
        return 0

    # Batch mode
    if args.batch:
        input_dir, output_dir = args.batch

        if not os.path.isdir(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return 1

        results = batch_convert_directory(
            input_dir, output_dir,
            file_pattern=args.pattern,
            parallel=args.parallel,
            max_workers=args.workers
        )

        # Summary
        successful = sum(1 for success in results.values() if success)
        print(f"\nBatch conversion complete:")
        print(f"  Successful: {successful}/{len(results)}")

        if successful < len(results):
            print("\nFailed conversions:")
            for filepath, success in results.items():
                if not success:
                    print(f"  - {os.path.basename(filepath)}")

        return 0 if successful == len(results) else 1

    # Single file mode
    if not args.input_file:
        parser.error("Input file required (or use -b for batch mode)")

    if not args.output_file:
        # Generate output filename
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = base_name + '.xml'
        if args.compress:
            args.output_file += '.gz'

    # Prepare CSV parameters
    csv_params = {}
    if args.format == 'csv' or args.input_file.lower().endswith(('.csv', '.txt')):
        csv_params = {
            'delimiter': args.csv_delimiter,
            'num_header_rows': args.csv_header_rows,
            'lead_names_row': args.csv_lead_names_row,
            'data_start_row': args.csv_data_start_row,
            'sampling_rate_hz': args.csv_sampling_rate
        }

    # Convert
    print(f"Converting {args.input_file} -> {args.output_file}")

    success = convert_ecg_to_xml(
        args.input_file,
        args.output_file,
        file_format=args.format,
        **csv_params
    )

    if success:
        print("✓ Conversion successful")

        # Show summary if verbose
        if args.verbose:
            # Read back XML to show summary
            try:
                tree = ET.parse(args.output_file if not args.compress else args.output_file.replace('.gz', ''))
                root = tree.getroot()

                print("\nConversion Summary:")
                print(f"  Format: {root.get('SourceFormat', 'Unknown')}")

                patient = root.find('.//PatientInfo')
                if patient is not None:
                    print(f"  Patient: {patient.findtext('Name', 'N/A')} (ID: {patient.findtext('ID', 'N/A')})")

                acq = root.find('.//AcquisitionDetails')
                if acq is not None:
                    print(f"  Sampling Rate: {acq.findtext('SamplingFrequency', 'N/A')} Hz")
                    print(f"  Duration: {acq.findtext('RecordDuration', 'N/A')} seconds")
                    print(f"  Channels: {acq.findtext('NumberOfChannels', 'N/A')}")

            except:
                pass

        return 0
    else:
        print("✗ Conversion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())