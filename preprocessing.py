import os
import mne
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from bisect import bisect_right
from scipy import signal
from scipy.signal import kaiserord, firwin, filtfilt

edf_path = './polysomnography/edfs/baseline'
annotation_path = './polysomnography/annotations-events-nsrr/baseline'

def parse_xml_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    onsets = []
    durations = []
    descriptions = []
    
    events = root.findall('.//ScoredEvent')[1:]  # Recording Start Time 제외

    for event in events:
        start = float(event.find('Start').text)
        duration = float(event.find('Duration').text)
        concept = event.find('EventConcept').text  # annotations 정보
        
        onsets.append(start)
        durations.append(duration)
        descriptions.append(concept)
        
    return onsets, durations, descriptions

def load_edf_and_annotations(edf_file_path, annotation_file_path):
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    onsets, durations, descriptions = parse_xml_annotations(annotation_file_path)
    
    return raw, onsets, durations, descriptions

def resample_data(data, orig_freq, target_freq=10):
    num_samples = int(len(data) * (target_freq / orig_freq))
    resampled_data = signal.resample(data, num_samples)
    return resampled_data

def low_pass_filter(data, sample_rate, cutoff=1.5, stopband=2, attenuation=100):
    nyq_rate = sample_rate / 2.0
    width = stopband - cutoff
    ripple_db = attenuation

    N, beta = kaiserord(ripple_db, width / nyq_rate)

    taps = firwin(N, cutoff / nyq_rate, window=('kaiser', beta))

    filtered_data = filtfilt(taps, 1.0, data, axis=0)
    
    return filtered_data

def normalize_amplitude(data):
    baseline = np.median(data, axis=0)
    normalized_data = (data - baseline) / (np.max(data, axis=0) - np.min(data, axis=0))
    return normalized_data

def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data

def preprocess_signals(raw, target_freq=10):
    # Channel 이름 대소문자 구분 X (SAO2, SaO2..)
    available_channels = {ch_name.upper(): ch_name for ch_name in raw.ch_names}
    
    required_channels = ['AIRFLOW', 'SAO2']
    
    selected_channels = []
    for ch in required_channels:
        if ch in available_channels:
            selected_channels.append(available_channels[ch])
        else:
            print(f"channel {ch} not found")
            return None, None, None
    
    # transpose -> axis=0에 대해 resampling
    data = raw.get_data(picks=selected_channels).T
    orig_freq = raw.info['sfreq']
    meas_date = raw.info['meas_date']

    data = resample_data(data, orig_freq, target_freq)

    data[:, 0] = low_pass_filter(data[:, 0], target_freq, cutoff=1.5, stopband=2, attenuation=100)

    data[:, 0] = normalize_amplitude(data[:, 0])

    data = standardize_data(data)

    times = np.arange(0, len(data) / target_freq, 1 / target_freq)
    
    return data, times, meas_date

def match_annotations_to_times(times, meas_date, onsets, durations, descriptions):
    event_times = [(onset, onset + duration, description) for onset, duration, description in zip(onsets, durations, descriptions)]
    event_times.sort()

    annotations = ['normal'] * len(times)  # 기본 : normal
    
    event_starts = [onset for onset, _, _ in event_times]
    for i, time in enumerate(times):
        event_index = bisect_right(event_starts, time) - 1
        if event_index >= 0:
            onset, end, description = event_times[event_index]
            if onset <= time <= end:
                annotations[i] = description
    
    return annotations

def save_to_csv(data, times, annotations, channels, meas_date, study_id):
    output_folder = 'chat_csv'
    
    output_file = os.path.join(output_folder, f'{study_id}.csv')
    
    df = pd.DataFrame(data, columns=channels)
    df['EDF_Time'] = meas_date + pd.to_timedelta(times, unit='s')
    df['Annotation'] = annotations
    
    # 예측 컬럼 생성 (apnea)
    df['apnea'] = df['Annotation'].apply(lambda x: 1 if 'apnea' in x.lower() or 'hypopnea' in x.lower() else 0)
    
    df.to_csv(output_file, index=False)
    print(f"Data saved for {study_id} in {output_file}")


def process_all_files_in_folder(edf_path, annotation_path):
    edf_files = [f for f in os.listdir(edf_path) if f.endswith('.edf')]
    
    for edf_file in edf_files:
        study_id = edf_file.replace('chat-baseline-', '').replace('.edf', '')
        edf_file_path = os.path.join(edf_path, edf_file)
        annotation_file_path = os.path.join(annotation_path, f'chat-baseline-{study_id}-nsrr.xml')
        
        if os.path.exists(annotation_file_path):
            raw, onsets, durations, descriptions = load_edf_and_annotations(edf_file_path, annotation_file_path)
            data, times, meas_date = preprocess_signals(raw, target_freq=10)
            if data is not None:
                annotations = match_annotations_to_times(times, meas_date, onsets, durations, descriptions)
                save_to_csv(data, times, annotations, ['Airflow', 'SAO2'], meas_date, study_id)
        else:
            print(f"file missing for {study_id}")

if __name__ == '__main__':
    process_all_files_in_folder(edf_path, annotation_path)
