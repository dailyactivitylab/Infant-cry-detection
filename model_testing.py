import librosa
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from joblib import load

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    [Fs, x] = audioBasicIO.read_audio_file(file_path)
    F, _ = ShortTermFeatures.feature_extraction(y, sr, 1 * sr, 0.5 * sr) 
    F = F[:, 0::2]
     
    print("len of audio",duration)
    feature_windows = []
    windows = []
    item = 0
    for i in range(0, int(duration) - 4):
        windows.append([item + i, item + i + 5])

    print("len of windows",len(windows))
    for item in windows:
        F_window = F[:, int(item[0]) : int(item[1])]
        F_feature = np.concatenate((np.mean(F_window, axis = 1), np.median(F_window, axis = 1), np.std(F_window, axis = 1)), axis = None)
        feature_windows.append(F_feature)
    return np.array(feature_windows)

from collections import Counter

def majority_voting(predictions):
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    return most_common

def predict_audio(features):
    model = load('') # Put model downloaded location
    predictions = model.predict(features)
    return predictions

file_path = '' #Put audio path
features = preprocess_audio(file_path)
print(features)
predictions = predict_audio(features)
print(predictions)
print("len of predictions",len(predictions))
final_label = majority_voting(predictions)
if final_label == 0:
    print("Prefiction is :", "Other")
else:
    print("Prefiction is :", "Cry")
