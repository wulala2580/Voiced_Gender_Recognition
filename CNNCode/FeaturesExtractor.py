import numpy as np
from sklearn import preprocessing
import warnings
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class FeaturesExtractor:
    def __init__(self):
        pass


    def extract_features(self, audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.

        Args:
            audio_path (str) : path to wave file without silent moments.
        Returns:
            (array) : Extracted features matrix.
        """
        y, sr = librosa.load(audio_path, sr=16000)
        # Preemphasis pre-emphasis，预加重
        preemphasis = .97
        y = np.append(y[0], y[1:] - preemphasis * y[:-1])

        timeseries_length = 32


        # mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=20)  # shape=(n_mfcc, t)
        melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)

        # convert to log scale
        logmelspec = librosa.power_to_db(melspec)
        # chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

        # mfccs = preprocessing.scale(mfccs)
        logmelspec = preprocessing.scale(logmelspec)




        cmap = plt.get_cmap("OrRd")
        rgb_img = cmap(logmelspec)
        rgb_img = np.delete(rgb_img, 3, 2)
        features = []
        numbers = np.shape(rgb_img)[1] // (timeseries_length // 2)

        for i in range(numbers-1):
            part_image = rgb_img[0:128, 16*i:(16*(i+2))]
            features.append(part_image)

        last_part_image = rgb_img[0:128, 16*(numbers-1):np.shape(rgb_img)[1]]
        last_part_image = cv2.resize(last_part_image, (32, 128))

        features.append(last_part_image)

        return features