import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
from sklearn.mixture import GMM
from sklearn.svm import SVC

import pydub
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from subprocess import Popen, PIPE
from pydub.silence import split_on_silence, detect_nonsilent

from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

warnings.filterwarnings("ignore")


class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path = males_files_path
        self.features_extractor = FeaturesExtractor()

    def ffmpeg_silence_eliminator(self, input_path, output_path):
        """
        Eliminate silence from voice file using ffmpeg library.
        Args:
            input_path  (str) : Path to get the original voice file from.
            output_path (str) : Path to save the processed file to.
        Returns:
            (list)  : List including True for successful authentication, False otherwise and a percentage value
                      representing the certainty of the decision.
        """
        # filter silence in mp3 file
        filter_command = ["ffmpeg", "-i", input_path, "-af", "silenceremove=1:0:0.05:-1:1:-36dB", "-ac", "1", "-ss",
                          "0", "-t", "90", output_path, "-y"]
        out = subprocess.Popen(filter_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out.wait()

        with_silence_duration = os.popen(
            "ffprobe -i '" + input_path + "' -show_format -v quiet | sed -n 's/duration=//p'").read()
        no_silence_duration = os.popen(
            "ffprobe -i '" + output_path + "' -show_format -v quiet | sed -n 's/duration=//p'").read()

        # print duration specs
        try:
            print("%-32s %-7s %-50s" % ("ORIGINAL SAMPLE DURATION", ":", float(with_silence_duration)))
            print("%-23s %-7s %-50s" % ("SILENCE FILTERED SAMPLE DURATION", ":", float(no_silence_duration)))
        except:
            print("WaveHandlerError: Cannot convert float to string", with_silence_duration, no_silence_duration)

        # convert file to wave and read array
        load_command = ["ffmpeg", "-i", output_path, "-f", "wav", "-"]
        p = Popen(load_command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        data = p.communicate()[0]
        audio_np = np.frombuffer(data[data.find(b'\x00data') + 9:], np.int16)

        # delete temp silence free file, as we only need the array
        # os.remove(output_path)
        return audio_np, no_silence_duration

    def process(self):
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        files = females + males
        files = sorted(files)
        # collect voice features
        features = []
        labels = []


        for file in files:
            print("%10s %8s %1s" % ("--> TRAINING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)

            # gender = file.split("/")[1][:-1]
            gender = file.split("\\")[0].split("/")[1][:-1]
            print(gender)
            for i in vector:
                features.append(i)
                if gender == "female":
                    labels.append(0)
                else:
                    labels.append(1)


        features = np.asarray(features)
        labels = np.asarray(labels)
        labels = tf.squeeze(labels)
        labels = tf.one_hot(labels,depth=2)


        self.X_train = features
        self.y_train = labels


        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 32, 3))
        for layer in vgg16_model.layers:
            layer.trainable = False

        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        top_model.add(Dense(512, activation='relu'))
        top_model.add(Dropout(0.4))
        top_model.add(Dense(2, activation='softmax'))

        model = Sequential()
        model.add(vgg16_model)
        model.add(top_model)
        # sgd = SGD(learning_rate=0.05, decay=1e-5)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.fit(self.X_train, self.y_train, batch_size=64, epochs=20)
        # model.evaluate(X_test, y_test)

        model.save("CNNModel.h5", save_format="h5")

    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path)]
        males = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path)]
        return females, males



if __name__ == "__main__":
    models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
    models_trainer.process()
