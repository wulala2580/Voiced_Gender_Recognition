import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm
from keras.models import load_model
from sklearn.mixture import GMM
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
warnings.filterwarnings("ignore")

import pydub
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from subprocess import Popen, PIPE
from pydub.silence import split_on_silence, detect_nonsilent

from sklearn.svm import SVC


class GenderIdentifier:

    def __init__(self, files_path, model_path):
        self.testing_path = files_path
        self.error = 0
        self.total_sample = 0
        self.features_extractor = FeaturesExtractor()
        self.model = load_model(model_path)

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
        files = self.get_file_paths(self.testing_path)
        # read the test directory and get the list of test audio files
        for file in files:
            self.total_sample += 1
            print("----------------------------------------------------")
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)
            feature = []
            genders = {0: "female", 1: "male"}
            for i in vector:
                feature.append(i)
            # feature.append(vector)
            feature = np.asarray(feature)
            pre_result = self.model.predict(feature, batch_size=64).argmax(axis=1)
            length = len(pre_result)
            initial = pre_result[0]
            print("The gender at  1 s is ",genders[initial])
            for i in range(2,length-1,2):
                if pre_result[i] == pre_result[i-1]:
                    print("The gender at ", i // 2 + 1, "s is ", genders[pre_result[i]] )
                elif pre_result[i] != pre_result[i+1]:
                    print("The gender at ", i // 2 + 1, "s is ", genders[pre_result[i-1]])
                else:
                    print("The gender at ", i // 2 + 1, "s is ", genders[pre_result[i]])



            print("----------------------------------------------------")



    def get_file_paths(self, testing_path):
        # get file paths
        files = [os.path.join(testing_path, f).replace("\\", "/") for f in os.listdir(testing_path)]
        return files



if __name__ == "__main__":
    gender_identifier = GenderIdentifier("TestingData", "CNNModel.h5")
    gender_identifier.process()
