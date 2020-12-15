#Voice Based Gender Recognition Model
This model can detect the gender of the current voice every second. It is based on CNN.
##Test The Code
In order to test the result of this code, just run

    python CNNCode/GenderIdentifier.py
##Train The Model
In order to train your own model, you should first have your labeled data, and place them separately in ```TrainingData/females``` and  ```TrainingData/males```, and then run the code

    python CNNCode/ModelsTrainer.py
