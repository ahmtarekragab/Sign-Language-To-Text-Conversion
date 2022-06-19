# Sign Language to Text Conversion


## Abstract

Sign language is one of the oldest and most natural form of language for communication, but since most people do not know sign language and interpreters are very difficult to come by we have come up with a real time method using neural networks for fingerspelling based american sign language.
 
In this method, the hand is first passed through a filter and after the filter is applied the hand is passed through a classifier which predicts the class of the hand gestures. This method provides 98.00 % accuracy for the 26 letters of the alphabet.

## Project Description

American sign language is a predominant sign language Since the only disability D&M people have is communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language. 

Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior and visuals. 

Deaf and Mute(Dumb)(D&M) people make use of their hands to express different gestures to express their ideas with other people. 

Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language. 

Sign language is a visual language and consists of 3 major components 

![components](images/components.jpg)

In this project I basically focus on producing a model which can recognize Fingerspelling based hand gestures in order to form a complete word by combining each gesture. 

The gestures I  trained are as given in the image below.

![Signs](images/signs.jpg)

### Prerequisites.

⦁	To complete this project, we needed the following: A local development environment for Python 3 with at least 1GB of RAM. 
⦁	A working webcam to do real-time image detection.

# Steps of building this project

### 1. Libraries Requirements - (Requires the latest pip version to install all the packages.


``` python
# Importing the Libraries Required

1. Lastest pip -> pip install --upgrade pip

2. numpy -> pip install numpy

3. string -> pip install strings

4. os-sys -> pip install os-sys

5. opencv -> pip install opencv-python

6. tensorFlow -> i) pip install tensorflow 
                 ii) pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl

7. keras -> pip install keras

8. tkinter -> pip install tk

9. PIL -> pip install Pillow

10. enchant -> pip install pyenchant (Python bindings for the Enchant spellchecking system)

11. hunspell -> pip install cyhunspell (A wrapper on hunspell for use in Python)
```

### 2. Preparing the Sign Language Classifications Dataset

⦁ We will build a sign language classifier using a neural network. Our goal is to produce a model that accepts a picture of a hand as input and outputs a letter.
⦁ First, we download the database to our current working directory
- As before, we imported the necessary utilities to create the class that will hold our data. For data processing here, we will create the train and test datasets folders.

![Capture](https://user-images.githubusercontent.com/56514238/174483629-59ebc726-c2de-42e7-b2e1-cda8a5902c8e.PNG)


#### 3. Building and Training the Sign Language Classifier Using Deep Learning

We built a neural network with layers, define a loss, an optimizer, and finally, optimize the loss function for our neural network predictions. At the end of this step, we will have a working sign language classifier.
⦁ Create a new file called train.py
⦁ Import the necessary utilities

``` python
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
```

### Data preprocessing using Keras

With Keras preprocessing layers, you can build and export models that are truly end-to-end: models that accept raw images or raw structured data as input; models that handle feature normalization or feature value indexing on their own. Image preprocessing These layers are for standardizing the inputs of an image model.

``` python
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=3, 
                                     padding="same", 
                                     activation="relu", 
                                     input_shape=[128, 128, 1]))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, 
                                         strides=2, 
                                         padding='valid'))
classifier.add(tf.keras.layers.Conv2D(filters=32, 
                                      kernel_size=3, 
                                      padding="same", 
                                      activation="relu"))

classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, 
                                         strides=2, 
                                         padding='valid'))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(units=128, 
                                     activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.40))
classifier.add(tf.keras.layers.Dense(units=96, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.40))
classifier.add(tf.keras.layers.Dense(units=64, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=27, activation='softmax')) # softmax for more than 2
classifier.compile(optimizer = 'adam', 
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
classifier.summary()
classifier.fit(training_set,
                  epochs = 5,
                  validation_data = test_set)
model_json = classifier.to_json()
```

### Saving Models and Weights

``` python
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model_new.h5')
print('Weights saved')
```

### Define Keras neural network that includes the layers

``` python
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.__version__ 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('E:/Graduation project/Sign-Language-To-Text-Conversion-main/dataSet/trainingData',                                
                                                 target_size = (128, 128),
                                                 batch_size = 10,
                                                 color_mode = 'grayscale',                                
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('E:/Graduation project/Sign-Language-To-Text-Conversion-main/dataSet/testingData',
                                            target_size = (128, 128),                                  
                                            batch_size = 10,        
                                            color_mode = 'grayscale',
                                            class_mode = 'categorical')
```

### 	After this we initialized the neural network, defined a loss function

Launching our train.py file to see our neural network trains:

![Model Summary](images/model_summary.PNG)

### ⦁	We made 5 epochs as the following

![Output](images/output.PNG)

To obtain lower loss, we could increase the number of epochs to 10, or even 20. However, after a certain period of training time, the network loss will cease to decrease with increased training time. To sidestep this issue, as training time increases.

###4. Evaluating the Sign Language Classifier

⦁ Create a new file called Application.py
⦁ Import the necessary utilities 

``` python
import numpy as np

import cv2
import os, sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

from hunspell import Hunspell
import enchant

from keras.models import model_from_json
```

⦁ First of all, we defined a class called Application then we defined __init__ method in it
⦁ then we imported all the models we used in our application

``` python
class Application:

    def __init__(self):

        self.hs = Hunspell('en_US')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open("E:/Graduation project/Sign-Language-To-Text-Conversion-main/Models/model_new.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("E:/Graduation project/Sign-Language-To-Text-Conversion-main/Models/model_new.h5")

        self.json_file_dru = open("E:/Graduation project/Sign-Language-To-Text-Conversion-main/Models/model-bw_dru.json" , "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()

        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("E:/Graduation project/Sign-Language-To-Text-Conversion-main/Models/model-bw_dru.h5")
        self.json_file_tkdi = open("E:/Graduation project/Sign-Language-To-Text-Conversion-main/Models/model-bw_tkdi.json" , "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()

        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights("E:/Graduation project/Sign-Language-To-Text-Conversion-main/Models/model-bw_tkdi.h5")
        self.json_file_smn = open("E:/Graduation project/Sign-Language-To-Text-Conversion-main/Models/model-bw_smn.json" , "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()

        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("E:/Graduation project/Sign-Language-To-Text-Conversion-main/Models/model-bw_smn.h5")
```

⦁ Then we made the GUI (We made it simple, easy to use for right-handed people)
⦁ we used tkinter library to build our project interface

``` python
 self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 10, width = 580, height = 580)
        
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 400, y = 65, width = 275, height = 275)

        self.T = tk.Label(self.root)
        self.T.place(x = 60, y = 5)
        self.T.config(text = "Sign Language To Text Conversion", font = ("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 500, y = 540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10, y = 540)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220, y = 595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 595)
        self.T2.config(text = "Word :", font = ("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 350, y = 645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10, y = 645)
        self.T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250, y = 690)
        self.T4.config(text = "Suggestions :", fg = "red", font = ("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command = self.action1, height = 0, width = 0)
        self.bt1.place(x = 26, y = 745)

        self.bt2 = tk.Button(self.root, command = self.action2, height = 0, width = 0)
        self.bt2.place(x = 325, y = 745)

        self.bt3 = tk.Button(self.root, command = self.action3, height = 0, width = 0)
        self.bt3.place(x = 625, y = 745)
```

### The GUI (Graphical User Interface) of the application is as shown below

![Capture](https://user-images.githubusercontent.com/56514238/174484342-3e90f6fa-110b-46bf-bbd0-6a0867d0f196.PNG)

⦁ Then we added a method loop, which reads from the camera at every timestep:

``` python
 def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)
```

⦁ We captured each frame shown by the webcam of our machine 
⦁ In each frame I defined a region of interest (ROI) which is denoted by a blue bounded square as shown in the image below.

![trainingdata](https://user-images.githubusercontent.com/56514238/174484396-2cf6b1b3-3839-43e1-b2c8-b095d616ce09.png)

⦁ The code for image processing is as following:

``` python
cv2image = cv2image[y1 : y2, x1 : x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

⦁ And this is how it looks like after applying gaussian blur filter on it:

![roi](https://user-images.githubusercontent.com/56514238/174484442-4fd8083f-95bd-4a6d-a664-18a8ca77a297.png)

### Testing

we defined a method to predict the input image 

``` python
 def predict(self, test_image):

        test_image = cv2.resize(test_image, (128, 128))

        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))


        result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))

        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))

        result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))

        prediction = {}

        prediction['blank'] = result[0][0]

        inde = 1
```

⦁ While testing the applications we found out that some of the symbol predictions were coming out wrong.
 So, we used two layers of algorithms to verify and predict symbols which are more similar to each other so that I can get close as I can to detect the symbol shown.
 In our testing the following symbols were not showing properly and were giving output as other symbols: 
1. For D: R and U 
2. For U: D and R 
3. For I: T, D, K and I 
4. For S: M and N 

So, to handle above cases we made three different classifiers for classifying these sets: 
1. {D, R, U}
2. {T, K, D, I} 
3. {S, M, N}

``` python
#LAYER 1

        prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        self.current_symbol = prediction[0][0]


        #LAYER 2

        if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):

        	prediction = {}

        	prediction['D'] = result_dru[0][0]
        	prediction['R'] = result_dru[0][1]
        	prediction['U'] = result_dru[0][2]

        	prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):

        	prediction = {}

        	prediction['D'] = result_tkdi[0][0]
        	prediction['I'] = result_tkdi[0][1]
        	prediction['K'] = result_tkdi[0][2]
        	prediction['T'] = result_tkdi[0][3]

        	prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):

        	prediction1 = {}

        	prediction1['M'] = result_smn[0][0]
        	prediction1['N'] = result_smn[0][1]
        	prediction1['S'] = result_smn[0][2]

        	prediction1 = sorted(prediction1.items(), key = operator.itemgetter(1), reverse = True)

        	if(prediction1[0][0] == 'S'):

        		self.current_symbol = prediction1[0][0]
```

⦁ We defined five methods to suggest words from the letters and display the words in 3 buttons at the bottom of the screen as shown below:

![Capture](https://user-images.githubusercontent.com/56514238/174484580-07ead4f9-d350-4510-86d5-00513fd814a1.PNG)

``` python
def action1(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 0):

            self.word = ""

            self.str += " "

            self.str += predicts[0]

    def action2(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 1):
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 2):
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def action4(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 3):
            self.word = ""
            self.str += " "
            self.str += predicts[3]

    def action5(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 4):
            self.word = ""
            self.str += " "
            self.str += predicts[4]
```

⦁ Finally, release the capture and close all windows by creating a method called destructor outside the video loop to end the main function

``` python
 def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
```

⦁ Once the script is run, a window will pop up with your live webcam feed. The predicted sign language letter will be shown in the character. Hold up your hand and make your favorite sign to see your classifier in action. Here are some sample results

![Capture](https://user-images.githubusercontent.com/56514238/174484716-dfb8c4b0-9155-4be5-b043-5b53cbe24914.PNG)

# License

Copyright (c) 2022 SH.A CS 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

  
