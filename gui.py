import tkinter as tk
from tkinter import filedialog
from tkinter import *

from sklearn import metrics

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"


# Define the emotions list
emotions_list = ['No','Yes']


top =tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    return model

model = load_model("tumour_model_a.json", "tumour_weights.h5")



def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the input size of the model
    resized_image = cv2.resize(gray_image, (180, 180))
    # Normalize the image
    normalized_image = resized_image / 255.0
    # Reshape the image to match the input shape of the model
    input_image = np.reshape(normalized_image, (1, 180, 180, 1))
    return input_image

# Function to detect the emotion in the image
def Detect(image_path):
    image = cv2.imread(image_path)
    input_image = preprocess_image(image)
    # Predict the emotion using the loaded model
    emotion_index = np.argmax(model.predict(input_image))
    pred = emotions_list[emotion_index]
    
    label1.configure(foreground="#011638",text = pred)
    
    
    



def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()
top.mainloop()