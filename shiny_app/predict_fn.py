from keras.utils import np_utils
import cv2
import numpy as np 
import os
from PIL import Image
from keras.models import load_model
import traceback

def model_load(model):
    """ Path where the weights file and json file are located """
    loaded_model = load_model(model)
    return loaded_model


def process_image(image):
    """ Pre-process image by resizing, reshaping, and standardizing """
    try:
      image = cv2.imread(image)
      image_from_array = Image.fromarray(image, 'RGB')
      resized = image_from_array.resize((50,50))
      resized = np.array(resized)
      resized = resized.astype('float32')/255
      resized = np.expand_dims(resized, axis=0)
      pred = cnn_predict(resized)
      return pred
    except Exception as e:
      traceback.print_exc()
    
    

def cnn_predict(image):
    # predictions = []
    model = model_load('cnnmodel.h5')
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)
    pred = np_utils.to_categorical(pred)
    pred = pred.astype(int)
    pred = pred.tolist()
    par = [[1]]
    unif = [[0, 1]]
    if pred == par:
        print("Parasitized")
        # predictions.append(1)
        return "pos"
    if pred == unif:
        print("Uninfected")
        # predictions.append(0)
        return "neg"
    # return predictions


# def main():
#     images = process_image('Uninfected')
#     predictions = cnn_predict(images)
#     print(predictions)
#
#
# if __name__ == "__main__":
#     main()
