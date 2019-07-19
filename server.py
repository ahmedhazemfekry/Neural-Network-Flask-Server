import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk
import helper as hp
import preprocessing as pre
import machine_learning as ml
import json
import os

from flask import Flask, redirect, url_for, request, jsonify
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ML_model       = None
ML_history     = None
graph          = None
titles         = None
classes        = None
targets        = None
categories     = None
training_type  = 0

app = Flask(__name__)

@app.route('/')
def initialize():
   return 'Use /Train or /Predict'

@app.route('/Train', methods = ['POST'])
def Train():
    global ML_model
    global ML_history
    global classes
    global titles
    global targets
    global categories
    global graph


    # Getting the POST Request Body Data
    Data = request.data
    # Converting Text/Plain to JSON Structure
    JsonData = json.loads(Data)
    # Extracting product titles and product classes
    titles = JsonData["products"]
    targets = JsonData["classes"]
    training_type = JsonData["training_type"]
    # 1 is Very Small (80  vec size, Hidden Layers (1024,512))
    # 2 is Small      (200 vec size, Hidden Layers (2048,1024))
    # 3 is Large      (200 vec size, Hidden Layers (2048,1024,1024))
    if(len(titles) == len(targets)):
    # Preprocessing of data
        # Converts target to multi classes array where [1,0,0,0,0,0,....] corresponds to class 1 and [0,1,0,0,0,0,....] corresponds to class 2
        labels, classes, categories = hp.Get_Targets_Arrays(targets)
        print(categories)
        # Converts products titles to vectors
        pre.Doc2Vectors(titles, training_type)
        # Creating Vectors List for all products -> Dataset
        Vectors_List = hp.Get_Product_Vectors(len(titles),training_type)
        # Splitting Data to Train, Validate and Test sets
        train_data, train_labels, val_data, val_labels, test_data, test_labels = pre.Data_Split(Vectors_List,labels)
        # Training
        if(training_type == 1):
            ML_model, ML_history = ml.Train_1(train_data, train_labels, val_data, val_labels, len(labels[0]))
        else:
            if(training_type ==2):
                ML_model, ML_history = ml.Train_2(train_data, train_labels, val_data, val_labels, len(labels[0]))
            else:
                ML_model, ML_history = ml.Train_3(train_data, train_labels, val_data, val_labels, len(labels[0]))

        graph = tf.get_default_graph()
        # Evaluating the trained model
        results = ML_model.evaluate(test_data, test_labels)

        response = "Training Completed with testing scores of " + str(results[1]) + " accuracy and " + str(results[0]) + " Loss"
        return response

    else:
        return "Products and Classes don't have the same length"

@app.route('/Predict',methods = ['POST'])
def Predict():
    global ML_model
    global classes
    global categories
    global training_type

    # Getting the POST Request Body Data
    Data = request.data
    # Converting Text/Plain to JSON Structure
    JsonData = json.loads(Data)
    # Extracting product titles and product classes
    titles = JsonData["products"]

    # Get the product title for prediction from the GET Request
    #title = request.args.get('product')
    # Convert the title to vector based on the titles vector model done in the training process
    #v = hp.Get_Products_Title_Vector(titles)

    # Load model weights for predictins
    ML_model = load_model("weights")
    ML_model._make_predict_function()

    predicted_classes = []

    for title in titles:
        v = hp.Get_Product_Title_Vector(title)
        # Predictions
        pred = ML_model.predict(v)
        max_index = np.argmax(pred)
        predicted_class = categories[max_index]
        predicted_classes.append(predicted_class)
    response = {
    "predictions":predicted_classes,
    }
    return jsonify(response)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5010)
