
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
targets         = None

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
    # Getting the POST Request Body Data
    Data = request.data
    # Converting Text/Plain to JSON Structure
    JsonData = json.loads(Data)
    # Extracting product titles and product classes
    titles = JsonData["products"]
    targets = JsonData["classes"]

    if(len(titles) == len(targets)):
    # Preprocessing of data
        # Converts target to multi classes array where [1,0,0,0,0,0,....] corresponds to class 1 and [0,1,0,0,0,0,....] corresponds to class 2
        labels, classes = hp.Get_Targets_Arrays(targets)
        # Converts products titles to vectors
        pre.Dov2Vectors(titles)
        # Loading the products vectors model done at the last step
        model= pre.Doc2Vec.load("d2v.model")
        # Creating Vectors List for all products -> Dataset
        Vectors_List = hp.Get_Product_Vectors(model,144,20)
        # Splitting Data to Train, Validate and Test sets
        train_data, train_labels, val_data, val_labels, test_data, test_labels = pre.Data_Split(Vectors_List,labels)
        # Training
        ML_model, ML_history = ml.Train(train_data, train_labels, val_data, val_labels, len(labels[0]))

        global graph
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
    global targets
    # Getting the POST Request Body Data
    Data = request.data
    # Converting Text/Plain to JSON Structure
    JsonData = json.loads(Data)
    # Extracting product titles and product classes
    titles = JsonData["products"]

    # Get the product title for prediction from the GET Request
    #title = request.args.get('product')
    # Convert the title to vector based on the titles vector model done in the training process
    v = hp.Get_Product_Title_Vector(titles)
    # Load model weights for predictins
    ML_model = load_model("weights")
    ML_model._make_predict_function()
    # Predictions
    preds = ML_model.predict(v)

    predicted_classes = []

    for p in preds:
        max_index = np.argmax(p)
        predicted_class = targets[max_index]
        predicted_classes.append(predicted_class)

    response = {"predictions":predicted_classes}
    return jsonify(response)



if __name__ == '__main__':
   app.run()
