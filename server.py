
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk
import helper as hp
import preprocessing as pre
import machine_learning as ml

from flask import Flask, redirect, url_for, request
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

ML_model = None
ML_history = None
graph = None
app = Flask(__name__)

@app.route('/')
def hello_world():
   return 'Hello World'

@app.route('/Train', methods = ['POST'])
def Train():
    global ML_model
    global ML_history

    titles, targets = hp.Read_Data("./Data/Data_With_Category.csv")
    labels, classes = hp.Get_Targets_Arrays(targets)
    model= pre.Doc2Vec.load("d2v.model")
    Vectors_List = hp.Get_Product_Vectors(model,144,20)

    train_data, train_labels, val_data, val_labels, test_data, test_labels = pre.Data_Split(Vectors_List,labels)
    ML_model, ML_history = ml.Train(train_data, train_labels, val_data, val_labels, len(labels[0]))
    global graph
    graph = tf.get_default_graph()

    results = ML_model.evaluate(test_data, test_labels)
    return "Train Completed"

@app.route('/Predict',methods = ['GET'])
def Predict():
    global ML_model
    
    title = request.args.get('product')
    v = hp.Get_Product_Title_Vector(title)

    ML_model = load_model("weights")
    ML_model._make_predict_function()

    preds = ML_model.predict(v)
    print(preds)

    return "True"


# @app.route('/Predict/<title>')
# def Predict(title):
#     global ML_model
#     v = hp.Get_Product_Title_Vector(title)
#     ML_model = load_model("weights")
#     ML_model._make_predict_function()
#
#     preds = ML_model.predict(v)
#     print(preds)
#     return "True"



if __name__ == '__main__':
   app.run()
