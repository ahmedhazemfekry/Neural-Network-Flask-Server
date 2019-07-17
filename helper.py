import nltk
import pandas as pd
import numpy as np
import preprocessing as pre

from nltk.tokenize import word_tokenize

# Get the Vector coressponds to a specific Product Title
def Get_Product_Title_Vector(product_title):
    Vector_PVDM   = Get_Product_Title_Vector_PVDM(product_title)
    Vector_PVDBOW = Get_Product_Title_Vector_PVDBOW(product_title)
    Vector        = np.concatenate((Vector_PVDM,Vector_PVDBOW),axis=1)
    return Vector

def Get_Product_Title_Vector_PVDM(product_title):
    model_PVDM= pre.Doc2Vec.load("d2vPVDM.model")
    #to find the vector of a document which is not in training data
    title_tokenized = word_tokenize(product_title.lower())
    vector =  model_PVDM.infer_vector(doc_words=title_tokenized, alpha=0.025, min_alpha=0.00025, epochs=100)
    vector = np.array([vector])
    return vector

def Get_Product_Title_Vector_PVDBOW(product_title):
    model_PVDBOW= pre.Doc2Vec.load("d2vPVDBOW.model")
    #to find the vector of a document which is not in training data
    title_tokenized = word_tokenize(product_title.lower())
    vector =  model_PVDBOW.infer_vector(doc_words=title_tokenized, alpha=0.025, min_alpha=0.00025, epochs=100)
    vector = np.array([vector])
    return vector

# Get Array of Vectors for all the Data (Products Titles)
def Get_Product_Vectors(input_size, training_type):
    vec_size = 200
    if(training_type == 1):
        vec_size = 80
    else:
        vec_size = 200
    model_PVDM   = pre.Doc2Vec.load("d2vPVDM.model")
    model_PVDBOW = pre.Doc2Vec.load("d2vPVDBOW.model")

    Vectors_List_PVDM = np.zeros([input_size,vec_size])
    Vectors_List_PVDBOW = np.zeros([input_size,vec_size])

    for idx in range(input_size):
        Vectors_List_PVDM[idx]=model_PVDM[idx]
        Vectors_List_PVDBOW[idx]=model_PVDBOW[idx]
    Vectors_List = np.concatenate((Vectors_List_PVDM,Vectors_List_PVDBOW),axis=1)
    return Vectors_List

# Get the unique set of the existed categories (Targets of the training data)
def Get_Unique_Category_Set(targets):
    Labels_List = []
    Categories  = []
    for Category in targets:
        if not Category in Categories:
           Categories.append(Category)
    return Categories

# Get the target vector suitable for training process [0,1,0,0,0,0,0,0,0]-> 2nd class, [0,0,0,0,1,0,0,0,0]-> 5th class
def Get_Targets_Arrays(targets):
    categories_set = Get_Unique_Category_Set(targets)
    print("Categories",categories_set)
    labels = np.zeros([len(targets), len(categories_set)])
    classes = np.zeros(len(targets))
    for idx,val in enumerate(targets):
      target_idx = categories_set.index(val)
      # define product category array like : [0,0,0,0,1,0,0,0,0]
      product_class = np.zeros(len(categories_set))
      product_class[target_idx]=1
      labels[idx]=product_class
      classes[idx]=target_idx
    return labels,classes,categories_set

def Read_Data(path):
    data = pd.read_csv(path)
    data = data.dropna(subset=['Product'])
    data = data.drop_duplicates(subset=['Product'])

    products = data["Product"].tolist()
    targets = data["Product type"].tolist()
    return products, targets
