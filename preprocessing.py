import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np
import sklearn.model_selection as sk

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Generate Vectors to Product Titles
def Dov2Vectors(products_titles):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(products_titles)]
    max_epochs = 100
    vec_size = 20
    alpha = 0.025
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm =0)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("d2v.model")
    print("Model Saved")
    return True


# Splitting Data for Training
def Data_Split(products_vectors,labels_vectors):
    train_data, test_data, train_labels, test_labels  = sk.train_test_split(products_vectors, labels_vectors, test_size=0.1, random_state=1)
    train_data, val_data, train_labels, val_labels  = sk.train_test_split(train_data, train_labels, test_size=0.1, random_state=1)
    return train_data, train_labels, val_data, val_labels, test_data, test_labels
