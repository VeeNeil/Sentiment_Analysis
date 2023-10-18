import keras.utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter
import joblib
import tensorflow as tf

# Ignore specific warning messages
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

newmodel = tf.keras.models.load_model("./saved_model/sentiment_model.keras")
# Text to predict
text_to_predict = "te"

# Tokenize and preprocess the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_to_predict])
sequences = tokenizer.texts_to_sequences([text_to_predict])
padded = pad_sequences(sequences, maxlen=1000)

# Make a prediction
prediction = newmodel.predict(padded)

# Assuming the model outputs probabilities for each class, you can get the class label with the highest probability
class_labels = ["Very Good", "Good", "Neutral", "Bad", "Very Bad", "Inappropriate"]
predicted_class = class_labels[prediction.argmax()]

print("Predicted Sentiment:", predicted_class)