import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files

dataset = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True,
)

# set path to dataset
IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')

classes = ['pos', 'neg']
train_data = load_files(os.path.join(IMDB_DATADIR, 'train'), shuffle=True, categories=classes)
test_data = load_files(os.path.join(IMDB_DATADIR, 'test'), shuffle=False, categories=classes)

x_train = np.array(train_data.data)
y_train = np.array(train_data.target)
x_test = np.array(test_data.data)
y_test = np.array(test_data.target)

print(x_train.shape)  # (25000,)
print(y_train.shape)  # (25000, 1)
print(x_train[0][:50])  # this film was just brilliant casting

import autokeras as ak

# Initialize the text classifier.
clf = ak.TextClassifier(
    overwrite=True,
    max_trials=10)  # It only tries 1 model as a quick demo.
# Feed the text classifier with training data.
clf.fit(x_train, y_train, epochs=5)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))

clf.fit(x_train,
        y_train,
        # Split the training data and use the last 15% as validation data.
        validation_split=0.15)

split = 5000
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
clf.fit(x_train,
        y_train,
        epochs=2,
        # Use your own validation set.
        validation_data=(x_val, y_val))
#782/782 [==============================] - 10s 13ms/step - loss: 0.2681 - accuracy: 0.8910
#[0.2680726647377014, 0.890999972820282]