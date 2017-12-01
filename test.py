from data_preparation import *
from network import *
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import pickle as pkl
import gzip

PROCESSED_DATA_FILE = 'data/processed_data.pkl.gz'
CRISTI_TEST_CSV = 'data/cristi_test.csv'
MODEL_WEIGHTS = 'model/sensim_adadelta_model_weights.h5'

#1. Load processed support data such as vocabulary and embeddings
print("Load processed support data")
f = gzip.open(PROCESSED_DATA_FILE)
processed_data = pkl.load(f)
f.close()

embeddings = processed_data['embeddings']
vocabulary = processed_data['vocabulary']
max_seq_length = processed_data['max_seq_length']
embedding_dim = processed_data['embedding_dim']

#2. Init neural net
print("Initialize neural net")
# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
optimizer = Adadelta(clipnorm=gradient_clipping_norm)
#optimizer = 'Adam'

model = create_network(n_hidden, optimizer, max_seq_length, embeddings, embedding_dim)
model.load_weights(MODEL_WEIGHTS)  

#3. Load test data
print("Load test data")
my_test_df = pd.read_csv(CRISTI_TEST_CSV)
X_test = vectorize_test_data(my_test_df, vocabulary, max_seq_length)

test_prediction = model.predict([X_test['left'], X_test['right']], verbose=False)
print(test_prediction)