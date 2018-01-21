from data_preparation import *
from network import *
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import pickle as pkl
import gzip

class Tester:
    def __init__(self):
        PROCESSED_DATA_FILE = 'data/processed_data.pkl.gz'
        MODEL_WEIGHTS = 'model/sensim_adadelta_model_weights.h5'
        
        #1. Load processed support data such as vocabulary and embeddings
        print("Load processed support data")
        f = gzip.open(PROCESSED_DATA_FILE)
        processed_data = pkl.load(f)
        f.close()
        
        embeddings = processed_data['embeddings']
        embedding_dim = processed_data['embedding_dim']
        self.vocabulary = processed_data['vocabulary']
        self.max_seq_length = processed_data['max_seq_length']
        
        #2. Init neural net
        print("Initialize neural net")
        # Model variables
        n_hidden = 50
        gradient_clipping_norm = 1.25
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)
        #optimizer = 'Adam'
        
        self.model = create_network(n_hidden, optimizer, self.max_seq_length, embeddings, embedding_dim)
        self.model.load_weights(MODEL_WEIGHTS)
    
    def compute_similarity(self, text1, text2):
        d = { 'test_id': [0], 'question1': [text1], 'question2': [text2]}
        df = pd.DataFrame(data=d)
        X_test = vectorize_test_data(df, self.vocabulary, self.max_seq_length)
        
        test_prediction = self.model.predict([X_test['left'], X_test['right']], verbose=False)
        return test_prediction