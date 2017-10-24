from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from data_preparation import *
from network import *
import datetime
import matplotlib.pyplot as plt
import pickle as pkl
import gzip

# File paths
TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test.csv'
EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
PROCESSED_DATA_FILE = 'data/processed_data.pkl.gz'
MODEL_FILE = 'model/sensim_adadelta_model_weights.h5'

# Load training and test set
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
embedding_dim = 300

embeddings, vocabulary = init_data_as_vectors(word2vec, [train_df, test_df], embedding_dim)

del word2vec

max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
                     train_df.question2.map(lambda x: len(x)).max(),
                     test_df.question1.map(lambda x: len(x)).max(),
                     test_df.question2.map(lambda x: len(x)).max())

# save processed data
data = {'embeddings': embeddings, 'vocabulary': vocabulary, 
        'max_seq_length': max_seq_length, 'embedding_dim': embedding_dim}

f = gzip.open(PROCESSED_DATA_FILE, 'wb')
pkl.dump(data, f)
f.close()
print('processed data saved')

del vocabulary

#prepare training data
validation_size = 40000
X_train, X_validation, Y_train, Y_validation = split_data(train_df, validation_size, max_seq_length)

# Model variables
n_hidden = 50
batch_size = 64
n_epoch = 15
gradient_clipping_norm = 1.25
optimizer = Adadelta(clipnorm=gradient_clipping_norm)
#optimizer = 'Adam'

model = create_network(n_hidden, optimizer, max_seq_length, embeddings, embedding_dim)

# Start training
training_start_time = time()

malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

#model.save('model/qusim_model.h5')
model.save_weights(MODEL_FILE)

# Plot accuracy
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()