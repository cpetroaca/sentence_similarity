import re
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import itertools
from sklearn.model_selection import train_test_split
import numpy as np
    
def init_data_as_vectors(word2vec_store, data_files, embedding_dim):
    stops = set(stopwords.words('english'))
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    questions_cols = ['question1', 'question2']

    # Iterate over the questions only of both training and test datasets
    for dataset in data_files:
        for index, row in dataset.iterrows():
    
            # Iterate through the text of both questions of the row
            for question in questions_cols:
    
                q2n = []  # q2n -> question numbers representation
                for word in text_to_word_list(row[question]):
    
                    # Check for unwanted words
                    if word in stops and word not in word2vec_store.vocab:
                        continue
    
                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])
    
                # Replace questions as word to question as number representation
                dataset.set_value(index, question, q2n)
    
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in word2vec_store.vocab:
            embeddings[index] = word2vec_store.word_vec(word)

    return embeddings, vocabulary

def vectorize_test_data(dataset, vocabulary, max_seq_length):
    stops = set(stopwords.words('english'))
    questions_cols = ['question1', 'question2']
    
    for index, row in dataset.iterrows():
            # Iterate through the text of both questions of the row
            for question in questions_cols:
    
                q2n = []  # q2n -> question numbers representation
                for word in text_to_word_list(row[question]):
    
                    # Check for unwanted words
                    if word in stops and word not in vocabulary:
                        continue
    
                    q2n.append(vocabulary[word])
    
                # Replace questions as word to question as number representation
                dataset.set_value(index, question, q2n)
    
    X = dataset[questions_cols]
    # Split to dicts
    X_test = {'left': X.question1, 'right': X.question2}
    # Zero padding
    for dataset, side in itertools.product([X_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
    
    return X_test

def split_data(train_df, validation_size, max_seq_length):
    questions_cols = ['question1', 'question2']
    training_size = len(train_df) - validation_size

    X = train_df[questions_cols]
    Y = train_df['is_duplicate']
    
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
    
    # Split to dicts
    X_train = {'left': X_train.question1, 'right': X_train.question2}
    X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
    
    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values
    
    # Zero padding
    for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
    
    # Make sure everything is ok
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)
    
    return X_train, X_validation, Y_train, Y_validation;
    
def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text