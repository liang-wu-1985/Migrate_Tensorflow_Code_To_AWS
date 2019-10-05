import numpy as np
import os
import tensorflow as tf


def get_embeddings(embedding_dir):
    
    embeddings = np.load(os.path.join(embedding_dir, 'embedding.npy'))
    print('embeddings shape:  ', embeddings.shape)

    return embeddings


def get_cnn_model(embedding_dir, NUM_WORDS, WORD_INDEX_LENGTH, LABELS_INDEX_LENGTH, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    
    embedding_matrix = get_embeddings(embedding_dir)
    filter_sizes = [1,2,3,5]
    num_filters = 36
    embed_size=embedding_matrix.shape[1]
    maxlen=MAX_SEQUENCE_LENGTH

    inp = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    x = tf.keras.layers.Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1], weights=[embedding_matrix])(inp)
    x = tf.keras.layers.Reshape((maxlen, embed_size, 1))(x)
    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = tf.keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(tf.keras.layers.MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = tf.keras.layers.Concatenate(axis=1)(maxpool_pool)   
    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dropout(0.2)(z)

    preds = tf.keras.layers.Dense(LABELS_INDEX_LENGTH, activation="sigmoid")(z)

    tf.keras.Model(inp, preds).summary()

    return tf.keras.Model(inp, preds)