import data_utils

import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx("float32")

ROOT = "/home/ec2-user/biods220/project/assign2" 

def build_masked_lstm_model(num_timesteps, num_features, no_label_cols, lstm_hidden_units=256, activation = None):
    model_lstm = tf.keras.Sequential()
    model_lstm.add(tf.keras.layers.Masking(mask_value=0, input_shape=(num_timesteps, num_features)))
    model_lstm.add(tf.keras.layers.LSTM(lstm_hidden_units, return_sequences = True))
    model_lstm.add(tf.keras.layers.Dropout(0.5))
    model_lstm.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(no_label_cols, activation = activation)))
    
    for layer in model_lstm.layers[1:]:
        layer.supports_masking = True
    return model_lstm

def run_pretrain(target, build = True, run = True):
    if build:
        data_utils.build_seq_dataset(ROOT, target)
    (
        train_x,
        val_x,
        train_y,
        val_y,
        no_feature_cols,
        no_label_cols,
        test_x,
        test_y,
        val_boolmat,
        test_boolmat,
        features,
        labels,
    ) = data_utils.load_seq_dataset(ROOT, target)
        
    # convert all float64 to float32
    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_y = val_y.astype(np.float32)
    test_y = test_y.astype(np.float32)

    train_x = np.concatenate([train_x, val_x])
    train_y = np.concatenate([train_y, val_y])

    print("train shapes ", train_x.shape, train_y.shape)
    print("test shapes  ", test_x.shape, test_y.shape)
    print("# features   ", len(features))

    num_timesteps, num_features = train_x.shape[-2:]
    model = build_masked_lstm_model(num_timesteps, num_features, no_label_cols, activation = 'sigmoid' if target['trend'] else None)

    if run:
        loss= tf.keras.losses.BinaryCrossentropy() if target['trend'] else tf.keras.losses.MeanSquaredError()
        model.compile(
            optimizer='adam',
            loss = loss, 
        )
        history = model.fit(
            x = train_x,
            y = train_y,
            epochs = 200, 
            validation_data = (test_x, test_y),
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=30,restore_best_weights=True)]
        )
    return model, no_label_cols, features
