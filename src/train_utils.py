import data_utils

from sklearn.metrics import roc_curve, auc as auc_function
import tensorflow as tf
import pandas as pd
import numpy as np

# config
tf.keras.backend.set_floatx("float32")

ROOT = "/home/ec2-user/biods220/project/assign2"  # Put your root path here

def load_masked_lstm_model(num_timesteps, num_pretrain_features, num_pretrain_labels, 
                            pretrain_feat_mask, pretrain_path = 'pretrained.h5',
                            lstm_hidden_units=256):
    model_pt = tf.keras.Sequential()
    model_pt.add(tf.keras.layers.Masking(mask_value=0, input_shape=(num_timesteps, num_pretrain_features)))
    model_pt.add(tf.keras.layers.LSTM(lstm_hidden_units, return_sequences = True))
    model_pt.add(tf.keras.layers.Dropout(0.5))
    model_pt.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_pretrain_labels)))
    for layer in model_pt.layers[1:]:
        layer.supports_masking = True
    
    if pretrain_path is not None:
        print('LOADING PRETRAINED WEIGHTS')
        model_pt.load_weights(pretrain_path)
    
    num_features = sum(pretrain_feat_mask)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(num_timesteps, num_features)))
    model.add(tf.keras.layers.LSTM(lstm_hidden_units, return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid')))
    for layer in model.layers[1:]:
        layer.supports_masking = True

    transfer_weights = model_pt.layers[1].trainable_weights
    transfer_weights[0] = tf.boolean_mask(transfer_weights[0], pretrain_feat_mask)
    model.layers[1].set_weights(transfer_weights)
    return model

def run_experiment(target, pretrain_target, pretrain_features, num_pretrain_labels, label_frac = 1, pretrain = False, seed = 0):
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

    np.random.seed(seed)
    num_train_samples = int(label_frac * len(train_x))
    train_ind = np.random.choice(len(train_x), size = num_train_samples, replace=False)

    train_x = train_x.astype(np.float32)[train_ind]
    train_y = train_y.astype(np.float32)[train_ind]

    val_x = val_x.astype(np.float32)
    val_y = val_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)
    
    print('RUNNING EXPERIMENT FOR {} with label_frac: {}, pretrain: {}, and seed: {}'.format(target, label_frac, pretrain, seed))

    print("train shapes ", train_x.shape, train_y.shape)
    print("val shapes   ", val_x.shape, val_y.shape)
    print("test shapes  ", test_x.shape, test_y.shape)
    
    pretrain_feat_mask = [True if x in features else False for x in pretrain_features]
    pretrain_path = 'output/' + str(pretrain_target) + '_pretrained.h5' if pretrain else None
    model = load_masked_lstm_model(train_x.shape[-2], len(pretrain_features), num_pretrain_labels, 
                                    pretrain_feat_mask, pretrain_path)

    # LE
    model.layers[1].trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC (curve='ROC', name='AUROC')],
    )
    history = model.fit(
        x = train_x, y = train_y, epochs = 50, 
        validation_data = (val_x, val_y),
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=20,restore_best_weights=True)]
    )
    test_y_pred = model.predict(test_x)
    y_pred_masked = test_y_pred[~test_boolmat]
    y_true_masked = test_y[~test_boolmat]
    fpr, tpr, thresholds = roc_curve(y_true_masked, y_pred_masked)
    le_auc = auc_function(fpr, tpr)
    
    # FT
    model.layers[1].trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC (curve='ROC', name='AUROC')],
    )
    history = model.fit(
        x = train_x, y = train_y, epochs = 50, 
        validation_data = (val_x, val_y),
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=20,restore_best_weights=True)]
    )

    test_y_pred = model.predict(test_x)
    y_pred_masked = test_y_pred[~test_boolmat]
    y_true_masked = test_y[~test_boolmat]
    fpr, tpr, thresholds = roc_curve(y_true_masked, y_pred_masked)
    ft_auc = auc_function(fpr, tpr)
    return le_auc, ft_auc