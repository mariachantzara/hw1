import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping



def train_neural_network(X_train, y_train, X_val, y_val):

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    #model = Sequential([
        #Dense(64, input_shape=(X_train.shape[1],)),
        #LeakyReLU(alpha=0.1),
        #Dropout(0.3),

        #Dense(32),
        #LeakyReLU(alpha=0.1),

        #Dense(1, activation='sigmoid')
    #])

#def train_neural_network_tanh(X_train, y_train, X_val, y_val):

    #model = Sequential([
        #Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
        #Dropout(0.3),
        #Dense(32, activation='tanh'),
        #Dense(1, activation='sigmoid')
    #])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )

    # save model
    os.makedirs("models", exist_ok=True)
    model.save("models/neural_network.h5")

    return model, history