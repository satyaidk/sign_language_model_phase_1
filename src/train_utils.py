import tensorflow as tf

def get_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            "../models/classify_model.h5",
            save_best_only=True,
            monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1
        )
    ]
