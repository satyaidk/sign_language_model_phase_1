import tensorflow as tf

def build_model(num_classes=35, input_shape=(299, 299, 3)):
    base_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet"
    )

    base_model.trainable = False  # freeze pretrained weights

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    # ✅ Compile the model here
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
