import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(batch_size=64, target_size=(299, 299)):
    train_df = pd.read_csv("../data/processed/train.csv")
    val_df = pd.read_csv("../data/processed/val.csv")
    test_df = pd.read_csv("../data/processed/test.csv")

    train_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,
        zoom_range=0.15,
        rotation_range=30,
        height_shift_range=0.1,
        width_shift_range=0.1
    )

    test_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
    )

    train = train_gen.flow_from_dataframe(
        train_df,
        x_col="file_paths",
        y_col="labels",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val = test_gen.flow_from_dataframe(
        val_df,
        x_col="file_paths",
        y_col="labels",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test = test_gen.flow_from_dataframe(
        test_df,
        x_col="file_paths",
        y_col="labels",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train, val, test
