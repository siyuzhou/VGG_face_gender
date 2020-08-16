import os
import argparse
import tensorflow as tf

from data_loader import *

SELF_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SELF_PATH, 'data/agegender_cleaned')
PRETRAINED_MODEL = os.path.join(SELF_PATH, 'saved_models/VGG_FACE/VGG_FACE.h5')
LOG_DIR = os.path.join(SELF_PATH, 'saved_models/VGG_FACE_gender')

IMAGE_SIZE = 224
CLASS_NAMES = ['F', 'M']


def load_pretrained_model(trainable=False):
    VGG_FACE = tf.keras.models.load_model(PRETRAINED_MODEL)
    VGG_FACE.trainable = trainable
    return VGG_FACE


def build_model(base_model, input_layer_name, feature_layer_name, new_layer_units, output_units):
    inputs = base_model.get_layer(input_layer_name).input
    features = base_model.get_layer(feature_layer_name).output

    x = features
    for unit in new_layer_units:
        x = tf.keras.layers.Dense(unit, 'relu')(x)
        # x = tf.keras.layers.Dropout()(x)

    outputs = tf.keras.layers.Dense(output_units, 'sigmoid')(x)

    new_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])
    return new_model


def main():
    # Load data
    files_pattern = 'aligned/*/*.jpg'
    files_pattern = os.path.join(DATA_DIR, files_pattern)

    dataset = load_dataset(files_pattern, IMAGE_SIZE, CLASS_NAMES)
    train_dataset, valid_dataset = split_dataset(dataset, split=0.2)

    train_dataset = batch_dataset(train_dataset, ARGS.batch_size)
    valid_dataset = batch_dataset(valid_dataset, ARGS.batch_size)

    if ARGS.eval_only:
        # Evaluation
        VGG_face_gender = tf.keras.models.load_model(LOG_DIR)
        VGG_face_gender.evaluate(valid_dataset)
    else:
        # Training
        print('Loading pretrained VGG_FACE model ... (model not compiled)')
        VGG_FACE = load_pretrained_model()
        VGG_face_gender = build_model(VGG_FACE, 'data', 'drop7', [128], 1)

        VGG_face_gender.fit(train_dataset, validation_data=valid_dataset,
                            epochs=ARGS.epochs,
                            callbacks=[tf.keras.callbacks.ModelCheckpoint(LOG_DIR)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--eval-only', action='store_true', default=False)
    ARGS = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    main()
