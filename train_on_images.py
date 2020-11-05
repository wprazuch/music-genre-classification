import tensorflow as tf
from ml_music import models
from ml_music import utils
from tensorflow.keras.optimizers import RMSprop

IMAGE_SIZE = (256, 256)
DATASET_PATH = r'datasets/gtzan/images_original'
BATCH_SIZE = 8


def main():
    data, labels = utils.load_img_by_classes(DATASET_PATH)
    data, labels = utils.preprocess_loaded_data(data, labels)

    model = models.common.ResNet50V2(num_classes=10, image_size=IMAGE_SIZE)

    dataset = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(512)

    TRAIN_VAL_SPLIT = (int(0.85*(len(dataset))))

    train = dataset.take(TRAIN_VAL_SPLIT).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    val = dataset.skip(TRAIN_VAL_SPLIT).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    model.compile(optimizer=RMSprop(learning_rate=0.003),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train, validation_data=val, epochs=10)


if __name__ == "__main__":
    main()
