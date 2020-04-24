import tensorflow as tf

from generator.data_generator import DataGenerator
from helper.box_encoder import SSDEncoder
from helper.loss import SSDLoss
from network.mobilenetv2 import mobilenet_v2
from helper.preprocessing import path_to_image_aug, path_to_image

AUTOTUNE = tf.data.experimental.AUTOTUNE

VOC_2007_images_dir = 'dataset/VOCdevkit/VOC2007/JPEGImages/'
VOC_2007_annotations_dir = 'dataset/VOCdevkit/VOC2007/Annotations/'
# VOC_2007_train_image_set_filename = 'dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
# VOC_2007_valid_image_set_filename = 'dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
VOC_2007_train_image_set_filename = 'dataset/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
VOC_2007_valid_image_set_filename = 'dataset/VOCdevkit/VOC2007/ImageSets/Main/val.txt'

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

EPOCHS = 5000
BATCH_SIZE = 32
INPUT_IMG_SIZE = (300, 300)
ASPECT_RATIOS = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
LAYERS_DEPTHS = [-1, -1, 512, 256, 256, 128]
SCALES_PASCAL = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]


def parse_func(p, a):
    return p, a


def main():
    model = mobilenet_v2(INPUT_IMG_SIZE, len(classes), ASPECT_RATIOS, LAYERS_DEPTHS, SCALES_PASCAL, 'train')
    model.summary()

    predictor_sizes = [model.get_layer('block_13_expand_relu').output_shape[1:3],  # 19x19
                       model.get_layer('out_relu').output_shape[1:3],  # 10x10
                       model.get_layer('BoxPredictor3_2_Conv2d_3x3_s2_conv').output_shape[1:3],  # 5x5
                       model.get_layer('BoxPredictor4_2_Conv2d_3x3_s2_conv').output_shape[1:3],  # 3x3
                       model.get_layer('BoxPredictor5_2_Conv2d_3x3_s2_conv').output_shape[1:3],  # 2x2
                       model.get_layer('BoxPredictor6_2_Conv2d_3x3_s2_conv').output_shape[1:3]]  # 1x1

    ssd_encoder = SSDEncoder(INPUT_IMG_SIZE, ASPECT_RATIOS, SCALES_PASCAL,
                             len(classes), predictor_sizes=predictor_sizes)
    train_dataset = DataGenerator(ssd_encoder)
    train_dataset.parse(VOC_2007_images_dir, VOC_2007_annotations_dir, VOC_2007_train_image_set_filename, classes)
    train_steps_per_epoch = train_dataset.sample_count() // BATCH_SIZE
    train_ds = tf.data.Dataset.from_generator(
        train_dataset.generate,
        (tf.string, tf.float32),
        (tf.TensorShape([]), tf.TensorShape([ssd_encoder.boxes_count(), len(classes) + 4])))

    # print(list(train_ds.take(1).as_numpy_iterator()))

    train_ds = train_ds.map(lambda p, a: (path_to_image_aug(p, INPUT_IMG_SIZE), a),
                            num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    # print(list(train_ds.take(1).as_numpy_iterator()))

    valid_dataset = DataGenerator(ssd_encoder)
    valid_dataset.parse(VOC_2007_images_dir, VOC_2007_annotations_dir, VOC_2007_valid_image_set_filename, classes)
    valid_steps_per_epoch = valid_dataset.sample_count() // BATCH_SIZE
    valid_ds = tf.data.Dataset.from_generator(
        valid_dataset.generate,
        (tf.string, tf.float32),
        (tf.TensorShape([]), tf.TensorShape([ssd_encoder.boxes_count(), len(classes) + 4])),
        args=([False]))
    valid_ds = valid_ds.map(lambda p, a: (path_to_image(p, INPUT_IMG_SIZE), a),
                            num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=300,
            verbose=1,
            mode='min',
            baseline=None,
            restore_best_weights=False),
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/{epoch:02d}-{val_loss:.2f}.h5',
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min")
    ]

    model.fit(
        train_ds,
        steps_per_epoch=train_steps_per_epoch,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_ds,
        validation_steps=valid_steps_per_epoch)


if __name__ == '__main__':
    main()
