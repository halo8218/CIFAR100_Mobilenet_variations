# based on https://github.com/TropComplique/ShuffleNet-tensorflow
import tensorflow as tf
from CONSTANTS import SHUFFLE_BUFFER_SIZE,\
    NUM_THREADS, PREFETCH_BUFFER_SIZE


def _get_data(num_classes, image_size):
    """Get images and targets in batches.

    Training data is augmented with random crops and flips.

    Arguments:
        num_classes: An integer.
        image_size: An integer, it is assumed that
            image_width = image_height = image_size.

    Returns:
        A dict with the following keys:
            'init_data': An op, initialize data sources and batch size.
            'train_init': An op, initialize train data iterator.
            'val_init': An op, initialize validation data iterator.
            'x_batch': A float tensor with shape [batch_size, image_size, image_size, 3],
                images have pixel values in range [0, 1].
            'y_batch': A float tensor with shape [batch_size, num_classes],
                targets are one-hot encoded.
    """

    batch_size = tf.Variable(
        tf.placeholder(tf.int64, [], 'batch_size'),
        trainable=False, collections=[]
    )
    train_file = tf.Variable(
        tf.placeholder(tf.string, [], 'train_file'),
        trainable=False, collections=[]
    )
    val_file = tf.Variable(
        tf.placeholder(tf.string, [], 'val_file'),
        trainable=False, collections=[]
    )
    init_data = tf.variables_initializer([train_file, val_file])
    init_batch_size = tf.variables_initializer([batch_size])

    train_dataset = tf.data.TFRecordDataset(train_file)
    train_dataset = train_dataset.map(
        lambda x: _parse_and_preprocess(x, image_size, augmentation=True),
        num_parallel_calls=NUM_THREADS
    ).prefetch(PREFETCH_BUFFER_SIZE)

    train_dataset = train_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.repeat()

    val_dataset = tf.data.TFRecordDataset(val_file)
    val_dataset = val_dataset.map(
        lambda x: _parse_and_preprocess(x, image_size)
    )
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.repeat()

    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
    )
    train_init = iterator.make_initializer(train_dataset)
    val_init = iterator.make_initializer(val_dataset)

    x_batch, y_batch = iterator.get_next()
    y_batch = tf.one_hot(y_batch, num_classes, axis=1, dtype=tf.float32)

    data = {
        'init_data': init_data,
        'init_batch_size': init_batch_size,
        'train_init': train_init, 'val_init': val_init,
        'x_batch': x_batch, 'y_batch': y_batch
    }
    return data


def _parse_and_preprocess(example_proto, image_size, augmentation=False):

    features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_png(parsed_features['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    target = parsed_features['image/class/label']

    if augmentation:
        image = tf.image.resize_image_with_crop_or_pad(image, image_size+4, image_size+4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
    return image, target
