import tensorflow as tf
from typing import Callable

from cms.modules.parameter_manager import DatasetParam

def gaussian_blur(image: tf.Tensor, kernel_size: int = 23, padding: str = 'SAME') -> tf.Tensor:
    sigma = tf.random.uniform((1,))* 1.9 + 0.1

    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.exp(
        -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


def color_jitter(x: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    x = tf.image.random_brightness(x, max_delta=0.8*s)
    x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_hue(x, max_delta=0.2*s)
    x = tf.clip_by_value(x, 0, 1)
    return x

def color_drop(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, p: float) -> tf.Tensor:
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)

@tf.function
def train_augment(image: tf.Tensor, config: DatasetParam) -> tf.Tensor:
    h = image.shape[0]
    w = image.shape[1]
    
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (int(h/config.crop_pct), int(w/config.crop_pct)))
    image = tf.image.random_crop(image, (config.input_size, config.input_size, config.input_channel))

    # Random flips
    image = random_apply(tf.image.flip_left_right, image, p=0.5)
    # Randomly apply transformation (color distortions) 
    image = random_apply(color_jitter, image, p=0.8)
    # Randomly apply grayscale
    image = random_apply(color_drop, image, p=0.2)
    
    return image

@tf.function
def test_augment(image: tf.Tensor, config: DatasetParam) -> tf.Tensor:
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (config.input_size, config.input_size))
    
    return image
