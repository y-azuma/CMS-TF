import tensorflow as tf
from typing import Callable

from cms.modules.parameter_manager import DatasetParam

def gaussian_blur(image: tf.Tensor, kernel_size: int = 23, padding: str = 'SAME') -> tf.Tensor:
    """
    Apply Gaussian blur to an image tensor

    Args:
        image (tf.Tensor): Input image tensor
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 23.
        padding (str, optional): Padding method for convolution. Default is 'SAME'.

    Returns:
        tf.Tensor: Blurred image tensor
    """
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
    """
    Apply color jittering to an image tensor

    Args:
        x (tf.Tensor): Input image tensor
        s (float, optional): Strength of color jittering. Default is 0.5.

    Returns:
        tf.Tensor: Color jittered image tensor
    """
    x = tf.image.random_brightness(x, max_delta=0.8*s)
    x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_hue(x, max_delta=0.2*s)
    x = tf.clip_by_value(x, 0, 1)
    return x

def color_drop(x: tf.Tensor) -> tf.Tensor:
    """
    Convert an RGB image to grayscale and replicate it to 3 channels

    Args:
        x (tf.Tensor): Input RGB image tensor

    Returns:
        tf.Tensor: Grayscale image tensor with 3 channels
    """
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, p: float) -> tf.Tensor:
    """
    Randomly apply a function to an image tensor with a given probability

    Args:
        func (Callable[[tf.Tensor], tf.Tensor]): Function to apply
        x (tf.Tensor):  Input image tensor
        p (float): Probability of applying the function

    Returns:
        tf.Tensor: Resulting image tensor
    """
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)

@tf.function
def train_augment(image: tf.Tensor, config: DatasetParam) -> tf.Tensor:
    """
    Apply training data augmentation to an image tensor

    Args:
        image (tf.Tensor): Input image tensor
        config (DatasetParam): Configuration parameters for augmentation

    Returns:
        tf.Tensor: Augmented image tensor
    """
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
    """
    Apply test data augmentation to an image tensor

    Args:
        image (tf.Tensor): Input image tensor
        config (DatasetParam): Configuration parameters for augmentation

    Returns:
        tf.Tensor: Augmented image tensor
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (config.input_size, config.input_size))
    
    return image
