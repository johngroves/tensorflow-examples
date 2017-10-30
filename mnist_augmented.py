# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# ==============================================================================
"""
Extending Tensorflow mnist tutorial with image augmentation.
"""

from tensorflow.examples.tutorials.mnist import input_data
from imgaug import augmenters as iaa
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

FLAGS = None
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Number of times the dataset will be multiplied
augment_factor = 5
# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# Add image augmentation steps
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 2.0)), # Blur images with a sigma of 0 to 2.0
    iaa.Affine(shear=(-36, 36)) # Add shear between -36 and 36 degrees
])

cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train for 10 epochs
for _ in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    for i in range(augment_factor):
        skewed_xs = seq.augment_images(batch_xs.reshape(batch_xs.shape[0], 28, 28, 1))
        skewed_xs = batch_xs.reshape(batch_xs.shape[0], 784)
        batch_xs = np.concatenate([skewed_xs, batch_xs])
        batch_ys = np.concatenate([batch_ys, batch_ys])
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    # Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

# Demonstration of augmentation

# Take four images, augment them
four_images = mnist.test.images[:4]
four_skewed = seq.augment_images(four_images.reshape(4, 28, 28, 1)).reshape(4, 784)
combined_images = np.concatenate([four_images, four_skewed])


# Display originals with augmented
columns = 4
plt.figure(figsize=(10,10))
for i, image in enumerate(combined_images):
    plt.subplot(len(combined_images) / columns + 1, columns, i + 1)
    plt.imshow(image.reshape(28,28))
plt.show()
