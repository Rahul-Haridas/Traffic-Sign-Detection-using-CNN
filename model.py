import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import tensorflow as tf
import csv



def load_data(data_dir):

    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]

        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels



ROOT_PATH = "dataset"
train_data_dir = os.path.join(ROOT_PATH, "train/Final_Training")
test_data_dir = os.path.join(ROOT_PATH, "test/Final_Test/Images")

images, labels = load_data(train_data_dir)


print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))




def display_images_and_labels(images, labels):

    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:

        image = images[labels.index(label)]
        plt.subplot(7, 7, i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
    plt.show()

display_images_and_labels(images, labels)


def display_label_images(images, label):

    limit = 24
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

display_label_images(images, 23)

print("--------Before Resizing---------")
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))


images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]
#images32 =[skimage.exposure.rescale_intensity(images32) for images32 in images32]

display_images_and_labels(images32, labels)

print("--------After Resizing---------")
for image in images32[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))



labels_a = np.array(labels)
images_a = np.array(images32)

print("labels: ", labels_a.shape, "images: ", images_a.shape)



graph = tf.Graph()


with graph.as_default():

    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    images_flat = tf.contrib.layers.flatten(images_ph)
    logits = tf.contrib.layers.fully_connected(images_flat, 43, tf.nn.relu)


    predicted_labels = tf.argmax(logits, 1)


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels_ph))


    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


    init = tf.global_variables_initializer()
print("-----Tensors-----")
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)


session = tf.Session(graph=graph)


_ = session.run([init])

print("--------Training----------")
for i in range(251):
    _, loss_value = session.run([train, loss],
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)



sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

predicted = session.run([predicted_labels],
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)




fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()

signs_class=[]
class_no=[]
with open('signnames.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        signs_class.append((row['SignName']))
        #class_no.append((row['ClassId']))


fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth: {0}\nPrediction: {1}".format(signs_class[truth], signs_class[prediction]),
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()




