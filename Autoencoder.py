import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.datasets import fashion_mnist
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
from keras.datasets import mnist 

# Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()#cifar10.load_data()
print("Size of training set", X_train.shape)
print("Size of test set", X_test.shape)
classes = ['A', 'B', 'C', 'D', 'E',
            'F', 'G', 'H', 'I', 'J']
num_classes = len(classes)
# input dimension = 32*32*3 = 3072
input_dim = np.prod(X_train.shape[1:])
print(input_dim)
#Showing some training images and labels
plt.figure(figsize=(9,3))
for index, (image, label) in enumerate(zip(X_train[0:5], y_train[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (32, 32, 3)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 12)

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats

# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))


# Train the model, iterating on the data in batches of 128 samples
history = autoencoder.fit(X_train, X_train, epochs=5, batch_size=128, shuffle=True,
                validation_data=(X_test, X_test))

print(autoencoder.summary())

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Let's also create a separate encoder model:
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
print(encoder.summary())

# As well as the decoder model:
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
print(decoder.summary())

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)  #=decoder.predict(encoded_imgs)

# Compare Original images (top row) with reconstructed ones (bottom row)
m = 10  # how many digits we will display
plt.figure(figsize=(9, 3))
for i in range(m):
    # display original
    ax = plt.subplot(2, m, i + 1)
    plt.imshow(X_test[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, m, i + 1 + m)
    plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Using encoding layer with softmax layer to train a classifier
# For a single-input model with 10 classes (categorical classification):
model = Sequential()
model.add(autoencoder)
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, num_classes=10)

# Train the model, iterating on the data in batches of 128 samples
model.fit(X_train, y_train, epochs=5, batch_size=128)

print(model.summary())

#predicted labels/classes
y_pred = model.predict_classes(X_test)

#Precision and recall
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)  
df = pd.DataFrame(cm, classes, classes)
plt.figure()
sns.set(font_scale=1.2)#for label size
#comap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)
ax = sns.heatmap(cm,annot=True,annot_kws={"size": 16},linewidths=.5,cbar=False,
        xticklabels=classes,yticklabels=classes,square=True, cmap='Blues_r', fmt="d")
# This sets the yticks "upright" with 0, as opposed to sideways with 90.
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
plt.xticks(rotation=90)
plt.show()

#accuracy score
acc = accuracy_score(y_test, y_pred)
print('\nAccuracy for the test data: {:.2%}\n'.format(acc))




#https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/


