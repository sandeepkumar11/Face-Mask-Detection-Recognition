# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint

# Part 1 - Building the classifier

# Initialising the classifier
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32,3,3, activation="relu", input_shape=[64, 64, 3]))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32,3,3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu'))

# Step 5 - Output Layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Part 2 - Training the classifier

# Compiling the classifier
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 3 - Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# Training the classifier on the Training set and evaluating it on the Test set
classifier.fit_generator(training_set,
                  steps_per_epoch = 334,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 334)
checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

classifier.fit_generator(training_set,
                         steps_per_epoch=40,
                              epochs=20,
                              validation_data=test_set,
                              callbacks=[checkpoint])
