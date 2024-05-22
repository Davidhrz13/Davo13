
#########################################################################
# Import required packages
#########################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters


#########################################################################
# Set Up flow For Training & Validation data
#########################################################################

# data flow parameters

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6 


# image generators


training_generator = ImageDataGenerator(rescale = 1./255, 
                                        rotation_range = 20, 
                                        width_shift_range = 0.2, 
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5,1.5), 
                                        fill_mode = 'nearest') 
                                        
validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows
# from hard drive to our network

training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')


#########################################################################
# Architecture Tuning
#########################################################################


# network architecture

def build_model(hp):#hyperparameters
    model = Sequential()

    model.add(Conv2D(filters = hp.Int("Input_Conv_Filters", min_value = 32, max_value = 128, step = 32), kernel_size = (3,3), padding = 'same', input_shape = (img_width, img_height, num_channels))) # hp parameter of type integer with the specified filter logic
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    for i in range(hp.Int("n_Conv_Layers", min_value = 1, max_value = 3, step = 1)): #Testing 1 to 3 conv layers

        model.add(Conv2D(filters = hp.Int(f"Conv_{i}_Filters", min_value = 32, max_value = 128, step = 32), kernel_size = (3,3), padding = 'same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D())


    model.add(Flatten())
    
    for j in range(hp.Int("n_Dense_Layers", min_value = 1, max_value = 4, step = 1)): #Testing 1 to 4 dense layers
    
        model.add(Dense(hp.Int(f"Dense_{j}_Neurons", min_value = 32, max_value = 128, step = 32)))
        model.add(Activation('relu'))
        
        if hp.Boolean("Dropout"): # randomly test true or false, if true, add dropout
            model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # compile model
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = hp.Choice("Optimizer", values = ['adam', 'RMSProp']),
                  metrics = ['accuracy'])
   
    return model

tuner = RandomSearch(hypermodel = build_model,
                     objective = 'val_accuracy',
                     max_trials = 3,#n random combinations of parameters we are testing
                     executions_per_trial = 2, #each combination is running twice - validation avg of those runs
                     directory = 'tuner_results', #saving directory
                     project_name = 'fruit-cnn',
                     overwrite = True) #if we run code again, overwrite results

tuner.search(x = training_set,
             validation_data = validation_set,
             epochs = 5,
             batch_size = 32)

# top networks
tuner.results_summary()

# best network - hyperparameters
tuner.get_best_hyperparameters()[0].values

# summary of best newtwork architecture

tuner.get_best_models()[0].summary()


#########################################################################
# Network Architecture
#########################################################################

# network architecture for Andrew's best combination using his GPU

model = Sequential()

model.add(Conv2D(filters = 96, kernel_size = (3,3), padding = 'same', input_shape = (img_width, img_height, num_channels))) #We are convolving over 2 dimensions - grid across and down (1d - langauge, video - 3d)
model.add(Activation('relu'))
model.add(MaxPooling2D()) #Leave default parametrs (pool size (2,2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')) #this Conv layer doesn't need pixels specs, it gets info from pooling layer
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')) #this Conv layer doesn't need pixels specs, it gets info from pooling layer
model.add(Activation('relu'))
model.add(MaxPooling2D())

# Now we need to stack info to pass it to dense layer - flattening

model.add(Flatten())
model.add(Dense(160))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Output layer - need # nuerons for each class we want to predict
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile network

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


# view network architecture

model.summary()


#########################################################################
# Train Our Network!
#########################################################################

# training parameters

num_epochs = 50
# images will pass 50 times in batches of 32 
model_file_name = 'models/frutis_cnn_tuned.h5'

# callbacks

save_best_model = ModelCheckpoint(filepath = model_file_name,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True)

# train the network

history = model.fit(x = training_set, 
                    validation_data = validation_set, 
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])



#########################################################################
# Visualise Training & Validation Performance
#########################################################################

import matplotlib.pyplot as plt

# plot validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])


#########################################################################
# Make Predictions On New Data (Test Set)
#########################################################################

# import required packages

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd 
from os import listdir
#create list of files of any specified directory - imaage files

# parameters for prediction

model_file_name = 'models/frutis_cnn_tuned.h5'
img_width = 128
img_height = 128
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']


# load model

model = load_model(model_file_name)


# import image & apply pre-processing

# image pre-processing function

def preprocess_image(filepath):
    image = load_img(filepath, target_size = (img_width,img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image * (1./255)
    
    return image
    

# image prediction function

def make_prediction(image):
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob

# loop through test data

source_dir = 'data/test/'
folder_names = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']
actual_labels = []
predicted_labels = []   
predicted_probabilities = []
filenames = []
    
for folder in folder_names:
    
    images = listdir(source_dir + '/' + folder)
    
    for image in images: 
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_probability = make_prediction(processed_image)
        
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_probability)
        filenames.append(image)
        

# create dataframe to analyse

predictions_df = pd.DataFrame({"actual_label" : actual_labels,
                               "predicted_label" : predicted_labels, 
                               "predicted_probability" : predicted_probabilities, 
                               "file_name" : filenames})

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'], 1, 0)

# overall test set accuracy

test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)
print(test_set_accuracy)
#Improved accuracy from 75% to 88% with D.Augmentation
#Improved to 95% with both dropout and data augmentation
# 95% of accuracy with tuned model

# confusion matrix (raw numbers)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'])
print(confusion_matrix)


# confusion matrix (percentages)
confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)









