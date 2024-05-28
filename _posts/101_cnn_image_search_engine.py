
#########################################################################
# Convolutional Neural Network - Image Search Engine
#########################################################################


###########################################################################################
# import packages
###########################################################################################

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from os import listdir
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle

###########################################################################################
# bring in pre-trained model (excluding top)
###########################################################################################

# image parameters
# We don't want top of network, only pooling layer - features will come from, we are using VGG for initial layers

img_width = 224
img_height = 224
num_channels = 3

# network architecture

vgg = VGG16(input_shape = (img_width, img_height, num_channels),
            include_top = False,
            pooling = 'avg') # single set of numbers to represent all those features (global avg pooling applied to final layer - 1 single array)

vgg.summary() # we can see the global_average_pooling2d as final layer - assess similarity for our task

model = Model(inputs = vgg.input,
              outputs = vgg.layers[-1].output) #denote that our ouput is the final layer of our VGG object

# save model file

model.save('models/vgg16_search_engine.h5')


###########################################################################################
# preprocessing & featurising functions
###########################################################################################


# image preproecessing function

def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width,img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    
    return image
    

# featurised image

def featurised_image(image):
    
    feature_vector = model.predict(image)
    
    
    return feature_vector
 
   
###########################################################################################
# featurise base images
###########################################################################################

# source directory for base images

source_dir = 'data/'


# empty objects to append to

file_name_store = []
feature_vector_store = np.empty((0,512))

# pass in & featurise base image set

for image in listdir(source_dir):
    
    print(image)
    
    # append image filename for future lookup
    file_name_store.append(source_dir + image)
    
    #preprocess the image
    preprocessed_image = preprocess_image(source_dir + image)
    
    #extract the feature vector
    feature_vector = featurised_image(preprocessed_image)
    
    #append feature vector for similarity calculations
    
    feature_vector_store = np.append(feature_vector_store, feature_vector, axis = 0)
    
    
# save key objects for future use

pickle.dump(file_name_store, open('models/file_name_store.p', 'wb'))
pickle.dump(feature_vector_store, open('models/feature_vector_store.p', 'wb'))
    
###########################################################################################
# pass in new image, and return similar images
###########################################################################################

# load in required objects

model = load_model('models/vgg16_search_engine.h5', compile = False) # compile just avoids warning message since we are not training model

file_name_store = pickle.load(open('models/file_name_store.p', 'rb'))
feature_vector_store = pickle.load(open('models/feature_vector_store.p', 'rb'))

# search parameters

search_results_n = 8 # how many search results we want return, when we feed a search image, we get 8 closest images
search_image = 'search_image_01.jpg' #filepath for search image itself
        
# preprocess & featurise search image

preprocessed_image = preprocess_image(search_image)# our search image needs to be proprocessed as well
search_feature_vector = featurised_image(preprocessed_image)      

# instantiate nearest neighbours logic
# we need to return 8 closeset matches, using nearast neighbor using cosine distance
# angle between 2 vectors, the smaller the angle, the lower the difference

image_neighbors = NearestNeighbors(n_neighbors = search_results_n, metric = 'cosine')


# apply to our feature vector store

image_neighbors.fit(feature_vector_store)

# return search results for search image (distances & indices)
# Backbone of image search engine- pass our search feature vector into image neighbor object, return: a) 8 nearest vectors and the distance and index from feature vector store

image_distances, image_indices = image_neighbors.kneighbors(search_feature_vector)

# convert closest image indices & distances to lists
# makes things easier for plotting

image_indices = list(image_indices[0])
image_distances = list(image_distances[0])

# get list of filenames for search results

search_result_files = [file_name_store[i] for i in image_indices]# list comprehension to extract each of image file names from file name object based on the index value from the list

# plot results

plt.figure(figsize=(12,9))
for counter, result_file in enumerate(search_result_files):    
    image = load_img(result_file)
    ax = plt.subplot(3, 3, counter+1)
    plt.imshow(image)
    plt.text(0, -5, round(image_distances[counter],3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()





