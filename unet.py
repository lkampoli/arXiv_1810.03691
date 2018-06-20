import keras
import tensorflow as tf
from keras import backend as K


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
keras.__version__
K.clear_session()
from keras.models import Model
import numpy as np
import h5py
import random
from scipy.stats import norm
from keras import regularizers, Input, optimizers, layers
from keras.layers import Activation
from keras.callbacks import History
from keras.models import load_model
import os

nb_epochs = 150
steps_train = 100
steps_valid = 50
steps_epochs = 1
b_size = 16
LR = 0.01
Decay = 0.8  #Decay applied every 10epochs
model_load = False
num_model_load = 43
adim_8 = 342.553


if not os.path.exists('model'):
    os.makedirs('model')

def pad_fields(passive, target, adim, size_block_x, size_block_yz):

    padded_passive = np.pad(passive, ((0, size_block_x), (size_block_yz // 2, size_block_yz // 2), (size_block_yz // 2, size_block_yz // 2)), mode="edge")
    #periodicity y
    padded_passive[:, :size_block_yz//2] = padded_passive[:, -2*(size_block_yz//2) : -(size_block_yz//2)]
    padded_passive[:, -(size_block_yz//2):] = padded_passive[:, (size_block_yz//2) : 2*(size_block_yz//2)]
    #periodicity z
    padded_passive[:, :, :size_block_yz//2] = padded_passive[:, :, -2*(size_block_yz//2) : -(size_block_yz//2)]
    padded_passive[:, :, -(size_block_yz//2):] = padded_passive[:, :, (size_block_yz//2) : 2*(size_block_yz//2)]

    target = target / adim
    padded_target = np.pad(target, ((0, size_block_x), (size_block_yz // 2, size_block_yz // 2), (size_block_yz // 2, size_block_yz // 2)), mode="edge")
    #periodicity y
    padded_target[:, :size_block_yz//2] = padded_target[:, -2*(size_block_yz//2) : -(size_block_yz//2)]
    padded_target[:, -(size_block_yz//2):] = padded_target[:, (size_block_yz//2) : 2*(size_block_yz//2)]
    #periodicity z
    padded_target[:, :, :size_block_yz//2] = padded_target[:, :, -2*(size_block_yz//2) : -(size_block_yz//2)]
    padded_target[:, :, -(size_block_yz//2):] = padded_target[:, :, (size_block_yz//2) : 2*(size_block_yz//2)]
    return(padded_passive, padded_target)

    
def generator_3D(begin_num, end_num, size_num, rotate, flip): #valid_or_train= 1 for validation, 2 for train
    while 1:

        size_block_yz = 16#np.random.randint(1, 4) * 8
        size_block_x = 16#np.random.randint(1, 4) * 8
        
        crops_per_block = 10

        batch_size = size_num*crops_per_block
        liste_input = np.zeros((batch_size, size_block_x, size_block_yz, size_block_yz, 1))
        liste_target = np.zeros((batch_size, size_block_x, size_block_yz, size_block_yz, 1))
        


        for num in range (size_num):
            new_begin_num = np.random.randint(begin_num, end_num) 
            number = str(116000 + (new_begin_num)*200)
            my_file = h5py.File("DATA/filtered19_00" + str(number) + ".h5", 'r');
            passive = my_file['filt_8'].value
            target = my_file['filt_grad_8'].value
            padded_passive, padded_target = pad_fields(passive, target, adim_8, size_block_x, size_block_yz)
            for k in range (crops_per_block):
          
                abs_x = np.random.randint(0, 64-size_block_x+1)
                abs_y = np.random.randint(0, 32)#with padding
                abs_z = np.random.randint(0, 32)                
                liste_input[num*crops_per_block + k, :, :, :, 0]=padded_passive[abs_x:abs_x+size_block_x, abs_y:abs_y+size_block_yz, abs_z:abs_z+size_block_yz]
                liste_target[num*crops_per_block + k, :, :, :, 0]=padded_target[abs_x:abs_x+size_block_x, abs_y:abs_y+size_block_yz, abs_z:abs_z+size_block_yz]

        if rotate:
            rot_times = np.random.randint(4, size=batch_size)
            rot_ax = np.random.randint(3, size=batch_size)
            ax_to_plane = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
            for i, (ti, ax) in enumerate(zip(rot_times, rot_ax)):
                print i
                print (ti, ax)
                print liste_input[i,].shape
                print np.rot90(liste_input[i,], k=ti, axes=ax_to_plane[ax]).shape
                liste_input[i,] = np.rot90(liste_input[i,], k=ti, axes=ax_to_plane[ax])
                liste_target[i,] = np.rot90(liste_target[i,], k=ti, axes=ax_to_plane[ax])

        if flip:
            for i, flip_by in enumerate(np.random.randint(6, size=batch_size)):
                if flip_by <= 2:
                    liste_input[i] = np.flip(liste_input[i], axis=flip_by)
                    liste_target[i] = np.flip(liste_target[i], axis=flip_by)

        indice = np.arange(liste_input.shape[0])
        np.random.shuffle(indice)   
        liste_input = liste_input[indice]
        liste_target = liste_target[indice]

        
        yield liste_input, liste_target

# better with cross ?
#train_generator = generator_3D(0, 75, 4, False, False)
#valid_generator = generator_3D(75, 95, 1, False, False)
train_generator = generator_3D(97, 98, 4, False, False)
valid_generator = generator_3D(97, 98, 1, False, False)

num_channels = 1
num_mask_channels = 1
img_shape = (None, None, None, 1)

inputs = keras.Input(shape = img_shape)
conv1 = layers.Conv3D(32, 3, padding='same')(inputs)
conv1 = layers.BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = layers.Conv3D(32, 3, padding='same')(conv1)
conv1 = layers.BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

conv2 = layers.Conv3D(64, 3, padding='same')(pool1)
conv2 = layers.BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = layers.Conv3D(64, 3, padding='same')(conv2)
conv2 = layers.BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

conv3 = layers.Conv3D(128, 3, padding='same')(pool2)
conv3 = layers.BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = layers.Conv3D(128, 3, padding='same')(conv3)
conv3 = layers.BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 =  layers.UpSampling3D(size=(2, 2, 2))(conv3)

up4  = layers.concatenate([conv3, conv2])
conv4 = layers.Conv3DTranspose(64, 3, padding='same')(up4)
conv4 = layers.BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)
conv4 = layers.Conv3DTranspose(64, 3, padding='same')(conv4)
conv4 = layers.BatchNormalization()(conv4)##conv ou crop
conv4 = Activation('relu')(conv4)
conv4 = layers.Conv3DTranspose(64, 1, padding='same')(conv4)
conv4 =  layers.UpSampling3D(size=(2, 2, 2))(conv4)

up5  = layers.concatenate([conv4, conv1])
conv5 = layers.Conv3DTranspose(32, 3, padding='same')(up5)
conv5 = layers.BatchNormalization()(conv5)
conv5 = Activation('relu')(conv5)
conv5 = layers.Conv3DTranspose(32, 3, padding='same')(conv5)
conv5 = layers.BatchNormalization()(conv5)##conv ou crop
conv5 = Activation('relu')(conv5)
conv5 = layers.Conv3DTranspose(1, 1, padding='same', activation='relu')(conv5)

model = Model(inputs=inputs, outputs=conv5)
#model.summary()



new_Adam = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=new_Adam, loss='mse', metrics = ['mae'])

if model_load :
    model.load_weights('model/unet_benzen_' + str(num_model_load) + '.h5')

    
min_loss = 0
val_min_loss = 1000000
for kk in range(nb_epochs/steps_epochs):
    k=kk+1

    if (k%10 == 0):
        LR = LR * Decay
        print "new LR : ", LR
        new_Adam = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=new_Adam, loss='mse', metrics = ['mae'])

    history = model.fit_generator(train_generator, steps_per_epoch=steps_train, epochs=steps_epochs, validation_data=valid_generator, validation_steps = steps_valid)

    loss = history.history['loss']
    loss = np.asarray(loss)
    val_loss = history.history['val_loss']
    val_loss = np.asarray(val_loss)
   
    if (val_loss < val_min_loss):
        min_loss = k
        val_min_loss = val_loss
        model.save('model/unet_benzen_' + str(k) +'.h5')
        np.save ('best_model', str(min_loss))

    print "Epoch number :" + str(k)
    
print "Best val_loss results for Epochs " + str(min_loss) + " with value " + str(val_min_loss)

np.save ('best_model', str(min_loss))
