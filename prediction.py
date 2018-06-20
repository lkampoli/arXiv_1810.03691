import keras
keras.__version__
import tensorflow as tf
from keras import backend as K
print "test allocation"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
keras.__version__
K.clear_session()

import numpy as np
import h5py
import random
from matplotlib import cm
from keras.models import load_model
import time
import matplotlib.pyplot as plt

num_best_model = np.load('best_model.npy')#load the best model number

model = load_model('model/unet_benzen_' + str(num_best_model) + '.h5')

deraf = 8
size_block = 16
size_num = 1
begin_num = 0
adim_8 = 342.553
dim_x = 64
dim_y = 32
dim_z = 32
div = dim_x / 1


liste_input = np.zeros((size_num, dim_x, dim_y, dim_z, 1))
liste_target = np.zeros((size_num, dim_x, dim_y, dim_z))
liste_LES = np.zeros((size_num, dim_x, dim_y, dim_z))


def true_integrale(field, factor=1):
    x, y, z = field.shape
    if (x==1):
        new_field = np.zeros((x, y-1, z-1))
        new_field += field[:, 1: , 1:]
        new_field += field[:, 1:, :-1]
        new_field += field[:, :-1, 1:]
        new_field += field[:, :-1, :-1]
        new_field = new_field/4
        new_field = new_field * (factor * x / 10000.) * (factor * y / 10000.) * (factor * z / 10000.)
    else:
        new_field = np.zeros((x-1, y-1, z-1))
        new_field += field[:-1, :-1, :-1]
        
        new_field += field[1:, :-1, :-1]
        new_field += field[:-1, 1:, :-1]
        new_field += field[1:, 1: , :-1]
        new_field += field[:-1, :-1, 1:]
        new_field += field[1:, :-1, 1:]
        new_field += field[:-1, 1:, 1:]
        new_field += field[1:, 1: , 1:]
        new_field = new_field/8
        new_field = new_field * (factor * x / 10000.) * (factor * y / 10000.) * (factor * z / 10000.)

    return np.mean(new_field)

def grad_n(arr, spacing=1):
    spacing = spacing/128.
    x, y, z = np.gradient(arr, spacing)
    return np.sqrt(x**2 + y**2 + z**2)

for num in range (size_num):
    
    number = str(135400 + (begin_num+num)*200)

    my_file = h5py.File("DATA/filtered19_00" + str(number) + ".h5", 'r');

    passive = my_file['filt_' + str(deraf)].value
    target = my_file['filt_grad_' + str(deraf)].value
    LES = my_file['grad_filt_' + str(deraf)].value

    target = target / adim_8
    LES = LES / adim_8

    liste_input[num, :, :, :, 0] = passive[:dim_x, :dim_y, :dim_z]
    liste_target[num] = target[:dim_x, :dim_y, :dim_z]
    liste_LES[num] = LES[:dim_x, :dim_y, :dim_z]


    predictions= model.predict(liste_input)

    true_vals = liste_target
    
     
#denormalization
surface_LES = np.zeros((size_num, div))
surface_real = np.zeros((size_num, div))
surface_pred = np.zeros((size_num, div))

mse = np.zeros((size_num, div))

for i in range(size_num):
    for k in range(div):
        surface_LES[i, k] =  325 * true_integrale(liste_LES[i, (k * dim_x/div):((k+1) * dim_x/div)], 8)
        surface_pred[i, k] = 325 * true_integrale(predictions[i, (k * dim_x/div):((k+1) * dim_x/div), :, :, 0], 8)

	surface_real[i, k] = 325 * true_integrale(true_vals[i, (k * dim_x/div):((k+1) * dim_x/div)], 8)
       

        mse[i, k] = np.mean( (predictions[i, (k * dim_x/div):((k+1) * dim_x/div), :, :, 0] - true_vals[i, (k * dim_x/div):((k+1) * dim_x/div)]) ** 2)
      
 
adim_surface = 0.0256*2*deraf*0.0001 #mesh : 256 points of size 0.0001 meter
surface_LES /= adim_surface
surface_real /= adim_surface
surface_pred /= adim_surface


print "Surface target moyenne = " + str((true_vals).mean(axis=None))
print "Surface prediction moyenne = " + str((predictions).mean(axis=None))
print "MSE U-net = " + str((np.abs(mse)).mean(axis=None))

pos_x = np.linspace(1, dim_x*8/10, div)


plt.rc('font', family='serif')
plt.rc('text', usetex=False)


plt.ylabel('FS adimensional')
plt.xlabel('x (mm)')


plt.plot(pos_x, surface_LES[0,], '--', color='0.70',  markevery=5, label='LES')
plt.plot(pos_x, surface_real[0,], 'k', markevery=5, label='DNS')
plt.plot(pos_x, surface_pred[0,], '--s', color='0.3',  markevery=5, label='U-net prediction')

plt.legend()
plt.show()
