import keras
keras.__version__
import tensorflow as tf
from keras import backend as K
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
from glob import glob



def true_integral(field, factor=1):#Integral for the center of each cell
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




def CNN_predict(distrib, model):

    deraf = 8
    size_block = 16
    nbr_pred = 1

    crop_size = 2*2*4
    batch_size = nbr_pred*crop_size

    grad_lam = 325
    adim_8 = 342.553

    dim_x = 64
    dim_y = 32
    dim_z = 32

    div = dim_x // 1

    liste_input = np.zeros((nbr_pred, dim_x, dim_y, dim_z, 1))
    liste_target = np.zeros((nbr_pred, dim_x, dim_y, dim_z))
    liste_LES = np.zeros((nbr_pred, dim_x, dim_y, dim_z))

    for num in range (nbr_pred):

        liste_name = sorted(glob("DATA/" + distrib + "*.h5"))
        choosen_num = np.random.randint(len(liste_name))
        my_file = h5py.File(liste_name[choosen_num], 'r');

        passive = my_file['filt_' + str(deraf)].value
        target = my_file['filt_grad_' + str(deraf)].value
        LES = my_file['grad_filt_' + str(deraf)].value

        target = target / adim_8
        LES = LES / adim_8

        liste_input[num, :, :, :, 0] = passive[:dim_x, :dim_y, :dim_z]
        liste_target[num] = target[:dim_x, :dim_y, :dim_z]
        liste_LES[num] = LES[:dim_x, :dim_y, :dim_z]

        predictions= model.predict(liste_input)

    #denormalization
    surface_LES = np.zeros((nbr_pred, div))
    surface_real = np.zeros((nbr_pred, div))
    surface_pred = np.zeros((nbr_pred, div))

    mse = np.zeros((nbr_pred, div))

    for i in range(nbr_pred):
        for k in range(div):
            surface_LES[i, k] =  grad_lam * true_integral(liste_LES[i, (k * dim_x//div):((k+1) * dim_x//div)], 8)
            surface_pred[i, k] = grad_lam * true_integral(predictions[i, (k * dim_x//div):((k+1) * dim_x//div), :, :, 0], 8)
            surface_real[i, k] = grad_lam * true_integral(liste_target[i, (k * dim_x//div):((k+1) * dim_x//div)], 8)
            mse[i, k] = np.mean( (predictions[i, (k * dim_x//div):((k+1) * dim_x//div), :, :, 0] - liste_target[i, (k * dim_x//div):((k+1) * dim_x//div)]) ** 2)
    
    adim_surface = 0.0256*2*deraf*0.0001 #mesh : 256 points of size 0.0001 meter
    surface_LES /= adim_surface
    surface_real /= adim_surface
    surface_pred /= adim_surface

    print ("Mean surface target = " + str((liste_target).mean(axis=None)))
    print ("Mean surface predicted  = " + str((predictions).mean(axis=None)))
    print ("MSE CNN = " + str((np.abs(mse)).mean(axis=None)))

    return(surface_LES, surface_real, surface_pred)

def plot_results(surface_LES, surface_real, surface_pred):
  
    dim_x = 64
    div = dim_x/1
    pos_x = np.linspace(1, dim_x*8/10, div)

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)
    plt.title('Flame surface')
    plt.ylabel('FS adimensional')
    plt.xlabel('x (mm)')
        
    plt.plot(pos_x, surface_LES[0,], '--', color='0.70',  markevery=5, label='LES')
    plt.plot(pos_x, surface_real[0,], 'k', markevery=5, label='DNS')
    plt.plot(pos_x, surface_pred[0,], '--s', color='0.3',  markevery=5, label='CNN prediction')
    plt.legend()
    plt.show()

    
if __name__ == "__main__":

    name_models = sorted(glob("save_model/*"))
    best_model = name_models[-1]
    print (name_models)
    print (best_model)
    model = load_model(best_model)#Load the model with the lowest loss

    surface_LES, surface_real, surface_pred = CNN_predict("DNS3", model) #Prediction on a field never seen during the training
    
    plot_results(surface_LES, surface_real, surface_pred)
    
    
