import data_utils_multi as utils
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

from CascadeAttention import CascadeAttention
from Cascade import Cascade
from MultiviewAttention import MultiviewAttention
from Multiview import Multiview



window_size = 10
    
conv1_filters = 4
conv2_filters = 8
conv3_filters = 16

conv1_kernel_shape = (7,7)
conv2_kernel_shape = conv1_kernel_shape
conv3_kernel_shape = conv1_kernel_shape

padding1 = "same"
padding2 = padding1
padding3 = padding1

conv1_activation = "relu"
conv2_activation = conv1_activation
conv3_activation = conv1_activation

dense_nodes = 500
dense_activation = "relu"
dense_dropout = 0.5

lstm1_cells = 10
lstm2_cells = lstm1_cells

dense3_nodes = dense_nodes
dense3_activation = "relu"
final_dropout = 0.5

def get_cascade_model():
    cascade_object = Cascade(window_size,conv1_filters,conv2_filters,conv3_filters,
                conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                padding1,padding2,padding3,conv1_activation,conv2_activation,
                conv3_activation,dense_nodes,dense_activation,dense_dropout,
                lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation,
                final_dropout)
    cascade_model = cascade_object.model
    return cascade_model, cascade_object

def get_cascade_model_attention():
    cascade_attention_object = CascadeAttention(window_size,conv1_filters,conv2_filters,conv3_filters,
                conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                padding1,padding2,padding3,conv1_activation,conv2_activation,
                conv3_activation,dense_nodes,dense_activation,dense_dropout,
                lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation,
                final_dropout)
    cascade_attention_model = cascade_attention_object.model
    return cascade_attention_model, cascade_attention_object

def get_multiview_model():
    multiview_object = Multiview(window_size,conv1_filters,conv2_filters,conv3_filters,
             conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
             padding1,padding2,padding3,conv1_activation,conv2_activation,
             conv3_activation,dense_nodes,dense_activation,
             lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation)
    multiview_model = multiview_object.model
    return multiview_model, multiview_object

def get_multiview_model_attention():
    multiview_attention_object = MultiviewAttention(window_size,conv1_filters,conv2_filters,conv3_filters,
             conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
             padding1,padding2,padding3,conv1_activation,conv2_activation,
             conv3_activation,dense_nodes,dense_activation,
             lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation)
    multiview_attention_model = multiview_attention_object.model
    return multiview_attention_model, multiview_attention_object






def find_first_index_rest(label_list):
    for i in range(len(label_list)):
        if label_list[i][0] == 1:
            return i

def find_first_index_math(label_list):
    for i in range(len(label_list)):
        if label_list[i][1] == 1:
            return i
        
def find_first_index_memory(label_list):
    for i in range(len(label_list)):
        if label_list[i][2] == 1:
            return i

def find_first_index_motor(label_list):
    for i in range(len(label_list)):
        if label_list[i][3] == 1:
            return i
    
    
def plot_feature_maps_conv1(type_state,model_type,model):
    if model_type == "Cascade":
        model_conv1 = Model(inputs=model.inputs, outputs=[model.layers[i].output for i in range(10,20)])
    if model_type == "Multiview":
        model_conv1 = Model(inputs=model.inputs, outputs=[model.layers[i].output for i in range(30,40)])
    if type_state == "Rest":
        feature_maps_conv1  = model_conv1.predict(dict_image_rest_to_predict)
    if type_state == "Math":
        feature_maps_conv1  = model_conv1.predict(dict_image_math_to_predict)
    if type_state == "Memory":
        feature_maps_conv1  = model_conv1.predict(dict_image_mem_to_predict)
    if type_state == "Motor":
        feature_maps_conv1  = model_conv1.predict(dict_image_motor_to_predict)
    else:
        feature_maps_conv1  = model_conv1.predict(dict_image_rest_to_predict)

    fig, axs = plt.subplots(10,4, figsize=(20, 21))
    fig.subplots_adjust(left=0, bottom=0.5, right=0.5, top=1.5, wspace=0, hspace=None)

    axs = axs.ravel()
    rows = 10
    cols = 4

    ax_number = 0
    for i in range(rows):
        for j in range(cols):
            axs[ax_number].imshow(feature_maps_conv1[i][0,:,:,j], cmap='summer')
            axs[ax_number].set_title("{} map {} for input {}".format(type_state,j+1,i+1))
            axs[ax_number].get_xaxis().set_visible(False)
            axs[ax_number].get_yaxis().set_visible(False)
            ax_number += 1
            
def plot_feature_maps_conv2(type_state,model_type,model):
    if model_type == "Cascade":
        model_conv2 = Model(inputs=model.inputs, outputs=[model.layers[i].output for i in range(20,30)])
    if model_type == "Multiview":
        model_conv2 = Model(inputs=model.inputs, outputs=[model.layers[i].output for i in range(50,60)])
         
    if type_state == "Rest":
        feature_maps_conv2  = model_conv2.predict(dict_image_rest_to_predict)
    if type_state == "Math":
        feature_maps_conv2  = model_conv2.predict(dict_image_math_to_predict)
    if type_state == "Memory":
        feature_maps_conv2  = model_conv2.predict(dict_image_mem_to_predict)
    if type_state == "Motor":
        feature_maps_conv2  = model_conv2.predict(dict_image_motor_to_predict)
    else:
        feature_maps_conv2  = model_conv2.predict(dict_image_rest_to_predict)
    fig, axs = plt.subplots(10,8, figsize=(20, 21))
    fig.subplots_adjust(left=0, bottom=0.5, right=0.9, top=1.5, wspace=0, hspace=None)

    axs = axs.ravel()
    rows = 10
    cols = 8

    ax_number = 0
    for i in range(rows):
        for j in range(cols):
            axs[ax_number].imshow(feature_maps_conv2[i][0,:,:,j], cmap='summer')
            axs[ax_number].set_title("{} Ft map {} for input {}".format(type_state,j,i+1))
            axs[ax_number].get_xaxis().set_visible(False)
            axs[ax_number].get_yaxis().set_visible(False)
            ax_number += 1
            
def plot_feature_maps_conv3(type_state,model_type,model):    
    if model_type == "Cascade":
        model_conv3 = Model(inputs=model.inputs, outputs=[model.layers[i].output for i in range(30,40)])
    if model_type == "Multiview":
        model_conv3 = Model(inputs=model.inputs, outputs=[model.layers[i].output for i in range(61,71)])
    
    if type_state == "Rest":
        feature_maps_conv3  = model_conv3.predict(dict_image_rest_to_predict)
    if type_state == "Math":
        feature_maps_conv3  = model_conv3.predict(dict_image_math_to_predict)
    if type_state == "Memory":
        feature_maps_conv3  = model_conv3.predict(dict_image_mem_to_predict)
    if type_state == "Motor":
        feature_maps_conv3  = model_conv3.predict(dict_image_motor_to_predict)
    else:
        feature_maps_conv3  = model_conv3.predict(dict_image_rest_to_predict)

    fig, axs = plt.subplots(10,16, figsize=(20, 21))
    fig.subplots_adjust(left=0, bottom=0.5, right=2, top=1.5, wspace=0, hspace=None)

    axs = axs.ravel()
    rows = 10
    cols = 16

    ax_number = 0
    for i in range(rows):
        for j in range(cols):
            axs[ax_number].imshow(feature_maps_conv3[i][0,:,:,j], cmap='summer')
            axs[ax_number].set_title("{} Ft map {} for input {}".format(type_state,j,i+1))
            axs[ax_number].get_xaxis().set_visible(False)
            axs[ax_number].get_yaxis().set_visible(False)
            ax_number += 1
            
            
subject_files_train = []
subjects = ['212318']
for subject in subjects:
    for item in utils.test_files_dirs:
        if subject in item:
            subject_files_train.append(item)
subject_files_train = subject_files_train[:4]


import argparse
parser = argparse.ArgumentParser()



parser.add_argument('-m','--model', type=str,help="Please choose the type of model \
                    you want to visualize (cascade or multiview)",choices=['cascade', 'multiview'])

parser.add_argument('-a','--attention',type=bool,help="Please choose whether you \
                    want to use self attention (True or False), by default no attention", default=False)

parser.add_argument('-s','--state',type=str,help="Please choose the type of \
                    state you want to plot (Rest,Math,Memory,Motor), by default\
                    Rest state", default="rest", choices=['Rest', 'Math', 'Memory', 'Motor'])

parser.add_argument('-w','--weights',type=str,help='Please indicate the weight files \
                    that you want the model to use (if none, will take by default the \
                    ones provided)')

parser.add_argument('-l','--layer',type=int,help="Please choose the layer \
                    number you want to visualize (1,2,3), by default 1.", default=1)


args = parser.parse_args()


if args.model.lower() == "cascade":
    model_type = "Cascade"
elif args.model.lower() == "multiview":
    model_type = "Multiview"
else:
    print("No model chosen, exiting ...")
    import sys
    sys.exit()
    
use_attention = args.attention

cascade_default_weights = "checkpoint-epoch_009-val_acc_0.997.hdf5"
cascade_attention_default_weights = "checkpoint-epoch_009-val_acc_0.997.hdf5"

multiview_default_weights = "checkpoint-epoch_009-val_acc_0.997.hdf5"
multiview_attention_default_weights = "checkpoint-epoch_009-val_acc_0.997.hdf5"

if model_type == "Cascade":
    if use_attention==False:
        model,model_object = get_cascade_model()
        if not args.weights :
            model.load_weights(cascade_default_weights)
        else:
            try:
                model.load_weights(args.weights)
            except Exception as e:
                print("Exception occured when trying to load the weights !")
                print("Please try loading default weights or passing absolute path of the weights")
                print("Exception message : {}".format(e))
    else:
        model,model_object = get_cascade_model_attention()
        if not args.weights :
            model.load_weights(cascade_attention_default_weights)
        else:
            try:
                model.load_weights(args.weights)
            except Exception as e:
                print("Exception occured when trying to load the weights !")
                print("Please try loading default weights or passing absolute path of the weights")
                print("Exception message : {}".format(e))
        
else:
    if use_attention == False:
        model,model_object = get_multiview_model()
        if not args.weights :
            model.load_weights(multiview_default_weights)
        else:
            try:
                model.load_weights(args.weights)
            except Exception as e:
                print("Exception occured when trying to load the weights !")
                print("Please try loading default weights or passing absolute path of the weights")
                print("Exception message : {}".format(e))
    else:
        model,model_object = get_multiview_model_attention()
        if not args.weights :
            model.load_weights(multiview_attention_default_weights)
        else:
            try:
                model.load_weights(args.weights)
            except Exception as e:
                print("Exception occured when trying to load the weights !")
                print("Please try loading default weights or passing absolute path of the weights")
                print("Exception message : {}".format(e))
        

        


if model_type == "Cascade":
    X_train, Y_train = utils.load_overlapped_data_cascade(subject_files_train)
else:
    X_train, Y_train = utils.load_overlapped_data_multiview(subject_files_train)
    
dict_image_rest_to_predict = {}
dict_image_math_to_predict = {}
dict_image_mem_to_predict = {}
dict_image_motor_to_predict = {}
            
  

num_rows_mesh = 20
num_cols_mesh = 21
num_meg_channels = 248   
num_inputs_cascade = 10
num_inputs_multiview = 20       
if model_type == "Cascade":        
    for i in range(num_inputs_cascade):
        dict_image_rest_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][0],(1,num_rows_mesh,num_cols_mesh,1))
    
    for i in range(num_inputs_cascade):
        dict_image_math_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][4],(1,num_rows_mesh,num_cols_mesh,1))
    
    for i in range(num_inputs_cascade):
        dict_image_mem_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][3],(1,num_rows_mesh,num_cols_mesh,1))
    
    for i in range(num_inputs_cascade):
        dict_image_motor_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][1],(1,num_rows_mesh,num_cols_mesh,1))
        
if model_type == "Multiview":
    for i in range(num_inputs_multiview):
        if i < 10:
            dict_image_rest_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][0],(1,num_rows_mesh,num_cols_mesh,1))
        if i >= 10 :
            dict_image_rest_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][0],(1,num_meg_channels,1))

    for i in range(num_inputs_multiview):
        if i < 10 :
            dict_image_math_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][4],(1,num_rows_mesh,num_cols_mesh,1))
        if i >= 10 :
            dict_image_math_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][4],(1,num_meg_channels,1))
    
    
    for i in range(num_inputs_multiview):
        if i < 10 :
            dict_image_mem_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][3],(1,num_rows_mesh,num_cols_mesh,1))
        if i >= 10:
            dict_image_mem_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][3],(1,num_meg_channels,1))
    
    for i in range(num_inputs_multiview):
        if i < 10 :
            dict_image_motor_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][1],(1,num_rows_mesh,num_cols_mesh,1))
        if i>= 10:
            dict_image_motor_to_predict["input"+str(i+1)] = np.reshape(X_train["input"+str(i+1)][1],(1,num_meg_channels,1))
            
if args.layer == 1:
    plot_feature_maps_conv1(args.state,model_type,model)
elif args.layer == 2:
    plot_feature_maps_conv2(args.state,model_type,model)
elif args.layer == 3:
    plot_feature_maps_conv3(args.state,model_type,model)


    
    
        



