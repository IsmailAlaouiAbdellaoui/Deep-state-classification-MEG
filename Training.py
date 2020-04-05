import time
import gc
import sys

import experiment_utils as eutils
import data_utils as utils

from tensorflow.keras.optimizers import Adam

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
    

train_loss_results = []
train_accuracy_results = []

batch_size = 64



def train(model_type,use_attention,setup,num_epochs):
    if setup == 0:#used for quick tests
        subjects = ['105923']
        list_subjects_test = ['212318']
    if setup == 1:
        subjects = ['105923','164636','133019']
        list_subjects_test = ['204521','212318','162935','707749','725751','735148']
    if setup == 2:
        subjects = ['105923','164636','133019','113922','116726','140117']
        list_subjects_test = ['204521','212318','162935','707749','725751','735148']
    if setup == 3:
        subjects = ['105923','164636','133019','113922','116726','140117','175237','177746','185442','191033','191437','192641']
        list_subjects_test = ['204521','212318','162935','707749','725751','735148']
        
    if model_type == "Cascade":
        if use_attention==False:
            model,model_object = get_cascade_model()
        else:
            model,model_object = get_cascade_model_attention()
    
    else:
        if use_attention == False:
            model,model_object = get_multiview_model()
        else:
            model,model_object = get_multiview_model_attention()
        
    
    subjects_string = ",".join([subject for subject in subjects])
    comment = "Training with subjects : " + subjects_string
    
    accuracies_temp_train = []
    losses_temp_train= []
    
    accuracies_train = []#per epoch
    losses_train = []#per epoch
    
    accuracies_temp_val = []
    losses_temp_val = []
    
    accuracies_val = []#per epoch
    losses_val = []#per epoch
    start_time = time.time()

    model.compile(optimizer = Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    
    experiment_number = eutils.on_train_begin(model_object,model_type)
    for epoch in range(num_epochs):
        print("\n\n Epoch",epoch+1," \n")
        for subject in subjects:
            start_subject_time = time.time()
            print("-- Training on subject", subject)
            subject_files_train = []
            for item in eutils.train_files_dirs:
                if subject in item:
                    subject_files_train.append(item)
            
            subject_files_val = []
            for item in eutils.validate_files_dirs:
                if subject in item:
                    subject_files_val.append(item)
            number_workers_training = 16
            number_files_per_worker = len(subject_files_train)//number_workers_training

            if model_type == 'Cascade':
                X_train, Y_train = utils.multi_processing_cascade(subject_files_train,number_files_per_worker,number_workers_training)            
            elif model_type == 'Multiview':
                X_train, Y_train = utils.multi_processing_multiview(subject_files_train,number_files_per_worker,number_workers_training)            

            X_train,Y_train = utils.reshape_input_dictionary(X_train, Y_train, batch_size)

            number_workers_validation = 8
            number_files_per_worker = len(subject_files_val)//number_workers_validation
            
            if model_type == 'Cascade':
                X_validate, Y_validate = utils.multi_processing_cascade(subject_files_val,number_files_per_worker,number_workers_validation)            
            elif model_type == 'Multiview':
                X_validate, Y_validate = utils.multi_processing_multiview(subject_files_val,number_files_per_worker,number_workers_validation)            
            
            X_validate, Y_validate = utils.reshape_input_dictionary(X_validate, Y_validate, batch_size)
            
            history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1, 
                                    verbose = 1, validation_data=(X_validate, Y_validate), 
                                    callbacks=None)
            subj_train_timespan = time.time() - start_subject_time
            print("Saving the model weights...")
            eutils.model_checkpoint(experiment_number,model,model_type) # saving model weights after each subject
            print("Model's weights saved, Epoch : {}, subject : {}".format(epoch+1,subject) )
            print("Timespan subject training is : {}".format(subj_train_timespan))
            history_dict = history.history
            accuracies_temp_train.append(history_dict['acc'][0])#its because its a list of 1 element
            losses_temp_train.append(history_dict['loss'][0])
            accuracies_temp_val.append(history_dict['val_acc'][0])
            losses_temp_val.append(history_dict['val_loss'][0])
            #Freeing memory
            X_train = None
            Y_train = None
            X_validate = None
            Y_validate = None
            gc.collect()
        
        print("Epoch {:03d}".format(epoch))

        ## Training Information ##
        average_loss_epoch_train = sum(losses_temp_train)/len(losses_temp_train)
        print("Epoch Training Loss : {:.3f}".format(average_loss_epoch_train))
        losses_train.append(average_loss_epoch_train)
        losses_temp_train = []

        average_accuracy_epoch_train = sum(accuracies_temp_train)/len(accuracies_temp_train)
        print("Epoch Training Accuracy: {:.3%}".format(average_accuracy_epoch_train))
        accuracies_train.append(average_accuracy_epoch_train)
        accuracies_temp_train = []

        ## Validation Information ##
        average_loss_epoch_validate = sum(losses_temp_val)/len(losses_temp_val)
        print("Epoch Validation Loss : {:.3f}".format(average_loss_epoch_validate))
        losses_val.append(average_loss_epoch_validate)
        losses_temp_val = []

        average_accuracy_epoch_validate = sum(accuracies_temp_val)/len(accuracies_temp_val)
        print("Epoch Validation Accuracy: {:.3%}".format(average_accuracy_epoch_validate))
        accuracies_val.append(average_accuracy_epoch_validate)
        accuracies_temp_val = []

        eutils.on_epoch_end(epoch, average_accuracy_epoch_train, average_loss_epoch_train, \
                        average_accuracy_epoch_validate, average_loss_epoch_validate, experiment_number, model,model_type)

        if (epoch+1) % 2 == 0 :
#            start_testing = time.time()
            print("Testing on subjects")
            accuracies_temp = []
            #Creating dataset for testing
            for subject in list_subjects_test:
                start_testing = time.time()
                print("Reading data from subject", subject)
                subject_files_test = []
                for item in eutils.test_files_dirs:
                        if subject in item:
                            subject_files_test.append(item)
                            
                number_workers_testing = 10
                number_files_per_worker = len(subject_files_test)//number_workers_testing
                print(number_files_per_worker)
                X_test, Y_test = eutils.multi_processing(subject_files_test,number_files_per_worker,number_workers_testing,model_type)

                print("\n\nEvaluation cross-subject: ")
                result = model.evaluate(X_test, Y_test, batch_size = batch_size,verbose=2)
                
                accuracies_temp.append(result[1])
                print("Recording the testing accuracy of '{}' in a file".format(subject))
                eutils.append_individual_test(experiment_number,epoch,subject,result[1],model_type)
                print("Timespan of testing is : {}".format(time.time() - start_testing))
            avg = sum(accuracies_temp)/len(accuracies_temp)
            print("Average testing accuracy : {0:.2f}".format(avg))
            print("Recording the average testing accuracy in a file")
            eutils.append_average_test(experiment_number,epoch,avg,model_type)

            X_test = None
            Y_test = None
                
    stop_time = time.time()
    time_span = stop_time - start_time
    print()
    print()
    print("training took {:.2f} seconds".format(time_span))
    eutils.on_train_end(experiment_number,model_type)
    eutils.save_training_time(experiment_number, time_span,model_type)
    eutils.write_comment(experiment_number,comment,model_type)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--setup', type=int, help="Please select a number between \
                    0 and 4 to choose the setup of the training")

parser.add_argument('-m','--model', type=str,help="Please choose the type of model \
                    you want to train (cascade or multiview)",choices=['cascade', 'multiview'])

parser.add_argument('-a','--attention',type=bool,help="Please choose whether you \
                    want to use self attention (True or False), by default no attention", default=False)

parser.add_argument('-e','--epochs',type=int,help="Please choose the number of \
                    epochs, by default 1 epoch", default=1)

args = parser.parse_args()



if args.model.lower() == "cascade":
    model_type = "Cascade"
elif args.model.lower() == "multiview":
    model_type = "Multiview"
else:
    print("No model chosen, exiting ...")
    sys.exit()
    
if args.attention == True:
    use_attention = True
    attention = "with"
elif args.attention == False:
    use_attention = False
    attention = "without"
else:
    print("Attention mechanism is not specified.")
    print("The model will be without attention.")
    use_attention = False
    attention = "without"

if args.setup:
    setup = args.setup
else:
    print("No setup has been chosen, basic training will start.")
    setup = 0

epochs = args.epochs
print("Training of {} model {} attention mechanism will begin with {} epochs and setup {}.".format(model_type,attention,epochs,setup))
train(model_type,use_attention,setup,epochs)
    
#Snippet might come useful later
#import tensorflow as tf
#print(type(tf.__version__))
#version_tf = float(tf.__version__.split('.')[0])
#if(version_tf >= 2):
#  print(history_dict['accuracy'][0])#its because its a list of 1 element
#  print(history_dict['loss'][0])
#  print(history_dict['val_accuracy'][0])
#  print(history_dict['val_loss'][0])
#else:
#  print(history_dict['acc'][0])
#  print(history_dict['loss'][0])
#  print(history_dict['val_acc'][0])
#  print(history_dict['val_loss'][0])
    
    
#To do for refactoring:


    
       
