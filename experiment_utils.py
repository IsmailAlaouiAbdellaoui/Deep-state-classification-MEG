from os import listdir
from os.path import isfile, join
import numpy as np
import data_utils as utils
import gc
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl



def create_summary_file(experiment_number,model_type):
        filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
        with open(filename, "w") as file:
            file.write("Summary of the model used for experiment"+str(experiment_number)+" : \n\n")

def plot_epochs_info(experiment_number,model_type):
    filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
    train_accuracies = []
    train_losses = []
    valid_accuracies = []
    valid_losses = []
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            number_epochs = len(lines)
            x_values = np.arange(start = 1, stop = number_epochs + 1)
            for line in lines:
                temp_parts = line.split(',')
                
                train_accuracy_part = temp_parts[1]
                train_accuracies.append(float(train_accuracy_part.split(':')[1]))
                
                train_loss_part = temp_parts[2]
                train_losses.append(float(train_loss_part.split(':')[1]))
                
                valid_accuracy_part = temp_parts[3]
                valid_accuracies.append(float(valid_accuracy_part.split(':')[1]))
                
                valid_loss_part = temp_parts[4]
                valid_losses.append(float(valid_loss_part.split(':')[1]))
                
    except Exception as e:
        print("Problem while reading the file {}".format(filename))
        print("Exception message : {}".format(e))
    # plt.figure(figsize=(10,10))
    mpl.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,10))
    # plt.figure()
    ax1.plot(x_values,train_accuracies,label="Training Accuracy",color="#4C72B0")
    ax1.plot(x_values,valid_accuracies,label="Validation Accuracy", color='#55A868')
    ax1.legend(loc="upper left",fontsize='small')
    
    ax2.plot(x_values,train_losses,label="Training Loss", color = "#DD8452" )
    ax2.plot(x_values,valid_losses,label="Validation Loss", color="#C44E52")
    ax2.legend(loc="upper right",fontsize='small')
    
    ax1.set_title("Accuracy during Training and Validation")
    ax2.set_title("Loss during Training and Validation")
    plt.xlabel("Epochs")
    ax1.set(ylabel ="Accuracy")
    ax2.set(ylabel="Loss")
    output_filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/plot_model"+str(experiment_number)+".png"
    fig.savefig(output_filename,dpi=100)
    plt.show()

def get_experiment_number(model_type):
    experiments_folders_list = os.listdir(path='Experiments/'+model_type)
    if(len(experiments_folders_list) == 0): #empty folder
        return 1
    else:  
        temp_numbers=[]
        for folder in experiments_folders_list:
            number = re.findall(r'\d+', folder)
            if(len(number)>0):
                temp_numbers.append(int(number[0]))
        return max(temp_numbers) + 1

def append_to_epochs_file(experiment_number, epoch_number, training_accuracy, training_loss, validation_accuracy, validation_loss,model_type):
    filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("Epoch {0},training_acuracy:{1:.2f},trainig_loss:{2:.2f},validation_accuracy:{3:.2f},validation_loss:{4:.2f}\n".format(epoch_number,training_accuracy,training_loss,validation_accuracy,validation_loss))

def create_model_folder(model_type):
    path_model = "Experiments/"+model_type
    if(not os.path.isdir(path_model)):
        try:
            os.mkdir(path_model)
        except Exception as e:
            print ("Creation of the main model experiment directory failed")
            print("Exception error: ",str(e))     
        
def create_experiment_folder(experiment_number,model_type):
    try:
        path_new_experiment = "Experiments/"+model_type+"/Experiment" + str(experiment_number)
        check_point_path = path_new_experiment+"/checkpoints"
        os.mkdir(path_new_experiment)
        os.mkdir(check_point_path)
    except Exception as e:
        print ("Creation of the directory {} or {} failed".format(path_new_experiment,check_point_path))
        print("Exception error: ",str(e))  


def create_main_experiment_folder():
    if(not os.path.isdir("Experiments")):
        try:
            os.mkdir("Experiments")
        except Exception as e:
            print ("Creation of the main experiment directory failed")
            print("Exception error: ",str(e))

def create_info_epochs_file(experiment_number,model_type):
        filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
        with open(filename, "w") as file:
            file.write("")

def on_train_begin(model_object,model_type):
    create_main_experiment_folder()
    create_model_folder(model_type)
    experiment_number = get_experiment_number(model_type)
    create_experiment_folder(experiment_number,model_type)
    print()
    print()
    print("-"*7 +" Beginning of Experiment {} of the {} model ".format(experiment_number,model_type) + "-"*7)            
    print()
    print()
    # self.create_experiment_folder(self.experiment_number)
    create_summary_file(experiment_number,model_type)
    append_to_summary_file(model_object, experiment_number,model_type)
    create_info_epochs_file(experiment_number,model_type)
    create_info_test_file(experiment_number,model_type)
    return experiment_number

def on_train_end(experiment_number,model_type):
    print()
    print()
    print("-"*7 +" End of Experiment {} ".format(experiment_number) + "-"*7)
    print()
    print()
    print("-"*7 +" Plotting and saving the epochs training/validation accuracy/loss " + "-"*7)
    plot_epochs_info(experiment_number,model_type)

def save_training_time(experiment_number,time,model_type):
    filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("\n\nTraining time: {:.2f} seconds".format(time))

def write_comment(experiment_number,comment,model_type):
    filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("\nAttention model, comment :  {}".format(comment))
##################### Add whether its an attention


def on_epoch_end(epoch, accuracy, loss, val_accuracy, val_loss,experiment_number,model,model_type):
    try:
        append_to_epochs_file(experiment_number,epoch, accuracy, loss, val_accuracy, val_loss,model_type)

    except Exception as e:
        print("Failed to append in epoch file or saving weights...")
        print("Exception error: ",str(e))

    
def model_checkpoint(experiment_number,model,model_type):
    exp_path = "Experiments/"+model_type+"/Experiment" + str(experiment_number)
    checkpoint_path = exp_path+"/checkpoints" 
    model.save_weights(checkpoint_path)
    

def append_to_summary_file(model_object, experiment_number,model_type):
        filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
        with open(filename, "a+") as file:
            file.write("window_size: "+str(model_object.window_size)+"\n")#.format(str(model_object.window_size)))
            file.write("conv1_filters: {}\n".format(str(model_object.conv1_filters)))
            file.write("conv2_filters: {}\n".format(str(model_object.conv2_filters)))
            file.write("conv3_filters: {}\n".format(str(model_object.conv3_filters)))
            file.write("conv1_kernel_shape: {}\n".format(str(model_object.conv1_kernel_shape)))
            file.write("conv2_kernel_shape: {}\n".format(str(model_object.conv2_kernel_shape)))
            file.write("conv3_kernel_shape: {}\n".format(str(model_object.conv3_kernel_shape)))
            file.write("padding1: {}\n".format(str(model_object.padding1)))
            file.write("padding2: {}\n".format(str(model_object.padding2)))
            file.write("padding3: {}\n".format(str(model_object.padding3)))
            file.write("conv1_activation: {}\n".format(str(model_object.conv1_activation)))
            file.write("conv2_activation: {}\n".format(str(model_object.conv2_activation)))
            file.write("conv3_activation: {}\n".format(str(model_object.conv3_activation)))
            file.write("dense_nodes: {}\n".format(str(model_object.dense_nodes)))
            file.write("dense_activation: {}\n".format(str(model_object.dense_activation)))
            file.write("lstm1_cells: {}\n".format(str(model_object.lstm1_cells)))
            file.write("lstm2_cells: {}\n".format(str(model_object.lstm2_cells)))
            file.write("dense3_nodes: {}\n".format(str(model_object.dense3_nodes)))
            file.write("dense3_activation: {}\n".format(str(model_object.dense3_activation)))
            if(model_type=="Cascade"):
                file.write("final_dropout: {}\n".format(str(model_object.final_dropout)))
                file.write("dense_dropout: {}\n".format(str(model_object.dense_dropout)))
                
            
def create_info_test_file(experiment_number,model_type):
    filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/info_test_model"+str(experiment_number)+".txt"
    with open(filename, "w") as file:
        file.write("")    
        
def append_individual_test(experiment_number,epoch_number,subject,testing_accuracy,model_type):
    filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/info_test_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("\nEpoch {0},test for subject '{1}',testing_accuracy:{2:.2f}".format(epoch_number,subject,testing_accuracy))

def append_average_test(experiment_number,epoch_number,testing_accuracy,model_type):
    filename = "Experiments/"+model_type+"/Experiment"+str(experiment_number)+"/info_test_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("\nEpoch {0},average_testing_accuracy:{1:.2f}\n\n".format(epoch_number,testing_accuracy))


training_file_dir = "Data/train"
all_train_files = [f for f in listdir(training_file_dir) if isfile(join(training_file_dir, f))]
train_files_dirs = []
for i in range(len(all_train_files)):
    train_files_dirs.append(training_file_dir+'/'+all_train_files[i])
rest_list, mem_list, math_list, motor_list = utils.separate_list(train_files_dirs)
train_files_dirs = utils.orderer_shuffling(rest_list, mem_list, math_list, motor_list)


validation_file_dir = "Data/validate"
all_validate_files = [f for f in listdir(validation_file_dir) if isfile(join(validation_file_dir, f))]
validate_files_dirs = []
for i in range(len(all_validate_files)):
    validate_files_dirs.append(validation_file_dir+'/'+all_validate_files[i])
rest_list, mem_list, math_list, motor_list = utils.separate_list(validate_files_dirs)
validate_files_dirs = utils.orderer_shuffling(rest_list, mem_list, math_list, motor_list)


test_file_dir = "Data/test"
all_test_files = [f for f in listdir(test_file_dir) if isfile(join(test_file_dir, f))]
test_files_dirs = []
for i in range(len(all_test_files)):
    test_files_dirs.append(test_file_dir+'/'+all_test_files[i])
rest_list, mem_list, math_list, motor_list = utils.separate_list(test_files_dirs)
test_files_dirs = utils.orderer_shuffling(rest_list, mem_list, math_list, motor_list)