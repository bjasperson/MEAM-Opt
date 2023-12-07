#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import datetime
from predNN import PredNet


plt.rcParams['figure.dpi'] = 150


class OptNet(nn.Module):
    """
    Topology neural network. Neural network will take in x,y location
    Returns two images with shape/normalized thickness as pixel values
    (pixel_shape = Num_layers, H_pixels, W_pixels)
    """
    
    def __init__(self, n_given_params, n_fit_params):
        """
        """
        super(OptNet, self).__init__() 
        
        # self.bn0 = nn.BatchNorm1d(n_given_params)
        # self.bn1 = nn.BatchNorm1d(20)
        # self.bn2 = nn.BatchNorm1d(20)
        # self.bn3 = nn.BatchNorm1d(20)
        # self.bn4 = nn.BatchNorm1d(20)
        # self.bn5 = nn.BatchNorm1d(20)
        
        self.fc1 = nn.Linear(n_given_params, 20)
        nn.init.xavier_normal_(self.fc1.weight) #xavier_normal_from TOuNN.py
        self.fc2 = nn.Linear(20, 20)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(20, 20)
        nn.init.xavier_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(20, 20)
        nn.init.xavier_normal_(self.fc4.weight)
        self.fc5 = nn.Linear(20, n_fit_params)  # use with softmax option
        nn.init.xavier_normal_(self.fc5.weight)
                
        self.l_relu1 = nn.LeakyReLU(0.01) #was 0.1
        self.l_relu2 = nn.LeakyReLU(0.01)
        self.l_relu3 = nn.LeakyReLU(0.01)
        self.l_relu4 = nn.LeakyReLU(0.01)
        self.l_relu5 = nn.LeakyReLU(0.01)

    def forward(self, x):        
        #option w/o dropout
        # x = self.bn0(x)
        x = self.l_relu1(self.fc1(x))
        # x = self.bn1(x)
        x = self.l_relu2(self.fc2(x))
        # x = self.bn2(x)
        x = self.l_relu3(self.fc3(x))
        # x = self.bn3(x)
        x = self.l_relu4(self.fc4(x))
        # x = self.bn4(x)
        x = self.fc5(x)

        return x

class Optimization():
    """
    Parameter optimization. Executes optimization using 
    pre-trained performance network and OptNet neural network.
    """
    def __init__(self, predNN_dict, given_params, fit_params, learning_rate, device):
        """initialize TopOpt
        """
        self.perfnn = predNN_dict['predNN']
        self.pipe = predNN_dict['pipe']
        self.label_scaler = predNN_dict['label_scaler']
        self.given_params = given_params
        self.fit_params = fit_params
        self.learning_rate = learning_rate
        self.device = device
        
        torch.set_grad_enabled(True)        
        
        #lock down perfnn
        self.perfnn.requires_grad = False
        for param in self.perfnn.parameters():
            param.requires_grad = False
        
        #initialize OptNet
        self.opt_net = OptNet(len(given_params), len(fit_params))
              
        self.optimizer = optim.Adam(
            self.opt_net.parameters(), amsgrad=True, lr=self.learning_rate)#, weight_decay=1e-5)


    def set_inputs(self, species):
        # give me a species string, 
        # import the raw inputs from csv
        # normlize with scaler
        df_opt_inputs = pd.read_csv("./data/opt_inputs.csv")
        df_opt_inputs = df_opt_inputs.set_index('species')
        scaled_inputs = self.pipe.transform(df_opt_inputs)[0][:len(self.given_params)]
        scaled_inputs = scaled_inputs.reshape(1,-1)
        # need to normalize using scalar, then set
        self.input_values = torch.tensor(scaled_inputs.astype('float32'))

        return
    
    def set_targets(self, species):
        df_opt_targets = pd.read_csv("./data/opt_targets.csv")
        df_opt_targets = df_opt_targets.set_index('species')
        self.target_prop_names = df_opt_targets.columns.tolist()
        df_opt_targets = df_opt_targets.loc[species].tolist()
        self.target_prop_values = df_opt_targets
        scaled_target_prop_values = self.label_scaler.transform([df_opt_targets])[0]
        self.scaled_target_prop_values = torch.tensor(scaled_target_prop_values.astype('float32'))
        return

    def pretrain(self, initial_density, num_epochs):
        """pretrain top_opt to output given, uniform density
        """
        self.perfnn.eval()  #set perf_net to eval mode
        self.opt_net.train() #opt_net is training (influences dropout)
        
        #make target rho array
        C,H,W = self.image_shape
        target = self.opt_net.weightx*float(initial_density)*torch.ones(1,int(H),int(W)) #density funct is 1 channel only
        target = target[None]
        
        for i in range(num_epochs):
            images = self.opt_net(self.input_xy, 1,symmetric=True)
            images = images[None]
            loss = ((target-images)**2).sum()
            
            if i % 100 == 0:
                print("pretrain loss: ",loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.plot_images(images, 'pretrained image')

    def output_line_fn(self,i):
        output_line = [i]
        pred_fit_params = self.opt_net(self.input_values)
        new_params = torch.cat((self.input_values, pred_fit_params),axis=1)
        pred_matl_prop = self.perfnn(new_params)
        pred_matl_prop = self.label_scaler.inverse_transform(pred_matl_prop.detach().numpy()).tolist()[0]
        new_params = self.pipe.named_steps.scale.inverse_transform(new_params.detach().numpy()).tolist()[0]
        output_line.extend(new_params)
        output_line.extend(pred_matl_prop)
        return output_line

   
    def optimize(self, max_epochs, output = True):
        """perform optimization/training of top_op

        """
        output_lines = []
        output_line = ["epoch"]
        output_line.extend(self.pipe.feature_names_in_.tolist())
        output_line.extend(self.target_prop_names)
        output_lines.append(output_line)

        self.perfnn.eval()  #set perf_net to eval mode
        self.opt_net.train() #opt_net is training (influences dropout)
        
        # add initial params and prediction
        output_lines.append(self.output_line_fn("initial"))

        optimize_loss = []
        error_terms = []
        loss_fn = nn.MSELoss()
        
        for i in range(max_epochs):
            # forward pass to obtain predicted images
            # no need to normalize images first b/c using "normalized" layer thicknesses
            
            pred_fit_params = self.opt_net(self.input_values)  # tensor [N_batch,2]
            #images = images[None]  #add axis
            new_params = torch.cat((self.input_values, pred_fit_params),axis=1)
            pred_matl_prop = self.perfnn(new_params)
        

            #backpropogation
            self.optimizer.zero_grad()
            loss = loss_fn(pred_matl_prop, self.scaled_target_prop_values.reshape(1,-1))
            loss.backward(retain_graph=True)
            self.optimizer.step()

            #error_terms.append(error_terms_in)
            if i % 10 == 0 or i == max_epochs-1:
                output_lines.append(self.output_line_fn(i))

            if output == True:
                if i % 100 == 0 or i == max_epochs-1:
                    print('---------------------')
                    print('epoch: ', i)
                    #print('objective loss: ', objective.tolist())
                    print('loss = ', loss.tolist())
            ##########

            optimize_loss.append(loss.tolist())
        
        output_lines_df = pd.DataFrame(output_lines[1:], columns = output_lines[0])
        output_lines_df.to_csv("./experiments/opt.csv")

        # get/show final performance
        self.opt_net.eval()
        opt_params = self.opt_net(self.input_values) 
        opt_params = torch.cat((self.input_values, opt_params),axis=1)
        self.pred_matl_prop = self.label_scaler.inverse_transform(self.perfnn(opt_params).detach().numpy()).tolist()[0]
        self.opt_prop_values_scaled = opt_params.detach().numpy()

        opt_params_rescaled = self.pipe.named_steps.scale.inverse_transform(self.opt_prop_values_scaled)
        self.opt_prop_values = opt_params_rescaled
        print(f"optimized params:\n{self.pipe.feature_names_in_} \n {opt_params_rescaled}")

        self.loss = optimize_loss
        self.error_terms = error_terms

       
    def print_predicted_performance(self): 
        print('--------')
        print('Final design pred perf:')
        
        label_names = self.target_prop_names
        pred_perf = self.opt_prop_values
        print("parameter: target : realized")
        for i in range(len(label_names)):
            print(f"{label_names[i]}: {self.target_prop_values[i]} : {self.pred_matl_prop[i]:.3f}")
        print('--------')    


def use_gpu(use_gpu):
    if use_gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    print("Using {} device".format(device))

    return device


# def plot_error(error_terms, error_labels, y_limit = None):
#     error_terms = np.array(error_terms)
#     for i in range(len(error_labels)):
#         plt.plot(error_terms[:, i], label=error_labels[i])
    
#     plt.xlabel('# of Epochs')
#     plt.ylabel('Loss')
#     plt.title('Breakdown of loss contributions')
#     plt.legend()
#     if y_limit != None:
#         plt.ylim(0, y_limit)
#     plt.show()

def plot_pred_vs_actual(y_train_pred, y_train,
                        y_test_pred, y_test,
                        property):
    plt.figure()
    plt.scatter(y_train,y_train_pred,label="train")
    plt.scatter(y_test,y_test_pred,label="test")
    plt.plot([min(y_train),max(y_train)],
            [min(y_train),max(y_train)])
    plt.xlabel('actual')
    plt.ylabel('pred')
    plt.legend()
    plt.title(property)
    plt.show()

def vis_model_weights(model,layer):
    """plot model weights
    
    ref:
    https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
    """
    
    model_weights = []
    conv_layers = []
    
    model_children = list(model.conv_layers.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) ==  nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
    
    
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[layer]):
        plt.subplot(6, 5, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()

def create_dataloader(features, labels, n_batch):    
    features_tf = torch.tensor(features.astype('float32'))
    labels_tf = torch.tensor(labels.astype('float32'))

    dataset = TensorDataset(features_tf, labels_tf)
    dataloader = DataLoader(dataset, batch_size=n_batch)
    return dataloader


def train(dataloader, model, loss_fn, optimizer, device, train_error):
    """
    """

    model.train()
    train_loss = 0
    batch_num = 0
    for batch, (X, y) in enumerate(dataloader):
        batch_num += 1
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= batch_num

    train_error.append(train_loss)

    return train_error

#########################################################
def test(dataloader, model, loss_fn, device, test_error, error_flag=False):
    """
    """

    model.eval()
    test_loss = 0
    error_calc = []
    batch_num = 0
    with torch.no_grad():
        for X, y in dataloader:
            batch_num += 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= batch_num

    test_error.append(test_loss)

    if error_flag == True:
        return(error_calc)
    else:
        return(test_error)


#########################################################
def get_all_preds(dataloader, model, device):
    #all_labels = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)
    #all_error = torch.tensor([]).to(device)

    model.eval()

    for batch in dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        #all_labels = torch.cat((all_labels, labels), dim=0)

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

        #error = 100*(preds - labels)/labels
        #all_error = torch.cat((all_error, error), dim=0)

    all_preds = all_preds.detach().numpy()
    return all_preds#, all_error, all_labels
#########################################################


def opt_funct(target_species,
              predNN_dict,
              given_params,
              fit_params,
              print_details = False,
              num_epochs = 3000,
              test_model = False
              ):
    # set True to use GPU, False to use CPU
    device = use_gpu(False)  
        
    # only use during debugging (slows code)
    torch.autograd.set_detect_anomaly(True)
    
    #initilize opt
    opt = Optimization(predNN_dict, given_params, fit_params, .001, device)
    
    if print_details == True:
        print('Trainable parameters:', sum(p.numel()
            for p in opt.opt_net.parameters() if p.requires_grad))
        
    
    #set given inputs and targets for provided species
    opt.set_inputs(target_species)
    opt.set_targets(target_species)
        
         
    opt.optimize(num_epochs,output = print_details)
    opt.print_predicted_performance()

    if print_details == True:    
        plt.plot(np.array(opt.loss))
        plt.title('Optimizing loss')
        plt.show()
    
        # plot_error(opt.error_terms, opt.error_labels)

    return opt


def import_data():
    df = pd.read_csv("data/df_meam_params.csv", index_col=0)
    #df = df.fillna("")
    df = df.set_index(["model","species"])
    # get list of meam params
    param_list = df.columns.to_list()[0:37]
    df = df.reset_index()

    return df, param_list


def main():
    # will need to bring sklearn SVR model into pytorch
    # https://pythonawesome.com/convert-scikit-learn-models-to-pytorch-modules/
    # https://github.com/unixpickle/sk2torch

    df, param_list = import_data()

    given_params = ['z','ielement','atwt','alat','ibar','rho0','re']
    fit_params = ['alpha','b0','b1','b2','b3','esub','asub',
				  't1','t2','t3','rozero','rc','delr','zbl',
				  'attrac','repuls','Cmin','Cmax','Ec']
    all_indicators = given_params + fit_params

    # property list to explore, commented out props with few data points
    matl_props = ['lattice_constant_fcc', 'bulk_modulus_fcc',
            'c44_fcc', 'c12_fcc', 'c11_fcc',
            'cohesive_energy_fcc', 
            #'thermal_expansion_coeff_fcc', # not a lot of points
            #'surface_energy_100_fcc', # not a lot of points
            #'extr_stack_fault_energy_fcc', 'intr_stack_fault_energy_fcc',
            #'unstable_stack_energy_fcc', 'unstable_twinning_energy_fcc',
            #'relaxed_formation_potential_energy_fcc',
            #'unrelaxed_formation_potential_energy_fcc',
            #'vacancy_migration_energy_fcc', 
            #'relaxation_volume_fcc'
            ]
    
    df = df.dropna(subset=matl_props)

    
    if True:
        with open('models/predNN_231208-1909.pkl','rb') as inp:
            predNN_dict = pickle.load(inp)

        opt = opt_funct('Ni',
                        predNN_dict,
                        given_params,
                        fit_params,
                        print_details = True,
                        test_model = "MEAM_LAMMPS_CostaAgrenClavaguera_2007_AlNi__MO_131642768288_002")

        # if input('save results? y to save:  ') == 'y':
        #     opt.save_results(perf_nn_folder)


if __name__ == '__main__':
    top_opt_out = main()