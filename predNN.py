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


plt.rcParams['figure.dpi'] = 150

class PredNet(nn.Module):

    def __init__(self, n_params, n_labels):
        super().__init__()
        self.fc1 = nn.Linear(n_params, n_params)
        self.fc2 = nn.Linear(n_params, n_labels)
        self.fc3 = nn.Linear(n_labels, n_labels)
        return

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def use_gpu(use_gpu):
    if use_gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    print("Using {} device".format(device))

    return device


def plot_pred_vs_actual(y_train_pred, y_train,
                        y_test_pred, y_test,
                        property,show_fig = True,
                        figsize = (2.5,2.5)):
    fig,ax = plt.subplots(figsize = figsize, layout = 'constrained')
    ax.scatter(y_train,y_train_pred,label="train")
    ax.scatter(y_test,y_test_pred,label="test",marker="X")
    max_value =max(np.concatenate([y_train,y_test,y_train_pred, y_test_pred]))
    min_value = min(np.concatenate([y_train,y_test,y_train_pred, y_test_pred]))
    ax.plot([min_value,max_value],
            [min_value,max_value])
    ax.set_xlim(.95*min_value,1.05*max_value)
    ax.set_ylim(.95*min_value,1.05*max_value)
    ax.set_xlabel('actual',fontsize=8)
    ax.set_ylabel('prediction',fontsize=8)
    ax.legend(fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    ax.text(0.95,0.05, f"{property}",
            transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    if show_fig == True:
        plt.show()
    return fig

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


def train_predNN(df,
                 meam_params,
                 matl_props,
                 num_epochs = 300,
                 learning_rate = 0.001,
                 n_batch = 2**3,
                 save_out = True,
                 plot_out = True,
                 df_train = "",
                 df_test = ""
                 ):
    """
    """
    #######################

    # split data into train/test
    if type(df_train) is str:
        df_train, df_test = train_test_split(df, shuffle=True)
    else:
        del(df)

    # normalize inputs based on training meam params
    # might need an imputer here too for features
    imput = KNNImputer(n_neighbors=2, weights="uniform",
                    keep_empty_features=True)
    pca = PCA()
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                    ('imp', imput),
                    #('pca', pca)
                    ])
    pipe.fit(df_train[meam_params])

    label_scaler = StandardScaler()
    label_scaler.fit(df_train[matl_props])

    X_train = pipe.transform(df_train[meam_params])
    y_train = label_scaler.transform(df_train[matl_props].to_numpy())
    X_test = pipe.transform(df_test[meam_params])
    y_test = label_scaler.transform(df_test[matl_props].to_numpy())

    # create separate train and test datasets and dataloaders
    train_dataloader = create_dataloader(X_train,
                                         y_train,
                                         n_batch)
    test_dataloader = create_dataloader(X_test,
                                         y_test,
                                         n_batch)
    
    network = PredNet(len(meam_params), len(matl_props))
    use_gpu = False  # manual override for gpu option; having issues with pixel_optim_nn on gpu    
    if use_gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'
    print("Using {} device".format(device))

    # turn off/on comutation graph
    torch.set_grad_enabled(True)  # debug 220218 turned off

    # output parameters of model
    params = list(network.parameters())
    print("length of parameters = ", len(params))
    # print("conv1's weight: ",params[0].size())  # conv1's .weight
    print('Trainable parameters:', sum(p.numel()
          for p in network.parameters() if p.requires_grad))
    print('---------------------------')
    
    optimizer_in = optim.Adam(network.parameters(),
                              lr=learning_rate)  # from deeplizard
    #momentum = 0.87
    #optimizer_in = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum) #from pytorch tutorial
    loss_fn_in = nn.MSELoss()  # MSE for continuous label
    test_error = []
    train_error = []

    for t in range(num_epochs):
        train_error = train(train_dataloader, network,
                            loss_fn_in, optimizer_in, device, train_error)
        test_error = test(test_dataloader, network,
                          loss_fn_in, device, test_error, error_flag=False)
        if (t+1)%10 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Avg training loss: {train_error[-1]:>7f}")
            print(f"Avg test loss: {test_error[-1]:>8f} \n")
    print("Done!")

    ########################################################
    if plot_out == True:
        # plot test/train error
        plt.plot(np.array(test_error), label='test_error')
        plt.plot(np.array(train_error), label='train_error')
        plt.legend()
        plt.show()

    ########################################################
    # get predictions for all data and plot

    print('----Evaluate ----')
    y_train = label_scaler.inverse_transform(y_train)
    y_test = label_scaler.inverse_transform(y_test)
    
    y_train_pred = get_all_preds(train_dataloader,
                                 network,
                                 device)
    
    y_train_pred = label_scaler.inverse_transform(y_train_pred)

    y_test_pred = get_all_preds(test_dataloader,
                                network,
                                device)
    
    y_test_pred = label_scaler.inverse_transform(y_test_pred)

    if False:
        for i,p in enumerate(matl_props):
            plot_pred_vs_actual(y_train_pred[:,i],
                                y_train[:,i],
                                y_test_pred[:,i],
                                y_test[:,i],
                                p)

    print('----Evaluate done ----')

    output = {'predNN':network,
            'pipe':pipe,
            'label_scaler':label_scaler,
            'y_test':y_test,
            'y_test_pred':y_test_pred}
    ###########################################
    if save_out == True:
        if input('save NN model + stats? y to save:    ') == 'y':
            # make timestamp
            date = datetime.datetime.now()
            timestamp = (str(date.year)[-2:] + str(date.month).rjust(2, '0') +
                        str(date.day).rjust(2, '0')
                        + '-' + str(date.hour).rjust(2, '0') +
                        str(date.minute).rjust(2, '0'))
            os.mkdir(f"./models/{timestamp}")
            for i,p in enumerate(matl_props):
                fig = plot_pred_vs_actual(y_train_pred[:,i],
                            y_train[:,i],
                            y_test_pred[:,i],
                            y_test[:,i],
                            p, show_fig = False)
                fig.savefig(f"./models/{timestamp}/pred_{p}.eps", dpi=300)
                plt.close()
            torch.save(network.state_dict(), f'./models/{timestamp}/predNN_{timestamp}.pth')
            with open(f'./models/{timestamp}/predNN_{timestamp}.pkl', 'wb') as outp:
                pickle.dump(output, outp, pickle.HIGHEST_PROTOCOL)

            np.save(f"./models/{timestamp}/y_train.npy",y_train)
            np.save(f"./models/{timestamp}/y_test.npy",y_test)
            np.save(f"./models/{timestamp}/y_train_pred.npy",y_train_pred)
            np.save(f"./models/{timestamp}/y_test_pred.npy",y_test_pred)
            with open(f"./models/{timestamp}/matl_props.txt",'w') as f:
                f.write(str(matl_props))

    # if plot_out == True:
    #     evaluate.plot_results()
    #     evaluate_train.plot_results()
        
    return output


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


def import_data():
    df = pd.read_csv("data/df_meam_params.csv", index_col=0)
    #df = df.fillna("")
    # df = df.set_index(["model","species"])
    # # get list of meam params
    # param_list = df.columns.to_list()[0:37]
    # df = df.reset_index()

    return df#, param_list


def apply_one_hot(df):
    df_lat = pd.get_dummies(df['lat'])
    df = pd.concat((df_lat,df),axis=1)
    return df


def prep_data_and_labels():
    df = import_data()
    # df = apply_one_hot(df) # not sure if this is better

    df = df[df['lat'] == 'fcc'] # limit to fcc for now

    given_params = ['z','ielement','atwt','alat','ibar','rho0','re','zbl']
    fit_params = ['alpha','b0','b1','b2','b3','esub','asub',
				  't1','t2','t3','rozero','rc','delr',
				  'attrac','repuls','Cmin','Cmax','Ec',
                  'emb_lin_neg','bkgd_dyn']
    
    all_indicators = given_params + fit_params

    # property list to explore, commented out props with few data points
    matl_props = ['lattice_constant_fcc', 'bulk_modulus_fcc',
            'c44_fcc', 'c12_fcc', 'c11_fcc',
            'cohesive_energy_fcc', 
            'thermal_expansion_coeff_fcc', # not a lot of points
            'surface_energy_100_fcc', # not a lot of points
            #'extr_stack_fault_energy_fcc', 'intr_stack_fault_energy_fcc',
            #'unstable_stack_energy_fcc', 'unstable_twinning_energy_fcc',
            #'relaxed_formation_potential_energy_fcc',
            #'unrelaxed_formation_potential_energy_fcc',
            #'vacancy_migration_energy_fcc', 
            #'relaxation_volume_fcc'
            ]
    
    df = df.dropna(subset=matl_props)
    return df, all_indicators, matl_props

#########################################################
def main():
    # will need to bring sklearn SVR model into pytorch
    # https://pythonawesome.com/convert-scikit-learn-models-to-pytorch-modules/
    # https://github.com/unixpickle/sk2torch

    df, all_indicators, matl_props = prep_data_and_labels()

    pred_nn_dict = train_predNN(df, 
                        all_indicators, 
                        matl_props,
                        n_batch = 2,
                        num_epochs = 100)
    


if __name__ == '__main__':
    top_opt_out = main()