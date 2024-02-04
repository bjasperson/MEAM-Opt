from predNN import prep_data_and_labels, train_predNN
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_pred_vs_actual_test(y_test_pred, y_test,
                             property,show_fig = True,
                             figsize = (2.5,2.5)):
    fig,ax = plt.subplots(figsize = figsize, layout = 'constrained')
    ax.scatter(y_test,y_test_pred)
    max_value =max(np.concatenate([y_test,y_test_pred]))
    min_value = min(np.concatenate([y_test, y_test_pred]))
    ax.plot([min_value,max_value],
            [min_value,max_value])
    ax.set_xlim(.95*min_value,1.05*max_value)
    ax.set_ylim(.95*min_value,1.05*max_value)
    ax.set_xlabel('actual',fontsize=8)
    ax.set_ylabel('prediction',fontsize=8)
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

def main():

    df, all_indicators, matl_props = prep_data_and_labels()
    kf = KFold(n_splits = 5)

    df_labels = pd.read_csv("./data/label_dict.csv")
    label_dict = df_labels.to_dict(orient="records")[0]



    for i, (train_index,test_index) in enumerate(kf.split(df)):
        df_train = df.iloc[train_index,:]
        df_test = df.iloc[test_index,:]
        output = train_predNN(df,
                              all_indicators,
                              matl_props,
                              n_batch = 2,
                              num_epochs = 100,
                              save_out = False,
                              plot_out = False,
                              df_train = df_train,
                              df_test = df_test
                              )
        
        if i == 0:
            y_test = output['y_test']
            y_test_pred = output['y_test_pred']
        else:
            y_test = np.concatenate((y_test, output['y_test']))
            y_test_pred = np.concatenate((y_test_pred, output['y_test_pred']))
    
    matl_props = [label_dict[i] for i in matl_props]
    
    for i, prop in enumerate(matl_props):
        fig = plot_pred_vs_actual_test(y_test_pred[:,i], 
                                       y_test[:,i],
                                       prop,
                                       show_fig = False)
        fig.savefig(f"./experiments/cv_study/pred_cv_{prop}.eps")

    return


if __name__ == "__main__":
    main()