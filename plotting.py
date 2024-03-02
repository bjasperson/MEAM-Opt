import predNN
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns


def pred_plots(label_dict, timestamp):

    y_train = np.load(f"./models/{timestamp}/y_train.npy")
    y_test = np.load(f"./models/{timestamp}/y_test.npy")
    y_train_pred = np.load(f"./models/{timestamp}/y_train_pred.npy")
    y_test_pred = np.load(f"./models/{timestamp}/y_test_pred.npy")
    with open(f"./models/{timestamp}/matl_props.txt") as f:
        matl_props = f.readlines()
    matl_props = literal_eval(matl_props[0])
    matl_props = [label_dict[i] for i in matl_props]

    for i,prop in enumerate(matl_props):
        fig = predNN.plot_pred_vs_actual(y_train_pred[:,i],
            y_train[:,i],
            y_test_pred[:,i],
            y_test[:,i],
            prop, show_fig = False)
        fig.savefig(f"./figs/pred_{prop}.eps")
        plt.close()

    return


def pairplot_func(df, to_compare, filename, height = 1, tick_fontsize = 8):
    sns.reset_defaults()
    sns.set(style="ticks")#, color_codes=True)
    sns.set(font_scale=0.85)
    sns.set_style("whitegrid")
    to_compare.extend(['species'])
    fig = sns.pairplot(data = df[to_compare], corner = 'true', height = height,
                        plot_kws = dict(s=10, facecolor='b', edgecolor="b"))

    fig.tick_params(axis='x', labelrotation=90) #labelsize = tick_fontsize
    for ax in fig.axes.flatten():
        if ax is not None:
            ax.get_xaxis().set_label_coords(0.5,-0.75)
    fig.savefig(f"./figs/{filename}.eps")
    plt.close()

def generate_pairplots(df, label_dict):    
    df.columns = [label_dict[i] if i in label_dict else i for i in df.columns]

    plot_list = [['Lattice Const','re','rc'],
                 ['Bulk Mod.','Ec','b0'],
                 ['Coh. Energy','Ec','esub'],
                 ['C44','rho0','asub'],
                 ['C44','Bulk Mod.','C11'],
                 ['S.E. 100', 'Ec','esub']
                ]
    
    filenames = ['Fig2b',
                'Fig2c',
                'Fig2d',
                'Fig2e',
                'Fig2f',
                'Fig2g']

    for current,filename in zip(plot_list,filenames):
        pairplot_func(df, current, filename)

    return 

def correlation_df(df):#, label_dict):
    df_corr = df.corr(numeric_only = True).round(2)
    columns = df_corr.columns.to_list()
    #columns = [label_dict[x] for x in columns]
    df_corr.columns = columns
    
    df_index = df_corr.index.to_list()
    #df_index = [label_dict[x] for x in df_index]
    df_corr.index = df_index

    #order = df_corr['Strength MPa'].sort_values(ascending=False).index.to_list()
    #df_corr = df_corr[order].reindex(order)
    return df_corr

def corr_coeff_heatmap_plot(df_corr, prop_list):
    colors = sns.color_palette("vlag", as_cmap=True)
    plt.figure(figsize = (7,3))
    ax = sns.heatmap(df_corr.loc[prop_list][[i for i in df_corr.columns if i not in prop_list]], 
                cmap=colors,
                square = True,
                cbar = False,
                )
    fig = ax.get_figure()
    ax.grid(False)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"./figs/Fig2a.eps", bbox_inches = 'tight')#, dpi=300)


def corr_coeff_heatmap(df, label_dict):
    df = df.set_index(["model","species"])
    # get list of meam params
    param_list = df.columns.to_list()[0:37]
    df = df.reset_index()
    df.columns = [label_dict[i] if i in label_dict else i for i in df.columns]
    print(param_list)
    df.groupby('lat')['lat'].count()

    # property list to explore
    prop_list = ['Lattice Const', 'Bulk Mod.',
                'C44', 'C12', 'C11',
                'Coh. Energy','S.E. 100'
                ]

    df_corr = correlation_df(df[prop_list+param_list])
    df_corr.to_csv("experiments/prop_correlations_manuscript/corr_coeff.csv")

    for prop in prop_list:
        prop_corr = abs(df_corr[prop]).sort_values(ascending=False).index.to_list()
        param_corr = [i for i in prop_corr if i in param_list][:5]
        print(f"{prop}: {prop_corr[0:5]}")
        print(f"{prop}:{param_corr}")
    
    corr_coeff_heatmap_plot(df_corr, prop_list)
    
    return 


def main():
    df_labels = pd.read_csv("./data/label_dict.csv")
    label_dict = df_labels.to_dict(orient="records")[0]

    df = pd.read_csv("data/df_meam_params.csv", index_col=0)

    
    # run helper functions
    pred_plots(label_dict, "240203-0923")
    generate_pairplots(df, label_dict)
    corr_coeff_heatmap(df, label_dict)

if __name__ == "__main__":
    main()