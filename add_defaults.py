import pandas as pd


def main():
    df = pd.read_csv("data/df_meam_params.csv",index_col = 0)
    df_defaults = pd.read_csv("data/df_meam_defaults.csv")
    df_defaults = df_defaults.set_index("parameter")
    for param in df_defaults.index.to_list():
        default_value = df_defaults.loc[param].default
        print(f"replacing {param} with {default_value}")
        df[param] = df[param].fillna(default_value)
        print("done")
    df.to_csv("data/df_meam_params.csv")
    return 

if __name__ == "__main__":
    main()