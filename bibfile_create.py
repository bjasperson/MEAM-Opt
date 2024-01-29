import pandas as pd
from predNN import import_data

def filter_df(df):
    # repeat filtering as done for manuscript
    df = df[df['lat'] == 'fcc'] # limit to fcc for now

    # property list to explore, commented out props with few data points
    matl_props = ['lattice_constant_fcc', 'bulk_modulus_fcc',
            'c44_fcc', 'c12_fcc', 'c11_fcc',
            'cohesive_energy_fcc', 
            'thermal_expansion_coeff_fcc', 
            'surface_energy_100_fcc', 
            ]
    
    df = df.dropna(subset=matl_props)

    return df

def main():
    '''
    Create combined bibfile of all models
    and drivers used for research
    '''
    # import list of models used
    df = import_data()
    df = filter_df(df)
    models_list = df.model.drop_duplicates().to_list()

    # combine into one bib
    combined_bib = ""
    for model in models_list:
        kimcode = model.split("__")[1]
        with open(f'./data/param_files/{model}/kimcite-{kimcode}.bib','r') as file:
            lines = file.readlines()
            for line in lines:
                combined_bib += line
            lines += "\n\n"

    # save to folder
    with open(f'./data/combined_bib.bib','w') as out:
        out.write(combined_bib)


    return


if __name__ == "__main__":
    main()