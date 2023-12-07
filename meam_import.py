import openKimInterface
import wget
import os
import tarfile
import glob
import pandas as pd

def get_MEAM_param_file(model):
    authors = model.authors
    kim_code = model.kim_code
    species = model.model_species
    year = model.year
    path = "data/param_files/" 
    url_path = ("https://openkim.org/download/MEAM_LAMMPS_"
                f"{authors}_{year}_{species}__{kim_code}.txz"
                )
    file = wget.download(url_path , out= path)
    with tarfile.open(file) as tar:
        content = tar.extractall(f"{path}")
    os.remove(file)
    return

def get_param_files():
    """download model parameter files from OpenKIM
    """
    # get model names 
    df_models = openKimInterface.get_all_IPs().drop_duplicates()

    # filter to only EAM dynamo
    df_models = df_models[(df_models["type1"] == "MEAM") &
                          (df_models["type2"] == "LAMMPS")]

    # pull params
    for i in range(len(df_models)):
        print(f"\n{i+1} of {(len(df_models))}")
        get_MEAM_param_file(df_models.iloc[i])
    return 

def meam_library_file(file, elem):
    header = ["elt", "lat", "z", "ielement", "atwt", 
              "alpha", "b0", "b1", "b2", "b3", "alat", 
              "esub", "asub", "t0", "t1", "t2", 
              "t3", "rozero", "ibar"]
    values = []
    n_params = len(header)
    with open(file, errors = 'ignore') as f:
        lines = f.readlines()
        values = []
        for i, line in enumerate(lines):
            if len(values) < n_params and len(line.split()) > 0 and line.split()[0].strip("'") == elem:
                i_elem = i
                break
        while len(values) < n_params:
            line = lines[i_elem]
            values.extend([a.strip("'") for a in line.split()])
            i_elem+=1
    line_dict = dict(zip(header, values))
    return line_dict

def meam_elems_file(file):
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            li = line.strip()
            if len(li) > 0 and not li.startswith("#"):
                elems_list = li.split()
    return elems_list

def meam_param_file(i_elem, file, line_dict):
    param_dict = {}
    with open(file) as f:
        # print(f"file: {file}")
        lines = f.readlines()
        for line in lines:
            li = line.strip()
            if len(li) > 0 and not li.startswith("#"):
                li = li.replace("="," ").split()
                keyword_index = li[0]
                value = li[-1]
                keyword_index = keyword_index.replace("("," ").replace(")"," ").split()
                if len(keyword_index) > 1:
                    keyword = keyword_index[0]
                    index_list = keyword_index[1]
                    index_list = index_list.split(",")
                    match = True
                    for index_current in index_list:
                        if int(index_current) != i_elem:
                            match = False
                    if match == True:
                        # print("Match", keyword, index_list, value)
                        param_dict[keyword] = value
                elif len(keyword_index) == 1:
                    keyword = keyword_index[0]
                    # print("Match: ", keyword, value)
                    param_dict[keyword] = value
    line_dict.update(param_dict)
    # print(line_dict)
    return line_dict

def get_meam_param_df():
    list_of_dicts = []
    for dir_path in glob.glob("./data/param_files/*"):
        filenames = os.listdir(dir_path)
        filenames = [i for i in filenames if "meam" in i]
        if len(filenames) == 3:
            model_name = dir_path.split("/")[-1]

            # get file names
            elems_file = [f"{dir_path}/{i}" for i in filenames if "elems" in i][0]
            library_file = [f"{dir_path}/{i}" for i in filenames if "library" in i][0]
            param_file = [f"{dir_path}/{i}" for i in filenames if "library" not in i and "elems" not in i][0]

            # get elem list            
            elems_list = meam_elems_file(elems_file)

            for i_elem, elem in enumerate(elems_list):
                line_dict = meam_library_file(library_file, elem)
                line_dict["model"] = model_name
                line_dict["species"] = line_dict["elt"]
                line_dict = meam_param_file(i_elem+1, param_file, line_dict)
                list_of_dicts.append(line_dict)
    df = pd.DataFrame(list_of_dicts)
    float_cols = df.columns.drop(['elt','lat','model','species']).to_list()
    for col in float_cols:
        df[col] = df[col].astype(float)
    return df


if __name__ == "__main__":
    #list of properties to get (gather all, can remove from saved df if needed)
    openkim_props = ['lattice_const', 'bulk_modulus', 'elastic_const', 
                    'cohesive_energy', 'thermal_expansion_coeff', 'surface_energy',
                    'extr_stack_fault_energy','intr_stack_fault_energy',
                    'unstable_stack_energy','unstable_twinning_energy',
                    'monovacancy_relaxed_formation_potential_energy',
                    'monovacancy_unrelaxed_formation_potential_energy',
                    'monovacancy_vacancy_migration_energy',
                    'monovacancy_relaxation_volume'] 
    #get_param_files()
    df = get_meam_param_df()
    df = openKimInterface.get_prop_df(openkim_props, df)
    df.to_csv("./data/df_meam_params.csv")
    print('done')