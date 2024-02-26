# MEAM-Opt

File Descriptions:
1. add_defaults.py: replaces missing values in df_meam_params.csv with default values for MEAM in LAMMPS. See [Model Driver Readme](https://openkim.org/files/MD_249792265679_002/README.md) for values.
2. bibfile_create.py: Create combined bibfile of all models and drivers used for research project.
3. meam_import.py: Creates dataframe of models, parameters and calculated properties (df_meam_params.csv).
4. openKimInterface.py: support functions to extract data from OpenKIM.
5. optNN.py: optimization neural network.
6. ploting.py: plotting functions for manuscript.
7. predNN.py: prediction neural network.
8. predNN_CV_study: cross-validation study of predNN performance.
