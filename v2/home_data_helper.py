import pymatgen.core.composition
import os
import pandas as pd

# for every line in home_data.csv divide the FE by the number of atoms in the formula
def calculate_FE_per_atom():
    PATH = os.getcwd()
    data_path = PATH+"/v2/data/home_data.csv"
    df = pd.read_csv(data_path)
    df['FE'] = df['FE']/df['formula'].apply(lambda x: pymatgen.core.composition.Composition(x).num_atoms)
    df.to_csv(data_path, index=False)
    return df
def ternary_fraction_FE(element):
    '''
    For every line in home_data.csv, divide the FE by the number of atoms in the formula
    But also split up the formula into its elements and calculate the fraction of each element in the formula
    '''
    PATH = os.getcwd()
    data_path = PATH+"/v2/data/home_data.csv"
    df = pd.read_csv(data_path)
    df['FE'] = df['FE']/df['formula'].apply(lambda x: pymatgen.core.composition.Composition(x).num_atoms)
    df[element] = df['formula'].apply(lambda x: pymatgen.core.composition.Composition(x).get(element)/pymatgen.core.composition.Composition(x).num_atoms)
    df.to_csv(data_path, index=False)
    return df
calculate_FE_per_atom()


