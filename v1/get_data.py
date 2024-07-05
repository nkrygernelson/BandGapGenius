from mp_api.client import MPRester
import pymatgen
import os
api_key = 'GPPXFjpPCxMe6U8Uz9CQa4TEhNbaaPCY'


def write_to_text_file(string):
    with open('data/big_data_band_gap.txt', 'a') as f:
        f.write(string)
    


def get_ternary_semiconductors():
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(num_elements=3, band_gap = (0.3,3),fields=['band_gap','structure']  )
        for i in range(len(docs)):
            #temporary removal of elements containg "("
            if "(" not in str(docs[i].structure.reduced_formula):
                write_to_text_file(str(docs[i].structure.reduced_formula) + ","+ str(docs[i].band_gap) + "\n")

def refined_query():
    include_list = ["P", "S"]
    exclude_list = ["H", "O", "N", "F", "I", "Br", "Cl", "C", "Se", "Si", "As"]
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(num_elements= 3,band_gap=(0.1,6),energy_above_hull=(0,0.2), fields=['formation_energy_per_atom','structure'],  )
        for i in range(len(docs)):
            if i == 1:
                for k in docs[i]:
                    print(k)    
            elements = pymatgen.core.composition.Composition(docs[i].structure.reduced_formula).get_el_amt_dict().keys()
            #we only write the compunds that contain the elements in the include list and do not contain the elements in the exclude list
            if all(elem in elements for elem in include_list) and not any(elem in elements for elem in exclude_list):
                write_to_text_file(str(docs[i].structure.reduced_formula) + ","+ str(docs[i].formation_energy_per_atom) + "\n")
def big_net():
    exclude_list = ["H", "O", "N", "F", "I", "Br", "Cl", ]
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(num_elements= 3,band_gap=(0.1,6),energy_above_hull=(0,0.2), fields=['band_gap','structure'],  )
        for i in range(len(docs)):
            if i == 1:
                for k in docs[i]:
                    print(k)    
            elements = pymatgen.core.composition.Composition(docs[i].structure.reduced_formula).get_el_amt_dict().keys()
            #we only write the compunds that contain the elements in the include list and do not contain the elements in the exclude list
            if not any(elem in elements for elem in exclude_list):
                write_to_text_file(str(docs[i].structure.reduced_formula) + ","+ str(docs[i].band_gap) + "\n")
def list_of_combos():
    elements_list = [
    ["P", "S"],
    ["As", "S"],
    ["As", "Se"],
    ["P", "Se"],
    ["P", "Te"],
    ["Sb", "Te"]]
    exclude_list = ["H", "O", "N", "F", "I", "Br", "Cl", "C", "Si" ]
    for include_list in elements_list:
        file_name_dir = "data/data_FE_" + "_".join(include_list) + ".txt"
        print(file_name_dir)
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(num_elements= 3,band_gap=(0.1,6),energy_above_hull=(0,0.2), fields=['formation_energy_per_atom','structure'],  )
            for i in range(len(docs)):
                if i == 1:
                    for k in docs[i]:
                        print(k)    
                elements = pymatgen.core.composition.Composition(docs[i].structure.reduced_formula).get_el_amt_dict().keys()
                #we only write the compunds that contain the elements in the include list and do not contain the elements in the exclude list
                if all(elem in elements for elem in include_list) and not any(elem in elements for elem in exclude_list):
                   
                    text_to_write = str(docs[i].structure.reduced_formula) + ","+ str(docs[i].formation_energy_per_atom) + "\n"
                    print(text_to_write)
                    #if the file does not exist, create it
                    if not os.path.exists(file_name_dir):
                        with open(file_name_dir, 'w') as f:
                            f.write(text_to_write)
                    else:
                        with open(file_name_dir, 'a') as f:
                            f.write(text_to_write)



list_of_combos()  
#refined_query()
