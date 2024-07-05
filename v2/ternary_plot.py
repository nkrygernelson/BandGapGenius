import os
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, PDEntry
import plotly.express as px
import pandas as pd

PATH = os.getcwd()
elements = ["Ag","Al","Au","B","Ba","Bi","Ca","Cd","Co","Cr","Cs","Cu","Fe","Ga","Ge","K","Hf","Hg","In","Ir","La","Li","Mg","Mn","Mo","Na","Nb","Ni","Os","Pb","Pd","Pt","Rb","Re","Rh","Ru","Sb","Sc","Sr","Sn","Ta","Tc","Ti","Tl","W","Y","Zr","Zn"]
for element in elements:
    Fl = []
    #import ternary data as csv
    ternary_data = pd.read_csv(PATH+f'/v2/data/ternary/ternary_data_{element}.csv')
    #we want to convert the data into a list of PDEntry objects where the first entry is the composition and the second entry is the energy
    el_list = ternary_data.to_numpy()
    print(el_list)
    for i in el_list:
        print(Composition(i[0]))
        Fl.append(PDEntry(i[0], i[1]*Composition(i[0]).num_atoms))

    PD = PhaseDiagram(Fl)
    print(PD)
    plotter = PDPlotter(PD,ternary_style="3d")
    plotter.get_plot(ordering=[element,"P","S"]).write_html(PATH+f'/v2/figures/ternary_plot_{element}.html')
