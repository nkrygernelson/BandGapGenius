import numpy as np
import pymatgen
import pymatgen.core
import pymatgen.core.composition
import mendeleev

def get_coeffs(formula):
    coeffs = []
    element_dict = pymatgen.core.composition.Composition(
        formula).get_el_amt_dict()
    for element in pymatgen.core.composition.Composition(formula).elements:
        coeffs.append(element_dict[str(element)])
    return coeffs


def stoichiometric_attributes(formula):

    coeffs = get_coeffs(formula)

    def Lp_norm(coeffs, p):
        return sum([abs(x/sum(coeffs))**p for x in coeffs])**(1/p)
    norms = [Lp_norm(coeffs, p) for p in [2, 3, 5, 7]]
    return norms


def fractional_weighted_average(formula, property):
    # property is a list of values for each element
    weights = [float(i/sum(get_coeffs(formula))) for i in get_coeffs(formula)]
    property = [float(i) for i in property]
    # element wise multiplication of the weights and the property
    return np.sum([i*j for i, j in zip(weights, property)])


def fractional_weighted_deviation(formula, property):
    # property is a list of values for each element
    weights = np.array([i/sum(get_coeffs(formula))
                       for i in get_coeffs(formula)])
    property = np.array(property)
    # element wise multiplication of the weights and the property
    avg = fractional_weighted_average(formula, property)
    property = np.array([np.sqrt((i-avg)**2) for i in property])
    return np.sum([i*j for i, j in zip(weights, property)])


def elemental_attributes(formula):
    """
    Calculates various elemental attributes based on the given chemical formula.

    Args:
        formula (str): The chemical formula of the compound.

    Returns:
        tuple: A tuple containing the calculated elemental attributes:
        fractional_weighted_average, fractional_weighted_deviation, min, max, range

    The elemental properties that are calculated include:
    - Atomic number
    - Row
    - d valence electrons
    - unfilled d states
    - magnetic moment of 0k ground state x
    - Menedeleev number
    - covalent radius
    - f valence electrons
    - unfilled f states
    - space group number at 0k ground state x
    - atomic weight
    - electronegativity
    - total number of valence electrons
    - total unfilled states
    - melting temperature
    - s valence electrons
    - unfilled s states
    - specific volume of 0k ground state
    - column
    - p valence electrons
    - unfilled p states
    - band gap energy of of 0k ground state
    """

    def get_electronic_config(element):
        electron_dict = {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6, "4s": 2, "3d": 10, "4p": 6, "5s": 2,
                         "4d": 10, "5p": 6, "6s": 2, "4f": 14, "5d": 10, "6p": 6, "7s": 2, "5f": 14, "6d": 10, "7p": 6}
        aufbau_order = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p",
                        "5s", "4d", "5p", "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p"]
        electrons_left = pymatgen.core.composition.Composition(
            element).elements[0].Z
        element_electron_dict = {}
        for k in aufbau_order:
            if electrons_left == 0:
                break
            if electron_dict[k] <= electrons_left:
                element_electron_dict[k] = electron_dict[k]
                electrons_left -= electron_dict[k]
            else:
                element_electron_dict[k] = electrons_left
                break
        return element_electron_dict

    def atomic_number_attributes(formula):
        atomic_numbers = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            atomic_numbers.append(element.Z)

        return fractional_weighted_average(formula, atomic_numbers), np.std(atomic_numbers), np.min(atomic_numbers), np.max(atomic_numbers), np.max(atomic_numbers)-np.min(atomic_numbers)

    def row_attributes(formula):
        rows = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            rows.append(element.row)
        return fractional_weighted_average(formula, rows), np.std(rows), np.min(rows), np.max(rows), np.max(rows)-np.min(rows)

    def column_attributes(formula):
        columns = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            columns.append(element.group)
        return fractional_weighted_average(formula, columns), np.std(columns), np.min(columns), np.max(columns), np.max(columns)-np.min(columns)

    def d_electrons_attributes(formula):
        d_electrons = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            # we must check if the element even has a d orbital before we look
            # we find the number of electrons in the partially filled d orbital
            if any(["d" in i for i in get_electronic_config(str(element)).keys()]):
                last_d = sorted([i for i in get_electronic_config(
                    str(element)).keys() if "d" in i])[-1]
                d_electrons.append(get_electronic_config(str(element))[last_d]) 
            else:
                d_electrons.append(0)
        return fractional_weighted_average(formula, d_electrons), fractional_weighted_deviation(formula, d_electrons), np.min(d_electrons), np.max(d_electrons), np.max(d_electrons)-np.min(d_electrons)

   
   
    def d_unfilled_attributes(formula):
        d_unfilled = []
        d_full = 10
        for element in pymatgen.core.composition.Composition(formula).elements:
        
            if any(["d" in i for i in get_electronic_config(str(element)).keys()]):
                last_d = sorted([i for i in get_electronic_config(
                    str(element)).keys() if "d" in i])[-1]
              
                d_unfilled.append(d_full-get_electronic_config(str(element))[last_d]) 
            else:
                d_unfilled.append(0)

        return fractional_weighted_average(formula, d_unfilled), fractional_weighted_deviation(formula, d_unfilled), np.min(d_unfilled), np.max(d_unfilled), np.max(d_unfilled)-np.min(d_unfilled)
    def orbital_unfilled_attributes(formula,orbital):
        full_dict = {"s":2,"p":6,"d":10,"f":14}
        orbital_unfilled = []
        for element in pymatgen.core.composition.Composition(formula).elements:
        
            if any([orbital in i for i in get_electronic_config(str(element)).keys()]):
                last_orbital = sorted([i for i in get_electronic_config(
                    str(element)).keys() if orbital in i])[-1]
              
                orbital_unfilled.append(full_dict[orbital]-get_electronic_config(str(element))[last_orbital]) 
            else:
                orbital_unfilled.append(0)
        return fractional_weighted_average(formula, orbital_unfilled), fractional_weighted_deviation(formula, orbital_unfilled), np.min(orbital_unfilled), np.max(orbital_unfilled), np.max(orbital_unfilled)-np.min(orbital_unfilled)
    def orbital_electrons_attributes(formula,orbital):
        orbital_electrons = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            # we must check if the element even has a d orbital before we look
            # we find the number of electrons in the partially filled d orbital
            if any(["d" in i for i in get_electronic_config(str(element)).keys()]):
                last_d = sorted([i for i in get_electronic_config(
                    str(element)).keys() if "d" in i])[-1]
                orbital_electrons.append(get_electronic_config(str(element))[last_d]) 
            else:
                orbital_electrons.append(0)
        return fractional_weighted_average(formula, orbital_electrons), fractional_weighted_deviation(formula, orbital_electrons), np.min(orbital_electrons), np.max(orbital_electrons), np.max(orbital_electrons)-np.min(orbital_electrons)
        

    def atomic_weight_attributes(formula):
        atomic_weights = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            atomic_weights.append(element.atomic_mass)
        return fractional_weighted_average(formula, atomic_weights), fractional_weighted_deviation(formula, atomic_weights), np.min(atomic_weights), np.max(atomic_weights), np.max(atomic_weights)-np.min(atomic_weights)

    def covalent_radius_attributes(formula):
        covalent_radii = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            covalent_radii.append(element.atomic_radius)
        return fractional_weighted_average(formula, covalent_radii), fractional_weighted_deviation(formula, covalent_radii), np.min(covalent_radii), np.max(covalent_radii), np.max(covalent_radii)-np.min(covalent_radii)

    def electronegativity_attributes(formula):
        electronegativities = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            electronegativities.append(element.X)
        return fractional_weighted_average(formula, electronegativities), fractional_weighted_deviation(formula, electronegativities), np.min(electronegativities), np.max(electronegativities), np.max(electronegativities)-np.min(electronegativities)
    '''
    def total_valence_attributes(formula):
        #we must compute the total number of valence electrons
        #how to do this? we create a dictionary containing the chemical composition of all the noble gases and a dictionary of the number of atoms of the noble gases.
        #we find the nearest (smaller) noble gas. we compute the electronic structure of the studied element and subtract the electronic structure of the noble gas.
        #we count the total number of electrons in the resulting configuation. This is the number of valence electrons.
        noble_gases = {"He": 2, "Ne": 10, "Ar": 18, "Kr": 36, "Xe": 54, "Rn": 86, "Og": 118}
        electronic_dict_noble_gases = {gas:get_electronic_config(gas) for gas in noble_gases.keys()}
        total_valence = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            # find nearest (smaller) noble gas
            noble_core = sorted(noble_gases.keys(), key = lambda x: noble_gases[x]-element.Z)[-1]

    '''
    def melting_point_attributes(formula):
        melting_points = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            melting_point = element.data['Melting point']
            #remove the units
            tmp = ""
            for char in melting_point:
                if char.isdigit() or char == ".":
                    tmp += char
            melting_point = float(tmp)
            melting_points.append(melting_point)
        return fractional_weighted_average(formula, melting_points), fractional_weighted_deviation(formula, melting_points), np.min(melting_points), np.max(melting_points), np.max(melting_points)-np.min(melting_points)
        
    def molar_volume_attributes(formula):
        molar_volumes = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            molar_volume = element.data['Molar volume']
            #remove the units
            molar_volume = float(molar_volume.split()[0])
            molar_volumes.append(molar_volume)
        return fractional_weighted_average(formula, molar_volumes), fractional_weighted_deviation(formula, molar_volumes), np.min(molar_volumes), np.max(molar_volumes), np.max(molar_volumes)-np.min(molar_volumes)
    def mendeleev_number_attributes(formula):
        mendeleev_numbers = []
        for element in pymatgen.core.composition.Composition(formula).elements:
            mendeleev_numbers.append(element.data["Mendeleev no"])
        return fractional_weighted_average(formula, mendeleev_numbers), fractional_weighted_deviation(formula, mendeleev_numbers), np.min(mendeleev_numbers), np.max(mendeleev_numbers), np.max(mendeleev_numbers)-np.min(mendeleev_numbers)
               
    #put it all together
    attributes = []
    attributes.extend(atomic_number_attributes(formula))
    attributes.extend(row_attributes(formula))
    attributes.extend(column_attributes(formula))
    attributes.extend(d_electrons_attributes(formula))
    attributes.extend(d_unfilled_attributes(formula))
    attributes.extend(orbital_unfilled_attributes(formula,"s"))
    attributes.extend(orbital_unfilled_attributes(formula,"p"))
    attributes.extend(orbital_unfilled_attributes(formula,"d"))
    attributes.extend(orbital_unfilled_attributes(formula,"f"))
    attributes.extend(orbital_electrons_attributes(formula,"s"))
    attributes.extend(orbital_electrons_attributes(formula,"p"))
    attributes.extend(orbital_electrons_attributes(formula,"d"))
    attributes.extend(orbital_electrons_attributes(formula,"f"))
    attributes.extend(atomic_weight_attributes(formula))
    attributes.extend(covalent_radius_attributes(formula))
    attributes.extend(electronegativity_attributes(formula))
    attributes.extend(melting_point_attributes(formula))
    attributes.extend(molar_volume_attributes(formula))
    attributes.extend(mendeleev_number_attributes(formula))
    return attributes




def all_attributes(formula):
    return stoichiometric_attributes(formula) + elemental_attributes(formula)

