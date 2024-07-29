import os
import sys
import argparse
import parmed as pmd
import numpy as np
import pandas as pd
import scipy as sp
import warnings

MAX_ANGLE = 0.20
BOND_PERIOD = 10
K_STEP = 1000
MASS_STEP = 1
MASS_DIFFERENCE_THRESHOLD = 2 * MASS_STEP
MASS_ADD_THRESHOLD = 2 * MASS_STEP

def get_bonds(topology):
    """
    Generate bond information DataFrame from the provided topology.

    Args:
    topology: List of bonds in the molecular topology.

    Returns:
    bonds_dataframe: DataFrame containing bond-related information such as atom indices, bond function,
                     equilibrium bond length (req), and force constant (k).
    """
    bonds_data = []

    for bond in topology:
        ai = bond.atom1.idx + 1
        aj = bond.atom2.idx + 1
        ni = bond.atom1.name
        nj = bond.atom2.name
        mi = bond.atom1.mass  # get mass of atom1
        mj = bond.atom2.mass
        funct = bond.funct
        if bond.type is not None:
            req = bond.type.req
            k = bond.type.k
        else:
            if bond.atom1.name == "SG" and bond.atom2.name == "SG":
                req = 0.204 * 10.0
                k = 2.5e05 / (4.184 * 100 * 2)
            else:
                req = None
                k = None
                print("WARNING: there is an unparametrized bond in your reference topology: ", bond)

        bonds_data.append({"ai": ai, "aj": aj, "ni": ni, "nj": nj, "mi": mi, "mj": mj, "funct": funct, "req": req, "k": k})

    if bonds_data:
        bonds_dataframe = pd.DataFrame(bonds_data)
        bonds_dataframe["req"] = bonds_dataframe["req"] / 10.0
        bonds_dataframe["k"] = bonds_dataframe["k"] * 4.184 * 100 * 2
        bonds_dataframe["k"] = bonds_dataframe["k"].map(lambda x: "{:.6e}".format(x))
    else:
        bonds_dataframe = pd.DataFrame(columns=["ai", "aj", "funct", "req", "k"])

    return bonds_dataframe

def get_urey_bradley(topology):
    """
    Extracts Urey-Bradley data from a topology object and constructs a pandas DataFrame.
    """
    bonds_data = []

    for bond in topology:
        ai = bond.atom1.idx + 1
        aj = bond.atom2.idx + 1
        ni = bond.atom1.name
        nj = bond.atom2.name
        mi = bond.atom1.mass  # get mass of atom1
        mj = bond.atom2.mass
        # funct = bond.funct
        if bond.type is not None:
            req = bond.type.req
            k = bond.type.k
        else:
            if bond.atom1.name == "SG" and bond.atom2.name == "SG":
                req = 0.204 * 10.0
                k = 2.5e05 / (4.184 * 100 * 2)
            else:
                req = None
                k = None
                print("WARNING: there is an unparametrized bond in your reference topology: ", bond)

        bonds_data.append({"ai": ai, "aj": aj, "ni": ni, "nj": nj, "mi": mi, "mj": mj, "req": req, "k": k})

    if bonds_data:
        bonds_dataframe = pd.DataFrame(bonds_data)
        bonds_dataframe["req"] = bonds_dataframe["req"] / 10.0
        bonds_dataframe["k"] = bonds_dataframe["k"] * 4.184 * 100 * 2
        bonds_dataframe["k"] = bonds_dataframe["k"].map(lambda x: "{:.6e}".format(x))
    else:
        bonds_dataframe = pd.DataFrame(columns=["ai", "aj", "funct", "req", "k"])

    return bonds_dataframe

def get_angles(topology):
    """
    Extracts angle data from a topology object and constructs a pandas DataFrame.

    Parameters:
    - topology (parmed.Structure): The ParmEd topology object containing angle information.

    Returns:
    - pandas.DataFrame: A DataFrame containing angle data, including atom indices (ai, aj, ak),
                        function type (funct), equilibrium angles (theteq), and force constants (k).
    """
    angles_data = []

    for angle in topology:
        ai = angle.atom1.idx + 1
        aj = angle.atom2.idx + 1
        ak = angle.atom3.idx + 1
        ni = angle.atom1.name
        nj = angle.atom2.name
        nk = angle.atom3.name
        mi = angle.atom1.mass
        mj = angle.atom2.mass
        mk = angle.atom3.mass

        funct = angle.funct

        if angle.type is not None:
            theteq = angle.type.theteq
            k = angle.type.k
        else:
            if angle.atom2.name == "SG" and (angle.atom1.name == "SG" or angle.atom3.name == "SG"):
                theteq = 104.0
                k = 4.613345e02 / (4.184 * 2)  # Converting kcal/(mol*rad^2) to kJ/(mol*rad^2)
            else:
                theteq = None
                k = None
                print("WARNING: there is an unparametrized angle in your reference topology:", angle)

        angles_data.append({
            "ai": ai, "aj": aj, "ak": ak, "ni": ni, "nj": nj, "nk": nk, "mi": mi,
            "mj": mj, "mk": mk, "funct": funct, "theteq": theteq, "k": k
        })

    if angles_data:
        angles_dataframe = pd.DataFrame(angles_data)
        angles_dataframe["k"] = angles_dataframe["k"] * 4.184 * 2
        angles_dataframe["k"] = angles_dataframe["k"].map(lambda x: "{:.6e}".format(x))
    else:
        angles_dataframe = pd.DataFrame(columns=["ai", "aj", "ak", "ni", "nj", "nk", "mi", "mj", "mk", "funct", "theteq", "k"])

    return angles_dataframe


def red_mass(mi, mj):
    return mi * mj / (mi + mj)

def freq(k, m):
    return 1e12 * np.sqrt(k / m) / (2 * np.pi)

def angular_freq(k, m):
    return 1e12 * np.sqrt(k / m)


def calculate_bond_freq(bonds):
    bonds = bonds.copy()
    bonds['ai'] = bonds['ai'].astype(int)
    bonds['aj'] = bonds['aj'].astype(int)
    bonds['mi'] = bonds['mi'].astype(float)
    bonds['mj'] = bonds['mj'].astype(float)
    bonds['k'] = bonds['k'].astype(float)
    bonds['m_red'] = red_mass(bonds['mi'], bonds['mj'])
    bonds['freq'] = freq(bonds['k'], bonds['m_red'])

    return bonds

def calculate_angle_freq(angles):
    angles = angles.copy()
    angles['ai'] = angles['ai'].astype(int)
    angles['aj'] = angles['aj'].astype(int)
    angles['ak'] = angles['ak'].astype(int)
    angles['mi'] = angles['mi'].astype(float)
    angles['mj'] = angles['mj'].astype(float)
    angles['mk'] = angles['mk'].astype(float)
    angles['k'] = angles['k'].astype(float)
    angles['m_red'] = red_mass(angles['mi'], angles['mk'])
    angles['freq'] = angular_freq(angles['k'], angles['m_red'])

    return angles

def repartition_masses(h_bonds, h_urey_bradley, h_angles, masses, original_masses, dt):
    calc_mass_add_threshold = lambda o1, o2, n1, n2: (n1 + n2) - (o1 + o2) > MASS_ADD_THRESHOLD
    for i, _ in h_bonds.iterrows():
        if 1 / freq(h_bonds.at[i, 'k'], red_mass(masses[h_bonds.at[i, 'ai']], masses[h_bonds.at[i, 'aj']])) > BOND_PERIOD * dt:
            continue
        if (masses[h_bonds.at[i, 'ai']] + masses[h_bonds.at[i, 'aj']]) - (original_masses[h_bonds.at[i, 'ai']] + original_masses[h_bonds.at[i, 'aj']]) > MASS_ADD_THRESHOLD:
            h_bonds.at[i, 'k'] -= K_STEP
        else:
            if h_bonds.at[i, 'ni'].startswith('H'):
                masses[h_bonds.at[i, 'ai']] += MASS_STEP
                masses[h_bonds.at[i, 'aj']] -= MASS_STEP
                if masses[h_bonds.at[i, 'aj']] - masses[h_bonds.at[i, 'ai']] < MASS_DIFFERENCE_THRESHOLD:
                    masses[h_bonds.at[i, 'aj']] += MASS_STEP
            if h_bonds.at[i, 'nj'].startswith('H'):
                masses[h_bonds.at[i, 'aj']] += MASS_STEP
                masses[h_bonds.at[i, 'ai']] -= MASS_STEP
                if masses[h_bonds.at[i, 'ai']] - masses[h_bonds.at[i, 'aj']] < MASS_DIFFERENCE_THRESHOLD:
                    masses[h_bonds.at[i, 'ai']] += MASS_STEP

    # check for mass_add_threshold
    for i, _ in h_bonds.iterrows():
        if calc_mass_add_threshold(masses[h_bonds.at[i, 'ai']], masses[h_bonds.at[i, 'aj']], original_masses[h_bonds.at[i, 'ai']], original_masses[h_bonds.at[i, 'aj']]):
            print(f'Mass add breach for bond {i} with ai = {h_bonds.at[i, "ai"]} and aj = {h_bonds.at[i, "aj"]}...')

    for i, _ in h_urey_bradley.iterrows():
        if 1 / freq(h_urey_bradley.at[i, 'k'], red_mass(masses[h_urey_bradley.at[i, 'ai']], masses[h_urey_bradley.at[i, 'aj']])) > BOND_PERIOD * dt:
            continue
        if original_masses[h_bonds.at[i, 'ai']] - masses[h_bonds.at[i, 'ai']] > MASS_ADD_THRESHOLD or original_masses[h_bonds.at[i, 'aj']] - masses[h_bonds.at[i, 'aj']] > MASS_ADD_THRESHOLD:
            h_bonds.at[i, 'k'] -= K_STEP
        else:
            if h_urey_bradley.at[i, 'ni'].startswith('H'):
                masses[h_urey_bradley.at[i, 'ai']] += MASS_STEP
                masses[h_urey_bradley.at[i, 'aj']] -= MASS_STEP
                if masses[h_urey_bradley.at[i, 'aj']] - masses[h_urey_bradley.at[i, 'ai']] < MASS_DIFFERENCE_THRESHOLD:
                    masses[h_urey_bradley.at[i, 'aj']] += MASS_STEP
            if h_urey_bradley.at[i, 'nj'].startswith('H'):
                masses[h_urey_bradley.at[i, 'aj']] += MASS_STEP
                masses[h_urey_bradley.at[i, 'ai']] -= MASS_STEP
                if masses[h_urey_bradley.at[i, 'ai']] - masses[h_urey_bradley.at[i, 'aj']] < MASS_DIFFERENCE_THRESHOLD:
                    masses[h_urey_bradley.at[i, 'ai']] += MASS_STEP

    # check for mass_add_threshold
    for i, _ in h_urey_bradley.iterrows():
        if calc_mass_add_threshold(masses[h_urey_bradley.at[i, 'ai']], masses[h_urey_bradley.at[i, 'aj']], original_masses[h_urey_bradley.at[i, 'ai']], original_masses[h_urey_bradley.at[i, 'aj']]):
            print(f'Mass add breach for Urey-Bradley term {i} with ai = {h_urey_bradley.at[i, "ai"]} and aj = {h_urey_bradley.at[i, "aj"]}...')

    for i, _ in h_angles.iterrows():
        if angular_freq(h_angles.at[i, 'k'], red_mass(masses[h_angles.at[i, 'ai']], masses[h_angles.at[i, 'ak']])) * dt < MAX_ANGLE:
            continue
        if h_angles.at[i, 'ni'].startswith('H'):
            masses[h_angles.at[i, 'ai']] += MASS_STEP
            masses[h_angles.at[i, 'aj']] -= MASS_STEP
            if masses[h_angles.at[i, 'aj']] - masses[h_angles.at[i, 'ai']] < MASS_DIFFERENCE_THRESHOLD:
                masses[h_angles.at[i, 'aj']] += MASS_STEP
        if h_angles.at[i, 'nk'].startswith('H'):
            masses[h_angles.at[i, 'ak']] += MASS_STEP
            masses[h_angles.at[i, 'aj']] -= MASS_STEP
            if masses[h_angles.at[i, 'aj']] - masses[h_angles.at[i, 'ak']] < MASS_DIFFERENCE_THRESHOLD:
                masses[h_angles.at[i, 'aj']] += MASS_STEP

    # check for mass_add_threshold
    for i, _ in h_angles.iterrows():
        if calc_mass_add_threshold(masses[h_angles.at[i, 'ai']], masses[h_angles.at[i, 'aj']], original_masses[h_angles.at[i, 'ai']], original_masses[h_angles.at[i, 'aj']]):
            print(f'Mass add breach for angle {i} with ai = {h_angles.at[i, "ai"]} and aj = {h_angles.at[i, "aj"]}...')

    return masses, h_bonds, h_urey_bradley, h_angles

def symmetrize_masses(h_bonds, masses):
    for i, _ in h_bonds.iterrows():
        if not h_bonds.at[i, 'ni'].startswith('H'):
            # get indices of the hydrogens bonded to atom i
            a_i = h_bonds.at[i, 'ai']
            h_i = h_bonds[ (h_bonds['ai'] == h_bonds.at[i, 'ai']) | (h_bonds['aj'] == h_bonds.at[i, 'ai']) ]
            max_h_mass = np.max([masses[h_i.at[j, 'aj']] for j, _ in h_i.iterrows()])
            for j, _ in h_i.iterrows():
                while masses[h_i.at[j, 'aj']] < max_h_mass:
                    masses[h_i.at[j, 'aj']] += MASS_STEP
                    masses[a_i] -= MASS_STEP
                    if masses[h_i.at[j, 'ai']] - masses[h_i.at[j, 'aj']] < MASS_DIFFERENCE_THRESHOLD:
                        masses[h_i.at[j, 'ai']] += MASS_STEP
        if not h_bonds.at[i, 'nj'].startswith('H'):
            # get indices of the hydrogens bonded to atom j
            a_j = h_bonds.at[i, 'aj']
            h_j = h_bonds[ (h_bonds['ai'] == h_bonds.at[i, 'aj']) | (h_bonds['aj'] == h_bonds.at[i, 'aj']) ]
            max_h_mass = np.max([masses[h_j.at[j, 'aj']] for j, _ in h_j.iterrows()])
            for j, _ in h_j.iterrows():
                while masses[h_j.at[j, 'aj']] < max_h_mass:
                    masses[h_j.at[j, 'aj']] += MASS_STEP
                    masses[a_j] -= MASS_STEP
                    if masses[h_j.at[j, 'ai']] - masses[h_j.at[j, 'aj']] < MASS_DIFFERENCE_THRESHOLD:
                        masses[h_j.at[j, 'ai']] += MASS_STEP

    return masses

def synchronize_masses(df, atom_mass_pairs, masses):
    new_df = df.copy()
    for i, row in df.iterrows():
        for a, m in atom_mass_pairs:
            new_df.loc[i, m] = masses[row[a]]
            
    return new_df

def hmr_run(topol, dt):
    """
    TODO return constants in some way
    """
    print('Reading bonds...')
    bonds = get_bonds(topol.bonds)
    bonds = bonds[~bonds['ni'].str.contains('W') & ~bonds['nj'].str.contains('W')] # remove water bonds 
    bonds = calculate_bond_freq(bonds)
    h_bonds = bonds[bonds['ni'].str.startswith('H') | bonds['nj'].str.startswith('H')]
    h_bonds = h_bonds[ 1 / h_bonds['freq'] < BOND_PERIOD * dt ]

    print('Reading Urey-Bradley angles...')
    urey_bradley = get_urey_bradley(topol.urey_bradleys)
    urey_bradley = urey_bradley[~urey_bradley['ni'].str.contains('W') & ~urey_bradley['nj'].str.contains('W')] # remove water urey-bradley terms
    urey_bradley = calculate_bond_freq(urey_bradley)
    h_urey_bradley = urey_bradley[urey_bradley['ni'].str.startswith('H') | urey_bradley['nj'].str.startswith('H')]
    h_urey_bradley = h_urey_bradley[ 1 / h_urey_bradley['freq'] < BOND_PERIOD * dt ]

    print('Reading angles...')
    angles = get_angles(topol.angles)
    angles = angles[~angles['ni'].str.contains('W') & ~angles['nj'].str.contains('W') & ~angles['nk'].str.contains('W')] # remove water angles
    angles = calculate_angle_freq(angles)
    h_angles = angles[angles['ni'].str.startswith('H') | angles['nk'].str.startswith('H')]
    # h_angles, _ = check_time_step(h_angles, dt)
    h_angles = h_angles[ 1 / h_angles['freq'] >= MAX_ANGLE ]

    print('Getting h-angle masses...')
    masses = get_masses(bonds)
    original_masses = { k: v for k, v in bonds[['ai', 'mi']].drop_duplicates().values }
    original_masses = {**original_masses, **{ k: v for k, v in bonds[['aj', 'mj']].drop_duplicates().values }}
    mass_counter = original_masses.copy()
    for k in mass_counter.keys():
        mass_counter[k] = 0

    print('Begining main routine...')
    while not h_angles.empty:
        print(f'Remainig angles: {h_angles.shape[0]}', end='\r')
        for i, _ in h_angles.iterrows():
            if angular_freq(h_angles.at[i, 'k'], red_mass(masses[h_angles.at[i, 'ai']], masses[h_angles.at[i, 'ak']])) * dt < MAX_ANGLE:
                continue
            # if (masses[h_angles.at[i, 'ai']] + masses[h_angles.at[i, 'ak']]) - (original_masses[h_angles.at[i, 'ai']] + original_masses[h_angles.at[i, 'ak']]) > MASS_ADD_THRESHOLD:
            if mass_counter[h_angles.at[i, 'ai']] >= MASS_ADD_THRESHOLD or mass_counter[h_angles.at[i, 'ak']] >= MASS_ADD_THRESHOLD:
                h_angles.at[i, 'k'] -= K_STEP
            else:
                if h_angles.at[i, 'ni'].startswith('H'):
                    masses[h_angles.at[i, 'ai']] += MASS_STEP
                    masses[h_angles.at[i, 'aj']] -= MASS_STEP
                    if masses[h_angles.at[i, 'aj']] - masses[h_angles.at[i, 'ai']] < MASS_DIFFERENCE_THRESHOLD:
                        masses[h_angles.at[i, 'aj']] += MASS_STEP
                        mass_counter[h_angles.at[i, 'aj']] += MASS_STEP
                if h_angles.at[i, 'nk'].startswith('H'):
                    masses[h_angles.at[i, 'ak']] += MASS_STEP
                    masses[h_angles.at[i, 'aj']] -= MASS_STEP
                    if masses[h_angles.at[i, 'aj']] - masses[h_angles.at[i, 'ak']] < MASS_DIFFERENCE_THRESHOLD:
                        masses[h_angles.at[i, 'aj']] += MASS_STEP
                        mass_counter[h_angles.at[i, 'aj']] += MASS_STEP

        h_angles = synchronize_masses(h_angles, [('ai', 'mi'), ('aj', 'mj'), ('ak', 'mk')], masses)
        masses = symmetrize_masses(h_angles, masses)
        # for i, row in h_angles.iterrows():
        #     h_angles.loc[i, 'mi'] = masses[row['ai']]
        #     h_angles.loc[i, 'mj'] = masses[row['aj']]
        #     h_angles.loc[i, 'mk'] = masses[row['ak']]


        h_angles = calculate_angle_freq(h_angles)
        h_angles = h_angles[ 1 / h_angles['freq'] >= MAX_ANGLE ]

    # check for mass_add_threshold
    for i, _ in h_angles.iterrows():
        if (masses[h_angles.at[i, 'ai']] + masses[h_angles.at[i, 'ak']]) - (original_masses[h_angles.at[i, 'ai']] + original_masses[h_angles.at[i, 'ak']]) > MASS_ADD_THRESHOLD:
            print(f'Mass add breach for angle {i} with ai = {h_angles.at[i, "ai"]} and ak = {h_angles.at[i, "ak"]}...')

    while not h_bonds.empty:
        print(f'Remainig bonds: {h_bonds.shape[0]}', end='\r')
        # masses, h_bonds, h_urey_bradley, h_angles = repartition_masses(h_bonds, h_urey_bradley, h_angles, masses, original_masses, dt)
        for i, _ in h_bonds.iterrows():
            print(f'Doing for bond {i} with ai = {h_bonds.at[i, "ai"]} and aj = {h_bonds.at[i, "aj"]}...')
            print(f'mass is {masses[h_bonds.at[i, "ai"]]} and {masses[h_bonds.at[i, "aj"]]}')
            if 1 / freq(h_bonds.at[i, 'k'], red_mass(masses[h_bonds.at[i, 'ai']], masses[h_bonds.at[i, 'aj']])) > BOND_PERIOD * dt:
                continue
            print(f'Current masses: {masses[h_bonds.at[i, "ai"]]} and {masses[h_bonds.at[i, "aj"]]}')
            print(f'Original masses: {original_masses[h_bonds.at[i, "ai"]]} and {original_masses[h_bonds.at[i, "aj"]]}')
            print(f'Mass difference: {(masses[h_bonds.at[i, "ai"]] + masses[h_bonds.at[i, "aj"]]) - (original_masses[h_bonds.at[i, "ai"]] + original_masses[h_bonds.at[i, "aj"]])}')
            # if (masses[h_bonds.at[i, 'ai']] + masses[h_bonds.at[i, 'aj']]) - (original_masses[h_bonds.at[i, 'ai']] + original_masses[h_bonds.at[i, 'aj']]) > MASS_ADD_THRESHOLD:
            if mass_counter[h_bonds.at[i, 'ai']] >= MASS_ADD_THRESHOLD or mass_counter[h_bonds.at[i, 'aj']] >= MASS_ADD_THRESHOLD:
                print(f'Decreasing k for {i} from {h_bonds.at[i, "k"]} to {h_bonds.at[i, "k"] - K_STEP}')
                h_bonds.at[i, 'k'] -= K_STEP
            else:
                if h_bonds.at[i, 'ni'].startswith('H'):
                    print(f'Increasing mass for {h_bonds.at[i, "ai"]} and decreasing for {h_bonds.at[i, "aj"]}')
                    print(f'Current masses: {masses[h_bonds.at[i, "ai"]]} and {masses[h_bonds.at[i, "aj"]]}')
                    masses[h_bonds.at[i, 'ai']] += MASS_STEP
                    masses[h_bonds.at[i, 'aj']] -= MASS_STEP
                    print(f'New masses: {masses[h_bonds.at[i, "ai"]]} and {masses[h_bonds.at[i, "aj"]]}')
                    if masses[h_bonds.at[i, 'aj']] - masses[h_bonds.at[i, 'ai']] < MASS_DIFFERENCE_THRESHOLD:
                        print(f'Mass difference threshold breach for bond {i}... increasing mass for {h_bonds.at[i, "aj"]} by {MASS_STEP}')
                        masses[h_bonds.at[i, 'aj']] += MASS_STEP
                        mass_counter[h_bonds.at[i, 'aj']] += MASS_STEP
                if h_bonds.at[i, 'nj'].startswith('H'):
                    print(f'Increasing mass for {h_bonds.at[i, "aj"]} and decreasing for {h_bonds.at[i, "ai"]}')
                    print(f'Current masses: {masses[h_bonds.at[i, "ai"]]} and {masses[h_bonds.at[i, "aj"]]}')
                    masses[h_bonds.at[i, 'aj']] += MASS_STEP
                    masses[h_bonds.at[i, 'ai']] -= MASS_STEP
                    print(f'New masses: {masses[h_bonds.at[i, "ai"]]} and {masses[h_bonds.at[i, "aj"]]}')
                    if masses[h_bonds.at[i, 'ai']] - masses[h_bonds.at[i, 'aj']] < MASS_DIFFERENCE_THRESHOLD:
                        print(f'Mass difference threshold breach for bond {i}... increasing mass for {h_bonds.at[i, "ai"]} by {MASS_STEP}')
                        masses[h_bonds.at[i, 'ai']] += MASS_STEP
                        mass_counter[h_bonds.at[i, 'ai']] += MASS_STEP

        h_bonds = synchronize_masses(h_bonds, [('ai', 'mi'), ('aj', 'mj')], masses)
        h_bonds = calculate_bond_freq(h_bonds)
        h_bonds = h_bonds[ 1 / h_bonds['freq'] <= BOND_PERIOD * dt ]

    # check for mass_add_threshold
    for i, _ in h_bonds.iterrows():
        if (masses[h_bonds.at[i, 'ai']] + masses[h_bonds.at[i, 'aj']]) - (original_masses[h_bonds.at[i, 'ai']] + original_masses[h_bonds.at[i, 'aj']]) > MASS_ADD_THRESHOLD:
            print(f'Mass add breach for bond {i} with ai = {h_bonds.at[i, "ai"]} and aj = {h_bonds.at[i, "aj"]}...')

    while not h_urey_bradley.empty:
        print(f'Remainig Urey-Bradley terms: {h_urey_bradley.shape[0]}')
        for i, _ in h_urey_bradley.iterrows():
            print(f'Doing for Urey-Bradley term {i} with ai = {h_urey_bradley.at[i, "ai"]} and aj = {h_urey_bradley.at[i, "aj"]}...')
            print(f'mass is {masses[h_urey_bradley.at[i, "ai"]]} and {masses[h_urey_bradley.at[i, "aj"]]}')
            if 1 / freq(h_urey_bradley.at[i, 'k'], red_mass(masses[h_urey_bradley.at[i, 'ai']], masses[h_urey_bradley.at[i, 'aj']])) > BOND_PERIOD * dt:
                continue
            # if (masses[h_urey_bradley.at[i, 'ai']] + masses[h_urey_bradley.at[i, 'aj']]) - (original_masses[h_urey_bradley.at[i, 'ai']] + original_masses[h_urey_bradley.at[i, 'aj']]) > MASS_ADD_THRESHOLD:
            if mass_counter[h_urey_bradley.at[i, 'ai']] >= MASS_ADD_THRESHOLD or mass_counter[h_urey_bradley.at[i, 'aj']] >= MASS_ADD_THRESHOLD:
                print(f'Decreasing k for {i} from {h_urey_bradley.at[i, "k"]} to {h_urey_bradley.at[i, "k"] - K_STEP}')
                h_urey_bradley.at[i, 'k'] -= K_STEP
            else:
                if h_urey_bradley.at[i, 'ni'].startswith('H'):
                    masses[h_urey_bradley.at[i, 'ai']] += MASS_STEP
                    masses[h_urey_bradley.at[i, 'aj']] -= MASS_STEP
                    if masses[h_urey_bradley.at[i, 'aj']] - masses[h_urey_bradley.at[i, 'ai']] < MASS_DIFFERENCE_THRESHOLD:
                        masses[h_urey_bradley.at[i, 'aj']] += MASS_STEP
                        mass_counter[h_urey_bradley.at[i, 'aj']] += MASS_STEP
                if h_urey_bradley.at[i, 'nj'].startswith('H'):
                    masses[h_urey_bradley.at[i, 'aj']] += MASS_STEP
                    masses[h_urey_bradley.at[i, 'ai']] -= MASS_STEP
                    if masses[h_urey_bradley.at[i, 'ai']] - masses[h_urey_bradley.at[i, 'aj']] < MASS_DIFFERENCE_THRESHOLD:
                        masses[h_urey_bradley.at[i, 'ai']] += MASS_STEP
                        mass_counter[h_urey_bradley.at[i, 'ai']] += MASS_STEP
    
        h_urey_bradley = synchronize_masses(h_urey_bradley, [('ai', 'mi'), ('aj', 'mj')], masses)
        h_urey_bradley = calculate_bond_freq(h_urey_bradley)
        h_urey_bradley = h_urey_bradley[ 1 / h_urey_bradley['freq'] <= BOND_PERIOD * dt ]

    # check for mass_add_threshold
    for i, _ in h_urey_bradley.iterrows():
        if (masses[h_urey_bradley.at[i, 'ai']] + masses[h_urey_bradley.at[i, 'aj']]) - (original_masses[h_urey_bradley.at[i, 'ai']] + original_masses[h_urey_bradley.at[i, 'aj']]) > MASS_ADD_THRESHOLD:
            print(f'Mass add breach for Urey-Bradley term {i} with ai = {h_urey_bradley.at[i, "ai"]} and aj = {h_urey_bradley.at[i, "aj"]}...')
    # h_bonds = calculate_bond_freq(h_bonds)
    # h_bonds = h_bonds[ 1 / h_bonds['freq'] <= BOND_PERIOD * dt ]

    # print('Repartitioning remaining bonds...')
    # while not h_bonds.empty:
    #     print(f'Remainig bonds: {h_bonds.shape[0]}', end='\r')
    #     for i, _ in h_bonds.iterrows():
    #         if 1 / freq(h_bonds.at[i, 'k'], red_mass(masses[h_bonds.at[i, 'ai']], masses[h_bonds.at[i, 'aj']])) > BOND_PERIOD * dt:
    #             continue
    #         if (masses[h_bonds.at[i, 'ai']] + masses[h_bonds.at[i, 'aj']]) - (original_masses[h_bonds.at[i, 'ai']] + original_masses[h_bonds.at[i, 'aj']]) > MASS_ADD_THRESHOLD:
    #             h_bonds.loc[i, 'k'] -= K_STEP
    #         else:
    #             if h_bonds.at[i, 'ni'].startswith('H'):
    #                 if masses[h_bonds.at[i, 'aj']] - masses[h_bonds.at[i, 'ai']] < MASS_DIFFERENCE_THRESHOLD:
    #                     masses[h_bonds.at[i, 'aj']] += MASS_STEP
    #                 else:
    #                     masses[h_bonds.at[i, 'ai']] += MASS_STEP
    #                     masses[h_bonds.at[i, 'aj']] -= MASS_STEP
                       
    #             if h_bonds.at[i, 'nj'].startswith('H'):
    #                 if masses[h_bonds.at[i, 'ai']] - masses[h_bonds.at[i, 'aj']] < MASS_DIFFERENCE_THRESHOLD:
    #                     masses[h_bonds.at[i, 'ai']] += MASS_STEP
    #                 else:
    #                     masses[h_bonds.at[i, 'aj']] += MASS_STEP
    #                     masses[h_bonds.at[i, 'ai']] -= MASS_STEP

    #     for i, row in h_bonds.iterrows():
    #         h_bonds.loc[i, 'mi'] = masses[row['ai']]
    #         h_bonds.loc[i, 'mj'] = masses[row['aj']]

    #     h_bonds = calculate_bond_freq(h_bonds)
    #     h_bonds = h_bonds[ 1 / h_bonds['freq'] <= BOND_PERIOD * dt ]

    h_bonds = bonds[bonds['ni'].str.startswith('H') | bonds['nj'].str.startswith('H')]
    masses = symmetrize_masses(h_bonds, masses)

    non_h_bonds = bonds[~bonds['ni'].str.startswith('H') & ~bonds['nj'].str.startswith('H')]
    non_h_masses = get_masses(non_h_bonds)

    print('Repartitioning remaining non-hydrogen bonds...')
    while not non_h_bonds.empty:
        print(f'Remainig bonds: {non_h_bonds.shape[0]}')
        print(f'columns: {non_h_bonds.columns}')
        for i, row in non_h_bonds.iterrows():
            if 1 / freq(non_h_bonds.at[i, 'k'], red_mass(non_h_masses[non_h_bonds.at[i, 'ai']], non_h_masses[non_h_bonds.at[i, 'aj']])) > BOND_PERIOD * dt:
                continue
            # if (non_h_masses[non_h_bonds.at[i, 'ai']] + non_h_masses[non_h_bonds.at[i, 'aj']]) - (original_masses[non_h_bonds.at[i, 'ai']] + original_masses[non_h_bonds.at[i, 'aj']]) > MASS_ADD_THRESHOLD:
            if mass_counter[non_h_bonds.at[i, 'ai']] >= MASS_ADD_THRESHOLD or mass_counter[non_h_bonds.at[i, 'aj']] >= MASS_ADD_THRESHOLD:
                non_h_bonds.at[i, 'k'] -= K_STEP
            else:
                non_h_masses[non_h_bonds.at[i, 'ai']] += 0.5 * MASS_STEP
                non_h_masses[non_h_bonds.at[i, 'aj']] += 0.5 * MASS_STEP
                mass_counter[non_h_bonds.at[i, 'ai']] += 0.5 * MASS_STEP
                mass_counter[non_h_bonds.at[i, 'aj']] += 0.5 * MASS_STEP

        for i, row in non_h_bonds.iterrows():
            non_h_bonds.loc[i, 'mi'] = non_h_masses[row['ai']]
            non_h_bonds.loc[i, 'mj'] = non_h_masses[row['aj']]
        non_h_bonds = calculate_bond_freq(non_h_bonds)
        non_h_bonds = non_h_bonds[ 1 / non_h_bonds['freq'] <= BOND_PERIOD * dt ]

    masses = {**masses, **non_h_masses}

    return masses

def get_masses(topology_df):
    masses = {}
    for i, row in topology_df.iterrows():
        masses[row['ai']] = row['mi']
        masses[row['aj']] = row['mj']
        if 'mk' in topology_df.columns: masses[row['ak']] = row['mk']

    return masses

def write_topol(topol, new_masses, output):
    for i, atom in enumerate(topol.atoms):
        if atom.idx+1 in new_masses: 
            topol.atoms[i].mass = new_masses[atom.idx+1]

    if os.path.isfile(output): os.remove(output)
    topol.save(output)

def get_protein_total_mass(topology):
    return np.sum([atom.mass for atom in topology.atoms if 'W' not in atom.name])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input topology file", type=str)
    parser.add_argument("-o", "--output", help="Output topology path", type=str)
    parser.add_argument("-t", "--time_step", help="Time step in fs", type=float)
    args = parser.parse_args()
    args.time_step = args.time_step * 1e-15
    if not args.output.endswith('.top'):
        args.output += '.top'

    with warnings.catch_warnings():
        print(f'Loading topology from {args.input}')
        warnings.simplefilter("ignore")
        topology = pmd.load_file(args.input)

    print('Starting HMR...')
    mass_dict = hmr_run(topology, args.time_step)
    print(f'Total mass of the protein: {get_protein_total_mass(topology)}')
    write_topol(topology, mass_dict, args.output)

    print(f'Total mass of the protein after HMR: {get_protein_total_mass(topology)}')
