import numpy as np
import networkx as nx
import pyscf
import openfermion
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.transforms.opconversions.remove_symmetry_qubits import symmetry_conserving_bravyi_kitaev

def qham_list(qham, data=0.):
    qham_in_list = []
    for key, value in qham.terms.items():
        if len(key) == 0:
            qham_in_list.append(value.real)
            qham_in_list.append([])
        else:
            tmp = []
            qham_in_list.append(value.real)
            for i in range(len(key)):
                t = str(key[i][1]) + str(key[i][0])
                tmp.append(t)
            qham_in_list.append(tmp)
    return qham_in_list

def qham_process(qham):
    qham_out = []
    qham_len = [0, 1, 2, 3, 4]
    len_max = max([len(qham[i]) for i in range(1, len(qham), 2)])
    for k in qham_len:
        for i in range(1, len(qham), 2):
            if len(qham[i]) == k:
                qham_out.append(qham[i-1])
                qham_out.append(qham[i])
    return qham_out

def get_qham_list(mol, mapping, z2_symmetry):
    ham = get_fermion_operator(mol.get_molecular_hamiltonian())
    if z2_symmetry:
        #print('Using Z2_symmetry to reduce two qubits.')
        qham = symmetry_conserving_bravyi_kitaev(ham, 2*mol.n_electrons, mol.n_electrons)
        qham = qham_process(qham_list(qham, ham.terms[()]))
        return qham
    else:
        if mapping == 'jordan_wigner':
            qham = jordan_wigner(ham)
        elif mapping == 'bravyi_kitaev':
            qham = bravyi_kitaev(ham)
        return qham_process(qham_list(qham))
    

def new_pauli(paulis):
    new_pauli = []
    for term in paulis:
        tmp = []
        for j in range(len(term)):
            tmp_str = ''
            if term[j] in ['X', 'Y', 'Z']:
                tmp_str += term[j] + term[j+1]
                tmp.append(tmp_str)
        new_pauli.append(tmp)
    return new_pauli

def sort_pauli(old_string):
    pauli_string_ele=old_string.split()
    index=[]
    for pauli_string in pauli_string_ele:
        index.append(int(pauli_string[1]))
    largest_qubit=max(index)
    new_string=''
    first=0
    for i in range(largest_qubit+1):
        for pauli_string in pauli_string_ele:
            if (int(pauli_string[1])==i):
                if (first==0):
                    first+=1
                else:
                    new_string+=' '
                new_string+=pauli_string
    return new_string

def group_hamiltonian(rawdata, verbose=0, graphing=0):
    print('Lengt of Hamiltonian: ',len(rawdata)//2)
    # these are parameters for pauli strings
    ecore=float(rawdata[0])
    parameters=[]
    for i in range(2, len(rawdata), 2):
        parameters.append(float(rawdata[i]))
        
    # Let's extract all info
    pauli_strings=[]
    pauli_labels=[]
    for i in range(3,len(rawdata),2):
        tmp = ''
        for j in range(len(rawdata[i])):
            tmp += rawdata[i][j] + ' '
        pauli_labels.append(tmp.rstrip())
        pauli_strings.append(rawdata[i]) 
        
    # pack them up
    para_dict=dict(zip(pauli_labels,parameters))
        
    # Let's start grouping process.
    G = nx.Graph()
    
    # Set up Nodes
    G.add_nodes_from(pauli_labels)

    # Construct edges, or connections for the Nodes
    pauli_edges=[]
    for i in range(0,len(pauli_strings)-1):
        for j in range(0,i):
            edge=1
            for terms_i in pauli_strings[i]:
                for terms_j in pauli_strings[j]:
                    if (terms_i[1]==terms_j[1] and not terms_i[0]==terms_j[0]):
                        edge=0
            if (edge==1):
                pauli_edges.append([pauli_labels[i],pauli_labels[j]])
    G.add_edges_from(pauli_edges)
    
    # Prepare complement graph
    C = nx.complement(G)
    coords = nx.spring_layout(C)
    
    if (graphing):
        nx.draw(
            C,
            coords,
            labels={node: node for node in pauli_labels},
            with_labels=True,
            node_size=500,
            font_size=8,
            node_color="#9eded1",
            edge_color="#c1c1c1"
        )
        
    # Color it up!
    groups = nx.coloring.greedy_color(C, strategy="independent_set")
    num_groups = len(set(groups.values()))
    print("Minimum number of QWC groupings found:", num_groups)
    
    # (Optional) Print grouping result
    if (verbose):
        for i in range(num_groups):
            print(f"\nGroup {i}:")
            for term, group_id in groups.items():
                if group_id == i:
                    print(term)
            
    # Then we will squeeze them!
    pauli_string_new=[]
    for i in range(num_groups):
        #print(f"\nGroup {i}:")
        term_group=[]
        for term, group_id in groups.items():
            if group_id == i:
                term_group.append(term)
        #print(term_group)
        squeezed_term=term_group[0]
        for j in range(1, len(term_group)):
            #print(squeezed_term)
            for pauli in term_group[j].split():
                #print(pauli)
                if not (pauli in squeezed_term):
                    squeezed_term+=' '
                    squeezed_term+=pauli
                    squeezed_term=sort_pauli(squeezed_term)
        pauli_string_new.append(squeezed_term)
    pauli_string_new = new_pauli(pauli_string_new)
        #print(squeezed_term)
        
    return pauli_string_new, num_groups, groups, para_dict, ecore

def overlap(pauli_1, pauli_2):
    ov = 0
    for j in range(len(pauli_2)):
        if pauli_2[j] in pauli_1:
            ov += 1
        else:
            return 0
    return ov
            

def get_qham_data(qham, qham_measure):
    data = []
    qham_tmp = qham.copy()
    for i in range(len(qham_measure)):
        tmp = []
        for j in range(1, len(qham_tmp), 2):
            if len(qham_tmp[j]) > 0:
                a = overlap(qham_measure[i], qham_tmp[j])
            else:
                continue
            if a == len(qham[j]):
                tmp.append(qham_tmp[j-1])
                tmp.append(qham_tmp[j])
                qham_tmp[j-1]=0
                qham_tmp[j]=[]
        if len(tmp) > 0:
            data.append(tmp)
    return data

    
if __name__=='__main__':
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    geometry = [("H", [0.0, 0.0, 0]), ("H", [0.0, 0.0, 0.74])]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule.filename = "./" + molecule.filename.split("/")[-1]
    # Run pyscf.
    molecule = run_pyscf(molecule, run_scf=True, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=True)

    hamiltonian = get_qham_list(molecule, 'jordan_wigner', False)
    print(hamiltonian)
