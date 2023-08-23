import rdkit.Chem as Chem
from rdkit import RDLogger
from collections import defaultdict


def get_fragment_mol(mol, atom_indices):
    edit_mol = Chem.EditableMol(Chem.Mol())
    for idx in atom_indices:
        edit_mol.AddAtom(mol.GetAtomWithIdx(idx))
    for i, idx1 in enumerate(atom_indices):
        for idx2 in atom_indices[i + 1:]:
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond is not None:
                edit_mol.AddBond(atom_indices.index(idx1), atom_indices.index(idx2), order=bond.GetBondType())

    submol = edit_mol.GetMol()

    return submol

def get_mol(smiles):
    RDLogger.DisableLog('rdApp.*')  
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True) # Add clearAromaticFlags to avoid error
    if mol is None:
        return None
    return mol

def get_smiles(mol):
    RDLogger.DisableLog('rdApp.*')  
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def sanitize_smiles(smiles):
    try:
        mol = get_mol(smiles)
        smiles = get_smiles(mol)
    except Exception as e:
        return None
    return smiles

def get_r_b(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    edges = []
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        for i in range(len(cnei)):
            for j in range(i + 1, len(cnei)):
                c1,c2 = cnei[i],cnei[j]
                edges.append((c1,c2))
    
    s_dict = defaultdict(list)
    v_dict = defaultdict(list)
    e_dict = defaultdict(tuple)
    for i, (c1, c2) in enumerate(edges):
        e_dict[i] = (c1, c2)
        s_dict[c1].append(i)
        s_dict[c2].append(i)
    for i, clique in enumerate(cliques):
        v_dict[i] = clique
    
    return s_dict, v_dict, e_dict


