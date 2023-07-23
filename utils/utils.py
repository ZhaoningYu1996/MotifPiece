import rdkit.Chem as Chem


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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True) # Add clearAromaticFlags to avoid error
    if mol is None:
        return None
    return mol

def get_smiles(mol):
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