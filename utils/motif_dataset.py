import torch
from typing import Callable, Optional, Any
from torch_geometric.data import InMemoryDataset, Data
from utils.utils import get_mol
from rdkit.Chem import rdmolops
# from utils.tu2smiles import EDGE
class MotifDataset(InMemoryDataset):
    def __init__(self, root, data_smiles=None, transform = None, pre_transform = None, pre_filter = None, log = True):
        self.data_smiles = data_smiles
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    @property
    def raw_file_names(self):
        return ["raw_motif_data.pt"]
    
    def process(self):
        data_list = []

        for smiles in self.data_smiles:
            mol = get_mol(smiles)
            rdmolops.AssignStereochemistry(mol)
             # Extract atom-level features
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(atom.GetAtomicNum())

             # Extract bond-level features
            bond_features = []
            for bond in mol.GetBonds():
                bond_type = bond.GetBondTypeAsDouble()
                if bond_type == 1.0:
                    bond_feat = 2
                elif bond_type == 1.5:
                    bond_feat = 0
                elif bond_type == 2.0:
                    bond_feat = 3
                elif bond_type == 3.0:
                    bond_feat = 4

                bond_features.append(bond_feat)
            
            x = torch.tensor(atom_features, dtype=torch.long)  # Node feature matrix
            edge_index = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()], dtype=torch.long).t().contiguous()  # Edge connectivity
            edge_attr = torch.tensor(bond_features, dtype=torch.long)  # Edge feature matrix
            # print(x.size())
            # print(edge_attr.size())
            # print(stop)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # data_list += raw_data
        
        print(f"Number of motif data: {len(data_list)}")


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
