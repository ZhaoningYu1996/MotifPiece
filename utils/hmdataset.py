from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset, Data
import os
import urllib.request
import torch
from torch.utils.data import Dataset
from utils.tu2smiles import to_smiles, convert_data
from utils.utils import sanitize_smiles
# from utils.motif_dataset import MotifDataset
from utils.motifpiece import MotifPiece
import json
import pandas as pd
import csv
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import json
from torch_geometric.datasets import MoleculeNet
import os

###### To-Do: Make it create new vocabulary based on smiles representation.

class HeterTUDataset(InMemoryDataset):
    # def __init__(self, root, name) -> object:
    def __init__(self, root, name, num_nodes, transform=None, pre_transform=None, pre_filter=None):
        # super().__init__(root, transform, pre_transform, pre_filter)
        self.name = name
        self.num_nodes = num_nodes
        self.motif_vocab = {}
        self.cliques_edge = {}
        self.check = {}
        if self.name in ["PTC_MR", "Mutagenicity", "COX2_MD", "COX2", "BZR", "BZR_MD", "DHFR_MD", "MUTAG", "NCI1", "ER_MD", "PTC_FR", "PTC_MM", "PTC_FM"]:
            self.dataset = TUDataset("dataset/", self.name)
            self.data_type = "TUData"
        else:
            smiles_list = []
            self.raw_dataset = MoleculeNet('dataset/', self.name)
            labels = []
            for data in self.raw_dataset:
                smiles_list.append(data.smiles)
                # if data.y.item() == 0:
                #     print('hh')
                # print(data.y.squeeze().tolist())
                labels.append(data.y.squeeze().tolist())
            labels = pd.DataFrame(labels)
            labels = labels.replace(0, -1)
            labels = labels.fillna(0).values.tolist()
            if len(self.raw_dataset) != len(smiles_list):
                print('Wrong raw data mapping!')
            self.dataset = tuple([smiles_list, labels])
            self.data_type = "MolNet"
            
        # if method == "MotifPiece":
        #     with open("checkpoints/"+self.name+"_bridge_nomerge_vocab_id.txt", 'r') as file:
        #         self.vocab_id = json.load(file)
        #     with open("checkpoints/"+self.name+"_bridge_nomerge_inv_vocab_mapping.txt", 'r') as file:
        #         self.inv_vocab_mapping = json.load(file)
        # else:
        #     self.vocab_id = {}
        #     self.inv_vocab_mapping = {}

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['heter_data.pt']
    
    @property
    def raw_file_names(self):
        return ["smiles.csv"]

    def add_id(self, mol):
        for atom in mol.GetAtoms():
            atom.SetIntProp("OriID", atom.GetIdx())
            # print(atom.GetIntProp("OriID"))
        return mol
    
    def get_bonds(self, mol):
        bonds_list = []
        for bond in mol.GetBonds():
            b1 = bond.GetBeginAtomIdx()
            b2 = bond.GetEndAtomIdx()
            bonds_list.append(sorted(tuple([b1, b2])))
            # bonds_list.append(tuple([b2, b1]))
        return bonds_list
    
    def generate_heter(self, smiles, count):
              
        return 0
    
    def process(self):
        if self.name == "clintox":
            self.labels = torch.empty((self.num_nodes, 2), dtype=torch.long)
        elif self.name == "sider":
            self.labels = torch.empty((self.num_nodes, 27), dtype=torch.long)
        elif self.name == "tox21":
            self.labels = torch.empty((self.num_nodes, 12), dtype=torch.long)
        elif self.name == "toxcast":
            self.labels = torch.empty((self.num_nodes, 617), dtype=torch.long)
        elif self.name == "muv":
            self.labels = torch.empty((self.num_nodes, 17), dtype=torch.long)
        else:
            self.labels = torch.empty((self.num_nodes, 1), dtype=torch.long)
        heter_edge_attr = torch.empty((0,))
        if self.data_type == "TUData":

            smiles_list = []
            label_list = []
            graph_indices = []
            for i, data in enumerate(self.dataset):
                smiles = to_smiles(data, True, self.name)
                smiles = sanitize_smiles(smiles)
                if smiles == None:
                    continue
                else:
                    smiles_list.append(smiles)
                    label = data.y
                    if label.item() == -1:
                        label_list.append(0)
                    else:
                        label_list.append(1)
                    graph_indices.append(i)

        elif self.data_type == "MolNet":

            smiles_list = []
            label_list = []
            graph_indices = []
            for i, (smiles, label) in tqdm(enumerate(zip(*self.dataset))):
                smiles = sanitize_smiles(smiles)
                if smiles is None:
                    continue
                else:
                    smiles_list.append(smiles)
                    graph_indices.append(i)
                    label_list.append(label)
        
        motifpiece = MotifPiece(smiles_list, "motif_vocabulary/"+self.name+"/")

        for i, smiles in enumerate(smiles_list):
            motif_smiles_list, edge_list = motifpiece.inference(smiles)
            for motif in motif_smiles_list:
                if motif not in self.motif_vocab:
                    self.motif_vocab[motif] = len(self.motif_vocab)

        x = torch.eye(len(self.motif_vocab))
        heter_edge_list = []
        for i, smiles in enumerate(smiles_list):
            new_x = torch.zeros(len(self.motif_vocab))
            motif_smiles_list, edge_list = motifpiece.inference(smiles)
            for motif in motif_smiles_list:
                index = self.motif_vocab[motif]
                new_x[index] = 1
                heter_edge_list.append((index, i+len(self.motif_vocab)))
                heter_edge_list.append((i+len(self.motif_vocab), index))

            x = torch.cat((x, new_x.unsqueeze(dim=0)), dim=0)
            for edge in edge_list:
                heter_edge_list.append((self.motif_vocab[motif_smiles_list[edge[0]]], self.motif_vocab[motif_smiles_list[edge[1]]]))
                heter_edge_list.append((self.motif_vocab[motif_smiles_list[edge[1]]], self.motif_vocab[motif_smiles_list[edge[0]]]))
        
        heter_edge_index = torch.tensor(heter_edge_list).t().contiguous()
        heter_edge_index = torch.unique(heter_edge_index, dim=1)

        motif_label = [-1 for x in range(len(self.motif_vocab))]
        label_list.extend(motif_label)
        y = torch.tensor(label_list)

        graph_indices = torch.tensor(graph_indices)

        motif_smiles = sorted(self.motif_vocab, key=self.motif_vocab.get)
        
        heter_data = Data(x=x, edge_index=heter_edge_index, y=y, motif_smiles=motif_smiles, graph_indices=graph_indices)
        print(f"heterogeneous graph: {heter_data}")

        data_smiles_series = pd.Series(smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False,header=False)

        torch.save(self.collate([heter_data]), self.processed_paths[0])