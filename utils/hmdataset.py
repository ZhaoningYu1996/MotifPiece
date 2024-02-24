from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset, Data
import os
import urllib.request
import torch
from torch.utils.data import Dataset
from utils.tu2smiles import to_smiles, convert_data
from utils.utils import sanitize_smiles, get_mol, sanitize
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
from rdkit.Chem import rdmolops
from utils.splitter import scaffold_split

###### To-Do: Make it create new vocabulary based on smiles representation.

class HeterTUDataset(InMemoryDataset):
    # def __init__(self, root, name) -> object:
    def __init__(self, root, name, threshold=None, score_method=None, merge_method=None, decomposition_method=None, extract_set=None, transform=None, pre_transform=None, pre_filter=None):
        # super().__init__(root, transform, pre_transform, pre_filter)
        self.name = name
        self.score_method = score_method
        self.merge_method = merge_method
        self.decomposition_method = decomposition_method
        self.extract_set = extract_set
        self.threshold = threshold
        self.motif_vocab = {}
        self.cliques_edge = {}
        self.check = {}
        if self.name in ["PTC_MR", "PTC_FR", "PTC_MM", "PTC_FM"]:
            self.dataset = TUDataset("dataset/", self.name)
            self.dataset_list = [TUDataset("dataset/", x) for x in ["PTC_MR", "PTC_FR", "PTC_MM", "PTC_FM"]]
            self.name_list = ["PTC_MR", "PTC_FR", "PTC_MM", "PTC_FM"]
            self.data_type = "PTC"
        elif self.name in ["COX2_MD", "BZR_MD", "DHFR_MD", "ER_MD", "Mutagenicity"]:
            self.dataset = TUDataset("dataset/", self.name)
            self.data_type = "TUData"
        else:
            smiles_list = []
            self.raw_dataset = MoleculeNet('dataset/', self.name)
            labels = []
            for data in self.raw_dataset:
                smiles_list.append(data.smiles)
                labels.append(data.y.squeeze().tolist())
            labels = pd.DataFrame(labels)
            labels = labels.replace(0, -1)
            labels = labels.fillna(0).values.tolist()
            if len(self.raw_dataset) != len(smiles_list):
                print('Wrong raw data mapping!')
            self.dataset = tuple([smiles_list, labels])
            self.data_type = "MolNet"

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['heter_data.pt']
    
    @property
    def raw_file_names(self):
        return ["smiles.csv"]
    
    def process(self):
        # heter_edge_attr = torch.empty((0,))
        if self.data_type == "PTC":
            all_smiles_list = []
            smiles_list = []
            label_list = []
            graph_indices = []
            for i, dataset in enumerate(self.dataset_list):
                for j, data in enumerate(dataset):
                    smiles = to_smiles(data, True, self.name_list[i])
                    smiles = sanitize_smiles(smiles)
                    if smiles == None:
                        continue
                    else:
                        all_smiles_list.append(smiles)
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
                        label_list.append(label.item())
                    graph_indices.append(i)
            
            label_list = torch.tensor(label_list).unsqueeze(1)
            print(f"Number of all smiles: {len(all_smiles_list)}")
            motifpiece = MotifPiece(all_smiles_list, train_indices=None, label_list=label_list, vocab_path="motif_piece/"+self.name+"/"+str(self.threshold)+"/"+self.merge_method+"/"+self.score_method+"/"+self.extract_set+"/", threshold=self.threshold, score_method=self.score_method, merge_method=self.merge_method, extract_set=self.extract_set)
        elif self.data_type == "TUData":
            all_smiles_list = []
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
                        label_list.append(label.item())
                    graph_indices.append(i)
            
            label_list = torch.tensor(label_list).unsqueeze(1)
            motifpiece = MotifPiece(smiles_list, "motif_piece/"+self.name+"/"+str(self.threshold)+"/", threshold=self.threshold, method=self.method)

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
            label_list = torch.tensor(label_list)
            graph_labels = torch.clone(label_list).detach()
            train_indices, validate_indices, test_indices = scaffold_split(smiles_list)
            train_indices = set(train_indices)
            print(f"number of training data: {len(train_indices)}")
            motifpiece = MotifPiece(smiles_list, train_indices, graph_labels, "motif_piece/"+self.name+"/"+str(self.threshold)+"/"+self.merge_method+"/"+self.score_method+"/"+self.extract_set+"/", threshold=self.threshold, score_method=self.score_method, merge_method=self.merge_method, extract_set=self.extract_set)
        
        heter_edge_list = []

        ## Use self inference method
        if self.decomposition_method == "self_decomposition":
            motifs_tuple = motifpiece.self_inference()
            all_atom_num = []
            for i, (motif_smiles_list, edge_list) in enumerate(zip(*motifs_tuple)):
                atom_num = []
                for motif in motif_smiles_list:
                    mol = get_mol(motif)
                    rdmolops.AssignStereochemistry(mol)
                    if mol.GetNumAtoms() > 1:
                        atom_num.append(mol.GetNumAtoms())
                    if motif not in self.motif_vocab:
                        self.motif_vocab[motif] = len(self.motif_vocab)
                mean = (sum(atom_num)+1) / (len(atom_num)+1)
                variance = (sum([((x - mean) ** 2) for x in atom_num])+1) / (len(atom_num)+1)
                res = variance ** 0.5
                all_atom_num.append(mean)
            mean = sum(all_atom_num)/len(all_atom_num)
            variance = sum([((x - mean) ** 2) for x in all_atom_num]) / len(all_atom_num)
            res = variance ** 0.5
            print(f"The average of atom num is {mean}, the standard deviation is {res}.")
            x = torch.eye(len(self.motif_vocab))

            for i, (motif_smiles_list, edge_list) in enumerate(zip(*motifs_tuple)):
                new_x = torch.zeros(len(self.motif_vocab))
                for motif in motif_smiles_list:
                    index = self.motif_vocab[motif]
                    new_x[index] = 1
                    heter_edge_list.append((index, i+len(self.motif_vocab)))
                    heter_edge_list.append((i+len(self.motif_vocab), index))

                x = torch.cat((x, new_x.unsqueeze(dim=0)), dim=0)
                for edge in edge_list:
                    heter_edge_list.append((self.motif_vocab[motif_smiles_list[edge[0]]], self.motif_vocab[motif_smiles_list[edge[1]]]))
                    heter_edge_list.append((self.motif_vocab[motif_smiles_list[edge[1]]], self.motif_vocab[motif_smiles_list[edge[0]]]))

        elif self.decomposition_method == "decomposition":
            
            ### Use inference method
            all_atom_num_train = []
            all_atom_num_valid = []
            all_atom_num_test = []
            train_motif = []
            valid_motif = []
            test_motif = []
            average_motif = 0
            for i, smiles in tqdm(enumerate(smiles_list)):
                # motif_smiles_list, edge_list = motifpiece.inference(smiles)
                motif_smiles_list, edge_list = motifpiece.decomposition(smiles, merge_method=self.merge_method)
                # print(motif_smiles_list)
                # print(stop)
                average_motif += len(motif_smiles_list)
                atom_num = []
                for motif in motif_smiles_list:
                    mol = get_mol(motif)
                    rdmolops.AssignStereochemistry(mol)
                    if mol.GetNumAtoms() > 1:
                        atom_num.append(mol.GetNumAtoms())
                    if motif not in self.motif_vocab:
                        self.motif_vocab[motif] = len(self.motif_vocab)
                mean = (sum(atom_num)+1) / (len(atom_num)+1)
                variance = (sum([((x - mean) ** 2) for x in atom_num])+1) / (len(atom_num)+1)
                res = variance ** 0.5
                # if i in train_indices:
                #     all_atom_num_train.append(mean)
                #     train_motif.extend(motif_smiles_list)
                # elif i in validate_indices:
                #     all_atom_num_valid.append(mean)
                #     valid_motif.extend(motif_smiles_list)
                # elif i in test_indices:
                #     all_atom_num_test.append(mean)
                #     test_motif.extend(motif_smiles_list)
            # average_motif /= len(smiles_list)


            # print(f"Average number of motifs: {average_motif}")
            # print("train motifs")
            # print(len(train_motif))
            # print(list(set(train_motif)))
            # print(len(list(set(train_motif))))
            # print("valid motifs")
            # print(len(valid_motif))
            # print(list(set(valid_motif)))
            # print(len(list(set(valid_motif))))
            # print("test motifs")
            # print(len(test_motif))
            # print(list(set(test_motif)))
            # print(len(list(set(test_motif))))
            
            # train_mean = sum(all_atom_num_train)/len(all_atom_num_train)
            # train_variance = sum([((x - train_mean) ** 2) for x in all_atom_num_train]) / len(all_atom_num_train)
            # train_res = train_variance ** 0.5
            # print(f"The average of atom num for train is {train_mean}, the standard deviation is {train_res}.")

            # valid_mean = sum(all_atom_num_valid)/len(all_atom_num_valid)
            # valid_variance = sum([((x - valid_mean) ** 2) for x in all_atom_num_valid]) / len(all_atom_num_valid)
            # valid_res = valid_variance ** 0.5
            # print(f"The average of atom num for valid is {valid_mean}, the standard deviation is {valid_res}.")

            # test_mean = sum(all_atom_num_test)/len(all_atom_num_test)
            # test_variance = sum([((x - test_mean) ** 2) for x in all_atom_num_test]) / len(all_atom_num_test)
            # test_res = test_variance ** 0.5
            # print(f"The average of atom num for test is {test_mean}, the standard deviation is {test_res}.")
            # print(len(all_atom_num_test))
            # # print(stop)
            
            x = torch.eye(len(self.motif_vocab))


            for i, smiles in tqdm(enumerate(smiles_list)):
                new_x = torch.zeros(len(self.motif_vocab))
                # motif_smiles_list, edge_list = motifpiece.inference(smiles)
                motif_smiles_list, edge_list = motifpiece.decomposition(smiles, merge_method=self.merge_method)
                # print(f"The id of the molecule: {i}")
                # print(f"The smiles: {smiles}")
                # print(motif_smiles_list)
                # if i == 20:
                #     print(stop)
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

        # if self.name == "clintox":
        #     self.labels = torch.empty((self.num_nodes, 2), dtype=torch.long)
        # elif self.name == "sider":
        #     self.labels = torch.empty((self.num_nodes, 27), dtype=torch.long)
        # elif self.name == "tox21":
        #     self.labels = torch.empty((self.num_nodes, 12), dtype=torch.long)
        # elif self.name == "toxcast":
        #     self.labels = torch.empty((self.num_nodes, 617), dtype=torch.long)
        # elif self.name == "muv":
        #     self.labels = torch.empty((self.num_nodes, 17), dtype=torch.long)
        # else:
        #     self.labels = torch.empty((self.num_nodes, 1), dtype=torch.long)

        # label_list = torch.tensor(label_list)

        motif_labels = torch.empty((len(self.motif_vocab), label_list.size(1)), dtype=torch.int64)
        # print(label_list[:100])
        # print(label_list.size())
        # print(stop)
        y = torch.cat((motif_labels, label_list), dim=0)
        print(f"y size: {y.size()}")

        graph_indices = torch.tensor(graph_indices)

        motif_smiles = sorted(self.motif_vocab, key=self.motif_vocab.get)
        
        heter_data = Data(x=x, edge_index=heter_edge_index, y=y, motif_smiles=motif_smiles, graph_indices=graph_indices, graph_smiles=smiles_list)
        print(f"heterogeneous graph: {heter_data}")

        data_smiles_series = pd.Series(smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False, header=False)

        torch.save(self.collate([heter_data]), self.processed_paths[0])