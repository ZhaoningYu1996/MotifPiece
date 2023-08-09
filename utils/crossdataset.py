from torch_geometric.datasets import TUDataset, MoleculeNet
from torch_geometric.data import InMemoryDataset, Data
import os
import urllib.request
import torch
from torch.utils.data import Dataset
from utils.tu2smiles import to_smiles, convert_data
from utils.utils import sanitize_smiles
from utils.motifpiece import MotifPiece
import json
import pandas as pd
import csv
from tqdm import tqdm

class CombinedDataset(InMemoryDataset):
    def __init__(self, root, data_list, threshold=None, transform=None, pre_transform=None, pre_filter=None):
        # super().__init__(root, transform, pre_transform, pre_filter)
        self.data_names = data_list
        self.threshold = threshold
        self.smiles_list = []
        self.new_vocab = {}
        self.dataset = []
        self.labels = []
        self.motif_vocab = {}
        self.raw_dataset = []

        for name in self.data_names:
            if name in ["PTC_MR", "Mutagenicity", "COX2_MD", "COX2", "BZR", "BZR_MD", "DHFR_MD", "ER_MD", "PTC_FR", "PTC_MM", "PTC_FM"]:
                self.dataset.append(TUDataset("combined_data/", name))
                self.data_type = "TUData"
                self.dataset_list = [TUDataset("dataset/", x) for x in ["PTC_MR", "PTC_FR", "PTC_MM", "PTC_FM"]]
                self.name_list = ["PTC_MR", "PTC_FR", "PTC_MM", "PTC_FM"]
                # self.labels = torch.empty((self.num_nodes,), dtype=torch.long)
            else:
                smiles_list = []
                dataset = MoleculeNet('dataset/', name)
                self.raw_dataset.append(dataset)
                labels = []
                for data in dataset:
                    smiles_list.append(data.smiles)
                    labels.append(data.y.squeeze().tolist())
                labels = pd.DataFrame(labels)
                labels = labels.replace(0, -1)
                labels = labels.fillna(0).values.tolist()
                if len(dataset) != len(smiles_list):
                    print(f"length of raw dataset: {len(dataset)}, length of smiles: {len(smiles_list)}.")
                    print('Wrong raw data mapping!')
                    print(stop)
                self.dataset.append(tuple([smiles_list, labels]))
                self.data_type = "MolNet"
            

        
    
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # def __getitem__(self, idx):
    #     return self.dataset1[idx], self.dataset2[idx]
    
    @property
    def processed_file_names(self):
        return ['combined_data.pt']
    
    def process(self):
        
        count = 0
        
        # self.heter_node_features = torch.eye(self.num_nodes)
        # self.heter_edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.data_type == "TUData":
            whole_smiles_list = []
            whole_graph_indices = []
            motifpiece_smiles_list = []
            label_list = []
            num_graph = []
            one_smiles_list = []
            for i, dataset in enumerate(self.dataset_list):
                for j, data in enumerate(dataset):
                    smiles = to_smiles(data, True, self.name_list[i])
                    smiles = sanitize_smiles(smiles)
                    if smiles == None:
                        continue
                    else:
                        motifpiece_smiles_list.append(smiles)
            # name = "PTC"
            
            motifpiece_list = []

            for i, (dataset, name) in enumerate(zip(self.dataset, self.data_names)):
                graph_count = 0
                graph_indices = []
                smiles_list = []
                labels = []
                for j, data in enumerate(dataset):
                    smiles = to_smiles(data, True, name)
                    smiles = sanitize_smiles(smiles)
                    if smiles == None:
                        continue
                    else:
                        graph_count += 1
                        smiles_list.append(smiles)
                        label = data.y
                        if label.item() == -1:
                            labels.append(0)
                        else:
                            labels.append(label.item())
                        graph_indices.append(j)
                one_smiles_list.extend(smiles_list)
                whole_smiles_list.append(smiles_list)
                labels = torch.tensor(labels).unsqueeze(1)
                label_list.append(labels)
                num_graph.append(graph_count)
                whole_graph_indices.append(graph_indices)

                motifpiece = MotifPiece(motifpiece_smiles_list, "motif_vocabulary/"+self.data_names[i]+"/", threshold=self.threshold[i])
                motifpiece_list.append(motifpiece)

                for smiles in smiles_list:
                    motif_smiles_list, edge_list = motifpiece.inference(smiles)
                    for motif in motif_smiles_list:
                        if motif not in self.motif_vocab:
                            self.motif_vocab[motif] = len(self.motif_vocab)
                
        elif self.data_type == "MolNet": 
            whole_graph_indices = []
            label_list = []
            motifpiece_list = []
            whole_smiles_list = []
            num_graph = []
            one_smiles_list = []
            for i, (dataset, name) in enumerate(zip(self.dataset, self.data_names)):
                graph_count = 0
                graph_indices = []
                smiles_list = []
                labels = []

                for j, (smiles, label) in tqdm(enumerate(zip(*dataset))):
                    smiles = sanitize_smiles(smiles)
                    if smiles is None:
                        continue
                    else:
                        if smiles == "*":
                            continue
                        graph_count += 1
                        smiles_list.append(smiles)
                        graph_indices.append(i)
                        labels.append(label)
                
                whole_graph_indices.append(graph_indices)
                num_graph.append(graph_count)
                motifpiece = MotifPiece(smiles_list, "motif_vocabulary/"+name+"/", threshold=self.threshold[i])
                motifpiece_list.append(motifpiece)
                whole_smiles_list.append(smiles_list)
                labels = torch.tensor(labels)
                label_list.append(labels)
                
                one_smiles_list.extend(smiles_list)

                for smiles in smiles_list:
                    motif_smiles_list, edge_list = motifpiece.inference(smiles)
                    for motif in motif_smiles_list:
                        if motif not in self.motif_vocab:
                            self.motif_vocab[motif] = len(self.motif_vocab)

        x = torch.eye(len(self.motif_vocab))
        heter_edge_list = []
        id = 0

        for i, smiles_list in enumerate(whole_smiles_list):
            for j, smiles in enumerate(smiles_list):
                # print(f"the dataset id: {i}, the graph id: {j}")
                new_x = torch.zeros(len(self.motif_vocab))
                # print(f"smiles: {smiles}")
                motif_smiles_list, edge_list = motifpiece_list[i].inference(smiles)
                # print(f"motif smiles list: {motif_smiles_list}")
                for motif in motif_smiles_list:
                    index = self.motif_vocab[motif]
                    new_x[index] = 1
                    heter_edge_list.append((index, id+len(self.motif_vocab)))
                    heter_edge_list.append((id+len(self.motif_vocab), index))
                id += 1

                x = torch.cat((x, new_x.unsqueeze(dim=0)), dim=0)
                for edge in edge_list:
                    heter_edge_list.append((self.motif_vocab[motif_smiles_list[edge[0]]], self.motif_vocab[motif_smiles_list[edge[1]]]))
                    heter_edge_list.append((self.motif_vocab[motif_smiles_list[edge[1]]], self.motif_vocab[motif_smiles_list[edge[0]]]))
        heter_edge_index = torch.tensor(heter_edge_list).t().contiguous()
        heter_edge_index = torch.unique(heter_edge_index, dim=1)

        if self.data_type == "TUData":
            motif_labels = torch.empty((len(self.motif_vocab), labels.size(1)), dtype=torch.int64)
            y = torch.cat((motif_labels, label_list[0], label_list[1]), dim=0)
        if self.data_type == "MolNet": 
            y = []
            for labels in label_list:
                motif_labels = torch.empty((len(self.motif_vocab), labels.size(1)), dtype=torch.int64)
                label_one_dataset = torch.cat((motif_labels, labels), dim=0)
                print(labels.size())
                print(label_one_dataset.size())
                y.append(label_one_dataset)
        

        graph_indices = torch.tensor(graph_indices)

        motif_smiles = sorted(self.motif_vocab, key=self.motif_vocab.get)
        
        heter_data = Data(x=x, edge_index=heter_edge_index, y=y, motif_smiles=motif_smiles, graph_indices=whole_graph_indices, graph_smiles=one_smiles_list, num_graph=num_graph)
        print(f"heterogeneous graph: {heter_data}")

        data_smiles_series = pd.Series(smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False,header=False)

        torch.save(self.collate([heter_data]), self.processed_paths[0])
                    

    