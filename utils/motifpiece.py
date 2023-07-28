import os
import json
from utils.utils import get_fragment_mol, get_smiles, sanitize_smiles, get_mol, sanitize
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000

class MotifPiece:
    def __init__(self, dataset = None, vocab_path = None, pre_transform = None):
        self.pre_transform = None
        self.dataset_size = 0
        self.threshold = 5
        if os.path.isfile(vocab_path+"motif_vocab.txt"):
            with open(vocab_path+"motif_vocab.txt", "r") as file:
                self.motif_vocab = json.load(file)
        else:
            self.motif_vocab = {}
            self.process(dataset)
            if not os.path.exists(vocab_path):
                os.makedirs(vocab_path)
            with open(vocab_path+"motif_vocab.txt", "w") as file:
                file.write(json.dumps(self.motif_vocab))
        print(f"motif vocabulary: {self.motif_vocab}")
    
    def process(self, dataset):
        """
        Generate a motif vocabulary from a dataset

        Parameters
        ---------
        dataset: a list of smiles
            Input dataset that will be processed to extract motifs
        """
        s_dict_list = []
        v_dict_list = []
        e_dict_list = []
        iteration = 0
        while True:
            motif_count = {}
            count_graph = 0
            for i, data in enumerate(dataset):
                mol = get_mol(data)
                mol = sanitize(mol)
                if mol == None:
                    continue
                else:
                    if iteration == 0:
                        s_dict, v_dict, e_dict = self.initialize(mol)
                        s_dict_list.append(s_dict)
                        v_dict_list.append(v_dict)
                        e_dict_list.append(e_dict)
                    else:
                        s_dict, v_dict, e_dict = s_dict_list[count_graph], v_dict_list[count_graph], e_dict_list[count_graph]
                    
                    count_graph += 1
                    
                    for subgraph_id, edge_list in s_dict.items():
                        for j in range(len(edge_list)-1):
                            for k in range(j+1, len(edge_list)):
                                e1 = e_dict[edge_list[j]]
                                e2 = e_dict[edge_list[k]]
                                v1 = v_dict[e1[0]] + v_dict[e1[1]]
                                v2 = v_dict[e2[0]] + v_dict[e2[1]]
                                m = list(set(v1 + v2))
                                m_mol = get_fragment_mol(mol, m)
                                m_smiles = get_smiles(m_mol)
                                m_smiles = sanitize_smiles(m_smiles)
                                if m_smiles not in motif_count:
                                    motif_count[m_smiles] = 1
                                else:
                                    motif_count[m_smiles] += 1
            
            # Select the best motif candidate
            selected_motif = max(motif_count, key=motif_count.get)
            # print(selected_motif)
            # print(motif_count)
            
            
            if motif_count[selected_motif] < self.threshold:
                print("Beak the program!!!")
                break
            else:
                if selected_motif not in self.motif_vocab:
                    self.motif_vocab[selected_motif] = 1
                else:
                    self.motif_vocab[selected_motif] += 1
                count_graph = 0
                for i, data in enumerate(dataset):
                    
                    mol = get_mol(data)
                    mol = sanitize(mol)

                    if mol == None:
                        continue
                    else:
                        s_dict, v_dict, e_dict = s_dict_list[count_graph], v_dict_list[count_graph], e_dict_list[count_graph]
                        count_graph += 1
                        
                        while True:
                            break_all = -1
                            for subgraph_id, edge_list in s_dict.items():
                                for j in range(len(edge_list)-1):
                                    for k in range(j+1, len(edge_list)):
                                        e1 = e_dict[edge_list[j]]
                                        e2 = e_dict[edge_list[k]]
                                        v1 = v_dict[e1[0]] + v_dict[e1[1]]
                                        v2 = v_dict[e2[0]] + v_dict[e2[1]]
                                        m = list(set(v1 + v2))
                                        m_mol = get_fragment_mol(mol, m)
                                        m_smiles = get_smiles(m_mol)
                                        m_smiles = sanitize_smiles(m_smiles)
                                        
                                        if m_smiles == selected_motif:

                                            key = max(s_dict)
                                            v_dict[key+1] = m
                                            # print(f"edge_list: {edge_list}")
                                            # print(f"m: {m}")
                                            # print(f"s_dict: {s_dict}")
                                            # print(f"v_dict: {v_dict}")
                                            # print(f"e_dict: {e_dict}")
                                            # print(f"e1: {e1}")
                                            # print(f"e2: {e2}")

                                            new_edge = set(s_dict[e1[0]]+s_dict[e1[1]]+s_dict[e2[0]]+s_dict[e2[1]])
                                            new_edge.remove(edge_list[j])
                                            new_edge.remove(edge_list[k])
                                            new_edge = list(new_edge)
                                            s_dict[key+1] = new_edge
                                            node_list = list(set(list(e1)+list(e2)))

                                            deleted_edge_set = [edge_list[j], edge_list[k]]

                                            for e in new_edge:
                                                edge = e_dict[e]
                                                element1, element2 = list(edge)
                                                if element1 not in node_list and element2 in node_list:
                                                    e_dict[e] = (element1, key+1)
                                                elif element1 in node_list and element2 not in node_list:
                                                    e_dict[e] = (element2, key+1)
                                                else:
                                                    new_value = set(s_dict[key+1])
                                                    new_value.remove(e)
                                                    s_dict[key+1] = list(new_value)
                                                    deleted_edge_set.append(e)

                                            # for e in new_edge:
                                            #     edge = e_dict[e]
                                            #     for element in edge:
                                            #         if element not in node_list:
                                            #             e_dict[e] = (element, key+1)

                                            deleted_subgraph_set = list(set(list(e1)+list(e2)))

                                            for key in deleted_subgraph_set:
                                                del s_dict[key]
                                            # del e_dict[edge_list[j]], e_dict[edge_list[k]]
                                            for key in deleted_edge_set:
                                                del e_dict[key]

                                            break_all = 1
                                            break
                                    if break_all == 1:
                                        break
                                if break_all == 1:
                                    break
                            if break_all == -1:
                                break
            iteration += 1
        
    

    def initialize(self, mol):
        """
        An initialization function for motif vocabulary generation

        Parameters
        ---------
        mol: RDKit object
            A mol that will be process for the next process
        
        Output
        -----
        s_dict: A dictionary
            A dictionary with format subgraph_id: edge_indices_list
        v_dict: A dictionary
            A dictionary with format subgraph_id: constituent origianl node indices of the subgraph
        e_dict: A dictionary
            A dictionary with format edge_id: (start_node_id, end_node_id)
        """
        s_dict = {}
        v_dict = {}
        e_dict = {}
        for atom in mol.GetAtoms():
            id = atom.GetIdx()
            v_dict[id] = [id]
        for bond in mol.GetBonds():
            id = bond.GetIdx()
            startid = bond.GetBeginAtomIdx()
            endid = bond.GetEndAtomIdx()
            e_dict[id] = (startid, endid)
            if startid not in s_dict:
                s_dict[startid] = [id]
            else:
                s_dict[startid].append(id)
            if endid not in s_dict:
                s_dict[endid] = [id]
            else:
                s_dict[endid].append(id)
            
        return s_dict, v_dict, e_dict
    
    def inference(self, input_smiles):
        mol = get_mol(input_smiles)
        mol = sanitize(mol)
        # print(f"Number of atoms: {mol.GetNumAtoms()}")
        if mol == None:
            print("The data can not be identified by RDKit!")
            return 0
        iteration = 0
        while True:
            if iteration == 0:
                s_dict, v_dict, e_dict = self.initialize(mol)
            
            break_all = 0
            for subgraph_id, edge_list in s_dict.items():
                for j in range(len(edge_list)-1):
                    for k in range(j+1, len(edge_list)):
                        e1 = e_dict[edge_list[j]]
                        e2 = e_dict[edge_list[k]]
                        v1 = v_dict[e1[0]] + v_dict[e1[1]]
                        v2 = v_dict[e2[0]] + v_dict[e2[1]]
                        m = list(set(v1 + v2))
                        m_mol = get_fragment_mol(mol, m)
                        m_smiles = get_smiles(m_mol)
                        m_smiles = sanitize_smiles(m_smiles)
                        if m_smiles in self.motif_vocab:
                            key = max(s_dict)
                            v_dict[key+1] = m
                            # print(f"edge_list: {edge_list}")
                            # print(f"m: {m}")
                            # print(f"s_dict: {s_dict}")
                            # print(f"v_dict: {v_dict}")
                            # print(f"e_dict: {e_dict}")
                            # print(f"e1: {e1}")
                            # print(f"e2: {e2}")

                            new_edge = set(s_dict[e1[0]]+s_dict[e1[1]]+s_dict[e2[0]]+s_dict[e2[1]])
                            new_edge.remove(edge_list[j])
                            new_edge.remove(edge_list[k])
                            new_edge = list(new_edge)
                            s_dict[key+1] = new_edge
                            node_list = list(set(list(e1)+list(e2)))
                            # print(f"node list:{node_list}")
                            # print(f"new edge: {new_edge}")
                            deleted_edge_set = [edge_list[j], edge_list[k]]

                            for e in new_edge:
                                edge = e_dict[e]
                                element1, element2 = list(edge)
                                if element1 not in node_list and element2 in node_list:
                                    e_dict[e] = (element1, key+1)
                                elif element1 in node_list and element2 not in node_list:
                                    e_dict[e] = (element2, key+1)
                                else:
                                    new_value = set(s_dict[key+1])
                                    new_value.remove(e)
                                    s_dict[key+1] = list(new_value)
                                    deleted_edge_set.append(e)

                            deleted_subgraph_set = list(set([e1[0], e1[1], e2[0], e2[1]]))


                            for key in deleted_subgraph_set:
                                del s_dict[key], v_dict[key]

                            for key in deleted_edge_set:
                                del e_dict[key]

                            break_all = 1
                            break
                    if break_all == 1:
                        break
                if break_all == 1:
                    break
            if break_all == 0:
                break
            iteration += 1
        extracted_motif = {}
        motif_smiles_list = []
        s_id_list = []
        # print(f"s dict: {s_dict}")
        # print(f"v dict: {v_dict}")
        # print(f"e dict: {e_dict}")
        for subgraph_id, atom_list in v_dict.items():
            m_mol = get_fragment_mol(mol, atom_list)
            m_smiles = get_smiles(m_mol)
            m_smiles = sanitize_smiles(m_smiles)
            motif_smiles_list.append(m_smiles)
            s_id_list.append(subgraph_id)
        
        edge_list = []
        for edge_id, edge in e_dict.items():
            edge_list.append((s_id_list.index(edge[0]), s_id_list.index(edge[1])))
        # print(motif_smiles_list)
        # print(f"edges: {edge_list}")
        return motif_smiles_list, edge_list
                                




