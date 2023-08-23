import os
import json
from utils.utils import get_fragment_mol, get_smiles, sanitize_smiles, get_mol, sanitize
from tqdm import tqdm
from collections import defaultdict
import torch
import math
import numpy as np


class MotifPiece:
    def __init__(self, dataset = None, label_list = None, vocab_path = None, threshold=None, score_method="frequency", merge_method="edge", pre_transform = None):
        self.pre_transform = None
        self.threshold = threshold
        self.smiles_list = dataset
        self.label_list = label_list
        self.score_method = score_method
        self.merge_method = merge_method

        if os.path.isfile(vocab_path+"motif_vocab.txt"):
            with open(vocab_path+"motif_vocab.txt", "r") as file:
                self.motif_vocab = json.load(file)
            with open(vocab_path+"s_dict_list.pkl", "r") as file:
                self.s_dict_list= json.load(file)
            with open(vocab_path+"v_dict_list.pkl", "r") as file:
                self.v_dict_list = json.load(file)
            with open(vocab_path+"e_dict_list.pkl", "r") as file:
                self.e_dict_list = json.load(file)
        else:
            self.motif_vocab = defaultdict(int)
            self.dataset_smiles = []
            self.process(dataset)
            if not os.path.exists(vocab_path):
                os.makedirs(vocab_path)
            with open(vocab_path+"motif_vocab.txt", "w") as file:
                file.write(json.dumps(self.motif_vocab))
            with open(vocab_path+"s_dict_list.pkl", "w") as file:
                file.write(json.dumps(self.s_dict_list))
            with open(vocab_path+"v_dict_list.pkl", "w") as file:
                file.write(json.dumps(self.v_dict_list))
            with open(vocab_path+"e_dict_list.pkl", "w") as file:
                file.write(json.dumps(self.e_dict_list))
        # print(f"motif vocabulary: {self.motif_vocab}")
    
    def def_value(self):
        return [0]*self.label_list.size(0)

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
        mol_list = []
        max_node_list = []
        max_edge_list = []
        iteration = 0
        while True:
            print(f"Iteration: {iteration}")

            motif_count = defaultdict(int)
            motif_indices = {}
            motif_element = defaultdict(set)
            unit_count = defaultdict(int)
            count_motif_candidate = 0

            count_positive_list = [defaultdict(self.def_value) for x in range(self.label_list.size(1))]
            count_negative_list = [defaultdict(self.def_value) for x in range(self.label_list.size(1))]
            # print(len(count_positive_list))
            # print(len(count_positive_list[0]["hh"]))
            # print(stop)

            # if self.merge

            # We assume mol is not None
            for i, data in tqdm(enumerate(dataset), desc="motif generation"):
                mol = get_mol(data)
                mol = sanitize(mol)
                # if mol == None:
                #     print("kk")
                #     continue
                # else:
                if iteration == 0:
                    self.dataset_smiles.append(data)
                    s_dict, v_dict, e_dict, max_node, max_edge = self.initialize(mol)
                    s_dict_list.append(s_dict)
                    v_dict_list.append(v_dict)
                    e_dict_list.append(e_dict)
                    max_node_list.append(max_node)
                    max_edge_list.append(max_edge)
                    mol_list.append(mol)
                else:
                    s_dict, v_dict, e_dict, max_node, max_edge = s_dict_list[i], v_dict_list[i], e_dict_list[i], max_node_list[i], max_edge_list[i]
                
                for subgraph_id, edge_list in s_dict.items():
                    if len(edge_list) == 0:
                        continue
                    for j in range(len(edge_list)-1):
                        for k in range(j+1, len(edge_list)):
                            count_motif_candidate += 1
                            e1 = e_dict[edge_list[j]]
                            e2 = e_dict[edge_list[k]]
                            v1 = v_dict[e1[0]] + v_dict[e1[1]]
                            v2 = v_dict[e2[0]] + v_dict[e2[1]]
                            unit1_mol = get_fragment_mol(mol, list(set(v1)))
                            unit2_mol = get_fragment_mol(mol, list(set(v2)))
                            unit1_smiles = get_smiles(unit1_mol)
                            unit2_smiles = get_smiles(unit2_mol)
                            unit_count[unit1_smiles] += 1
                            unit_count[unit2_smiles] += 1
                            


                            m = list(set(v1 + v2))
                            m_mol = get_fragment_mol(mol, m)
                            m_smiles = get_smiles(m_mol)
                            m_smiles = sanitize_smiles(m_smiles)
                            # motif_count[m_smiles][i].append((edge_list[j], edge_list[k]))
                            motif_count[m_smiles] += 1
                            for p in range(len(count_positive_list)):
                                label = self.label_list[i][p]
                                # print(label)
                                # print(stop)
                                if label == -1:
                                    count_negative_list[p][m_smiles][i] = 1
                                elif label == 1:
                                    count_positive_list[p][m_smiles][i] = 1
                                else:
                                    print("label error!")
                                    print(stop)
                            motif_element[m_smiles].add(tuple(sorted([unit1_smiles, unit2_smiles])))

                            if m_smiles not in motif_indices:
                                motif_indices[m_smiles] = defaultdict(list)
                                motif_indices[m_smiles][i] = [(edge_list[j], edge_list[k])]
                            else:
                                motif_indices[m_smiles][i].append((edge_list[j], edge_list[k]))
                            # print(len(motif_count))
                            # print(motif_indices['CCC'][0])

                            # print(stop)
            
            # Select the best motif candidate
            selected_motif = None
            max_score = 0
            print(f"Number of motif candidate: {count_motif_candidate}")
            count_graph = 0
            for motif, count in motif_count.items():
                if count > self.threshold:
                    count_graph += 1
                    demon = 0
                    for i, (unit1_smiles, unit2_smiles) in enumerate(motif_element[motif]):
                        demon += unit_count[unit1_smiles]*unit_count[unit2_smiles]
                    idf = 0
                    for i in range(len(count_positive_list)):
                        positive_count = sum(count_positive_list[i][motif])
                        negative_count = sum(count_negative_list[i][motif])
                        label = self.label_list[:, i]
                        # print(label.size())
                        # print(stop)
                        label_positive_count = 0
                        label_negative_count = 0
                        for value in label:
                            if value == 1:
                                label_positive_count += 1
                            elif value == -1:
                                label_negative_count += 1
                            else:
                                print(value)
                                print("Error!")
                                print(stop)
                        # label[label<0] = 0
                        # label_positive_count = torch.count_nonzero(label).item()
                        # label_negative_count = label.size(0) - label_positive_count
                        if positive_count>label_positive_count or negative_count>label_negative_count:
                            print(f"positive count: {positive_count}")
                            print(f"all positive: {label_positive_count}")
                            print(f"negative count: {negative_count}")
                            print(f"all negative: {label_negative_count}")
                            print(stop)
                        dis = math.sqrt((positive_count/label_positive_count - negative_count/label_negative_count)**2)
                        idf += dis
                    idf /= len(count_positive_list)


                    
                    demon /= (i+1)
                    # score = count*count/demon
                    # score = count*idf
                    # score = count
                    # score = 1/(1+np.exp(-count))*idf
                    score = math.log(count)*idf
                    if score > max_score:
                        max_score = score
                        selected_motif = motif
            print(f"Number of motifs: {count_graph}")
            print(f"max score: {max_score}")

            # if len(motif_count)==0:
            #     break
            # selected_motif = max(motif_count, key=motif_count.get)
            # print(selected_motif)
            # print(score)
            # print(stop)
            # print(motif_count)
            # print(motif_indices["N[SH](=O)=O"][148])
            # # print(stop)
            # print(s_dict_list[149][8])
            # print(stop)

            count_motif = 0
            if selected_motif == None:
                break
            else:
                # print(f"motif count: {count_motif}")
                count_motif += 1
                self.motif_vocab[selected_motif] += 1
                print(f"motif vocabulary: {self.motif_vocab}")
                print(f"motif count: {motif_count[selected_motif]}")
                need_merge_indices = motif_indices[selected_motif]
                # print(selected_motif)
                del motif_count[selected_motif]
                del motif_indices[selected_motif]
                for id, edge_list in need_merge_indices.items():
                    merged_set = set()
                    merged_edge = set()
                    s_dict, v_dict, e_dict, max_node, max_edge, mol = s_dict_list[id], v_dict_list[id], e_dict_list[id], max_node_list[id], max_edge_list[id], mol_list[id]

                    # print(f"sdict: {s_dict}")
                    # print(f"vdict: {v_dict}")
                    # print(f"edict: {e_dict}")
                    # print(f"edgelist: {edge_list}")
                    common_node_list = []
                    edge_nodes_list = []
                    for edge in edge_list:
                        node_list = e_dict[edge[0]] + e_dict[edge[1]]
                        edge_nodes_list.append(node_list)
                    for p, edge in enumerate(edge_list):
                        # print(f"edge: {edge}")
                        if not set(edge).isdisjoint(merged_edge):
                            continue
                        merged_edge.update(set(edge))
                        # print(f"node list: {node_list}")
                        node_list = edge_nodes_list[p]

                        ### Merge edges to a node!
                        union_nodes = set(node_list)
                        if len(union_nodes) != 3:
                            print(union_nodes)
                            print(node_list)
                            print(stop)

                        if not union_nodes.isdisjoint(merged_set):
                            continue
                        merged_set.update(union_nodes)
                        
                        union_nodes = list(union_nodes)

                        union_edge = set(s_dict[union_nodes[0]]).union(set(s_dict[union_nodes[1]])).union(set(s_dict[union_nodes[2]]))
                        intersection_edge = (set(s_dict[union_nodes[0]]).intersection(set(s_dict[union_nodes[1]]))).union(set(s_dict[union_nodes[1]]).intersection(set(s_dict[union_nodes[2]]))).union(set(s_dict[union_nodes[0]]).intersection(set(s_dict[union_nodes[2]])))
                        difference_edge = union_edge - intersection_edge
                        to_change_nodes = []
                        for e_id in difference_edge:
                            edge = e_dict[e_id]
                            if edge[0] in union_nodes and edge[1] not in union_nodes:
                                to_change_nodes.append(edge[1])
                            elif edge[0] not in union_nodes and edge[1] in union_nodes:
                                to_change_nodes.append(edge[0])
                            else:
                                print(edge)
                                print(union_nodes)
                                print("error!!!")
                                print(stop)

                        union_edge = list(union_edge)
                        to_change_nodes = list(set(to_change_nodes))
                        v_dict[max_node] = list(set(v_dict[union_nodes[0]]+v_dict[union_nodes[1]]+v_dict[union_nodes[2]]))
                        if len(v_dict[max_node]) == 0:
                            print(v_dict)
                            print(s_dict)
                            print(union_nodes)
                            print(stop)

                        count_added_edge = 0
                        for node in to_change_nodes:
                            e_dict[max_edge+count_added_edge] = (node, max_node)
                            s_dict[max_node].append(max_edge+count_added_edge)
                            s_dict[node] = [x for x in s_dict[node] if x not in union_edge]
                            s_dict[node].append(max_edge+count_added_edge)
                            count_added_edge += 1
                        for e in union_edge:
                            del e_dict[e]

                        for n in union_nodes:
                            del s_dict[n]
                            del v_dict[n]

                        max_edge += count_added_edge
                        max_node += 1

                    s_dict_list[id], v_dict_list[id], e_dict_list[id], max_node_list[id], max_edge_list[id], mol_list[id] = s_dict, v_dict, e_dict, max_node, max_edge, mol
                        

                    ### Merge edges to an edge!!!


                        # Update unit_count, delete merged units
                        # e1 = e_dict[edge[0]]
                        # e2 = e_dict[edge[1]]
                        # v1 = v_dict[e1[0]] + v_dict[e1[1]]
                        # v2 = v_dict[e2[0]] + v_dict[e2[1]]

                        # unit1_mol = get_fragment_mol(mol, list(set(v1)))
                        # unit2_mol = get_fragment_mol(mol, list(set(v2)))
                        # unit1_smiles = get_smiles(unit1_mol)
                        # unit2_smiles = get_smiles(unit2_mol)
                        # unit_count[unit1_smiles] -= 1
                        # unit_count[unit2_smiles] -= 1

                    #     merged_set.update(set(node_list))
                    #     exist = set()

                    #     for node in node_list:
                    #         if node in exist:
                    #             common_node = node
                    #             break
                    #         exist.add(node)
                    #     # print(f"common node: {common_node}")
                    #     common_node_list.append(common_node)
                    #     need_to_merge = [x for x in node_list if x != common_node]
                    #     if len(need_to_merge) != 2:
                    #         print(edge)
                    #         print(e_dict)
                    #         print(need_to_merge)
                    #         print(node_list)
                    #         print(common_node)
                    #         print("ERROR!!!")
                    #         print(id)
                    #         print(selected_motif)
                    #         print(stop)

                    #     node1, node2 = need_to_merge[0], need_to_merge[1]
                    #     v_dict[node1] = list(set(v_dict[node1] + v_dict[common_node]))
                    #     # print(s_dict[node1])
                    #     s_dict[node1] = [x for x in s_dict[node1] if x not in edge]
                    #     # print(s_dict[node1])
                        
                    #     v_dict[node2] = list(set(v_dict[node2] + v_dict[common_node]))
                    #     s_dict[node2] = [x for x in s_dict[node2] if x not in edge]
                    #     to_add_node = []
                    #     # print(f"origin sdict: {s_dict[common_node]}")
                    #     # print(edge)
                    #     # print(e_dict[edge[0]])
                    #     # print(e_dict[edge[1]])
                    #     # print(common_node)
                    #     common_node_edge = [x for x in s_dict[common_node] if x not in edge]
                    #     # print(f"origin sdict: {common_node_edge}")
                    #     # print(common_node_edge)
                    #     if len(s_dict[common_node]) != len(set(s_dict[common_node])):
                    #         print(id)
                    #         print(common_node)
                    #         print(s_dict[common_node])
                    #         print(stop)
                    #     for e in common_node_edge:
                    #         other_edge = e_dict[e]
                    #         if other_edge == (node1, node2) or other_edge == (node2, node1):
                    #             continue
                    #         # print('----->')
                    #         # print(other_edge)
                    #         # print(common_node)
                    #         # print(e_dict)
                    #         if other_edge[0] == common_node:
                    #             to_add_node.append(other_edge[1])
                    #             s_dict[other_edge[1]].remove(e)
                    #         elif other_edge[1] == common_node:
                    #             to_add_node.append(other_edge[0])
                    #             s_dict[other_edge[0]].remove(e)
                    #         else:
                    #             print("Error other edges!!!")
                    #             print(stop)
                    #         del e_dict[e]
                    #     # print(s_dict)
                        
                    #     del e_dict[edge[0]], e_dict[edge[1]], v_dict[common_node], s_dict[common_node]
                    #     if node1 == node2:
                    #         print(node1)
                    #         print(node_list)
                    #         print(stop)
                    #     if set(s_dict[node1]).isdisjoint(set(s_dict[node2])):
                    #         e_dict[max_edge] = (node1, node2)
                    #         s_dict[node1].append(max_edge)
                    #         s_dict[node2].append(max_edge)
                    #         max_edge += 1
                    #     for node in to_add_node:
                    #         # print("------------->")
                    #         # print(node)
                    #         if node == node1 or node == node2:
                    #             print(node, node1, node2)
                    #             print(stop)
                    #         count_exist = 0
                    #         if set(s_dict[node]).isdisjoint(set(s_dict[node1])):
                    #             e_dict[max_edge+count_exist] = (node, node1)
                    #             s_dict[node1].append(max_edge+count_exist)
                    #             s_dict[node].append(max_edge+count_exist)
                    #             count_exist += 1
                    #         if set(s_dict[node]).isdisjoint(set(s_dict[node2])):
                    #             e_dict[max_edge+count_exist] = (node, node2)
                    #             s_dict[node2].append(max_edge+count_exist)
                    #             s_dict[node].append(max_edge+count_exist)
                    #             count_exist += 1
                    #         max_edge += count_exist
                    #     # print(s_dict[node1])
                    #     # print(s_dict[node2])
                    #     # print(stop)
                    # s_dict_list[id], v_dict_list[id], e_dict_list[id], max_edge_list[id] = s_dict, v_dict, e_dict, max_edge
                

            #     # Update
            #     # print(common_node)
            #     for node in common_node_list:
            #         merged_set.remove(node)
            #     print(s_dict)
            #     print(v_dict)
            #     print(e_dict)
            #     print(merged_set)
            #     to_check_nodes = set()
            #     for node in merged_set:
            #         to_check_nodes.add(node)
            #         edge_list = s_dict[node]
            #         for e in edge_list:
            #             to_check_nodes.update(set(e_dict[e]))
            #     print(to_check_nodes)

            #     for node in to_check_nodes:
            #         print(f"node: {node}")
            #         edge_list = s_dict[node]
            #         print(f"edge_list: {edge_list}")
            #         if len(edge_list) == 0:
            #             continue
            #         for j in range(len(edge_list)-1):
            #             for k in range(j+1, len(edge_list)):
            #                 count_motif_candidate += 1
            #                 e1 = e_dict[edge_list[j]]
            #                 e2 = e_dict[edge_list[k]]
            #                 v1 = list(set(v_dict[e1[0]] + v_dict[e1[1]]))
            #                 v2 = list(set(v_dict[e2[0]] + v_dict[e2[1]]))

            #                 unit1_mol = get_fragment_mol(mol, v1)
            #                 unit2_mol = get_fragment_mol(mol, v2)
            #                 unit1_smiles = get_smiles(unit1_mol)
            #                 unit2_smiles = get_smiles(unit2_mol)

            #                 if e1[0] in merged_set or e1[1] in merged_set:
            #                     unit_count[unit1_smiles] += 1
            #                 if e2[0] in merged_set or e2[1] in merged_set:
            #                     unit_count[unit2_smiles] += 1
                            
            #                 m = list(set(v1 + v2))
            #                 m_mol = get_fragment_mol(mol, m)
            #                 m_smiles = get_smiles(m_mol)
            #                 m_smiles = sanitize_smiles(m_smiles)

            #                 motif_count[m_smiles] += 1
            #                 motif_element[m_smiles].add(tuple(sorted([unit1_smiles, unit2_smiles])))

            #                 if m_smiles not in motif_indices:
            #                     motif_indices[m_smiles] = defaultdict(list)
            #                     motif_indices[m_smiles][id] = [(edge_list[j], edge_list[k])]
            #                 else:
            #                     motif_indices[m_smiles][id].append((edge_list[j], edge_list[k]))
            #                 # print(motif_indices[m_smiles])

            # # del motif_indices[selected_motif]
            # selected_motif = None
            # max_score = 0
            # for motif, count in motif_count.items():
            #     if count > self.threshold:
            #         demon = 0
            #         for i, (unit1_smiles, unit2_smiles) in enumerate(motif_element[motif]):
            #             demon += unit_count[unit1_smiles]*unit_count[unit2_smiles]
            #         demon /= (i+1)
            #         score = count*count_motif_candidate/demon
            #         if score > max_score:
            #             max_score = score
            #             selected_motif = motif

            
            
            # if selected_motif == None:
            #     print("Beak the program!!!")
            #     break
            # else:
            #     if selected_motif not in self.motif_vocab:
            #         self.motif_vocab[selected_motif] = 1
            #     else:
            #         self.motif_vocab[selected_motif] += 1
                # count_graph = 0
                # for i, data in tqdm(enumerate(dataset), desc="merge motifs"):
                    
                #     mol = get_mol(data)
                #     mol = sanitize(mol)

                #     if mol == None:
                #         continue
                #     else:
                #         s_dict, v_dict, e_dict = s_dict_list[count_graph], v_dict_list[count_graph], e_dict_list[count_graph]
                #         count_graph += 1
                        
                #         while True:
                #             break_all = -1
                #             for subgraph_id, edge_list in s_dict.items():
                #                 for j in range(len(edge_list)-1):
                #                     for k in range(j+1, len(edge_list)):
                #                         e1 = e_dict[edge_list[j]]
                #                         e2 = e_dict[edge_list[k]]
                #                         v1 = v_dict[e1[0]] + v_dict[e1[1]]
                #                         v2 = v_dict[e2[0]] + v_dict[e2[1]]
                #                         m = list(set(v1 + v2))
                #                         m_mol = get_fragment_mol(mol, m)
                #                         m_smiles = get_smiles(m_mol)
                #                         m_smiles = sanitize_smiles(m_smiles)
                                        
                #                         if m_smiles == selected_motif:

                #                             key = max(s_dict)
                #                             v_dict[key+1] = m

                #                             new_edge = set(s_dict[e1[0]]+s_dict[e1[1]]+s_dict[e2[0]]+s_dict[e2[1]])
                #                             new_edge.remove(edge_list[j])
                #                             new_edge.remove(edge_list[k])
                #                             new_edge = list(new_edge)
                #                             s_dict[key+1] = new_edge
                #                             node_list = list(set(list(e1)+list(e2)))

                #                             deleted_edge_set = [edge_list[j], edge_list[k]]

                #                             for e in new_edge:
                #                                 edge = e_dict[e]
                #                                 element1, element2 = list(edge)
                #                                 if element1 not in node_list and element2 in node_list:
                #                                     e_dict[e] = (element1, key+1)
                #                                 elif element1 in node_list and element2 not in node_list:
                #                                     e_dict[e] = (element2, key+1)
                #                                 else:
                #                                     new_value = set(s_dict[key+1])
                #                                     new_value.remove(e)
                #                                     s_dict[key+1] = list(new_value)
                #                                     deleted_edge_set.append(e)

                #                             deleted_subgraph_set = list(set(list(e1)+list(e2)))

                #                             for key in deleted_subgraph_set:
                #                                 del s_dict[key]
                #                             for key in deleted_edge_set:
                #                                 del e_dict[key]

                #                             break_all = 1
                #                             break
                #                     if break_all == 1:
                #                         break
                #                 if break_all == 1:
                #                     break
                #             if break_all == -1:
                #                 break
            iteration += 1
        self.s_dict_list = s_dict_list
        self.v_dict_list = v_dict_list
        self.e_dict_list = e_dict_list
        # print(self.s_dict_list[1145])
        # print(self.v_dict_list[1145])
        # print(self.e_dict_list[1145])
        # print(stop)

        
    

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
        s_dict = defaultdict(list)
        v_dict = {}
        e_dict = defaultdict(tuple)
        
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
        
        for atom in mol.GetAtoms():
            id = atom.GetIdx()
            v_dict[id] = [id]
            if id not in s_dict:
                s_dict[id] = []
            
        return s_dict, v_dict, e_dict, len(v_dict), len(e_dict)
    
    def inference(self, input_smiles):
        # print(f"smiles: {input_smiles}")
        mol = get_mol(input_smiles)
        mol = sanitize(mol)
        # print(f"Number of atoms: {mol.GetNumAtoms()}")
        if mol == None:
            print("The data can not be identified by RDKit!")
            return 0
        iteration = 0
        ori_e_dict = {}
        while True:
            if iteration == 0:
                s_dict, v_dict, e_dict = self.initialize(mol)
            
            break_all = 0
            # print(f"s_dict: {s_dict}")
            # print(f"v_dict: {v_dict}")
            # print(f"e_dict: {e_dict}")
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
                                    if (element1, element2) not in ori_e_dict and (element2, element1) not in ori_e_dict:
                                        ori_e_dict[(element1, key+1)] = (element1, element2)
                                    elif (element1, element2) in ori_e_dict:
                                        ori_e_dict[(element1, key+1)] = ori_e_dict[element1, element2]
                                    elif (element2, element1) in ori_e_dict:
                                        ori_e_dict[(element1, key+1)] = ori_e_dict[element2, element1]
                                    e_dict[e] = (element1, key+1)
                                elif element1 in node_list and element2 not in node_list:
                                    if (element1, element2) not in ori_e_dict and (element2, element1) not in ori_e_dict:
                                        ori_e_dict[(element2, key+1)] = (element1, element2)
                                    elif (element1, element2) in ori_e_dict:
                                        ori_e_dict[(element2, key+1)] = ori_e_dict[element1, element2]
                                    elif (element2, element1) in ori_e_dict:
                                        ori_e_dict[(element2, key+1)] = ori_e_dict[element2, element1]
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
            if len(atom_list) > 1:
                m_mol = get_fragment_mol(mol, atom_list)
                m_smiles = get_smiles(m_mol)
                m_smiles = sanitize_smiles(m_smiles)
                motif_smiles_list.append(m_smiles)
                s_id_list.append(subgraph_id)
            else:
                if len(s_dict[subgraph_id]) == 0:
                    m_mol = get_fragment_mol(mol, atom_list)
                    m_smiles = get_smiles(m_mol)
                    m_smiles = sanitize_smiles(m_smiles)
                    motif_smiles_list.append(m_smiles)
                    s_id_list.append(subgraph_id)
        
        edge_list = []

        # Miss one situation that the edge motif is original edge, and no edges for this motif
        for edge_id, edge in e_dict.items():
            if edge[0] in s_id_list and edge[1] in s_id_list:
                edge_list.append((s_id_list.index(edge[0]), s_id_list.index(edge[1])))
            # elif edge[0] in s_id_list and edge[1] not in s_id_list:
            #     ori_edge = ori_e_dict[edge]
            #     m_mol = get_fragment_mol(mol, [edge[1]])
            #     m_smiles = get_smiles(m_mol)
            #     m_smiles = sanitize_smiles(m_smiles)
            #     motif_smiles_list.append(m_smiles)
            #     edge_list.append((s_id_list.index(edge[0]), len(motif_smiles_list)-1))
            # elif edge[0] not in s_id_list and edge[1] in s_id_list:
            #     ori_edge = ori_e_dict[edge]
            #     m_mol = get_fragment_mol(mol, [edge[0]])
            #     m_smiles = get_smiles(m_mol)
            #     m_smiles = sanitize_smiles(m_smiles)
            #     motif_smiles_list.append(m_smiles)
            #     edge_list.append((len(motif_smiles_list)-1, s_id_list.index(edge[1])))
            # elif edge[0] not in s_id_list and edge[1] not in s_id_list:
            #     for e in edge:
            #         m_mol = get_fragment_mol(mol, [e])
            #         m_smiles = get_smiles(m_mol)
            #         m_smiles = sanitize_smiles(m_smiles)
            #         motif_smiles_list.append(m_smiles)
            #     edge_list.append((len(motif_smiles_list)-2, len(motif_smiles_list)-1))

        # print(motif_smiles_list)
        # print(f"edges: {edge_list}")
        return motif_smiles_list, edge_list
    
    def self_inference(self):
        all_motif_smiles = []
        all_edge_list = []
        print("hhh")
        for i in tqdm(range(len(self.smiles_list))):
            s_dict, v_dict, e_dict, smiles = self.s_dict_list[i], self.v_dict_list[i], self.e_dict_list[i], self.smiles_list[i]
            mol = get_mol(smiles)
            mol = sanitize(mol)
            motif_smiles_list = []
            s_id_list = []
            # print(f"num of graph: {i}")
            # print(f"s dict: {s_dict}")
            # print(f"v dict: {v_dict}")
            # print(f"e dict: {e_dict}")
            for subgraph_id, atom_list in v_dict.items():
                if len(atom_list) > 1:
                    m_mol = get_fragment_mol(mol, atom_list)
                    m_smiles = get_smiles(m_mol)
                    m_smiles = sanitize_smiles(m_smiles)
                    motif_smiles_list.append(m_smiles)
                    s_id_list.append(subgraph_id)
                else:
                    if len(s_dict[subgraph_id]) == 0:
                        m_mol = get_fragment_mol(mol, atom_list)
                        m_smiles = get_smiles(m_mol)
                        m_smiles = sanitize_smiles(m_smiles)
                        motif_smiles_list.append(m_smiles)
                        s_id_list.append(subgraph_id)
            
            edge_list = []

            # Miss one situation that the edge motif is original edge, and no edges for this motif
            for edge_id, edge in e_dict.items():
                if edge[0] in s_id_list and edge[1] in s_id_list:
                    edge_list.append((s_id_list.index(edge[0]), s_id_list.index(edge[1])))
            all_motif_smiles.append(motif_smiles_list)
            all_edge_list.append(edge_list)
        # print(all_motif_smiles)
        # print(all_edge_list)
        return all_motif_smiles, all_edge_list
                                



