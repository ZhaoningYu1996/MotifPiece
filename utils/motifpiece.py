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

            if self.merge_method == "edge":
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
                                if self.score_method in ["log_tf_df_ig", "ig"]:
                                    for p in range(len(count_positive_list)):
                                        label = self.label_list[i][p]
                                        # print(label)
                                        # print(stop)
                                        if label == -1:
                                            count_negative_list[p][m_smiles][i] = 1
                                        elif label == 1:
                                        # else:
                                            count_positive_list[p][m_smiles][i] = 1
                                        # else:
                                        #     print("label error!")
                                        #     print(stop)
                                motif_element[m_smiles].add(tuple(sorted([unit1_smiles, unit2_smiles])))

                                if m_smiles not in motif_indices:
                                    motif_indices[m_smiles] = defaultdict(list)
                                    motif_indices[m_smiles][i] = [(edge_list[j], edge_list[k])]
                                else:
                                    motif_indices[m_smiles][i].append((edge_list[j], edge_list[k]))
                                # print(len(motif_count))
                                # print(motif_indices['CCC'][0])

                                # print(stop)
                
                # print(f"s dict list: {s_dict_list}")
                # print(stop)
                # Select the best motif candidate
                selected_motif = None
                max_score = 0
                print(f"Number of motif candidate: {count_motif_candidate}")
                count_graph = 0
                for motif, count in motif_count.items():
                    if count > self.threshold:
                        count_graph += 1
                        if self.score_method == "mi":
                            demon = 0
                            for i, (unit1_smiles, unit2_smiles) in enumerate(motif_element[motif]):
                                demon += unit_count[unit1_smiles]*unit_count[unit2_smiles]
                            demon /= (i+1)
                        elif self.score_method in ["log_tf_df_ig", "ig"]:
                            idf = 0
                            df = 0
                            information_gain = 0
                            for i in range(len(count_positive_list)):
                                positive_count = sum(count_positive_list[i][motif])
                                negative_count = sum(count_negative_list[i][motif])
                                label = self.label_list[:, i]

                                label_positive_count = 0
                                label_negative_count = 0
                                for value in label:
                                    if value == 1:
                                        label_positive_count += 1
                                    elif value == -1:
                                        label_negative_count += 1

                                N = self.label_list.size(0)
                                
                                h_label = -(label_positive_count+1)/(N+1)*math.log2((label_positive_count+1)/(N+1))-(label_negative_count+1)/(N+1)*math.log2((label_negative_count+1)/(N+1))

                                p_m_0 = (N-positive_count-negative_count+1)/(N+1)
                                p_m_1 = (positive_count+negative_count+1)/(N+1)

                                p_0_0 = (label_negative_count-negative_count+1)/(N+1)
                                # p_0_0 = (label_negative_count-negative_count)/N
                                p_0_1 = (label_positive_count-positive_count+1)/(N+1)
                                p_1_0 = (negative_count+1)/(N+1)
                                p_1_1 = (positive_count+1)/(N+1)
                                h_label_motif_0 = -p_0_0*math.log2(p_0_0/p_m_0) - p_0_1*math.log2(p_0_1/p_m_0)
                                h_label_motif_1 = - p_1_0*math.log2(p_1_0/p_m_1) - p_1_1*math.log2(p_1_1/p_m_1)

                                information_gain += (h_label - p_m_0*h_label_motif_0 - p_m_1*h_label_motif_1)
                                
                                df += (positive_count+negative_count)/(label_positive_count+label_negative_count)
                                dis = math.sqrt((positive_count/label_positive_count - negative_count/label_negative_count)**2)
                                idf += dis
                            idf /= len(count_positive_list)
                            df /= len(count_positive_list)
                            information_gain /= len(count_positive_list)

                            if information_gain > 1 or information_gain < 0:
                                print(h_label)
                                print(h_label_motif_0)
                                print(h_label_motif_1)
                                print(p_m_0)
                                print(p_m_1)
                                print(positive_count)
                                print(negative_count)
                                print(label_positive_count)
                                print(label_negative_count)
                                print(information_gain)
                                print("Error!!")
                                print(stop)

                        # if information_gain < 0.5:
                        #     continue


                        # score = count*count/demon
                        # score = count*idf
                        if self.score_method == "frequency":
                            score = count
                        elif self.score_method == "ig":
                            score = information_gain
                        elif self.score_method == "log_tf_df_ig":
                            score = math.log(count)*df*information_gain
                        # score = 1/(1+np.exp(-count))*idf
                        # score = math.log(count)*idf
                        # score = count*idf
                        # score = math.log(count)*df
                        # score = math.log(count)*df*idf
                        # score = math.log(count)*idf
                        # if iteration == 0:
                        #     score = math.log(count)*idf
                        # else:
                        #     score = math.log(count)*df
                        # score = count*df*idf
                        # score = count * df # Not good
                        # score = df
                        # score = df*idf
                        # score = 1/(1+np.exp(-count))*df 
                        # score = math.log(count)*information_gain
                        # score = count*information_gain
                        # score = df*information_gain
                        # 
                        # score = df*information_gain
                        # 
                        if score > max_score:
                            max_score = score
                            selected_motif = motif
                print(f"Number of motifs: {count_graph}")
                print(f"max score: {max_score}")

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
                            
            elif self.merge_method == "node":
                for i, data in tqdm(enumerate(dataset), desc="motif generation"):
                    mol = get_mol(data)
                    mol = sanitize(mol)
                    # if mol == None:
                    #     print("kk")
                    #     continue
                    # else:
                    if iteration == 0:
                        self.dataset_smiles.append(data)
                        s_dict, v_dict, e_dict, max_node, max_edge = self.initialize_node(mol)
                        s_dict_list.append(s_dict)
                        v_dict_list.append(v_dict)
                        e_dict_list.append(e_dict)
                        max_node_list.append(max_node)
                        max_edge_list.append(max_edge)
                        mol_list.append(mol)
                    else:
                        v_dict, e_dict, max_node, max_edge = v_dict_list[i], e_dict_list[i], max_node_list[i], max_edge_list[i]
                    
                    ### Check all possible motif candidate which is a pair of node
                    # print(f"e dict: {e_dict}")
                    for e_id, node_pair in e_dict.items():
                        # print("-------------->")
                        # print(node_pair)
                        node_1 = node_pair[0]
                        node_2 = node_pair[1]
                        ori_node_1 = v_dict[node_1]
                        ori_node_2 = v_dict[node_2]
                        m = list(set(ori_node_1+ori_node_2))
                        m_mol = get_fragment_mol(mol, m)
                        m_smiles = get_smiles(m_mol)
                        m_smiles = sanitize_smiles(m_smiles)

                        motif_count[m_smiles] += 1

                        if self.score_method in ["log_tf_df_ig", "ig"]:
                            for p in range(len(count_positive_list)):
                                label = self.label_list[i][p]

                                if label == -1:
                                    count_negative_list[p][m_smiles][i] = 1
                                elif label == 1:
                                    count_positive_list[p][m_smiles][i] = 1
                                else:
                                    print("label error!")
                                    print(stop)

                        if m_smiles not in motif_indices:
                            motif_indices[m_smiles] = defaultdict(list)
                            motif_indices[m_smiles][i] = [e_id]
                        else:
                            motif_indices[m_smiles][i].append(e_id)

                # Select the best motif candidate
                selected_motif = None
                max_score = 0
                print(f"Number of motif candidate: {count_motif_candidate}")
                count_graph = 0
                for motif, count in motif_count.items():
                    if count > self.threshold:
                        count_graph += 1
                        if self.score_method in ["log_tf_df_ig", "ig"]:
                            idf = 0
                            df = 0
                            information_gain = 0
                            for i in range(len(count_positive_list)):
                                positive_count = sum(count_positive_list[i][motif])
                                negative_count = sum(count_negative_list[i][motif])
                                label = self.label_list[:, i]

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

                                if positive_count>label_positive_count or negative_count>label_negative_count:
                                    print(f"positive count: {positive_count}")
                                    print(f"all positive: {label_positive_count}")
                                    print(f"negative count: {negative_count}")
                                    print(f"all negative: {label_negative_count}")
                                    print(stop)
                                
                                N = self.label_list.size(0)
                                
                                h_label = -(label_positive_count+1)/(N+1)*math.log2((label_positive_count+1)/(N+1))-(label_negative_count+1)/(N+1)*math.log2((label_negative_count+1)/(N+1))

                                p_m_0 = (N-positive_count-negative_count+1)/(N+1)
                                p_m_1 = (positive_count+negative_count+1)/(N+1)
                                p_0_0 = (label_negative_count-negative_count+1)/(N+1)
                                p_0_1 = (label_positive_count-positive_count+1)/(N+1)
                                p_1_0 = (negative_count+1)/(N+1)
                                p_1_1 = (positive_count+1)/(N+1)
                                h_label_motif_0 = -p_0_0*math.log2(p_0_0/p_m_0) - p_0_1*math.log2(p_0_1/p_m_0)
                                h_label_motif_1 = - p_1_0*math.log2(p_1_0/p_m_1) - p_1_1*math.log2(p_1_1/p_m_1)

                                information_gain += (h_label - p_m_0*h_label_motif_0 - p_m_1*h_label_motif_1)
                                
                                df += (positive_count+negative_count)/(label_positive_count+label_negative_count)
                                dis = math.sqrt((positive_count/label_positive_count - negative_count/label_negative_count)**2)
                                idf += dis
                            idf /= len(count_positive_list)
                            df /= len(count_positive_list)
                            information_gain /= len(count_positive_list)
                            if information_gain > 1 or information_gain < 0:
                                print(h_label)
                                print(h_label_motif_0)
                                print(h_label_motif_1)
                                print(p_m_0)
                                print(p_m_1)
                                print(positive_count)
                                print(negative_count)
                                print(label_positive_count)
                                print(label_negative_count)
                                print(information_gain)
                                print("Error!!")
                                print(stop)

                        # if information_gain < 0.5:
                        #     continue

                        # score = count*count/demon
                        # score = count*idf
                        if self.score_method == "frequency":
                            score = count
                        elif self.score_method == "ig":
                            score = information_gain
                        elif self.score_method == "log_tf_df_ig":
                            score = math.log(count)*df*information_gain
                        # score = 1/(1+np.exp(-count))*idf
                        # score = math.log(count)*idf
                        # score = count*idf
                        # score = math.log(count)*df
                        # score = math.log(count)*df*idf
                        # score = math.log(count)*idf
                        # if iteration == 0:
                        #     score = math.log(count)*idf
                        # else:
                        #     score = math.log(count)*df
                        # score = count*df*idf
                        # score = count * df # Not good
                        # score = df
                        # score = df*idf
                        # score = 1/(1+np.exp(-count))*df 
                        # score = math.log(count)*information_gain
                        # score = count*information_gain
                        # score = df*information_gain
                        # 
                        # score = df*information_gain
                        # score = information_gain
                        if score > max_score:
                            max_score = score
                            selected_motif = motif
                print(f"Number of motifs: {count_graph}")
                print(f"max score: {max_score}")
                if selected_motif == None:
                    break
                else:
                    self.motif_vocab[selected_motif] += 1
                    print(f"motif vocabulary: {self.motif_vocab}")
                    print(f"motif count: {motif_count[selected_motif]}")
                    need_merge_indices = motif_indices[selected_motif]
                    # print(f"need merge indices: {need_merge_indices}")
                    for id, edge_list in need_merge_indices.items():
                        # print(f"edgelist: {edge_list}")
                        
                        merged_set = set()
                        s_dict, v_dict, e_dict, max_node, max_edge = s_dict_list[id], v_dict_list[id], e_dict_list[id], max_node_list[id], max_edge_list[id]
                        # print(e_dict)
                        node_list_edge = []
                        for e in edge_list:
                            node_pair = e_dict[e]
                            node_1 = node_pair[0]
                            node_2 = node_pair[1]
                            node_list_edge.append((node_1, node_2))
                            if node_1 not in v_dict or node_2 not in v_dict:
                                print("Huge Error!")
                                print(stop)
                            if node_1 not in s_dict or node_2 not in s_dict:
                                print("Huge Error!")
                                print(stop)
                        for i, e in enumerate(edge_list):
                            node_pair = node_list_edge[i]
                            node_1 = node_pair[0]
                            node_2 = node_pair[1]
                            if node_1 in merged_set or node_2 in merged_set:
                                continue
                            merged_set.add(node_1)
                            merged_set.add(node_2)
                            merged_node_list = list(set(v_dict[node_1]+v_dict[node_2]))

                            v_dict[max_node] = merged_node_list

                            edge_node_1 = s_dict[node_1]
                            edge_node_2 = s_dict[node_2]
                            need_to_del_edge = []
                            need_to_change_node = []
                            for other_e in list(set(edge_node_1+edge_node_2)):
                                first_node = e_dict[other_e][0]
                                second_node = e_dict[other_e][1]
                                need_to_del_edge.append(other_e)
                                if first_node in [node_1, node_2] and second_node not in [node_1, node_2]:
                                    need_to_change_node.append(second_node)
                                elif first_node not in [node_1, node_2] and second_node in [node_1, node_2]:
                                    need_to_change_node.append(first_node)
                                elif first_node in [node_1, node_2] and second_node in [node_1, node_2]:
                                    pass
                                else:
                                    print("Wrong merge!!!")
                                    print(stop)
                            
                            count_added_edge = 0
                            for node in need_to_change_node:
                                s_dict[node] = [x for x in s_dict[node] if x not in need_to_del_edge]
                                e_dict[max_edge+count_added_edge] = (node, max_node)
                                s_dict[max_node].append(max_edge+count_added_edge)
                                s_dict[node].append(max_edge+count_added_edge)
                                count_added_edge += 1

                            for item in need_to_del_edge:
                                del e_dict[item]
                            
                            del v_dict[node_1], v_dict[node_2], s_dict[node_1], s_dict[node_2]
                            max_node += 1
                            max_edge += count_added_edge
                        s_dict_list[id], v_dict_list[id], e_dict_list[id], max_node_list[id], max_edge_list[id] = s_dict, v_dict, e_dict, max_node, max_edge

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
        An initialization function for edge merge motif vocabulary generation

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
            s_dict[startid].append(id)
            s_dict[endid].append(id)
        
        for atom in mol.GetAtoms():
            id = atom.GetIdx()
            v_dict[id] = [id]
            if id not in s_dict:
                s_dict[id] = []
            
        return s_dict, v_dict, e_dict, len(v_dict), len(e_dict)
    
    def initialize_node(self, mol):
        """
        An initialization function for node merge motif vocabulary generation

        Parameters
        ---------
        mol: RDKit object
            A mol that will be process for the next process
        
        Output
        -----
        v_dict: A dictionary
            A dictionary with format subgraph_id: constituent origianl node indices of the subgraph
        e_dict: A dictionary
            A dictionary with format edge_id: (start_node_id, end_node_id)
        """
        s_dict = defaultdict(list)
        v_dict = defaultdict(list)
        e_dict = defaultdict(tuple)

        for bond in mol.GetBonds():
            id = bond.GetIdx()
            startid = bond.GetBeginAtomIdx()
            endid = bond.GetEndAtomIdx()
            e_dict[id] = (startid, endid)
            s_dict[startid].append(id)
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
                if self.merge_method == "edge":
                    s_dict, v_dict, e_dict, max_node, max_edge = self.initialize(mol)
                elif self.merge_method == "node":
                    s_dict, v_dict, e_dict, max_node, max_edge = self.initialize_node(mol)
            
            break_all = 0
            
            if self.merge_method == "edge":
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

            elif self.merge_method == "node":
                for e_id, node_pair in e_dict.items():
                    node_1 = node_pair[0]
                    node_2 = node_pair[1]
                    ori_node_1 = v_dict[node_1]
                    ori_node_2 = v_dict[node_2]
                    m = list(set(ori_node_1+ori_node_2))
                    m_mol = get_fragment_mol(mol, m)
                    m_smiles = get_smiles(m_mol)
                    m_smiles = sanitize_smiles(m_smiles)
                    if m_smiles in self.motif_vocab:
                        merged_node_list = list(set(v_dict[node_1]+v_dict[node_2]))
                        v_dict[max_node] = merged_node_list
                        edge_node_1 = s_dict[node_1]
                        edge_node_2 = s_dict[node_2]
                        need_to_del_edge = []
                        need_to_change_node = []
                        for other_e in list(set(edge_node_1+edge_node_2)):
                            first_node = e_dict[other_e][0]
                            second_node = e_dict[other_e][1]
                            need_to_del_edge.append(other_e)
                            if first_node in [node_1, node_2] and second_node not in [node_1, node_2]:
                                need_to_change_node.append(second_node)
                            elif first_node not in [node_1, node_2] and second_node in [node_1, node_2]:
                                need_to_change_node.append(first_node)
                            elif first_node in [node_1, node_2] and second_node in [node_1, node_2]:
                                pass
                            else:
                                print("Wrong merge!!!")
                                print(stop)
                        
                        count_added_edge = 0
                        for node in need_to_change_node:
                            s_dict[node] = [x for x in s_dict[node] if x not in need_to_del_edge]
                            e_dict[max_edge+count_added_edge] = (node, max_node)
                            s_dict[max_node].append(max_edge+count_added_edge)
                            s_dict[node].append(max_edge+count_added_edge)
                            count_added_edge += 1

                        for item in need_to_del_edge:
                            del e_dict[item]
                        
                        del v_dict[node_1], v_dict[node_2], s_dict[node_1], s_dict[node_2]
                        max_node += 1
                        max_edge += count_added_edge

                        break_all = 1
                        break
                # print("check")
            if break_all == 0:
                break
            iteration += 1

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
                # m_mol = get_fragment_mol(mol, atom_list)
                # m_smiles = get_smiles(m_mol)
                # m_smiles = sanitize_smiles(m_smiles)
                # motif_smiles_list.append(m_smiles)
                # s_id_list.append(subgraph_id)
                if len(atom_list) > 1:
                    m_mol = get_fragment_mol(mol, atom_list)
                    m_smiles = get_smiles(m_mol)
                    m_smiles = sanitize_smiles(m_smiles)
                    motif_smiles_list.append(m_smiles)
                    s_id_list.append(subgraph_id)
                else:
                    try:
                        a = len(s_dict[subgraph_id])
                    except:
                        print(self.smiles_list[i])
                        print(s_dict)
                        print(v_dict)
                        print(e_dict)
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
                                



