from utils.utils import get_fragment_mol, get_smiles, sanitize_smiles, get_mol, sanitize
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000

class MotifPiece:
    def __init__(self, dataset, pre_transform = None):
        self.pre_transform = None
        self.dataset_size = 0
        self.threshold = 100
        self.motif_vocab = {}
        self.process(dataset)
        print(f"motif vocabulary: {self.motif_vocab}")
    
    def process(self, dataset):
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
            print(motif_count)
            
            
            if motif_count[selected_motif] < 5:
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
                            deleted_subgraph_set = []
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

                                            new_edge = set(s_dict[e1[0]]+s_dict[e1[1]]+s_dict[e2[0]]+s_dict[e2[1]])
                                            new_edge.remove(edge_list[j])
                                            new_edge.remove(edge_list[k])
                                            new_edge = list(new_edge)
                                            s_dict[key+1] = new_edge

                                            for e in new_edge:
                                                edge = e_dict[e]
                                                for element in edge:
                                                    if element not in m:
                                                        e_dict[e] = (element, key+1)

                                            deleted_subgraph_set += list(set([e1[0], e1[1], e2[0], e2[1]]))

                                            for key in deleted_subgraph_set:
                                                del s_dict[key]
                                            del e_dict[edge_list[j]], e_dict[edge_list[k]]
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


