import torch
from utils.combined_dataset import CombinedDataset
from utils.motif_dataset import MotifDataset
from torch_geometric.datasets import Planetoid, MoleculeNet
from utils.model import GCN, GIN, Classifier, MTL, GINModel
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.splitter import scaffold_split
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import os
import random
from tqdm import tqdm
from utils.raw_dataset import RawDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

def create_node_folds(selected_nodes, num_folds=10):
    node_indices = selected_nodes
    np.random.shuffle(node_indices)
    return np.array_split(node_indices, num_folds)

def train(data, mask1, mask2):
    model.train()
    motif_model.train()
    data1_model.train()
    data2_model.train()
    for batch in motif_loader:
        batch.to(device)
        motif_x = batch.x
        motif_edge_index = batch.edge_index
        motif_edge_attr = batch.edge_attr
        motif_batch = batch.batch
        motif_out = motif_model(motif_x, motif_edge_index, motif_edge_attr, motif_batch)
    for batch in data1_loader:
        batch.to(device)
        data1_x = batch.x
        data1_edge_index = batch.edge_index
        data1_edge_attr = batch.edge_attr
        data1_batch = batch.batch
        data1_out = data1_model(data1_x, data1_edge_index, data1_edge_attr, data1_batch)
    for batch in data2_loader:
        batch.to(device)
        data2_x = batch.x
        data2_edge_index = batch.edge_index
        data2_edge_attr = batch.edge_attr
        data2_batch = batch.batch
        data2_out = data2_model(data2_x, data2_edge_index, data2_edge_attr, data2_batch)
    # motif_emb = data1_out[:num_nodes]
    node_feature = torch.cat((motif_out, data1_out, data2_out), dim=0)
    out1, out2 = model(node_feature, data)  # Perform a single forward pass.

    
    optimizer_motif.zero_grad()
    optimizer_data1.zero_grad()
    optimizer_data2.zero_grad()
    optimizer.zero_grad()
    # optimizer_linear_1.zero_grad()
    # optimizer_linear_2.zero_grad()
    # optimizer_gnn.zero_grad()  # Clear gradients.
    # optimizer_linear_1.zero_grad()
    # optimizer_linear_2.zero_grad()
    loss = 1*criterion(out1[mask1], data.y[mask1]) + 1*criterion(out2[mask2], data.y[mask2])  # TUDataset
    
    loss.backward()  # Derive gradients.
    # Update parameters based on gradients.
    optimizer_motif.step()
    optimizer_data1.step()
    optimizer_data2.step()
    optimizer.step()
    # optimizer_linear_1.step()
    # optimizer_linear_2.step()
    # optimizer_gnn.step()

    return loss

def test(data, mask1, mask2):
    model.eval()
    motif_model.eval()
    data1_model.eval()
    data2_model.eval()
#   classifier1.eval()
#   classifier2.eval()
    with torch.no_grad():
        for batch in motif_loader:
            batch.to(device)
            motif_x = batch.x
            motif_edge_index = batch.edge_index
            motif_edge_attr = batch.edge_attr
            motif_batch = batch.batch
            motif_out = motif_model(motif_x, motif_edge_index, motif_edge_attr, motif_batch)
        for batch in data1_loader:
            batch.to(device)
            data1_x = batch.x
            data1_edge_index = batch.edge_index
            data1_edge_attr = batch.edge_attr
            data1_batch = batch.batch
            data1_out = data1_model(data1_x, data1_edge_index, data1_edge_attr, data1_batch)
        for batch in data2_loader:
            batch.to(device)
            data2_x = batch.x
            data2_edge_index = batch.edge_index
            data2_edge_attr = batch.edge_attr
            data2_batch = batch.batch
            data2_out = data2_model(data2_x, data2_edge_index, data2_edge_attr, data2_batch)
        # motif_emb = data1_out[:num_nodes]
        # node_feature = torch.cat((motif_emb, data1_out[num_nodes:], data2_out[num_nodes:]), dim=0)
        node_feature = torch.cat((motif_out, data1_out, data2_out), dim=0)
        out1, out2 = model(node_feature, data)
        # out1 = classifier1(out)
        # out2 = classifier2(out)
        pred1 = out1.argmax(dim=1)  # Use the class with highest probability.
        pred2 = out2.argmax(dim=1)
        test_correct_1 = pred1[mask1] == data.y[mask1]  # Check against ground-truth labels.
        test_correct_2 = pred2[mask2] == data.y[mask2]
        test_acc_1 = int(test_correct_1.sum()) / len(mask1) # Derive ratio of correct predictions.
        test_acc_2 = int(test_correct_2.sum()) / len(mask2)
        return test_acc_1, test_acc_2

# set_seed(0)
data_name = ["PTC_FM", "PTC_FR"]
# num_data1 = 297
# num_data2 = 210
num_nodes = 107
# num_nodes = 103                   # PTC_FM, PTC_MR
# num_nodes = 101                   # PTC_FM, PTC_MM
# num_nodes = 108                   # PTC_FR, PTC_MM
# num_nodes = 104                   # PTC_MR, PTC_MM
# num_nodes = 106                   # PTC_MR, PTC_FR
# num_nodes = 98                    # DHFR_MD, ER_MD
# num_nodes = 73                    # DHFR_MD, COX2_MD
# num_nodes = 150                   # COX2_MD, ER_MD
# num_nodes = 77                    # COX2_MD, BZR_MD
# num_nodes = 105                   # ER_MD, BZR_MD
# for data_name = 
if data_name == "bbbp":
    num_nodes = 600
elif data_name == "toxcast":
    num_nodes = 1100
elif data_name == "PTC_MR":
    num_nodes = 160
elif data_name == "COX2":
    num_nodes = 60
elif data_name == "COX2_MD":
    num_nodes = 90
elif data_name == "BZR":
    num_nodes = 102
elif data_name == "BZR_MD":
    num_nodes = 150
# num_nodes = 2000
dataset = CombinedDataset('combined_data/' + data_name[0] + "_" + data_name[1], data_name[0], data_name[1], num_nodes)
data = dataset[0]
print(data)
graph_count = data.num_graphs
num_data1 = graph_count[0]
num_data2 = graph_count[1]

motif_dataset = MotifDataset("motif_data/"+data_name[0]+"_"+data_name[1])
motif_batch_size = len(motif_dataset)
# if num_nodes+num_data1+num_data2 != motif_batch_size:
#     print("Error!")
#     print(stop)

if num_data1 != graph_count[0] or num_data2 != graph_count[1]:
    print("Hugh Error!")
    print(stop)

motif_loader = DataLoader(motif_dataset, batch_size=num_nodes)
raw_dataset1 = RawDataset("raw_data/"+data_name[0])
raw_dataset2 = RawDataset("raw_data/"+data_name[1])
data1_loader = DataLoader(raw_dataset1, batch_size=len(raw_dataset1))
data2_loader = DataLoader(raw_dataset2, batch_size=len(raw_dataset2))
print(motif_batch_size)
print(motif_dataset[0])
print(motif_dataset[-1])
# print(stop)


start_index_1, end_index_1 = num_nodes, num_nodes+graph_count[0]
start_index_2, end_index_2 = end_index_1, end_index_1+graph_count[1]

selected_nodes_1 = np.arange(start_index_1, end_index_1)
num_folds = 10
node_folds = create_node_folds(selected_nodes_1, num_folds)
splits_1 = []
for fold, test_nodes_1 in enumerate(node_folds):
    train_nodes_1 = np.concatenate([node_folds[i] for i in range(num_folds) if i != fold])
    splits_1.append((train_nodes_1, test_nodes_1))

selected_nodes_2 = np.arange(start_index_2, end_index_2)
num_folds = 10
node_folds = create_node_folds(selected_nodes_2, num_folds)
splits_2 = []
for fold, test_nodes_2 in enumerate(node_folds):
    train_nodes_2 = np.concatenate([node_folds[i] for i in range(num_folds) if i != fold])
    splits_2.append((train_nodes_2, test_nodes_2))
    # print(train_nodes_2.shape, test_nodes_2.shape)
    # print(stop)
train_acc_list_1 = []
test_acc_list_1 = []
test_acc_list_2 = []
num_epoch = 2000
data.to(device)
for i, (split_1, split_2) in enumerate(zip(splits_1, splits_2)):
    train_mask_1 = torch.tensor(split_1[0]).to(device)
    train_mask_2 = torch.tensor(split_2[0]).to(device)
    test_mask_1 = torch.tensor(split_1[1]).to(device)
    test_mask_2 = torch.tensor(split_2[1]).to(device)
    best_test_1 = 0.0
    best_test_2 = 0.0
    best_train_1 = 0.0
    best_train_2 = 0.0
    # model = GIN(num_nodes, 16, 2, 6).to(device)
    model = MTL(16, 16, 2, 3).to(device)
    motif_model = GINModel(1, 1, 16, 16, 2, 0).to(device)
    data1_model = GINModel(1, 1, 16, 16, 2, 0).to(device)
    data2_model = GINModel(1, 1, 16, 16, 2, 0).to(device)
    # classifier1 = Classifier(16, 2).to(device)
    # classifier2 = Classifier(16, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer_motif = torch.optim.Adam(motif_model.parameters(), lr=0.01)
    optimizer_data1 = torch.optim.Adam(data1_model.parameters(), lr=0.01)
    optimizer_data2 = torch.optim.Adam(data2_model.parameters(), lr=0.01)
    # optimizer_gnn = torch.optim.Adam(model.parameters(), lr=0.005)
    # optimizer_linear_1 = torch.optim.Adam(model.classifier1.parameters(), lr=0.01)
    # optimizer_linear_2 = torch.optim.Adam(model.classifier2.parameters(), lr=0.02)
    # optimizer = torch.optim.Adam(list(model.parameters())+list(classifier1.parameters())+list(classifier2.parameters()), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, num_epoch)):

        loss = train(data, train_mask_1, train_mask_2)
        # scheduler.step()
        train_acc_1, train_acc_2 = test(data, train_mask_1, train_mask_2)
        test_acc_1, test_acc_2 = test(data, test_mask_1, test_mask_2)
        if test_acc_1 > best_test_1:
            best_test_1 = test_acc_1
        if test_acc_2 > best_test_2:
            best_test_2 = test_acc_2
        if train_acc_1 > best_train_1:
            best_train_1 = train_acc_1
        if train_acc_2 > best_train_2:
            best_train_2 = train_acc_2
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc 1: {train_acc_1:.4f}, Test acc 1: {test_acc_1:.4f}, Train acc 2: {train_acc_2:.4f}, Test acc 2: {test_acc_2:.4f}.')
    test_acc_list_1.append(best_test_1)
    test_acc_list_2.append(best_test_2)
    print(f'Best Test acc 1: {best_test_1} Best Test acc 2: {best_test_2}.')
    print(f'Best Train acc 1: {best_train_1} Best Train acc 2: {best_train_2}.')
mean_acc_1 = np.mean(test_acc_list_1)
std_1 = np.std(test_acc_list_1)
mean_acc_2 = np.mean(test_acc_list_2)
std_2 = np.std(test_acc_list_2)
print(f"Mean acc 1: {mean_acc_1}, std 1: {std_1}, Mean acc 2: {mean_acc_2}, std 2: {std_2}.")
