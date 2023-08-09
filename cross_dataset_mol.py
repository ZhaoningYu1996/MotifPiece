import torch
from utils.crossdataset import CombinedDataset
from utils.motif_dataset import MotifDataset
from torch_geometric.datasets import Planetoid, MoleculeNet
from utils.model import CGIN, GINModel, Classifier, CrossDatasetsGIN
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

def train(loader):
    model.train()
    # heter_model.train()
    # motif_model.train()
    # raw_model_1.train()
    # raw_model_2.train()
    # classifier1.train()
    # classifier2.train()
    count = 0
    total_loss = 0
    for data in loader:
        data.to(device)
        mask = data.n_id
        # threshold_1 = (mask >= num_nodes) and (mask < num_nodes+num_data1)
        raw_mask_indices_1 = torch.where((mask >= num_motifs) & (mask < num_motifs+num_data1))
        raw_mask_1 = mask[raw_mask_indices_1]-num_motifs
        # raw_mask_indices_1 = threshold_1.nonzero().squeeze().to(device)
        # threshold_2 = (mask >= num_nodes+num_data1) and (mask < num_nodes+num_data1+num_data2)
        raw_mask_indices_2 = torch.where((mask >= num_motifs+num_data1) & (mask < num_motifs+num_data1+num_data2))
        raw_mask_2 = mask[raw_mask_indices_2]-num_motifs-num_data1
        # raw_mask_indices_2 = raw_mask_indices_2.nonzero().squeeze().to(device)
        # threshold_3 = mask < num_nodes
        motif_mask_indices = torch.where(mask < num_motifs)
        motif_mask = mask[motif_mask_indices]
        # motif_mask_indices = motif_mask_indices.nonzero().squeeze().to(device)
        motif_loader = DataLoader(motif_dataset[motif_mask], batch_size=len(motif_dataset[motif_mask]))
        
        for batch in motif_loader:
            motif_data = batch.to(device)

            # motif_x = batch.x
            # motif_edge_index = batch.edge_index
            # motif_edge_attr = batch.edge_attr
            # motif_batch = batch.batch
            # motif_out = motif_model(motif_x, motif_edge_index, motif_edge_attr, motif_batch)
            
        data1_loader = DataLoader(raw_dataset1[raw_mask_1], batch_size=len(raw_dataset1[raw_mask_1]))
        
        for batch in data1_loader:
            raw_data_1 = batch.to(device)
            # raw_x = batch.x
            # raw_edge_index = batch.edge_index
            # raw_edge_attr = batch.edge_attr
            # raw_batch = batch.batch
            # raw_out_1 = raw_model_1(raw_x, raw_edge_index, raw_edge_attr, raw_batch)
            
        data2_loader = DataLoader(raw_dataset2[raw_mask_2], batch_size=len(raw_dataset2[raw_mask_2]))
        for batch in data2_loader:
            raw_data_2 = batch.to(device)
            # raw_x = batch.x
            # raw_edge_index = batch.edge_index
            # raw_edge_attr = batch.edge_attr
            # raw_batch = batch.batch
            # raw_out_2 = raw_model_2(raw_x, raw_edge_index, raw_edge_attr, raw_batch)
        
        # node_input = torch.cat((motif_out, raw_out_1, raw_out_2), dim=0)
        
        # num_dim = raw_out_1.size(1)
        # node_feature = torch.empty((data.n_id.size(0), num_dim)).to(device)
        # node_feature[motif_mask_indices] = motif_out
        # node_feature[raw_mask_indices_1] = raw_out_1
        # node_feature[raw_mask_indices_2] = raw_out_2

        count += 1
        data.to(device)

        # Get motif level graph embeddings
        # rep = heter_model(node_feature, data)

        # train_indices_1 = torch.where(data.n_id[:data.batch_size]<(num_data1+num_motifs))
        # train_indices_2 = torch.where(data.n_id[:data.batch_size]>=(num_data1+num_motifs))

        # pred1 = classifier1(rep[train_indices_1])
        # pred2 = classifier2(rep[train_indices_2])

        pred1, pred2 = model(data, num_motifs, num_data1, motif_data, raw_data_1, raw_data_2, motif_mask_indices, raw_mask_indices_1, raw_mask_indices_2)

        # optimizer_heter.zero_grad()
        # optimizer_motif.zero_grad()
        # optimizer_raw_1.zero_grad()
        # optimizer_raw_2.zero_grad()
        # optimizer_classifier1.zero_grad()
        # optimizer_classifier2.zero_grad()
        optimizer.zero_grad()

        n_id_1 = data.n_id[:data.batch_size][data.n_id[:data.batch_size]<(num_data1+num_motifs)]
        n_id_2 = data.n_id[:data.batch_size][data.n_id[:data.batch_size]>=(num_data1+num_motifs)]-num_data1

        n_id_1.to(device)
        n_id_2.to(device)

        is_valid_1 = labels_1[n_id_1]**2 > 0
        is_valid_2 = labels_2[n_id_2]**2 > 0

        loss_mat_1 = criterion(pred1.double().squeeze(), (labels_1[n_id_1].squeeze().double()+1)/2)
        loss_mat_1 = torch.where(is_valid_1, loss_mat_1, torch.zeros(loss_mat_1.shape).to(loss_mat_1.device).to(loss_mat_1.dtype))
        loss_1 = torch.sum(loss_mat_1)/torch.sum(is_valid_1)

        loss_mat_2 = criterion(pred2.double().squeeze(), (labels_2[n_id_2].squeeze().double()+1)/2)
        loss_mat_2 = torch.where(is_valid_2, loss_mat_2, torch.zeros(loss_mat_2.shape).to(loss_mat_2.device).to(loss_mat_2.dtype))
        loss_2 = torch.sum(loss_mat_2)/torch.sum(is_valid_2)

        loss = loss_1

        total_loss += loss
        loss.backward()  # Derive gradients.
        optimizer.step()
        # optimizer_heter.step()  # Update parameters based on gradients.
        # optimizer_motif.step()
        # optimizer_raw_1.step()
        # optimizer_raw_2.step()
        # optimizer_classifier1.step()
        # optimizer_classifier2.step()


    return total_loss/count

def test(loader):
    model.eval()
    # heter_model.eval()
    # motif_model.eval()
    # raw_model_1.eval()
    # raw_model_2.eval()
    # classifier1.eval()
    # classifier2.eval()
    y_true_1 = []
    y_scores_1 = []
    roc_list_1 = []
    y_true_2 = []
    y_scores_2 = []
    roc_list_2 = []
    with torch.no_grad():
        for data in loader:
            
            data.to(device)
            mask = data.n_id
            # threshold_1 = (mask >= num_nodes) and (mask < num_nodes+num_data1)
            raw_mask_indices_1 = torch.where((mask >= num_motifs) & (mask < num_motifs+num_data1))
            raw_mask_1 = mask[raw_mask_indices_1]-num_motifs
            # raw_mask_indices_1 = threshold_1.nonzero().squeeze().to(device)
            # threshold_2 = (mask >= num_nodes+num_data1) and (mask < num_nodes+num_data1+num_data2)
            raw_mask_indices_2 = torch.where((mask >= num_motifs+num_data1) & (mask < num_motifs+num_data1+num_data2))
            raw_mask_2 = mask[raw_mask_indices_2]-num_motifs-num_data1
            # raw_mask_indices_2 = threshold_2.nonzero().squeeze().to(device)
            # threshold_3 = mask < num_nodes
            motif_mask_indices = torch.where(mask < num_motifs)
            motif_mask = mask[motif_mask_indices]
            # motif_mask_indices = threshold_3.nonzero().squeeze().to(device)
            motif_loader = DataLoader(motif_dataset[motif_mask], batch_size=len(motif_dataset[motif_mask]))
            
            for batch in motif_loader:
                motif_data = batch.to(device)
                # motif_x = batch.x
                # motif_edge_index = batch.edge_index
                # motif_edge_attr = batch.edge_attr
                # motif_batch = batch.batch
                # motif_out = motif_model(motif_x, motif_edge_index, motif_edge_attr, motif_batch)
                
            data1_loader = DataLoader(raw_dataset1[raw_mask_1], batch_size=len(raw_dataset1[raw_mask_1]))
            for batch in data1_loader:
                raw_data_1 = batch.to(device)
                # raw_x = batch.x
                # raw_edge_index = batch.edge_index
                # raw_edge_attr = batch.edge_attr
                # raw_batch = batch.batch
                # raw_out_1 = raw_model_1(raw_x, raw_edge_index, raw_edge_attr, raw_batch)
                
            data2_loader = DataLoader(raw_dataset2[raw_mask_2], batch_size=len(raw_dataset2[raw_mask_2]))
            for batch in data2_loader:
                raw_data_2 = batch.to(device)
                # raw_x = batch.x
                # raw_edge_index = batch.edge_index
                # raw_edge_attr = batch.edge_attr
                # raw_batch = batch.batch
                # raw_out_2 = raw_model_2(raw_x, raw_edge_index, raw_edge_attr, raw_batch)
            
            # num_dim = raw_out_1.size(1)
            # node_feature = torch.empty((data.n_id.size(0), num_dim)).to(device)
            # node_feature[motif_mask_indices] = motif_out
            # node_feature[raw_mask_indices_1] = raw_out_1
            # node_feature[raw_mask_indices_2] = raw_out_2
            # print(node_input.size())
            # print(stop)
            data.to(device)

            # Get motif level graph embeddings
            # rep = heter_model(node_feature, data)

            # indices_1 = torch.where(data.n_id[:data.batch_size]<(num_data1+num_motifs))
            # indices_2 = torch.where(data.n_id[:data.batch_size]>=(num_data1+num_motifs))

            # pred1 = classifier1(rep[indices_1])
            # pred2 = classifier2(rep[indices_2])

            pred1, pred2 = model(data, num_motifs, num_data1, motif_data, raw_data_1, raw_data_2, motif_mask_indices, raw_mask_indices_1, raw_mask_indices_2)

            # n_id_1 = data.n_id[data.n_id<(num_data1+num_nodes)]-num_nodes
            # n_id_2 = data.n_id[data.n_id>=(num_data1+num_nodes)]-num_nodes-num_data1
            n_id_1 = data.n_id[:data.batch_size][data.n_id[:data.batch_size]<(num_data1+num_motifs)]
            n_id_2 = data.n_id[:data.batch_size][data.n_id[:data.batch_size]>=(num_data1+num_motifs)]-num_data1
            # if torch.max(labels_1[n_id_1]) > 1:
            #     print('Error!')
            # if torch.max(labels_2[n_id_2]) > 1:
            #     print('Error!')
            y_true_1.append(labels_1[n_id_1].squeeze().double().view(pred1.shape))
            y_scores_1.append(pred1)
            y_true_2.append(labels_2[n_id_2].squeeze().double().view(pred2.shape))
            y_scores_2.append(pred2)
        y_true_1 = torch.cat(y_true_1, dim=0).cpu().numpy()
        y_scores_1 = torch.cat(y_scores_1, dim=0).cpu().numpy()
        roc_list_1 = []
        for i in range(y_true_1.shape[1]):
            if np.sum(y_true_1[:,i] == 1) > 0 and np.sum(y_true_1[:,i] == -1) > 0:
                is_valid_1 = y_true_1[:,i]**2 > 0
                roc_list_1.append(roc_auc_score((y_true_1[is_valid_1,i] + 1)/2, y_scores_1[is_valid_1,i]))
        y_true_2 = torch.cat(y_true_2, dim=0).cpu().numpy()
        y_scores_2 = torch.cat(y_scores_2, dim=0).cpu().numpy()

        roc_list_2 = []
        for i in range(y_true_2.shape[1]):
            if np.sum(y_true_2[:,i] == 1) > 0 and np.sum(y_true_2[:,i] == -1) > 0:
                is_valid_2 = y_true_2[:,i]**2 > 0
                roc_list_2.append(roc_auc_score((y_true_2[is_valid_2,i] + 1)/2, y_scores_2[is_valid_2,i]))

        test_acc_1 = sum(roc_list_1)/len(roc_list_1)
        test_acc_2 = sum(roc_list_2)/len(roc_list_2)
        return test_acc_1, test_acc_2

# set_seed(0)
data_name = ["bbbp", "clintox"]

dataset = CombinedDataset('combined_data/' + data_name[0] + "_" + data_name[1], data_name, threshold=[50, 5])
heter_data = dataset[0]
motif_smiles = heter_data.motif_smiles
selected_graphs = heter_data.graph_indices
num_motifs = len(motif_smiles)
del heter_data.motif_smiles, heter_data.graph_indices

graph_count = heter_data.num_graph
num_data1 = graph_count[0]
num_data2 = graph_count[1]

labels = heter_data.y
labels_1 = labels[0].to(device)
labels_2 = labels[1].to(device)
print(f"labels size: {labels_1.size()}, {labels_2.size()}")

smiles_list_1 = heter_data.graph_smiles[:graph_count[0]]
smiles_list_2 = heter_data.graph_smiles[graph_count[0]:graph_count[0]+graph_count[1]]
print(graph_count)
# print(heter_data.y)
# print(len(smiles_list_1), len(smiles_list_2))
# print(heter_data)


motif_dataset = MotifDataset("motif_data/"+data_name[0]+"_"+data_name[1], motif_smiles)
motif_batch_size = len(motif_dataset)
print(f"motif batch size: {motif_batch_size}")

motif_loader = DataLoader(motif_dataset, batch_size=num_motifs)
raw_dataset1 = MoleculeNet('dataset/', data_name[0])[selected_graphs[0]]
raw_dataset2 = MoleculeNet('dataset/', data_name[1])[selected_graphs[1]]
# data1_loader = DataLoader(raw_dataset1, batch_size=len(raw_dataset1))
# data2_loader = DataLoader(raw_dataset2, batch_size=len(raw_dataset2))
# print(motif_batch_size)
# print(motif_dataset[0])
# print(motif_dataset[-1])
# print(stop)

train_idx_1, val_idx_1, test_idx_1 = scaffold_split(smiles_list_1)
train_idx_2, val_idx_2, test_idx_2 = scaffold_split(smiles_list_2)

train_idx_1 = torch.tensor([x + num_motifs for x in train_idx_1])
val_idx_1 = torch.tensor([x + num_motifs for x in val_idx_1])
test_idx_1 = torch.tensor([x + num_motifs for x in test_idx_1])

train_idx_2 = torch.tensor([x + num_motifs + num_data1 for x in train_idx_2])
val_idx_2 = torch.tensor([x + num_motifs + num_data1 for x in val_idx_2])
test_idx_2 = torch.tensor([x + num_motifs + num_data1 for x in test_idx_2])

train_idx = torch.cat((train_idx_1, train_idx_2), dim=0)
val_idx = torch.cat((val_idx_1, val_idx_2), dim=0)
test_idx = torch.cat((test_idx_1, test_idx_2), dim=0)

perm = torch.randperm(train_idx.size(0))
train_idx = train_idx[perm]
perm = torch.randperm(val_idx.size(0))
val_idx = val_idx[perm]
perm = torch.randperm(test_idx.size(0))
test_idx = test_idx[perm]

batch_size = 64
heter_data.x = heter_data.x.contiguous()
heter_data.edge_index = heter_data.edge_index.contiguous()
train_loader = NeighborLoader(
heter_data,
num_neighbors=[30] * 4,
batch_size=batch_size,
input_nodes=train_idx,
)
val_loader = NeighborLoader(
heter_data,
num_neighbors=[30] * 4,
batch_size=batch_size,
input_nodes=val_idx,
)
test_loader = NeighborLoader(
heter_data,
num_neighbors=[30] * 4,
batch_size=batch_size,
input_nodes=test_idx,
)

# heter_model = CGIN(300, 300, 2).to(device)
# motif_model = GINModel(1, 1, 300, 300, 2, 0.5).to(device)
# raw_model_1 = GINModel(9, 3, 300, 300, 2, 0.5).to(device)
# raw_model_2 = GINModel(9, 3, 300, 300, 2, 0.5).to(device)
# classifier1 = Classifier(300, 1).to(device)
# classifier2 = Classifier(300, 2).to(device)
model = CrossDatasetsGIN(300, 1, 2, device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
# optimizer_heter = torch.optim.Adam(heter_model.parameters(), lr=0.0002)
# optimizer_motif = torch.optim.Adam(motif_model.parameters(), lr=0.0002)
# optimizer_raw_1 = torch.optim.Adam(raw_model_1.parameters(), lr=0.0002)
# optimizer_raw_2 = torch.optim.Adam(raw_model_2.parameters(), lr=0.0002)
# optimizer_classifier1 = torch.optim.Adam(classifier1.parameters(), lr=0.0002)
# optimizer_classifier2 = torch.optim.Adam(classifier2.parameters(), lr=0.0002)

criterion = torch.nn.BCEWithLogitsLoss(reduction = "none")

num_epoch = 100
best_val_1 = 0.0
best_val_2 = 0.0
best_test_1 = 0.0
best_test_2 = 0.0
best_train_1 = 0.0
best_train_2 = 0.0
for epoch in range(1, num_epoch+1):
    loss = train(train_loader)
    train_acc_1, train_acc_2 = test(train_loader)
    val_acc_1, val_acc_2 = test(val_loader)
    test_acc_1, test_acc_2 = test(test_loader)
    if val_acc_1 > best_val_1:
        best_train_1 = train_acc_1
        best_val_1 = val_acc_1
        best_test_1 = test_acc_1
    if val_acc_2 > best_val_2:
        best_train_2 = train_acc_2
        best_val_2 = val_acc_2
        best_test_2 = test_acc_2
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc 1: {train_acc_1:.4f}, Validate acc 1: {val_acc_1}, Test acc 1: {test_acc_1:.4f}, Train acc 2: {train_acc_2}, Validate acc 2: {val_acc_2}, Test acc 2: {test_acc_2}.')

print(f'Best Test acc 1: {best_test_1} Best Test acc 2: {best_test_2}.')
print(f"Best Val acc 1: {best_val_1}, Best Val acc 2: {best_val_2}")
print(f'Best Train acc 1: {best_train_1} Best Train acc 2: {best_train_2}.')

