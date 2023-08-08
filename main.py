import torch
from utils.hmdataset import HeterTUDataset
from utils.motif_dataset import MotifDataset
# from utils.raw_dataset import RawDataset
from torch_geometric.datasets import MoleculeNet, TUDataset
from utils.model import GIN, GINModel
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.splitter import scaffold_split
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.data import Batch
import random
import os
from tqdm import tqdm


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

def train_mol(loader):
    # atom_model.train()
    motif_model.train()
    raw_model.train()
    heter_model.train()
    # n_f_model.train()
    # classifier.train()
    total_loss = 0
    count = 0

    for data in loader:
        data.to(device)
        mask = data.n_id
        threshold_1 = mask >= num_motifs
        raw_mask = mask[threshold_1]-num_motifs
        raw_mask_indices = threshold_1.nonzero().squeeze().to(device)
        threshold_2 = mask < num_motifs
        motif_mask = mask[threshold_2]
        motif_mask_indices = threshold_2.nonzero().squeeze().to(device)
        motif_loader = DataLoader(motif_dataset[motif_mask], batch_size=len(motif_dataset[motif_mask]))
        
        for batch in motif_loader:
            batch.to(device)
            motif_x = batch.x
            motif_edge_index = batch.edge_index
            motif_edge_attr = batch.edge_attr
            motif_batch = batch.batch
            motif_out = motif_model(motif_x, motif_edge_index, motif_edge_attr, motif_batch)
            # motif_out = motif_model(motif_x, motif_edge_index, motif_batch)
        
        raw_loader = DataLoader(graph_dataset[raw_mask], batch_size=len(graph_dataset[raw_mask]))
        for batch in raw_loader:
            batch.to(device)
            raw_x = batch.x
            raw_edge_index = batch.edge_index
            raw_edge_attr = batch.edge_attr
            raw_batch = batch.batch
            raw_out = raw_model(raw_x, raw_edge_index, raw_edge_attr, raw_batch)
            # raw_out = raw_model(raw_x, raw_edge_index, raw_batch)

        num_dim = raw_out.size(1)
        node_feature = torch.empty((data.n_id.size(0), num_dim)).to(device)
        node_feature[motif_mask_indices] = motif_out
        node_feature[raw_mask_indices] = raw_out
        
        # node_feature = data.x

        # node_feature = torch.cat((motif_out, raw_out), dim=0)

        count += 1
        
        # n_f = n_f_model(data.x)
        # node_feature = torch.cat((node_feature, n_f), dim=1)

        # Get motif level graph embeddings
        pred = heter_model(node_feature, data)[:data.batch_size]

        # # Create atom level batch based on sampled target nodes in hetergeneous graph
        # selected_indices = data.n_id[:data.batch_size]
        # selected_graphs = [graph_dataset[i-num_nodes] for i in selected_indices]
        # batch = Batch.from_data_list(selected_graphs).to(device)
        # if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
        #     continue

        # # Get atom level graph embeddings
        # atom_out = atom_model(batch)

        # # Concatenate atom level embeddings and motif level embeddings
        # out = torch.cat((heter_out[:data.batch_size], atom_out), dim=1)
        # pred = classifier(out)
        optimizer_heter.zero_grad()
        # optimizer_atom.zero_grad()
        optimizer_motif.zero_grad()
        optimizer_raw.zero_grad()
        # optimizer_n_f.zero_grad()
        # optimizer_classifier.zero_grad()

        is_valid = data.y[:data.batch_size]**2 > 0
        loss_mat = criterion(pred.double().squeeze(), (data.y[:data.batch_size].squeeze().double()+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat)/torch.sum(is_valid)

        total_loss += loss
        loss.backward()  # Derive gradients.
        optimizer_heter.step()  # Update parameters based on gradients.
        optimizer_motif.step()
        optimizer_raw.step()
        # optimizer_n_f.step()
        # optimizer_atom.step()
        # optimizer_classifier.step()
    return total_loss/count

def test_mol(loader):
    heter_model.eval()
    # atom_model.eval()
    motif_model.eval()
    raw_model.eval()
    # n_f_model.eval()
    # classifier.eval()
    
    y_true = []
    y_scores = []
    roc_list = []
    with torch.no_grad():
        for data in loader:
            data.to(device)
            mask = data.n_id
            threshold_1 = mask >= num_motifs
            raw_mask = mask[threshold_1]-num_motifs
            raw_mask_indices = threshold_1.nonzero().squeeze().to(device)
            threshold_2 = mask < num_motifs
            motif_mask = mask[threshold_2]
            motif_mask_indices = threshold_2.nonzero().squeeze().to(device)
            motif_loader = DataLoader(motif_dataset[motif_mask], batch_size=len(motif_dataset[motif_mask]))
            for batch in motif_loader:
                batch.to(device)
                motif_x = batch.x
                motif_edge_index = batch.edge_index
                motif_edge_attr = batch.edge_attr
                motif_batch = batch.batch
                motif_out = motif_model(motif_x, motif_edge_index, motif_edge_attr, motif_batch)
                # motif_out = motif_model(motif_x, motif_edge_index, motif_batch)
            
            raw_loader = DataLoader(graph_dataset[raw_mask], batch_size=len(graph_dataset[raw_mask]))
            for batch in raw_loader:
                batch.to(device)
                raw_x = batch.x
                raw_edge_index = batch.edge_index
                raw_edge_attr = batch.edge_attr
                raw_batch = batch.batch
                raw_out = raw_model(raw_x, raw_edge_index, raw_edge_attr, raw_batch)
                # raw_out = raw_model(raw_x, raw_edge_index, raw_batch)
            # node_feature = torch.cat((motif_out, raw_out), dim=0)
            num_dim = raw_out.size(1)
            node_feature = torch.empty((data.n_id.size(0), num_dim)).to(device)
            node_feature[motif_mask_indices] = motif_out
            node_feature[raw_mask_indices] = raw_out

            # node_feature = data.x
        
            
            # n_f = n_f_model(data.x)
            # node_feature = torch.cat((node_feature, n_f), dim=1)
            # selected_indices = data.n_id[:data.batch_size]
            # selected_graphs = [graph_dataset[i-num_nodes] for i in selected_indices]
            # batch = Batch.from_data_list(selected_graphs).to(device)
            
            pred = heter_model(node_feature, data)[:data.batch_size]
            # atom_out = atom_model(batch)
            # out = torch.cat((heter_out[:data.batch_size], atom_out), dim=1)
                # pred = classifier(out)
            y_true.append(data.y[:data.batch_size].view(pred.shape))
            y_scores.append(pred)
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
        roc_list = []
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
    return sum(roc_list)/len(roc_list) #y_true.shape[1]
        

def train(data, mask):
    model.train()
    motif_model.train()
    raw_model.train()
    # n_f_model.train()
    # print(data_type)
    # print(data.y)
    for batch in motif_loader:
        batch.to(device)
        motif_x = batch.x
        motif_edge_index = batch.edge_index
        motif_edge_attr = batch.edge_attr
        motif_batch = batch.batch
        motif_out = motif_model(motif_x, motif_edge_index, motif_edge_attr, motif_batch)
  
    for batch in raw_loader:
        batch.to(device)
        raw_x = batch.x
        raw_edge_index = batch.edge_index
        raw_edge_attr = batch.edge_attr
        raw_batch = batch.batch
        raw_out = raw_model(raw_x, raw_edge_index, raw_edge_attr, raw_batch)
    node_feature = torch.cat((motif_out, raw_out), dim=0)
    # n_f = n_f_model(data.x)
    # node_feature = torch.cat((node_feature, n_f), dim=1)

    out = model(node_feature, data)
    optimizer.zero_grad()  # Clear gradients.
    optimizer_motif.zero_grad()
    optimizer_raw.zero_grad()
    # optimizer_n_f.zero_grad()
    loss = criterion(out[mask], data.y[mask].squeeze())  # TUDataset
    
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer_motif.step()
    optimizer_raw.step()
    # optimizer_n_f.step()
    return loss

def test(data, mask):
      model.eval()
      motif_model.eval()
      raw_model.eval()
    #   n_f_model.eval()
      with torch.no_grad():
        for batch in motif_loader:
            batch.to(device)
            motif_x = batch.x
            motif_edge_index = batch.edge_index
            motif_edge_attr = batch.edge_attr
            motif_batch = batch.batch
            motif_out = motif_model(motif_x, motif_edge_index, motif_edge_attr, motif_batch)
        for batch in raw_loader:
            batch.to(device)
            raw_x = batch.x
            raw_edge_index = batch.edge_index
            raw_edge_attr = batch.edge_attr
            raw_batch = batch.batch
            raw_out = raw_model(raw_x, raw_edge_index, raw_edge_attr, raw_batch)
        node_feature = torch.cat((motif_out, raw_out), dim=0)
        # n_f = n_f_model(data.x)
        # node_feature = torch.cat((node_feature, n_f), dim=1)
        out = model(node_feature, data)
        
        # out = model(data.x, data, False)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[mask] == data.y[mask].squeeze()  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / len(mask)  # Derive ratio of correct predictions.
        return test_acc


# set_seed(0)

data_name ="PTC_FR"
if data_name == "bbbp":
    # num_nodes = 3153         # bridge
    # num_nodes = 2242         # BRICS
    # num_nodes = 287           # MotifPiece
    # num_nodes = 284
    num_nodes = 275          # hmgnn
    num_classes = 1
elif data_name == "toxcast":
    num_nodes = 595
    # num_nodes = 626          # MotifPiece
    # num_nodes = 7659         # BRICS
    # num_nodes = 9723         # bridge
    num_classes = 617


    # num_nodes = 429          #bridge
    # num_nodes = 330          # BRICS
    # num_nodes = 93           # hmgnn
    

elif data_name == "COX2":
    num_nodes = 60

elif data_name == "BZR":
    num_nodes = 102
elif data_name == "ER_MD":
    num_nodes = 599
    # num_nodes = 456           # BRICS
    # num_nodes = 74            # R&B
    # num_nodes = 77 # threshold = 100
    # num_nodes = 85 # threshold = 10
    start_index, end_index = num_nodes, num_nodes+422
elif data_name == "MUTAG":
    num_nodes = 60
elif data_name == "NCI1":
    num_nodes = 150
elif data_name == "sider":
    # num_nodes = 3091             # bridge
    # num_nodes = 2005             # BRICS
    # num_nodes = 354              # MotifPiece
    num_nodes = 345
    num_classes = 27
elif data_name == "clintox":
    num_nodes = 2886             # bridge
    # num_nodes = 1961             # BRICS
    # num_nodes = 341              # hmgnn
    # num_nodes = 350              # MorifPiece
    # num_nodes = 229
    num_classes = 2
elif data_name == "bace":
    num_nodes = 190              # MotifPiece
    # num_nodes = 300
    # num_nodes = 179              # hmgnn
    # num_nodes = 1436             # BRICS
    # num_nodes = 2499             # bridge
    num_classes = 1
elif data_name == "hiv":
    num_nodes = 2581
    num_classes = 1
elif data_name == "muv":
    num_nodes = 648
    num_classes = 17
elif data_name == "tox21":
    num_nodes = 577              # MotifPiece
    # num_nodes = 547              # hmgnn
    # num_nodes = 7120             # BRICS
    # num_nodes = 8773             # bridge
    num_classes = 12
# num_nodes = 2000
dataset = HeterTUDataset('dataset/' + data_name, data_name)
heter_data = dataset[0]
motif_smiles = heter_data.motif_smiles
selected_graphs = heter_data.graph_indices
num_motifs = len(motif_smiles)
del heter_data.motif_smiles, heter_data.graph_indices
# print(heter_data.edge_index)
motif_dataset = MotifDataset("motif_data/"+data_name, motif_smiles)
motif_batch_size = len(motif_dataset)
print(f"number of motifs: {motif_batch_size}")

if data_name in ["PTC_MR", "Mutagenicity", "COX2_MD", "COX2", "BZR", "BZR_MD", "DHFR_MD", "ER_MD", "PTC_FR", "PTC_MM", "PTC_FM"]:
    data_type = "TUData"
    if data_name == "PTC_MR":
        start_index, end_index = num_motifs, num_motifs+307
        # start_index, end_index = num_nodes, num_nodes+348
    elif data_name == "PTC_MM":
        num_graphs = 290
        start_index, end_index = num_motifs, num_motifs+num_graphs
    elif data_name == "PTC_FR":
        start_index, end_index = num_motifs, num_motifs+309
    elif data_name == "PTC_FM":
        num_graphs = 305
        start_index, end_index = num_motifs, num_motifs+num_graphs
    elif data_name == "MUTAG":
        start_index, end_index = num_motifs, num_motifs
    elif data_name == "Mutagenicity":
        start_index, end_index = num_motifs, num_motifs+3552
    elif data_name == "COX2_MD":
        start_index, end_index = num_motifs, num_motifs+297
    elif data_name == "COX2":
        start_index, end_index = num_motifs, num_motifs+467
    elif data_name == "BZR":
        start_index, end_index = num_motifs, num_motifs+405
    elif data_name == "BZR_MD":
        start_index, end_index = num_motifs, num_motifs+210
    elif data_name == "DHFR_MD":
        start_index, end_index = num_motifs, num_motifs+345

    # print(selected_graphs)
    raw_dataset = TUDataset("raw_data/"+data_name, data_name)[selected_graphs]
    raw_loader = DataLoader(raw_dataset, batch_size=len(raw_dataset))
    motif_loader = DataLoader(motif_dataset, batch_size=len(motif_dataset))
    selected_nodes = np.arange(start_index, end_index)
    num_folds = 10
    node_folds = create_node_folds(selected_nodes, num_folds)
    splits = []

    for fold, test_nodes in enumerate(node_folds):
        train_nodes = np.concatenate([node_folds[i] for i in range(num_folds) if i != fold])
        splits.append((train_nodes, test_nodes))


    train_acc_list = []
    test_acc_list = []
    num_epoch = 2000
    heter_data.to(device)
    for i, split in enumerate(splits):
        train_mask = torch.tensor(split[0]).to(device)
        test_mask = torch.tensor(split[1]).to(device)
        best_test = 0.0
        dim_n_f = 2
        dim_motif = 16
        # model = GCN(4000, 16, 2).to(device)
        # model = GIN(93, 16, 2, 3).to(device)
        model = GIN(dim_motif, dim_motif, 2, 3).to(device)
        raw_model = GINModel(1, 1, dim_motif, dim_motif, 2, 0.0).to(device)
        motif_model = GINModel(1, 1, dim_motif, dim_motif, 2, 0.0).to(device)
        # n_f_model = torch.nn.Linear(num_nodes, dim_n_f).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer_motif = torch.optim.Adam(motif_model.parameters(), lr=0.01)
        optimizer_raw = torch.optim.Adam(raw_model.parameters(), lr=0.01)
        # optimizer_n_f = torch.optim.Adam(n_f_model.parameters(), lr=0.001)
        # Set up the learning rate scheduler
        # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss()

        # print(f"train_mask: {train_mask}")
        # print(f"Test mask: {test_mask}")

        for epoch in tqdm(range(1, num_epoch)):
            loss = train(heter_data, train_mask)
            # scheduler.step()
            
            train_acc = test(heter_data, train_mask)
            test_acc = test(heter_data, test_mask)
            if test_acc > best_test:
                best_test = test_acc
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}.')
        print(f"Best test accuracy: {best_test}!")
        test_acc_list.append(best_test)
    mean_acc = np.mean(test_acc_list)
    std = np.std(test_acc_list)
    print(f"Mean acc: {mean_acc}, std: {std}.")

elif data_name in ["bbbp", "bace", "clintox", "muv", "hiv", "sider", "tox21", "toxcast"]:
    print("hh")
    data_type = "MolNet"
    graph_smiles = heter_data.graph_smiles
    print(f"Number of samples: {len(graph_smiles)}")
    graph_dataset = MoleculeNet('raw_data/', data_name)[selected_graphs]
    sample_data = graph_dataset[0]
    print(f"number of raw graphs: {len(graph_dataset)}")
    
    train_idx, val_idx, test_idx = scaffold_split(graph_smiles)

    train_idx = torch.tensor([x + num_motifs for x in train_idx])
    val_idx = torch.tensor([x + num_motifs for x in val_idx])
    test_idx = torch.tensor([x + num_motifs for x in test_idx])
    perm = torch.randperm(train_idx.size(0))
    train_idx = train_idx[perm]
    perm = torch.randperm(val_idx.size(0))
    val_idx = val_idx[perm]
    perm = torch.randperm(test_idx.size(0))
    test_idx = test_idx[perm]
    train_mask = torch.zeros((num_motifs+len(graph_smiles)), dtype=torch.bool)
    # print(f"train idx: {train_idx}")
    # print(f"val idx: {val_idx}")
    # print(f"test idx: {test_idx}")
    # print(stop)
    # for i in train_idx:
    #     train_mask[i] = True
    # val_mask = torch.zeros((num_nodes+len(smiles_list)), dtype=torch.bool)
    # for i in val_idx:
    #     val_mask[i] = True
    # test_mask = torch.zeros((num_nodes+len(smiles_list)), dtype=torch.bool)
    # for i in test_idx:
    #     test_mask[i] = True
    # train_mask = torch.tensor(train_idx)
    # val_mask = torch.tensor(val_idx)
    # test_mask = torch.tensor(test_idx)
    del heter_data.graph_smiles
    # print(data.edge_index)
    # print(data.edge_index.contiguous())
    batch_size = 32
    heter_data.x = heter_data.x.contiguous()
    heter_data.edge_index = heter_data.edge_index.contiguous()
    train_loader = NeighborLoader(
    heter_data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30]*4,
    # Use a batch size of 128 for sampling training nodes
    batch_size=batch_size,
    input_nodes=train_idx,
    )
    val_loader = NeighborLoader(
    heter_data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30]*4,
    # Use a batch size of 128 for sampling training nodes
    batch_size=batch_size,
    input_nodes=val_idx,
    )
    test_loader = NeighborLoader(
    heter_data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30]*4,
    # Use a batch size of 128 for sampling training nodes
    batch_size=batch_size,
    input_nodes=test_idx,
    )

    dim_motif = 300
    dim_n_f = 300

    heter_model = GIN(dim_motif, dim_motif, num_classes, 2).to(device)
    motif_model = GINModel(1, 1, dim_motif, dim_motif, 2, 0.5).to(device)
    raw_model = GINModel(9, 3, dim_motif, dim_motif, 2, 0.5).to(device)

    # heter_model = GCN(dim_motif, num_classes, 2, 0.5).to(device)
    # motif_model = GCNModel(1, dim_motif, dim_motif, 3, 0.5).to(device)
    # raw_model = GCNModel(9, dim_motif, dim_motif, 3, 0.5).to(device)

    optimizer_heter = torch.optim.Adam(heter_model.parameters(), lr=0.00005)
    # optimizer_atom = torch.optim.Adam(atom_model.parameters(), lr=0.001)
    optimizer_motif = torch.optim.Adam(motif_model.parameters(), lr=0.00005)
    optimizer_raw = torch.optim.Adam(raw_model.parameters(), lr=0.00005)
    # optimizer_n_f = torch.optim.Adam(n_f_model.parameters(), lr=0.00005)
    # optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=0.001)
    # optimizer2 = torch.optim.Adam(atom_model.parameters(), lr=0.001)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss(reduction = "none")
    num_epoch = 100
    best_val = 0.0
    best_test = 0.0
    for epoch in range(1, num_epoch+1):
        loss = train_mol(train_loader)
        train_acc = test_mol(train_loader)
        val_acc = test_mol(val_loader)
        test_acc = test_mol(test_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc: {train_acc:.4f}, Validate acc: {val_acc}, Test acc: {test_acc:.4f}.')
    print(f"Best validate acc: {best_val}, Best test acc; {best_test}.")

