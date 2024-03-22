import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from flask import Flask, send_file
from io import BytesIO
import cloudinary
import cloudinary.uploader
import cloudinary.api
import json

import torch_geometric


from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

# Load data from the folder
df_features = pd.read_csv("elliptic_txs_features.csv", header=None)
df_edges = pd.read_csv("elliptic_txs_edgelist.csv")
df_classes = pd.read_csv("elliptic_txs_classes.csv")

df_classes['class'] = df_classes['class'].map({'unknown': 2, '1': 1, '2': 0})

group_class = df_classes.groupby('class').count()

df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
df_merge = df_merge.sort_values(0).reset_index(drop=True)

nodes = df_merge[0].values
map_id = {j: i for i, j in enumerate(nodes)}

edges = df_edges.copy()
edges.txId1 = edges.txId1.map(map_id)
edges.txId2 = edges.txId2.map(map_id)

edges = edges.astype(int)
edge_index = np.array(edges.values).T
edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)
labels = df_merge['class'].values

node_features = df_merge.drop(['txId'], axis=1).copy()

classified_idx = node_features['class'].loc[node_features['class'] != 2].index
unclassified_idx = node_features['class'].loc[node_features['class'] == 2].index

classified_illicit_idx = node_features['class'].loc[node_features['class'] == 1].index
classified_licit_idx = node_features['class'].loc[node_features['class'] == 0].index

node_features = node_features.drop(columns=[0, 1, 'class'])
node_features_t = torch.tensor(np.array(node_features.values, dtype=np.double), dtype=torch.double)

train_idx, valid_idx = train_test_split(classified_idx.values, test_size=0.15)

data_train = Data(x=node_features_t, edge_index=edge_index, edge_attr=weights, y=torch.tensor(labels, dtype=torch.double))
data_train.train_idx = train_idx
data_train.valid_idx = valid_idx
data_train.test_idx = unclassified_idx

class GATv2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=args['heads'])
        self.conv2 = GATv2Conv(args['heads'] * hidden_dim, hidden_dim, heads=args['heads'])
        self.post_mp = nn.Sequential(
            nn.Linear(args['heads'] * hidden_dim, hidden_dim), nn.Dropout(args['dropout']),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        x = self.post_mp(x)
        return F.sigmoid(x)

class GnnTrainer(object):
    def __init__(self, model):
        self.model = model

    def train(self, data_train, optimizer, criterion, scheduler, args):
        self.data_train = data_train
        for epoch in range(args['epochs']):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data_train)
            out = out.reshape((data_train.x.shape[0]))
            loss = criterion(out[data_train.train_idx], data_train.y[data_train.train_idx])

            target_labels = data_train.y.detach().cpu().numpy()[data_train.train_idx]
            pred_scores = out.detach().cpu().numpy()[data_train.train_idx]
            train_acc, train_f1, train_f1macro, train_aucroc, train_recall, train_precision, train_cm = self.metric_manager.store_metrics("train", pred_scores, target_labels)

            loss.backward()
            optimizer.step()

            self.model.eval()
            target_labels = data_train.y.detach().cpu().numpy()[data_train.valid_idx]
            pred_scores = out.detach().cpu().numpy()[data_train.valid_idx]
            val_acc, val_f1, val_f1macro, val_aucroc, val_recall, val_precision, val_cm = self.metric_manager.store_metrics("val", pred_scores, target_labels)

            if epoch % 5 == 0:
                print("epoch: {} - loss: {:.4f} - accuracy train: {:.4f} -accuracy valid: {:.4f}  - val roc: {:.4f}  - val f1micro: {:.4f}".format(epoch, loss.item(), train_acc, val_acc, val_aucroc, val_f1))

    def predict(self, data=None, unclassified_only=True, threshold=0.5):
        self.model.eval()
        if data is not None:
            self.data_train = data
        out = self.model(self.data_train)
        out = out.reshape((self.data_train.x.shape[0]))
        if unclassified_only:
            pred_scores = out.detach().cpu().numpy()[self.data_train.test_idx]
        else:
            pred_scores = out.detach().cpu().numpy()
        pred_labels = pred_scores > threshold
        return {"pred_scores": pred_scores, "pred_labels": pred_labels}

    def save_metrics(self, save_name, path="./save/"):
        file_to_store = open(path + save_name, "wb")
        pickle.dump(self.metric_manager, file_to_store)
        file_to_store.close()

    def save_model(self, save_name, path="./save/"):
        torch.save(self.model.state_dict(), path + save_name)

class MetricManager(object):
    def __init__(self, modes=["train", "val"]):
        self.output = {}
        for mode in modes:
            self.output[mode] = {}
            self.output[mode]["accuracy"] = []
            self.output[mode]["f1micro"] = []
            self.output[mode]["f1macro"] = []
            self.output[mode]["aucroc"] = []
            self.output[mode]["recall"] = []
            self.output[mode]["precision"] = []
            self.output[mode]["cm"] = []

    def store_metrics(self, mode, pred_scores, target_labels):
        pred_labels = pred_scores > 0.5
        acc = accuracy_score(target_labels, pred_labels)
        f1_micro = f1_score(target_labels, pred_labels, average='micro')
        f1_macro = f1_score(target_labels, pred_labels, average='macro')
        aucroc = roc_auc_score(target_labels, pred_scores)
        recall = recall_score(target_labels, pred_labels)
        precision = precision_score(target_labels, pred_labels)
        cm = confusion_matrix(target_labels, pred_labels)
        self.output[mode]["accuracy"].append(acc)
        self.output[mode]["f1micro"].append(f1_micro)
        self.output[mode]["f1macro"].append(f1_macro)
        self.output[mode]["aucroc"].append(aucroc)
        self.output[mode]["recall"].append(recall)
        self.output[mode]["precision"].append(precision)
        self.output[mode]["cm"].append(cm)
        return acc, f1_micro, f1_macro, aucroc, recall, precision, cm

# Cloudinary configuration
cloudinary.config( 
  cloud_name = "df4gpwc0y", 
  api_key = "568149677141638", 
  api_secret = "i-ivcnbd_49gWa5ghpAvvQgY9QY" 
)

app = Flask(__name__)

@app.route('/')
def generate_and_upload_image():
    # Generate the graph image
    G = nx.erdos_renyi_graph(30, 0.05)
    nx.draw(G, with_labels=True)
    plt.savefig("output_graph_1.jpeg")
    plt.close()

    # Upload the image to Cloudinary
    cloudinary.uploader.upload("output_graph_1.jpeg", public_id="output_graph_1", unique_filename=False, overwrite=True)
    srcURL = cloudinary.CloudinaryImage("output_graph_1").build_url()

    return f"Graph image uploaded to Cloudinary. Delivery URL: {srcURL}"

if __name__ == "__main__":
    app.run(debug=True)
