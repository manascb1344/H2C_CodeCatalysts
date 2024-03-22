from flask import Flask, request, jsonify
import cloudinary
import cloudinary.uploader
import cloudinary.api
import json
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

cloudinary.config( 
  cloud_name = "df4gpwc0y", 
  api_key = "568149677141638", 
  api_secret = "i-ivcnbd_49gWa5ghpAvvQgY9QY" 
)

import torch
import torch_geometric
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch_geometric

from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, precision_score, recall_score, confusion_matrix

import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv
FOLDERNAME='/mnt/c/Users/mcb76/OneDrive/Desktop/'
# Load data from the folder
df_features = pd.read_csv(FOLDERNAME+'elliptic_txs_features.csv',header=None)
df_edges = pd.read_csv(FOLDERNAME+"elliptic_txs_edgelist.csv")
df_classes =  pd.read_csv(FOLDERNAME+"elliptic_txs_classes.csv")
df_classes['class'] = df_classes['class'].map({'unknown': 2, '1':1, '2':0}) 
group_class = df_classes.groupby('class').count()
plt.title("# of nodes per class")
plt.barh([ 'Licit','Illicit', 'Unknown'], group_class['txId'].values, color=['g', 'orange', 'r'] )
df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
df_merge = df_merge.sort_values(0).reset_index(drop=True)
df_merge.head()
# Setup trans ID to node ID mapping
nodes = df_merge[0].values

map_id = {j:i for i,j in enumerate(nodes)} # mapping nodes to indexes

# Create edge df that has transID mapped to nodeIDs
edges = df_edges.copy()
edges.txId1 = edges.txId1.map(map_id) #get nodes idx1 from edges list and filtered data
edges.txId2 = edges.txId2.map(map_id)

edges = edges.astype(int)

edge_index = np.array(edges.values).T #convert into an array
edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous() # create a tensor

weights = torch.tensor([1]* edge_index.shape[1] , dtype=torch.double) 
labels = df_merge['class'].values
# mapping txIds to corresponding indices, to pass node features to the model

node_features = df_merge.drop(['txId'], axis=1).copy()
# node_features[0] = node_features[0].map(map_id) # Convert transaction ID to node ID \
# print("unique=",node_features["class"].unique())

# Retain known vs unknown IDs
classified_idx = node_features['class'].loc[node_features['class']!=2].index # filter on known labels
unclassified_idx = node_features['class'].loc[node_features['class']==2].index

classified_illicit_idx = node_features['class'].loc[node_features['class']==1].index # filter on illicit labels
classified_licit_idx = node_features['class'].loc[node_features['class']==0].index # filter on licit labels

# Drop unwanted columns, 0 = transID, 1=time period, class = labels
node_features = node_features.drop(columns=[0, 1, 'class'])

# Convert to tensor
node_features_t = torch.tensor(np.array(node_features.values, dtype=np.double), dtype=torch.double)# drop unused columns
train_idx, valid_idx = train_test_split(classified_idx.values, test_size=0.15)
data_train = Data(x=node_features_t, edge_index=edge_index, edge_attr=weights, 
                               y=torch.tensor(labels, dtype=torch.double))
# Add in the train and valid idx
data_train.train_idx = train_idx
data_train.valid_idx = valid_idx
data_train.test_idx = unclassified_idx
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

from torch_geometric.nn import GCNConv,GATConv,GATv2Conv
import pickle

class GATv2(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,args):
        super(GATv2, self).__init__()
        #use our gat message passing 
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=args['heads'])
        self.conv2 = GATv2Conv(args['heads'] * hidden_dim, hidden_dim, heads=args['heads'])
        
        self.post_mp = nn.Sequential(
            nn.Linear(args['heads'] * hidden_dim, hidden_dim), nn.Dropout(args['dropout'] ), 
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        # MLP output
        x = self.post_mp(x)
        return F.sigmoid(x)

class GnnTrainer(object):
  
  def __init__(self, model):
    self.model = model
    self.metric_manager = MetricManager(modes=["train", "val"])

  def train(self, data_train, optimizer, criterion, scheduler, args):
  
    self.data_train = data_train
    for epoch in range(args['epochs']):
        self.model.train()
        optimizer.zero_grad()
        out = self.model(data_train)

        out = out.reshape((data_train.x.shape[0]))
        loss = criterion(out[data_train.train_idx], data_train.y[data_train.train_idx])
        ## Metric calculations
        # train data
        target_labels = data_train.y.detach().cpu().numpy()[data_train.train_idx]
        pred_scores = out.detach().cpu().numpy()[data_train.train_idx]
        train_acc, train_f1,train_f1macro, train_aucroc, train_recall, train_precision, train_cm = self.metric_manager.store_metrics("train", pred_scores, target_labels)


        ## Training Step
        loss.backward()
        optimizer.step()

        # validation data
        self.model.eval()
        target_labels = data_train.y.detach().cpu().numpy()[data_train.valid_idx]
        pred_scores = out.detach().cpu().numpy()[data_train.valid_idx]
        val_acc, val_f1,val_f1macro, val_aucroc, val_recall, val_precision, val_cm = self.metric_manager.store_metrics("val", pred_scores, target_labels)

        if epoch%5 == 0:
          print("epoch: {} - loss: {:.4f} - accuracy train: {:.4f} -accuracy valid: {:.4f}  - val roc: {:.4f}  - val f1micro: {:.4f}".format(epoch, loss.item(), train_acc, val_acc, val_aucroc,val_f1))

  # To predict labels
  def predict(self, data=None, unclassified_only=True, threshold=0.5):
    # evaluate model:
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

    return {"pred_scores":pred_scores, "pred_labels":pred_labels}

  # To save metrics
  def save_metrics(self, save_name, path="./save/"):
    file_to_store = open(path + save_name, "wb")
    pickle.dump(self.metric_manager, file_to_store)
    file_to_store.close()
  
  # To save model
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
      #new
      self.output[mode]["precision"] = []
      self.output[mode]["recall"] = []
      self.output[mode]["cm"] = []

  def store_metrics(self, mode, pred_scores, target_labels, threshold=0.5):

    # calculate metrics
    pred_labels = pred_scores > threshold
    accuracy = accuracy_score(target_labels, pred_labels)
    f1micro = f1_score(target_labels, pred_labels,average='micro')
    f1macro = f1_score(target_labels, pred_labels,average='macro')
    aucroc = roc_auc_score(target_labels, pred_scores)
    #new
    recall = recall_score(target_labels, pred_labels)
    precision = precision_score(target_labels, pred_labels)
    cm = confusion_matrix(target_labels, pred_labels)

    # Collect results
    self.output[mode]["accuracy"].append(accuracy)
    self.output[mode]["f1micro"].append(f1micro)
    self.output[mode]["f1macro"].append(f1macro)
    self.output[mode]["aucroc"].append(aucroc)
    #new
    self.output[mode]["recall"].append(recall)
    self.output[mode]["precision"].append(precision)
    self.output[mode]["cm"].append(cm)
    
    return accuracy, f1micro,f1macro, aucroc,recall,precision,cm
  
  # Get best results
  def get_best(self, metric, mode="val"):

    # Get best results index
    best_results = {}
    i = np.array(self.output[mode][metric]).argmax()

    # Output
    for m in self.output[mode].keys():
      best_results[m] = self.output[mode][m][i]
    
    return best_results
  

# Set training arguments, set prebuild=True to use builtin PyG models otherwise False
args={"epochs":100,
      'lr':0.01,
      'weight_decay':1e-5,
      'prebuild':True,
      'heads':2,
      'hidden_dim': 128, 
      'dropout': 0.5
      }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model selector GAT or GATv2
net = "GATv2"

if net == "GAT":
    if args['prebuild']==True:
      model = GAT(data_train.num_node_features, args['hidden_dim'], 1, args)
      print("Prebuilt GAT from PyG ")
    else:
      model = GATmodif(data_train.num_node_features, args['hidden_dim'], 1,args)
      print("Custom GAT implemented")
elif net == "GATv2":
    # args['heads'] = 1
    if args['prebuild']==True:
      model = GATv2(data_train.num_node_features, args['hidden_dim'], 1, args) 
      print("Prebuilt GATv2 from PyG ")
    else:
      model = GATv2modif(data_train.num_node_features,  args['hidden_dim'], 1,args) 
      print("Custom GATv2 implemented")

model.double().to(device)
model = GATv2(data_train.num_node_features, args['hidden_dim'], 1, args)
model.double().to(device)
# Push data to GPU
data_train = data_train.to(device)

# Setup training settings
# optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
# criterion = torch.nn.BCELoss()
# # Train
# gnn_trainer_gatv2 = GnnTrainer(model)
# gnn_trainer_gatv2.train(data_train, optimizer, criterion, scheduler, args)

# gnn_trainer_gatv2.save_metrics("GATv2prebuilt-results", path=FOLDERNAME + "save_results-metrics")
# gnn_trainer_gatv2.save_model("GATv2prebuilt-pth", path=FOLDERNAME + "save_results-model")
import networkx as nx
import matplotlib.pyplot as plt

# Load one model 
m1 = GATv2(data_train.num_node_features, args['hidden_dim'], 1, args).to(device).double()
# m1.load_state_dict(torch.load(FOLDERNAME + "/resultsmodel"))
m1.load_state_dict(torch.load("./resultsmodel"))

gnn_t2 = GnnTrainer(m1)
output = gnn_t2.predict(data=data_train, unclassified_only=False)




import time  # Add this import statement

# Define the function to generate the graph and upload the image
def generate_and_upload_graph(time_period):
    # Get index for the given time period
    
    sub_node_list = df_merge.index[df_merge.loc[:, 1] == time_period].tolist()
		
		

		

    # Fetch list of edges for that time period
    edge_tuples = []
    for row in data_train.cpu().edge_index.view(-1, 2).numpy():
        if (row[0] in sub_node_list) or (row[1] in sub_node_list):
            edge_tuples.append(tuple(row))
		
    # Fetch predicted results for that time period
    node_color = []
    for node_id in sub_node_list:
        if node_id in classified_illicit_idx:
            label = "red"  # fraud
        elif node_id in classified_licit_idx:
            label = "green"  # not fraud
        else:
            if output['pred_labels'][node_id]:
                label = "orange"  # Predicted fraud
            else:
                label = "blue"  # Not fraud predicted

        node_color.append(label)

    # Setup networkx graph
    G = nx.Graph()
    G.add_edges_from(edge_tuples)

    # Plot the graph
    plt.figure(3, figsize=(16, 16))
    plt.title("Time period:" + str(time_period))
    nx.draw_networkx(G, nodelist=sub_node_list, node_color=node_color, node_size=6, with_labels=False)

    # Generate a unique filename based on the current time
    current_time = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"output_graph.jpeg"

    # Save the graph as an image with the unique filename
    # plt.savefig(output_filename)

    # Generate unique public_id
    public_id = f"output_graph_{time_period}_{current_time}"

    # Upload the image to Cloudinary
    cloudinary.uploader.upload(output_filename, public_id=public_id, unique_filename=False, overwrite=True)
    srcURL = cloudinary.CloudinaryImage(public_id).build_url()

    # Get asset information
    image_info = cloudinary.api.resource(public_id)
    if image_info["width"] > 900:
        update_resp = cloudinary.api.update(public_id, tags="large")
    elif image_info["width"] > 500:
        update_resp = cloudinary.api.update(public_id, tags="medium")
    else:
        update_resp = cloudinary.api.update(public_id, tags="small")

    nx.draw_networkx(G, nodelist=sub_node_list, node_color=node_color, node_size=6, with_labels=False)
    return srcURL, json.dumps(image_info, indent=2)


# Define the API endpoint
@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    # Get the time period from the request data
    req_data = request.get_json()
    time_period = req_data['time_period']

    # Generate and upload the graph
    delivery_url, asset_info = generate_and_upload_graph(time_period)

    # Return the delivery URL and asset information
    return jsonify({"delivery_url": delivery_url})


if __name__ == '__main__':
    # Start the Flask app

    app.run(debug=False, port=5050)
