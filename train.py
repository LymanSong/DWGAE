import os
root_path = os.getcwd()
model_path = os.path.join(root_path, 'model')
data_path = os.path.join(root_path, 'data')
summary_path = os.path.join(root_path, 'summary')
os.chdir(model_path)
from libs_funcs import *
from GAE import *
from DWGAE import *
os.chdir(root_path)

import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import argparse
import time
from dgl import DGLGraph
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default= "bus_network", help='name of dataset (default: bus_network)')
parser.add_argument('--n_epochs', '-e', type=int, default=2000, help='number of epochs')
#parser.add_argument('--save_dir', '-s', type=str, default='../result', help='result directry')
parser.add_argument('--in_dim', '-i', type=int, default=10, help='input dimension')
#parser.add_argument('--hidden_dims', metavar='N', type=int, nargs='+', help='list of hidden dimensions')
parser.add_argument('--gnn_model', type=str, default ='dwgnn', choices = ['gcn', 'dwgnn'], help = 'gnn model to use')
parser.add_argument('--aggregator_type', type=str, default='lstm', choices=["lstm", "mean", "max", "sum"], help='an aggregator function for GNN model')
parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
parser.add_argument('--edge_func', type = str, default='softmax', choices = ['softmax', 'softmin'], help='normalization function to use during calculating neighbor attention weight vector')
args = parser.parse_args()
scaler = preprocessing.StandardScaler()

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

#%% sample data import
data = gpd.read_file(os.path.join(data_path, args.dataset + '.geojson'), driver = 'GeoJSON')
G = load_data(os.path.join(data_path, args.dataset + '_graph'))

if args.dataset == 'bus_network':
    cols_pop = ['pop_sum', 'pop_worker', 'pop_elder', 'pop_child']
    cols_passengers = ['passengers', 'passengers_init', 'passengers_transfer']
    cols_f_pop = ['TMST_00', 'TMST_01', 'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05',
           'TMST_06', 'TMST_07', 'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11',
           'TMST_12', 'TMST_13', 'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17',
           'TMST_18', 'TMST_19', 'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23',
           'MAN_FLOW_POP_CNT_10G', 'MAN_FLOW_POP_CNT_20G', 'MAN_FLOW_POP_CNT_30G',
           'MAN_FLOW_POP_CNT_40G', 'MAN_FLOW_POP_CNT_50G', 'MAN_FLOW_POP_CNT_60GU',
           'WMAN_FLOW_POP_CNT_10G', 'WMAN_FLOW_POP_CNT_20G', 'WMAN_FLOW_POP_CNT_30G',
           'WMAN_FLOW_POP_CNT_40G', 'WMAN_FLOW_POP_CNT_50G', 'WMAN_FLOW_POP_CNT_60GU', 
           'FLOW_POP_CNT_MON', 'FLOW_POP_CNT_TUS', 'FLOW_POP_CNT_WED', 'FLOW_POP_CNT_THU',
           'FLOW_POP_CNT_FRI', 'FLOW_POP_CNT_SAT', 'FLOW_POP_CNT_SUN']
    cols_traf = ['평균길이', '혼잡시간강도', '혼잡빈도강도']
    cols_weath = ['미세먼지(㎍/㎥)', '초미세먼지(㎍/㎥)', '오존(ppm)', '이산화질소(ppm)', '아황산가스(ppm)', '일산화탄소(ppm)']
    node_attrs = cols_passengers + cols_pop + cols_weath + cols_traf
    df_train = data[node_attrs]
    


#%% dgl graph conversion

key_col = [i for i in data.columns if 'ID' in i][0]
edge_attrs = list(list(G.edges(data = True))[-1][-1].keys())

node_dict = dict()
for i, r in data.iterrows():
    node_dict[i] = r[key_col]
    
node_dict_ = dict()
for k, v in node_dict.items():
    node_dict_[v] = k
    
g = DGLGraph()
g = dgl.add_nodes(g, len(df_train))

src_idx = []
dst_idx = []
for u, v in G.edges:
    g = dgl.add_edges(g, node_dict_[u], node_dict_[v])
    src_idx.append(node_dict_[u])
    dst_idx.append(node_dict_[v])
    
# construct node feature matrix
for i in df_train.columns:
    g.ndata[i] = torch.FloatTensor(df_train[i].values)
    
df_node_features = torch.stack([g.ndata[j] for j in [i for i in g.ndata]]).T
df_node_features = torch.FloatTensor(scaler.fit_transform(df_node_features.numpy()))

for i in range(len(df_train.columns)):
    g.ndata[df_train.columns[i]] = torch.FloatTensor(df_node_features[:,i])    

# construct edge feature matrix
df_edge_features = pd.DataFrame(columns = ['u', 'v'] + edge_attrs)
for u, v, item in G.edges(data = True):
    # print(u, v, item)
    df_edge_features = df_edge_features.append(pd.Series([u, v] + list(item.values()), index = df_edge_features.columns), ignore_index = True)

df_edge_features['u'] = df_edge_features['u'].apply(lambda x: node_dict_[int(x)])
df_edge_features['v'] = df_edge_features['v'].apply(lambda x: node_dict_[int(x)])


for fn in edge_attrs:
    g.edata[fn] = torch.FloatTensor(df_edge_features[fn].values)

#%% train GAE

def train(g, df_node_features, df_edge_features, hidden_dims, \
          gnn_model, n_epochs, lr, e_func = None, e_feature = 'path_length', session_record = ''):
    if gnn_model == 'gcn':
        g = dgl.add_self_loop(g)
        
    features = df_node_features
    in_feats = df_node_features.shape[1]

    if gnn_model == 'gcn':
        model = GAE(in_feats, hidden_dims)

    elif gnn_model == 'dwgnn':
        model = GAE_DWGNN(in_feats, hidden_dims, e_func, 'lstm')
        
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)
    adj = g.adjacency_matrix().to_dense()
    if gnn_model == 'dwgnn':
        w_adj = torch.sparse_coo_tensor(indices = g.adjacency_matrix().coalesce().indices(),\
                            values = torch.tensor(df_edge_features[e_feature]), size = (len(g.nodes()), len(g.nodes()))).to_dense()
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])

    losses = []
    print('Training Start')
    for epoch in tqdm(range(n_epochs)):
        if gnn_model == 'gcn':
            adj_logits = model.forward(g, features)
            loss = F.binary_cross_entropy_with_logits(adj_logits, adj, pos_weight=pos_weight)
        elif gnn_model == 'dwgnn':
            edge_weight_vec = torch.FloatTensor(scaler.fit_transform(g.edata[e_feature].numpy().reshape(-1, 1)))
            adj_logits = model.forward(g, features, edge_weight_vec)
            adj_logits[adj_logits != adj_logits] = 0
            loss = F.binary_cross_entropy_with_logits(adj_logits, w_adj, pos_weight=pos_weight)
        
        

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if epoch%(n_epochs/10) == 0 or epoch == n_epochs - 1:
            session_record += 'Epoch: {:02d} | Loss: {:.5f}\n'.format(epoch, loss)
            print('Epoch: {:02d} | Loss: {:.5f}'.format(epoch, loss))
        # print(torch.sigmoid(adj_logits))

    result = g.ndata['h'].detach().numpy()
    
    return result, model, losses, session_record

'''
gcn
hidden_dims = [64, 128, 64, 16, 8, 3] # loss ~= 1.01 for 2000 iterations
hidden_dims = [32, 64, 32, 16, 8] # loss ~= 0.895 for 2000 iterations

dwgnn
hidden_dims = [64, 128, 64, 16, 8, 3] # loss ~= 0.91 for 2000 iterations
hidden_dims = [32, 64, 32, 16, 8] # loss ~= 0.835 for 2000 iterations
'''

hidden_dims = [32, 64, 32, 16, 8] 
session_record = '\nhidden_dims : {}\n'.format(hidden_dims)
arguments_str = print_options(args)

print(session_record)

if args.gnn_model == 'gcn':
    result, model, losses, session_record = train(g, df_node_features, df_edge_features, \
                                                  hidden_dims, args.gnn_model, args.n_epochs, args.lr, session_record = session_record)
    s_filename = os.path.join(time.ctime().replace(':', ';') + '_' + args.dataset + '_' + args.gnn_model + '.txt')
elif args.gnn_model == 'dwgnn':
    if args.edge_func == 'softmax':
        e_func = nn.Softmax(dim = 1)
    else:
        e_func = nn.Softmin(dim = 1)
    result, model, losses, session_record = train(g, df_node_features, df_edge_features, hidden_dims, args.gnn_model,\
                                                  args.n_epochs, args.lr, e_func = e_func, e_feature = edge_attrs[0], session_record = session_record)
    s_filename = os.path.join(time.ctime().replace(':', ';') + '_' + args.dataset + '_' + args.gnn_model + '_' + args.aggregator_type + '.txt')

with open(os.path.join(summary_path, s_filename), 'w') as f:
    f.write(arguments_str)
    f.write(session_record)





