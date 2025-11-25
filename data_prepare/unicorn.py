import os
import time
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def show(str):
	print (str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

def parse_data():
    for i in range(3):
        os.system('tar -zxvf camflow-attack-' + str(i) + '.gz.tar')
    for i in range(13):
        os.system('tar -zxvf camflow-benign-' + str(i) + '.gz.tar')

    os.system('rm error.log')
    os.system('rm parse-error-camflow-*')
    show('Start processing.')
    for i in range(25):
        show('Attack graph ' + str(i+125))
        f = open('camflow-attack.txt.'+str(i), 'r')
        fw = open('datasets/unicorn/'+str(i+125)+'.txt', 'w')
        for line in f:
                tempp = line.strip('\n').split('\t')
                temp = []
                temp.append(tempp[0])
                temp.append(tempp[2].split(':')[0])
                temp.append(tempp[1])
                temp.append(tempp[2].split(':')[1])
                temp.append(tempp[2].split(':')[2])
                temp.append(tempp[2].split(':')[3])
                fw.write(temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\t'+temp[3]+'\t'+temp[4]+'\t'+temp[5]+'\n')
        f.close()
        fw.close()
        os.system('rm camflow-attack.txt.' + str(i))

    for i in range(125):
        show('Benign graph ' + str(i))
        f = open('camflow-normal.txt.'+str(i), 'r')
        fw = open('datasets/unicorn/'+str(i)+'.txt', 'w')
        for line in f:
                tempp = line.strip('\n').split('\t')
                temp = []
                temp.append(tempp[0])
                temp.append(tempp[2].split(':')[0])
                temp.append(tempp[1])
                temp.append(tempp[2].split(':')[1])
                temp.append(tempp[2].split(':')[2])
                temp.append(tempp[2].split(':')[3])
                fw.write(temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\t'+temp[3]+'\t'+temp[4]+'\t'+temp[5]+'\n')
        f.close()
        fw.close()
        os.system('rm camflow-normal.txt.' + str(i))
    show('Done.')

def prepare_graph(df):
    def process_node(node, action, node_dict, label_dict, dummies, node_type):
        node_dict.setdefault(node, []).append(action)
        label_dict[node] = dummies.get(getattr(row, node_type), -1)  

    nodes = {}
    labels = {}
    edges = []
    dummies = {
        "7998762093665332071": 0, "14709879154498484854": 1, "10991425273196493354": 2,
        "14871526952859113360": 3, "8771628573506871447": 4, "7877121489144997480": 5,
        "17841021884467483934": 6, "7895447931126725167": 7, "15125250455093594050": 8,
        "8664433583651064836": 9, "14377490526132269506": 10, "15554536683409451879": 11,
        "8204541918505434145": 12, "14356114695140920775": 13
    }

    for row in df.itertuples():
        process_node(row.actorID, row.action, nodes, labels, dummies, 'actor_type')
        process_node(row.objectID, row.action, nodes, labels, dummies, 'object')

        edges.append((row.actorID, row.objectID))

    features = [nodes[node] for node in nodes]
    feat_labels = [labels[node] for node in nodes]
    edge_index = [[], []]
    for src, dst in edges:
        src_index = list(nodes.keys()).index(src)
        dst_index = list(nodes.keys()).index(dst)
        edge_index[0].append(src_index)
        edge_index[1].append(dst_index)

    return features, feat_labels, edge_index, list(nodes.keys())

def load_unicorn(infer, type="train"):

    if type == "train":
        file_range = range(0, 95)
    elif type == "val":
        file_range = range(95, 100)
    else:
        file_range = range(100, 125)

    for i in file_range:
        if os.path.exists(f'datasets/unicorn/cache/{i}.pt'):
             graph_dict = torch.load(f"datasets/unicorn/cache/{i}.pt")
        else:
            with open(f"datasets/unicorn/{i}.txt") as f:
                data = [line.split('\t') for line in f.read().split('\n') if line]

            df = pd.DataFrame (data, columns = ['actorID', 'actor_type','objectID','object','action','timestamp'])
            df.sort_values(by='timestamp', ascending=True,inplace=True)
            df = df.dropna()
            phrases,labels,edges,mapp = prepare_graph(df)

            nodes = np.array([infer(x) for x in phrases])

            graph_dict = {
                "nodes": nodes,
                "labels": labels,
                "edges": edges
            }
            if os.path.exists('datasets/unicorn/cache/') == False:
                os.makedirs('datasets/unicorn/cache/')
            torch.save(graph_dict, f'datasets/unicorn/cache/{i}.pt')
        yield Data(
            x=torch.tensor(graph_dict["nodes"], dtype=torch.float),
            y=torch.tensor(graph_dict["labels"], dtype=torch.long),
            edge_index=torch.tensor(graph_dict["edges"], dtype=torch.long)
        )