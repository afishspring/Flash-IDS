import torch
from torch import nn
from gensim.models import Word2Vec
from models.gcn import GCN
from models.positionalEncoder import PositionalEncoder, DocumentInferrer
from data_prepare.unicorn import load_unicorn


if __name__ == "__main__":
    encoder = PositionalEncoder(30)
    w2vmodel = Word2Vec.load("trained_weights/unicorn/unicorn.model")
    infer = DocumentInferrer(encoder, w2vmodel)

    train_graphs = list(load_unicorn(infer, type="train"))
    val_graphs = list(load_unicorn(infer, type="val"))

    nfeat = train_graphs[0].x.shape[1]
    nhid = 32
    nclass = train_graphs[0].y.max().item() + 1

    model = GCN(nfeat,nhid,nclass)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 201):
        model.train()
        train_loss = 0
        for graph in train_graphs:
            optimizer.zero_grad()
            output = model(graph.x, graph.edge_index)
            loss = criterion(output, graph.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*graph.num_nodes
        train_loss /= sum([g.num_nodes for g in train_graphs])

        model.eval()
        correct = 0
        with torch.no_grad():
            for graph in val_graphs:
                output = model(graph.x, graph.edge_index)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == graph.y).sum().item()
        correct /= sum([g.num_nodes for g in val_graphs])
        print(f"Epoch {epoch}, Validation Accuracy: {100 * correct}%")
    
    # Test
    test_graphs = list(load_unicorn(infer, type="test"))
    model.eval()
    correct = 0
    with torch.no_grad():
        for graph in test_graphs:
            output = model(graph.x, graph.edge_index)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == graph.y).sum().item()
    correct /= sum([g.num_nodes for g in test_graphs])
    print(f"Test Accuracy: {100 * correct}%")