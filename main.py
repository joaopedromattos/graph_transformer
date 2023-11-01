import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import NodeLoader
import torch_geometric.transforms as T

from vanilla_transformer import VanillaTransformer

from utils import load_dataset

loss_fn = nn.CrossEntropyLoss()

def train(train_loader, model, optim, device):
    total_loss = 0
    
    model.train()
    for batch in train_loader:
        optim.zero_grad()
        out = model(batch.to(device))
        loss = loss_fn(out, batch.y)
        loss.backward()
        total_loss += loss.item()
        optim.step()
        
    return total_loss / len(train_loader)
        
    
def evaluate(val_loader, model, device):
    pass

if __name__ == '__main__':
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data, dataset = load_dataset('Cora')
    
    model_params = {
        'num_features' : dataset.num_features,
        'nhead' : 1,
        'num_encoder_layers': 6
    }
    
    model = VanillaTransformer(**model_params)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

    data = transform(data)
    
    data = T.RandomNodeSplit(num_val=0.05, num_test=0.1)(data)
    
    train_loader = NodeLoader(data.subgraph(data.train_mask), batch_size=32, shuffle=True, num_workers=0)
    val_loader = NodeLoader(data.subgraph(data.val_mask), batch_size=32, shuffle=True, num_workers=0)
    test_loader = NodeLoader(data.subgraph(data.test_mask), batch_size=32, shuffle=True, num_workers=0)
    
    runs = 3
    for run in runs:
        
        train_params = {
            'train_loader' : train_loader,
            'model' : model, 
            'optim' : optim,
            'device' : device
        }
        
        epoch_loss = train(**train_params)
        
        