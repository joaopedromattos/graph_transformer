import os 
import pickle
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset')
    parser.add_argument('--attention_type', type=str, default='Random-d', help='Attention type')
    parser.add_argument('--epoch_step', type=int, default=10, help='Path to activations')
    parser.add_argument('--path', type=str, default='/home/jrm28/graph_transformer/Exphormer/activations/Cora/Random-d/', help='Path to activations')
    return parser.parse_args()


args = parse_args()

dataset = args.dataset
path = f'/home/jrm28/graph_transformer/Exphormer/activations/{dataset}/{args.attention_type}/'
epochs = len(list(filter(lambda x: "Q" in x, os.listdir(path))))

Q, K, V, scores, edge_indexes = [], [], [], [], []

print(f"reading {epochs} epochs of activations from {path}")
for epoch in range(100):
    for layer in ('Q', 'K', 'V', 'scores', 'edge_indexes'):
        file = f'{epoch}_{layer}_activation.pkl'
        outp = pickle.load(open(os.path.join(path, file), 'rb'))
        
        if layer == 'Q':
            Q.append(outp)
        elif layer == 'K':
            K.append(outp)
        elif layer == 'V':
            V.append(outp)
        elif layer == 'edge_indexes':
            edge_indexes.append(outp)
        else:
            scores.append(outp)


def create_attention_matrices(scores, edge_indexes, num_nodes):
    attention_matrices = torch.zeros(len(scores), num_nodes, num_nodes)
    for i, (score, edge_index) in enumerate(zip(scores, edge_indexes)):
        print(f"Epoch {i} - score shape", score[0][:, 0].shape)
        attention_matrices[i, edge_index[0][0], edge_index[0][1]] = score[0][:, 0].squeeze().cpu().detach()
    return attention_matrices


import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt


step = args.epoch_step
tensors = create_attention_matrices(scores[::step], edge_indexes, Q[0][0].shape[0])

# File path for the PDF
pdf_path = f'./tensors_heatmap_{dataset}_{args.attention_type}.pdf'

# import code
# code.interact(local={**locals(), **globals()})

# Create a PDF file with each page containing one heatmap
with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
    for idx, tensor in enumerate(tensors):
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow((tensor - tensor.min())/ (tensor.max() - tensor.min()), cmap='viridis')
        fig.colorbar(cax)
        ax.set_title(f'Heatmap at {idx * step}')
        pdf.savefig(fig, dpi=500)  # Save the current figure into the PDF
        plt.close(fig)  # Close the figure to free memory

pdf_path



