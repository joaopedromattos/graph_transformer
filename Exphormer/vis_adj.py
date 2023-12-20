import os 
import pickle
import torch
import argparse
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid, Amazon, KarateClub
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt


def load_dataset(dataset):
    """
    Load the dataset from PyG.

    :param dataset: name of the dataset. Options: 'Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers'
    :return: PyG dataset data.
    """
    data_folder = f'/scratch/jrm28/tmp/'
    if dataset in ('Cora', 'CiteSeer', 'PubMed'):
        pyg_dataset = Planetoid(data_folder, dataset)
    elif dataset in ('Photo', 'Computers'):
        pyg_dataset = Amazon(data_folder, dataset)
    elif dataset == 'Karate':
        pyg_dataset = KarateClub(data_folder)
    else:
        raise NotImplementedError(f'{dataset} not supported. ')
    data = pyg_dataset.data
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset')
    parser.add_argument('--attention_type', type=str, default='Random-d', help='Attention type')
    parser.add_argument('--epoch_step', type=int, default=10, help='Path to activations')
    parser.add_argument('--path', type=str, default='/home/jrm28/graph_transformer/Exphormer/activations/Cora/Random-d/', help='Path to activations')
    return parser.parse_args()


args = parse_args()

pdf_path = f'./adjacency_matrix_{args.dataset}.pdf'

data = load_dataset(args.dataset)

adj = SparseTensor.from_edge_index(data.edge_index)

tensor = adj.to_dense()

with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(tensor, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title(f'Adjacency Matrix - {args.dataset}')
    pdf.savefig(fig)  # Save the current figure into the PDF
    plt.close(fig)  # Close the figure to free memory





