import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
import pickle
import random
from statsmodels.stats.multitest import multipletests
from matplotlib.collections import EllipseCollection
import scipy




PathWay_genes='../Data/pathway_genes.txt'
# ------------------
# LOAD_DB
# ------------------

def rnaseqdb_df(file_path):
    """
    Loads DB expression dataset
    :param file: DB expression file name
    :return: Pandas dataframe with RNA-Seq values (rows: genes, cols: samples)
    """ 
    df = pd.read_csv(file_path,index_col=[0])
    expr_data = df.values.T
    All_gene_symbols =(df.index.values).tolist()
    samples_ids = df.columns.values
    
    selected_gene = []
    gene_idxs = []
    symb = []
    with open(PathWay_genes) as f:
      for line in f:
        selected_gene.append(line.strip())
         
    for gene in selected_gene:
        try:
            gene_idxs.append(All_gene_symbols.index(gene))
            symb.append(gene)
        except ValueError:
            pass
  
    expr = expr_data[:, gene_idxs]
    gene_symbols = symb
    
    return expr, gene_symbols, np.array(samples_ids),np.array(selected_gene)
   















# ---------------------
# DATA UTILITIES
# ---------------------

def standardize(x, mean=None, std=None):
    """
    Shape x: (nb_samples, nb_vars)
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std


def split_train_test(x, train_rate=0.8):
    """
    Split data into a train and a test sets
    :param train_rate: percentage of training samples
    :return: x_train, x_test
    """
    nb_samples = x.shape[0]
    split_point = int(train_rate * nb_samples)
    x_train = x[:split_point]
    x_test = x[split_point:]

    return x_train, x_test



def save_synthetic(name, data, symbols, datadir):
    """
    Saves data with Shape=(nb_samples, nb_genes) to pickle file with the given name in datadir.
    :param name: name of the file in SYNTHETIC_DIR where the expression data will be saved
    :param data: np.array of data with Shape=(nb_samples, nb_genes)
    :param symbols: list of gene symbols matching the columns of data
    """
    file = '{}/{}.pkl'.format(datadir, name)
    data = {'data': data,
            'symbols': symbols}
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_synthetic(name, datadir):
    """
    Loads data from pickle file with the given name (produced by save_synthetic function)
    :param name: name of the pickle file in datadir containing the expression data
    :return: np.array of expression with Shape=(nb_samples, nb_genes) and list of gene symbols matching the columns
    of data
    """
    file = '{}/{}.pkl'.format(datadir, name)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['data'], data['symbols']




# ---------------------
# CORRELATION UTILITIES
# ---------------------

def pearson_correlation(x, y):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :return: Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        return (a - a_off) / a_std

    assert x.shape[0] == y.shape[0]
    x_ = standardize(x)
    y_ = standardize(y)
    return np.dot(x_.T, y_) / x.shape[0]


def upper_diag_list(m_):
    """
    Returns the condensed list of all the values in the upper-diagonal of m_
    :param m_: numpy array of float. Shape=(N, N)
    :return: list of values in the upper-diagonal of m_ (from top to bottom and from
             left to right). Shape=(N*(N-1)/2,)
    """
    m = np.triu(m_, k=1)  # upper-diagonal matrix
    tril = np.zeros_like(m_) + np.nan
    tril = np.tril(tril)
    m += tril
    m = np.ravel(m)
    return m[~np.isnan(m)]



def correlations_list(x, y, corr_fn=pearson_correlation):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :param corr_fn: correlation function taking x and y as inputs
    """
    corr = corr_fn(x, y)
    return upper_diag_list(corr)


def gamma_coef(x, y):
    """
    Compute gamma coefficients for two given expression matrices
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :return: Gamma(D^X, D^Z)
    """
    dists_x = 1 - correlations_list(x, x)
    dists_y = 1 - correlations_list(y, y)
    gamma_dx_dy = pearson_correlation(dists_x, dists_y)
    return gamma_dx_dy











# ---------------------
# CLUSTERING UTILITIES
# ---------------------

def hierarchical_clustering(data, corr_fun=pearson_correlation):
    """
    Performs hierarchical clustering to cluster genes according to a gene similarity
    metric.
    Reference: Cluster analysis and display of genome-wide expression patterns
    :param data: numpy array. Shape=(nb_samples, nb_genes)
    :param corr_fun: function that computes the pairwise correlations between each pair
                     of genes in data
    :return scipy linkage matrix
    """
    # Perform hierarchical clustering
    y = 1 - correlations_list(data, data, corr_fun)
    l_matrix = linkage(y, 'complete')  # 'correlation'
    return l_matrix








# ---------------------
# PLOTTING UTILITIES
# ---------------------

def plot_distribution(data, label, color='royalblue', linestyle='-', ax=None, plot_legend=True,
                      xlabel=None, ylabel=None):
    """
    Plot a distribution
    :param data: data for which the distribution of its flattened values will be plotted
    :param label: label for this distribution
    :param color: line color
    :param linestyle: type of line
    :param ax: matplotlib axes
    :param plot_legend: whether to plot a legend
    :param xlabel: label of the x axis (or None)
    :param ylabel: label of the y axis (or None)
    :return matplotlib axes
    """
    x = np.ravel(data)
    ax = sns.distplot(x,
                      hist=False,
                      kde_kws={'linestyle': linestyle, 'color': color, 'linewidth': 2, 'bw': .15},
                      label=label,
                      ax=ax)
    if plot_legend:
        plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    return ax


def plot_distance_matrix(dist_m, v_min, v_max, symbols, title='Distance matrix'):
    ax = plt.gca()
    im = ax.imshow(dist_m, vmin=v_min, vmax=v_max)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_yticks(np.arange(len(symbols)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            text = ax.text(j, i, '{:.2f}'.format(dist_m[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title(title)


def plot_distance_matrices(x, y, symbols, corr_fn=pearson_correlation):
    """
    Plots distance matrices of both datasets x and y.
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :symbols: array of gene symbols. Shape=(nb_genes,)
    :param corr_fn: 2-d correlation function
    """

    dist_x = 1 - np.abs(corr_fn(x, x))
    dist_y = 1 - np.abs(corr_fn(y, y))
    v_min = min(np.min(dist_x), np.min(dist_y))
    v_max = min(np.max(dist_x), np.max(dist_y))

    # fig = plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plot_distance_matrix(dist_x, v_min, v_max, symbols, title='Distance matrix, real')
    plt.subplot(1, 2, 2)
    plot_distance_matrix(dist_y, v_min, v_max, symbols, title='Distance matrix, synthetic')
    # fig.tight_layout()
    return plt.gca()


def plot_individual_distrs(x, y, symbols, nrows=4, xlabel='X', ylabel='Y'):
    """
    Plots individual distributions for each gene
    """
    nb_symbols = len(symbols)
    ncols = 1 + (nb_symbols - 1) // nrows

    # plt.figure(figsize=(18, 12))
    plt.subplots_adjust(left=0, bottom=0, right=None, top=1.3, wspace=None, hspace=None)
    for r in range(nrows):
        for c in range(ncols):
            idx = (nrows - 1) * r + c
            plt.subplot(nrows, ncols, idx + 1)

            plt.title(symbols[idx])
            plot_distribution(x[:, idx], xlabel='', ylabel='', label=xlabel, color='black')
            plot_distribution(y[:, idx], xlabel='', ylabel='', label=ylabel, color='royalblue')

            if idx + 1 == nb_symbols:
                break




    




















    

