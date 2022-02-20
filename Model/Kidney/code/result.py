from utils import *
from tf_utils import *
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EPOCHS = 10000
BATCH_SIZE = 64
LATENT_DIM = 10
MODELS_DIR = '../checkpoints/models/'

gen = tfk.models.load_model(MODELS_DIR + 'gen_rnaseqdb.h5')

#load dataset expression levels
expr_data, gene_symbols = rnaseqdb_df_Kidney()

#load clinical data
info_df=pd.read_csv('../data/KIRC_clinical_core.csv')

gender= info_df['gender'].values
tissues = info_df['stage'].values


print(Counter(tissues))
print(Counter(gender))

#Process categorical metadata
cat_dicts = []
tissues_dict_inv = np.array(list(sorted(set(tissues))))
tissues_dict = {t: i for i, t in enumerate(tissues_dict_inv)}
new_tissues = np.vectorize(lambda t: tissues_dict[t])(tissues)
cat_dicts.append(tissues_dict_inv)


gender_dict_inv = np.array(list(sorted(set(gender))))
gender_dict = {d: i for i, d in enumerate(gender_dict_inv)}
new_gender = np.vectorize(lambda t: gender_dict[t])(gender)
cat_dicts.append(gender_dict_inv)



cat_covs = np.concatenate((new_tissues[:, None], new_gender[:, None]), axis=-1)
cat_covs = (np.int32(cat_covs))


# Process numerical metadata
num_covs = np.zeros((expr_data.shape[0], 1), dtype=np.float32)


# Log-transform data
expr_data_log = np.log(1 + expr_data)
expr_data_log = np.float32(expr_data_log)

# Train/test split
np.random.seed(0)
idx = np.arange(expr_data_log.shape[0])
np.random.shuffle(idx)

expr_data_log_new_idx = expr_data_log[idx, :]


num_covs_new_idx = num_covs[idx, :]
cat_covs_new_idx = cat_covs[idx, :]

expr_data_train ,expr_data_test = split_train_test(expr_data_log_new_idx)
num_covs_train, num_covs_test = split_train_test(num_covs_new_idx)
cat_covs_train, cat_covs_test = split_train_test(cat_covs_new_idx)

# Normalise data
expr_data_mean = np.mean(expr_data_train, axis=0)
expr_data_std = np.std(expr_data_train, axis=0)
expr_data_train_norm = standardize(expr_data_train, mean=expr_data_mean, std=expr_data_std)
expr_data_test_norm = standardize(expr_data_test, mean=expr_data_mean, std=expr_data_std)


x_gen = predict(cc=cat_covs_test,
                nc=num_covs_test,
                gen=gen)


gene_list=gene_symbols[0:12]
gene_idxs=[]
for gene in gene_list:
        try:
            gene_idxs.append(gene_symbols.index(gene))
                      
        except ValueError:
            pass

x_gen_s = x_gen[:, gene_idxs]
x_test_s = expr_data_test_norm [:, gene_idxs]
x_train_s = expr_data_train_norm[:, gene_idxs]


# plt.figure(figsize=(14, 8))
# plot_individual_distrs(x_test_s, x_gen_s, gene_list)
# plt.savefig('distribution.png')
#plt.show()



plt.figure(figsize=(14, 8))
plot_distance_matrices(x_test_s, x_gen_s, gene_list)

plt.savefig('distance_matrics.png')
plt.show()

f, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'height_ratios': [2, 1]})
a0 = axes[0, 0]
a1 = axes[0, 1]
a2 = axes[0, 2]

Z = hierarchical_clustering(x_test_s)
plt.title('Real')
with plt.rc_context({'lines.linewidth': 0.5}):
    dn = dendrogram(Z, count_sort='ascending', labels=gene_list, leaf_rotation=45, link_color_func=lambda _: '#000000', ax=a0)
a0.get_yaxis().set_visible(False)
a0.set_title('$\\bf{a)}$    Dendrogram, test')
a0.spines["top"].set_visible(False)
a0.spines["right"].set_visible(False)
a0.spines["left"].set_visible(False)
a0.spines["bottom"].set_visible(False)

Z = hierarchical_clustering(x_train_s)
plt.title('Real')
with plt.rc_context({'lines.linewidth': 0.5}):
    dn = dendrogram(Z, count_sort='ascending', labels=gene_list, leaf_rotation=45, link_color_func=lambda _: '#000000', ax=a1)
a1.get_yaxis().set_visible(False)
a1.set_title('$\\bf{b)}$    Dendrogram, train')
a1.spines["top"].set_visible(False)
a1.spines["right"].set_visible(False)
a1.spines["left"].set_visible(False)
a1.spines["bottom"].set_visible(False)


Z = hierarchical_clustering(x_gen_s)
with plt.rc_context({'lines.linewidth': 0.5}):
    dn = dendrogram(Z, count_sort='ascending', labels=gene_list, leaf_rotation=45, link_color_func=lambda _: '#000000', ax=a2)
a2.get_yaxis().set_visible(False)
a2.set_title('$\\bf{c)}$    Dendrogram, gen')
a2.spines["top"].set_visible(False)
a2.spines["right"].set_visible(False)
a2.spines["left"].set_visible(False)
a2.spines["bottom"].set_visible(False)

f.tight_layout(pad=3.0)

plt.savefig('dendrogram.png')
plt.show()