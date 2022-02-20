from utils import *
from tf_utils import *
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold


EPOCHS = 10000
BATCH_SIZE = 64
LATENT_DIM = 10
MODELS_DIR = '../checkpoints/models/'




#load dataset expression levels
expr_data, gene_symbols, sample_names,genes_pathways = rnaseqdb_df('../Data/GeneExpressionData.csv')

print(expr_data.shape)



#load clinical data
info_df=pd.read_csv('../Data/new_clinical.csv')

tissues=info_df['PRS_type'].values
gender=info_df['Gender'].values

print(Counter(tissues))
print(Counter(gender))


#Process categorical metadata
cat_dicts = []
tissues_dict_inv = np.array(list(sorted(set(tissues))))
tissues_dict = {t: i for i, t in enumerate(tissues_dict_inv)}
new_tissues = np.vectorize(lambda t: tissues_dict[t])(tissues)
cat_dicts.append(tissues_dict_inv)
#print(tissues_dict_inv)


gender_dict_inv = np.array(list(sorted(set(gender))))
gender_dict = {d: i for i, d in enumerate(gender_dict_inv)}
new_gender = np.vectorize(lambda t: gender_dict[t])(gender)
cat_dicts.append(gender_dict_inv)
#print(gender_dict_inv)


cat_covs = np.concatenate((new_tissues[:, None], new_gender[:, None]), axis=-1)
cat_covs = (np.int32(cat_covs))
print('Cat covs: ', cat_covs.shape)

# Process numerical metadata
num_covs = np.zeros((expr_data.shape[0], 1), dtype=np.float32)
print('num covs: ',num_covs.shape)

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


# Define model
vocab_sizes = [len(c) for c in cat_dicts]
nb_numeric = num_covs.shape[-1]
expr_data_dim = expr_data.shape[-1]

gen = make_generator(expr_data_dim, vocab_sizes, nb_numeric)
disc = make_discriminator(expr_data_dim, vocab_sizes, nb_numeric)

# Evaluation metrics
def score_fn(expr_data_test_norm, cat_covs_test, num_covs_test):
    def _score(gen):
        x_gen = predict(cc=cat_covs_test,
                        nc=num_covs_test,
                        gen=gen)
        
        print(expr_data_test_norm , x_gen)
        gamma_dx_dz = gamma_coef(expr_data_test_norm, x_gen)
        return gamma_dx_dz
        
    return _score


# Function to save models
def save_fn(models_dir=MODELS_DIR):
    gen.save(models_dir + 'gen_rnaseqdb.h5')

# Train model
gen_opt = tfk.optimizers.RMSprop(5e-4)
disc_opt = tfk.optimizers.RMSprop(5e-4)

train(dataset=expr_data_train_norm,
    cat_covs=cat_covs_train,
    num_covs=num_covs_train,
    z_dim=LATENT_DIM,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    gen=gen,
    disc=disc,
    gen_opt=gen_opt,
    disc_opt=disc_opt,
    score_fn=score_fn(expr_data_test_norm, cat_covs_test, num_covs_test),
    save_fn=save_fn)

# Evaluate data

score = score_fn(expr_data_test_norm, cat_covs_test, num_covs_test)(gen)
print('Gamma(Dx, Dz): {:.2f}'.format(score))























