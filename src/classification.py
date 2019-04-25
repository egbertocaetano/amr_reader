import pickle as pkl
import scipy.sparse
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from scipy.sparse.linalg import svds
from numpy.linalg import inv, pinv
from numpy import zeros, diagflat


target_path = '/home/forrest/workspace/LINE/Baselines/AMR/reader/Dataset/MSRParaphraseCorpus/msrpc_target.txt'
lsa_path_1 = '/home/forrest/workspace/LINE/Baselines/AMR/results/19-04-10__22-08-45__MSRParaphraseCorpus/matrix/LSA_document-concept-matrix_11542.pkl'
doc_conc_mtx_path = '/home/forrest/workspace/LINE/Baselines/AMR/results/19-04-10__22-08-45__MSRParaphraseCorpus/matrix/document-concept-matrix.npz'
settings_path = '/home/forrest/workspace/LINE/Baselines/AMR/reader/Dataset/MSRParaphraseCorpus/settings.txt'

with open(target_path, 'rb') as f:
    targets = pkl.load(f, encoding="latin1")

with open(settings_path, 'rb') as f:
    settings = pkl.load(f, encoding="latin1")


# with open(lsa_path_1, "rb") as f:
#     lsa_matrix_1 = pkl.load(f)

doc_conc_mtx = scipy.sparse.load_npz(doc_conc_mtx_path)

# U, s, Vt = svds(doc_conc_mtx, 11542)

path_lsa_components = '/home/forrest/workspace/LINE/Baselines/AMR/results/19-04-10__22-08-45__MSRParaphraseCorpus/matrix/LSA_document-concept-matrix_11542_components.npy'

# U, s, Vt = np.load(path_lsa_components)

# sigma = diagflat(s)

# transform = Vt.T.dot(pinv(sigma))

path_transform = '/home/forrest/workspace/LINE/Baselines/AMR/results/19-04-10__22-08-45__MSRParaphraseCorpus/matrix/transform_11542_components.npy'

transform = np.load(path_transform)

dataset = []

length = doc_conc_mtx.shape[0]

for idx in range(0, length, 2):

    row_1 = doc_conc_mtx[idx] * transform
    row_2 = doc_conc_mtx[idx+1] * transform

    sum_rows = np.sum([row_1, row_2], axis=0)

    absolute_value = np.absolute(np.subtract(row_1, row_2))

    new_vector = np.concatenate((sum_rows, absolute_value), axis=1)

    dataset.append(new_vector[0])


"""
length = lsa_matrix_1.shape[0]

for idx in range(0, length, 2):

    row_1 = lsa_matrix_1[idx]
    row_2 = lsa_matrix_1[idx+1]

    sum_rows = np.sum([row_1, row_2], axis=0)

    absolute_value = np.absolute(np.subtract(row_1, row_2)).tolist()

    new_vector = np.concatenate((sum_rows, absolute_value))

    dataset.append(new_vector)
"""
print("Rows: %d" % len(dataset))

settings[0] = 4060
settings[1] = 1741

training_data = dataset[0: settings[0]]
training_targets = targets[0: settings[0]]

testing_data = dataset[settings[0]:]
testing_targets = targets[settings[0]:]

print(len(training_data))
print(len(testing_data))

svc = SVC(kernel='linear')
modelSVC = svc.fit(training_data, training_targets)
predicted = modelSVC.predict(testing_data)
print('SVC:')
print(metrics.classification_report(testing_targets, predicted))
print('accuracy: %f', metrics.accuracy_score(testing_targets, predicted))
print('\n')
print(metrics.confusion_matrix(testing_targets, predicted))
print('==============================================================')
