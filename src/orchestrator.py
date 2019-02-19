from amr_tools import AMRTools
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD


def build_document_concept_matrix(amr_document_list):
    """

    :param amr_document_list:
    :return:
    """

    document_list = []
    amr_tool = AMRTools()

    # The bag of concept must be save and loaded in matrix building
    concept_vocabulary = amr_tool.generate_bag_of_concepts(amr_document_list)

    document_quantity = len(amr_document_list)
    vocabulary_quantity = len(concept_vocabulary)

    document_concept_matrix = dok_matrix((document_quantity, vocabulary_quantity), dtype=np.float64)

    # Generate document-concept matrix
    for doc_index in range(0, document_quantity):

        document_name = amr_document_list[doc_index].split("/")[-1].split('\n')[0].replace('.amr', '')
        document_list.append(document_name)

        amr_list = amr_tool.amr_graph_reader(amr_document_list[doc_index])

        # Building amr graph set to represent document
        amr_graph_set = amr_tool.parse_graph(parse__graph_str_list=amr_list)

        for graph in amr_graph_set:

            for concept in graph.nodes:

                concept_name = concept.split('/')[-1].strip()

                concept_index = concept_vocabulary.index(concept_name)

                document_concept_matrix[doc_index, concept_index] = document_concept_matrix[doc_index, concept_index] + 1

    return document_concept_matrix, concept_vocabulary, document_list


def reweighting_document_concept_matrix(document_concept_matrix, vocabulary):

    """

    :param document_concept_matrix:
    :return:
    """
    # Getting column length
    row_length, column_length = document_concept_matrix.shape

    for col_index in range(0, column_length):

        appear_in_document = document_concept_matrix.getcol(col_index).nonzero()[0].size

        document_concept_matrix[:, col_index] = document_concept_matrix[:, col_index].dot(row_length/appear_in_document)

    return document_concept_matrix


if __name__ == '__main__':

    amr_document_list = [
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/source-document00015_splitted.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document02968.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document10403.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document01501.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document06001.amr'
    ]

    matrix, vocabulary, document_list = build_document_concept_matrix(amr_document_list=amr_document_list)

    matrix_l = reweighting_document_concept_matrix(matrix, vocabulary)

    # LSA Truncated SVD
    svd = TruncatedSVD()

    matrix_l_truncated = svd.fit(matrix_l)

    print(svd.explained_variance_ratio_)

    print('End...')

