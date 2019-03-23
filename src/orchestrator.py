from amr_tools import AMRTools
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from glossary import *
from datetime import datetime
import json
from optparse import OptionParser
import sys
import os
import errno
import pickle as pkl

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

    document_concept_matrix = csr_matrix((document_quantity, vocabulary_quantity), dtype=np.float64)

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

                document_concept_matrix[doc_index, concept_index] = document_concept_matrix[
                                                                        doc_index, concept_index] + 1

    return document_concept_matrix, concept_vocabulary, document_list


# TODO: Test -
def build_matrix_zeros(document_length, concept_vocabulary_length):
    """

    :param document_length:
    :param concept_vocabulary_length:
    :return:
    """
    matriz_zeros = csr_matrix(np.zeros((document_length, concept_vocabulary_length), dtype=np.float64),
                              dtype=np.float64)

    return matriz_zeros


# TODO: Test - Done
def document_concept_matrix(document_concept_matrix):
    """

    :param document_concept_matrix:
    :return:
    """
    # Getting column length
    row_length, column_length = document_concept_matrix.shape

    for col_index in range(0, column_length):
        appear_in_document = document_concept_matrix.getcol(col_index).nonzero()[0].size

        document_concept_matrix[:, col_index] = document_concept_matrix[:, col_index].dot(
            row_length / appear_in_document)

    return document_concept_matrix


# TODO: Test - Done
def collapse_merge_document(cmd__doc1, cmd__doc2, cmd__tool):
    # TODO: load documents graph str
    graph_list_1 = cmd__tool.amr_graph_reader(cmd__doc1)
    graph_list_2 = cmd__tool.amr_graph_reader(cmd__doc2)

    # TODO: Building graph list
    graph_list_1 = cmd__tool.parse_graph(graph_list_1)
    graph_list_2 = cmd__tool.parse_graph(graph_list_2)

    collapsed_merged_graphs = []

    for G in graph_list_1:

        for H in graph_list_2:

            graph = cmd__tool.collapse_merge_graphs(G, H)

            if graph is None:
                continue

            collapsed_merged_graphs.append(graph)

    del graph_list_1, graph_list_2

    return collapsed_merged_graphs


# TODO: Test - Done
def concept_rank(G, cr__amr_tool):
    """

    :param G:
    :param cr__amr_tool:
    :return:
    """

    rank = cr__amr_tool.concept_rank(G)

    return rank


# TODO: Test - Done
def reweighting_document_concept_matrix(document_concept_matrix, rows_to_reweighting, pagerank_graph,
                                        concept_vocabulary):
    """

    :param document_concept_matrix:
    :param rows_to_reweighting:
    :param pagerank_graph
    :return:
    """

    for row_index in rows_to_reweighting:

        for node_name, score in pagerank_graph.items():
            col_index = concept_vocabulary.index(node_name.split('/')[-1])

            document_concept_matrix[row_index, col_index] = document_concept_matrix[row_index, col_index] + score

    return document_concept_matrix


# TODO: Test -
def lsa(document_concept_matrix):
    """
    LSA

    :param document_concept_matrix:
    :return:
    """
    svd = TruncatedSVD(n_components=100)

    return svd.fit(document_concept_matrix)


# TODO: Test -
def split_documents(split__document_list, split__tool, split__output_path):
    """

    :param split__document_list:
    :param split__tool:
    :param split__output_path:
    :return:
    """

    # Creating path if doesn't exists
    create_path(split__output_path)

    document_list = []

    with open(split__document_list, 'r') as f:
        for line in f.readlines():
            document_list.append(line.replace('\n', ''))

    return split__tool.document_to_splitted_sentences(document__file_list=document_list,
                                                      document__output_path=split__output_path)


# TODO: Test -
def parse_amr_file_list(parse_amr__jamr_path, parse_amr__file_list, parse_amr__tool, parse_amr__output_path):
    """
    :param parse_amr__jamr_path:
    :param parse_amr__file_list:
    :param parse_amr__tool:
    :param parse_amr__output_path:
    :return:
    """

    # Creating path if doesn't exists
    create_path(parse_amr__output_path)

    return parse_amr__tool.amr_parse(parse_amr__jamr_path, parse_amr__file_list, parse_amr__output_path)


def parse_amr_in_file(parse_amr__jamr_path, parse_amr__file, parse_amr__tool, parse_amr__output_path):
    """

    :param parse_amr__jamr_path:
    :param parse_amr__file:
    :param parse_amr__tool:
    :param parse_amr__output_path:
    :return:
    """
    # Creating path if doesn't exists
    create_path(parse_amr__output_path)

    return parse_amr__tool.amr_parse_corpus_in_one_file(parse_amr__jamr_path,
                                                        parse_amr__file,
                                                        parse_amr__output_path
                                                        )


# TODO: Test -
def merge_graph_and_page_rank_build_matrix(mgpr__document_list, mgpr__vocabulary, mgpr__tool):
    """

    :param mgpr__document_list:
    :param mgpr__vocabulary:
    :param mgpr__tool:
    :return: document-concept matrix
    """

    document_length = len(mgpr__document_list)
    vocabulary_length = len(mgpr__vocabulary)

    # Building document-concept matrix filled with zeros
    doc_concept_matrix = build_matrix_zeros(document_length, vocabulary_length)

    for idx in range(0, document_length, 2):

        idx_1 = idx
        idx_2 = idx + 1

        # Getting document paths
        document_1 = mgpr__document_list[idx_1]
        document_2 = mgpr__document_list[idx_2]

        # Applying collapse and merge in documents
        collapsed_merged_document = collapse_merge_document(document_1, document_2, mgpr__tool)

        for G in collapsed_merged_document:
            rows_coordinate = [idx_1, idx_2]

            # Calculating page rank of document graphs
            documents_page_rank = concept_rank(G, mgpr__tool)

            # reweighting documents
            doc_concept_matrix = reweighting_document_concept_matrix(
                doc_concept_matrix,
                rows_coordinate,
                documents_page_rank,
                mgpr__vocabulary
            )

    return doc_concept_matrix


def handling_with_args(options, args):
    """

    :param options:
    :param args:
    :return:
    """
    idx = 0

    specific_path = None

    if options.config_path:
        config_path = args[idx]
        idx += 1
    else:
        raise Exception("Path config isn't specified!")

    # TODO: Test this -
    if options.specific_path:
        specific_path = args[idx]

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    return config_data, specific_path


def create_path(cp__path):
    """

    :param cp__path:
    :return:
    """
    if not os.path.exists(os.path.dirname(cp__path)):
        try:
            os.makedirs(os.path.dirname(cp__path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# TODO: Test -
def orchestrate(options, args):
    """

    :param execution_config:
    :return:
    """

    execution_config, specific_path = handling_with_args(options, args)

    # Instance AMR Tool
    amr_tool = AMRTools(execution_config[AMR][JMAR_ROOT_PATH])

    dataset_name = execution_config[DATASET][NAME]

    # TOOD: Test this - 
    # Building Execution folder
    if specific_path is not None:

        execution_folder = specific_path

    else:

        execution_folder = datetime.now().strftime('%y-%m-%d__%H-%M-%S') + '__' + dataset_name
        execution_folder = execution_config[PREPROCESS][OUTPUT_PATH] + '/' + execution_folder

    # Mounting spltted documents path
    splitted_documents_path = execution_folder + '/' + SPLITTED_DOCUMENTS_PATH + '/'

    # Mounting amr path
    amr_output_path = execution_folder + '/' + AMRS + '/'

    # TODO: 1 - Texts in documents must be splitted in sentences;
    # Verifying if is file list or one file
    if execution_config[DATASET][PROCESS_TYPE] == FILE_LIST:

        documents_splitted_path = split_documents(split__document_list=execution_config[PREPROCESS][PATH_LIST],
                                                  split__tool=amr_tool,
                                                  split__output_path=splitted_documents_path)

        # TODO: 2 - Apply AMR Parser in documents splitted or document-sentences (MSRParaphraseCorpus)
        amr_files_path = parse_amr_file_list(parse_amr__jamr_path=execution_config[AMR][JMAR_ROOT_PATH],
                                             parse_amr__file_list=documents_splitted_path,
                                             parse_amr__tool=amr_tool,
                                             parse_amr__output_path=amr_output_path)

    else:

        amr_files_path = parse_amr_in_file(parse_amr__jamr_path=execution_config[AMR][JMAR_ROOT_PATH],
                                           parse_amr__file=execution_config[PREPROCESS][PATH_LIST],
                                           parse_amr__tool=amr_tool,
                                           parse_amr__output_path=amr_output_path)

    # Generating bag of concepts
    vocabulary = amr_tool.generate_bag_of_concepts(generate__path_list=amr_files_path,
                                                   only_main_concept=False,
                                                   with_prefix=False)

    # TODO: 3 - PageRank; 4 - Build Matrix with page rank, summing all page rank values concepts occurrences document
    document_concept_mtx = merge_graph_and_page_rank_build_matrix(mgpr__document_list=amr_files_path,
                                                                  mgpr__vocabulary=vocabulary,
                                                                  mgpr__tool=amr_tool)

    # Salving document-concept-matrix
    matrix_output_path = execution_folder + '/' + MATRIX

    if not os.path.exists(matrix_output_path):
        os.mkdir(matrix_output_path)

    matrix_output_path = execution_folder + '/' + MATRIX + '/document-concept-matrix.npz'

    scipy.sparse.save_npz(matrix_output_path, document_concept_mtx)

    # TODO: 5 - Apply LSA in built matrix
    document_concept_mtx = lsa(document_concept_mtx)

    # Save SVD
    svd_output_path = execution_folder + '/' + MATRIX + '/LSA_document-concept-matrix.pkl'

    with open(svd_output_path, "wb") as f:
        pkl.dump(document_concept_mtx, f)


    # TODO: 6 - Classification


if __name__ == '__main__':
    """
    amr_document_list = [
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document02968.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document10403.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document01501.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document06001.amr'
    ]

    matrix, vocabulary, document_list = build_document_concept_matrix(amr_document_list=amr_document_list)

    matrix_l = document_concept_matrix(matrix)

    # print(matrix_l)

    for i in range(0, 4, 2):

        doc1 = amr_document_list[i]
        doc2 = amr_document_list[i + 1]

        merged_graphs = collapse_merge_document(doc1, doc2)

        for G in merged_graphs:
            rows = [i, i + 1]

            # Testing method
            page_rank = concept_rank(G)

            # Testing method
            matrix_l = reweighting_document_concept_matrix(
                matrix_l,
                rows,
                page_rank,
                vocabulary
            )

    print('End...')
    """

    usage = "Usage: %prog [options] input_file/dir"
    v = '1.0'

    optParser = OptionParser(usage=usage, version="%prog " + v)

    optParser.add_option("-c", "--config_path",
                         action="store_true",
                         dest="config_path",
                         default=False,
                         help="Indicates the config file location")

    optParser.add_option("-s", "--specific_path",
                         action="store_true",
                         dest="specific_path",
                         default=False,
                         help="To specific folder to store file while execute or to continue previus execution")

    (options, args) = optParser.parse_args()

    if len(args) == 0:
        optParser.print_help()
        sys.exit(1)

    orchestrate(options, args)
