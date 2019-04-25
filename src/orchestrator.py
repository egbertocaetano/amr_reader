from amr_tools import AMRTools
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, lil_matrix
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
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from model.flow import Flow


class ConceptLSAPageRankFlow(Flow):

    def execute(self, options, args):

        execution_config, specific_path = handling_with_args(options, args)

        # Instance AMR Tool
        amr_tool = AMRTools(execution_config[AMR][JMAR_ROOT_PATH])

        dataset_name = execution_config[DATASET][NAME]

        # TODO: Test this -
        # Building Execution folder
        if specific_path is not None:

            execution_folder = specific_path

        else:

            execution_folder = datetime.now().strftime('%y-%m-%d__%H-%M-%S') + '__' + dataset_name
            execution_folder = execution_config[PREPROCESS][OUTPUT_PATH] + '/' + execution_folder

        # Mounting splitted documents path
        splitted_documents_path = execution_folder + '/' + SPLITTED_DOCUMENTS_PATH + '/'

        # Mounting amr path
        amr_output_path = execution_folder + '/' + AMRS + '/'

        # TODO: 1 - Texts in documents must be splitted in sentences;
        # Verifying if is file list or one file
        if execution_config[DATASET][PROCESS_TYPE] == FILE_LIST:

            # TODO: implement to document_list
            document_concept_mtx = None

        else:

            amr_files_path = parse_amr_in_file(parse_amr__jamr_path=execution_config[AMR][JMAR_ROOT_PATH],
                                               parse_amr__file=execution_config[PREPROCESS][PATH_LIST],
                                               parse_amr__tool=amr_tool,
                                               parse_amr__output_path=amr_output_path)

            document_concept_mtx = build_document_concept_matrix_term_frequency(bdcm__amr_document_list=amr_files_path,
                                                                                amr_tool=amr_tool)

            document_concept_mtx = build_document_concept_matrix_modified_term_frequency(document_concept_matrix=document_concept_mtx)

        # Salving document-concept-matrix
        matrix_output_path = execution_folder + '/' + MATRIX

        if not os.path.exists(matrix_output_path):
            os.mkdir(matrix_output_path)

        matrix_output_path = execution_folder + '/' + MATRIX + '/document-concept-matrix.npz'

        scipy.sparse.save_npz(matrix_output_path, document_concept_mtx)

        n_components = 13554

        # TODO: 5 - Apply LSA in built matrix
        document_concept_mtx = lsa(document_concept_mtx, n_components)

        # Save SVD
        svd_output_path = execution_folder + '/' + MATRIX + '/LSA_document-concept-matrix_' + str(n_components) + '.pkl'

        with open(svd_output_path, "wb") as f:
            pkl.dump(document_concept_mtx, f)

        # TODO: 6 - Classification


class WordFlow(Flow):

    def execute(self, options, args):

        execution_config, specific_path = handling_with_args(options, args)

        # Instance AMR Tool
        amr_tool = AMRTools(execution_config[AMR][JMAR_ROOT_PATH])

        dataset_name = execution_config[DATASET][NAME]

        # TODO: Test this -
        # Building Execution folder
        if specific_path is not None:

            execution_folder = specific_path

        else:

            execution_folder = datetime.now().strftime('%y-%m-%d__%H-%M-%S') + '__' + dataset_name
            execution_folder = execution_config[PREPROCESS][OUTPUT_PATH] + '/' + execution_folder

        # Mounting splitted documents path
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

            # Generating bag of words
            vocabulary = generate_bag_of_words(execution_config[PREPROCESS][PATH_LIST], input_type='filename')

            document_concept_mtx = merge_graph_and_page_rank_build_matrix_with_vocabulary(mgpr__amr_list=amr_files_path,
                                                                                          mgpr__vocabulary=vocabulary,
                                                                                          mgpr__tool=amr_tool)

        else:

            amr_files_path = parse_amr_in_file(parse_amr__jamr_path=execution_config[AMR][JMAR_ROOT_PATH],
                                               parse_amr__file=execution_config[PREPROCESS][PATH_LIST],
                                               parse_amr__tool=amr_tool,
                                               parse_amr__output_path=amr_output_path)

            # vocabulary = generate_bag_of_words([execution_config[PREPROCESS][PATH_LIST]], input_type='filename')

            with open(execution_config[PREPROCESS][PATH_LIST], 'r') as file:
                document = file.readlines()
            
            # TODO: 3 - PageRank; 4 - Build Matrix with page rank, summing all page rank values concepts occurrences document
            document_concept_mtx = merge_graph_and_page_rank_build_matrix(mgpr__amr_list=amr_files_path,
                                                                          mgpr__document_list=document,
                                                                          mgpr__tool=amr_tool,
                                                                          input_type='content')
            
            # document_concept_mtx, vocabulary = build_matrix_tfidf(document, 'content')

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


class ConceptFlow(Flow):

    def execute(self, options, args):

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

            document_concept_mtx = None

        else:

            amr_files_path = parse_amr_in_file(parse_amr__jamr_path=execution_config[AMR][JMAR_ROOT_PATH],
                                               parse_amr__file=execution_config[PREPROCESS][PATH_LIST],
                                               parse_amr__tool=amr_tool,
                                               parse_amr__output_path=amr_output_path)

            # TODO: 3 - PageRank; 4 - Build Matrix with page rank, summing all page rank values concepts occurrences document
            document_concept_mtx = merge_graph_and_page_rank_build_matrix(mgpr__amr_list=amr_files_path,
                                                                          mgpr__document_list=amr_files_path,
                                                                          mgpr__tool=amr_tool,
                                                                          input_type='filename')

        # Salving document-concept-matrix
        matrix_output_path = execution_folder + '/' + MATRIX

        if not os.path.exists(matrix_output_path):
            os.mkdir(matrix_output_path)

        matrix_output_path = execution_folder + '/' + MATRIX + '/document-concept-matrix.npz'

        scipy.sparse.save_npz(matrix_output_path, document_concept_mtx)

        n_components = 11542

        # TODO: 5 - Apply LSA in built matrix
        document_concept_mtx = lsa(document_concept_mtx, n_components)

        # Save SVD
        svd_output_path = execution_folder + '/' + MATRIX + '/LSA_document-concept-matrix_' + str(n_components) + '.pkl'

        with open(svd_output_path, "wb") as f:
            pkl.dump(document_concept_mtx, f)

    # TODO: 6 - Classification


def build_document_concept_matrix_term_frequency(bdcm__amr_document_list, amr_tool):
    """

    :param amr_tool:
    :param bdcm__amr_document_list:
    :return:
    """

    document_list = []

    # The bag of concept must be save and loaded in matrix building
    concept_vocabulary = amr_tool.generate_bag_of_concepts(bdcm__amr_document_list)

    document_quantity = len(bdcm__amr_document_list)
    vocabulary_quantity = len(concept_vocabulary)

    document_concept_matrix = build_matrix_zeros(document_quantity, vocabulary_quantity)

    # Generate document-concept matrix
    for doc_index in range(0, document_quantity):

        document_name = bdcm__amr_document_list[doc_index].split("/")[-1].split('\n')[0].replace('.amr', '')
        document_list.append(document_name)

        amr_list = amr_tool.amr_graph_reader(bdcm__amr_document_list[doc_index])

        # Building amr graph set to represent document
        amr_graph_set = amr_tool.parse_graph(parse__graph_str_list=amr_list)

        for graph in amr_graph_set:

            for concept in graph.nodes:

                concept_name = clean_node_name(concept)

                concept_col_index = concept_vocabulary.index(concept_name)

                document_concept_matrix[doc_index, concept_col_index] = document_concept_matrix[doc_index, concept_col_index] + 1

    return document_concept_matrix


def generate_bag_of_words(bdtm__document_list, input_type='filename'):
    """

    :param input_type:
    :param bdtm__document_list:
    :return:
    """

    vectorizer = CountVectorizer(input=input_type)

    analyzer = vectorizer.build_analyzer()

    def stemm(doc):
        stemmer = PorterStemmer()
        return (stemmer.stem(word) for word in analyzer(doc))

    vectorizer.analyzer = stemm

    vectorizer.fit(bdtm__document_list)

    vocabulary = vectorizer.vocabulary_

    return vocabulary


# TODO: Test -
def build_matrix_zeros(document_length, concept_vocabulary_length):
    """

    :param document_length:
    :param concept_vocabulary_length:
    :return:
    """
    matriz_zeros = lil_matrix(np.zeros((document_length, concept_vocabulary_length), dtype=np.float64),
                              dtype=np.float64)

    return matriz_zeros


def build_matrix_count(bmt__document_list, input_type='filename', with_analyzer=False, amr_tool=None):
    """

    :param input_type:
    :param bmt__document_list:
    :return:
    """

    vectorizer = CountVectorizer(input=input_type, dtype=np.float64)

    analyzer = vectorizer.build_analyzer()

    def stemm(doc):
        stemmer = PorterStemmer()
        return (stemmer.stem(word) for word in analyzer(doc))

    def nodes(doc):

        graph_str = amr_tool.amr_graph_reader(doc)

        graph_list = amr_tool.parse_graph(graph_str)

        _nodes = []

        for graph in graph_list:

            _nodes.extend(graph.nodes)

        return _nodes

    if with_analyzer:

        vectorizer.analyzer = stemm
    else:

        vectorizer.analyzer = nodes

    term_document_matrix = vectorizer.fit_transform(bmt__document_list)

    vocabulary = vectorizer.vocabulary_

    return term_document_matrix, vocabulary


# TODO: Test - Done
def build_document_concept_matrix_modified_term_frequency(document_concept_matrix):
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

    return document_concept_matrix.tocsr()


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
def reweighting_document_concept_matrix(document_concept_matrix, rows_to_reweighting, pagerank_graph,concept_vocabulary):
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


# TODO: improvement this to regular expression 'generalize'
def clean_node_name(node_name):
    """

    :param node_name:
    :return:
    """

    name = node_name.split('/')[-1]

    # name = name.split('-')[0]

    name = name.replace('"', '')

    name = name.strip()

    return name


def reweighting_document_term_matrix(document_term_matrix, rows_to_reweighting, pagerank_graph, word_vocabulary):
    """

    :param document_term_matrix:
    :param rows_to_reweighting:
    :param pagerank_graph:
    :param word_vocabulary:
    :return:
    """

    for row_index in rows_to_reweighting:

        for node_name, score in pagerank_graph.items():

            name = clean_node_name(node_name=node_name)

            if name in word_vocabulary.keys():

                col_index = word_vocabulary[name]

                # document_term_matrix[row_index, col_index] = document_term_matrix[row_index, col_index] + score
                document_term_matrix[row_index, col_index] = document_term_matrix[row_index, col_index] * score

    return document_term_matrix


# TODO: Test -
def lsa(matrix, n_components=100):
    """
    LSA

    :param n_components
    :param matrix:
    :return:
    """
    svd = TruncatedSVD(n_components=n_components)

    return svd.fit_transform(matrix)


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
def merge_graph_and_page_rank_build_matrix_with_vocabulary(mgpr__document_list, mgpr__vocabulary, mgpr__tool):
    """

    :param mgpr__document_list:
    :param mgpr__vocabulary:
    :param mgpr__tool:
    :return: document-concept matrix
    """

    document_length = len(mgpr__document_list)
    vocabulary_length = len(mgpr__vocabulary)

    # Building document-concept matrix filled with zeros
    doc_term_matrix = build_matrix_zeros(document_length, vocabulary_length)

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
            doc_term_matrix = reweighting_document_term_matrix(
                doc_term_matrix,
                rows_coordinate,
                documents_page_rank,
                mgpr__vocabulary
            )

    return doc_term_matrix.tocsr()


def merge_graph_and_page_rank_build_matrix(mgpr__amr_list, mgpr__document_list, mgpr__tool, input_type='filename'):
    """

    :param mgpr__amr_list:
    :param mgpr__vocabulary:
    :param mgpr__tool:
    :return: document-concept matrix
    """

    # Building document-concept matrix filled with zeros
    doc_term_matrix, vocabulary = build_matrix_count(bmt__document_list=mgpr__document_list,
                                                     input_type=input_type,
                                                     amr_tool=mgpr__tool)

    amr_length = len(mgpr__amr_list)

    for idx in range(0, amr_length, 2):

        idx_1 = idx
        idx_2 = idx + 1

        # Getting document paths
        document_1 = mgpr__amr_list[idx_1]
        document_2 = mgpr__amr_list[idx_2]

        # Applying collapse and merge in documents
        collapsed_merged_document = collapse_merge_document(document_1, document_2, mgpr__tool)

        for G in collapsed_merged_document:
            rows_coordinate = [idx_1, idx_2]

            # Calculating page rank of document graphs
            documents_page_rank = concept_rank(G, mgpr__tool)

            # reweighting documents
            doc_term_matrix = reweighting_document_term_matrix(
                doc_term_matrix,
                rows_coordinate,
                documents_page_rank,
                vocabulary
            )

    return doc_term_matrix.tocsr()


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

    :param options:
    :param args:
    :return:
    """

    flow = ConceptFlow()

    flow.execute(options=options, args=args)

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
    vocabulary = amr_tool.generate_bag_of_concepts(generate_boc__path_list=amr_files_path,
                                                   generate_boc__only_main_concept=False,
                                                   generate_boc__with_prefix=False)

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
    """


if __name__ == '__main__':

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
