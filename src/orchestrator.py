from amr_tools import AMRTools


def build_document_term_matrix(amr_document_list):
    """

    :param amr_document_list:
    :return:
    """

    vocabulary = {}
    documents_quantity = 0

    amr_tool = AMRTools()

    bag_of_concept = amr_tool.generate_bag_of_concepts(amr_document_list)

    # Generate document-concept matrix
    for amr_document_path in amr_document_list:

        documents_quantity = documents_quantity + 1
        amr_list = amr_tool.read_graph_list_in_file(amr_document_path)

        # Building amr graph set to represent document
        amr_graph_set = amr_tool.parse_graph(
            parse__graph_str_list=amr_list, 
            vocabulary=vocabulary
        )

        # TODO: Continue from here. You must find a way yo build matrix document term with only one iteration over list



if __name__ == '__main__':

    amr_document_list = [
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document02968.amr'
    ]

    build_document_term_matrix(amr_document_list=amr_document_list)
