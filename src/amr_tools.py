import re
import subprocess
import os
import networkx as nx
from preprocessor.preprocessor import Preprocessor
import pickle as pkl


class AMRTools(object):
    """
    . {scripts_path}PARSE.sh < source-document00015_splitted.txt > {output_path} 2> out.out
    """

    def __init__(self):

        self.tab = "      "  # tab no AMR Parser é igual a 6 espaços
        self.space = " "
        self.open_parentheses = "("
        self.close_parentheses = ")"
        self.octothorp = "#"
        self.label_pattern = re.compile(":[A-Z|a-z|\-|of|0-9]+")
        self.jamr_root_path = "/home/forrest/workspace/LINE/Baselines/AMR/jamr"
        self.script_folder = os.path.join(self.jamr_root_path, "scripts")
        self.parser_path = os.path.join(self.jamr_root_path, self.script_folder)
        self.jamr_parse_config_cmd = ". {scripts_path}/config.sh"
        self.log_out_path = "/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/amr_out/"
        self.root_path = "/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader"

    # TODO: Improvement this method - OK
    # TODO:Must be return a graph set - OK
    def amr_graph_reader(self, path):
        """

        :param path:
        :param dictionary,
        :return:
        """
        stack_vertex = []
        stack_nodes = []
        edge_list = []
        graph_list = []

        lines = None

        with open(path, "r") as f:

            lines = f.readlines()

        size = len(lines)
        i = 0

        while i < size:

            if lines[i][0] == self.open_parentheses:

                # Getting root concept
                line = lines[i].replace("(", "").replace(")", "").replace("\n", "")

                vertex = {"name": line.strip(), "children_vertices": {}}

                stack_vertex.append(vertex)
                stack_nodes.append(vertex["name"])

                i = i + 1

                # Iterating over graph
                while i < size:

                    # Verifying end of graph
                    if lines[i][0] == "\n" or lines[i][0] == self.octothorp:
                        # Setting graph in list
                        graph_list.append(edge_list)

                        stack_vertex = []
                        stack_nodes = []
                        edge_list = []

                        i = i + 1
                        # Returning to processed new graph
                        break

                    # Getting line to be processed
                    line = lines[i].replace("(", "").replace(")", "").replace("\n", "")

                    # Counting the node's depth
                    lvl = line.count(self.tab)

                    # Catching label position in line
                    label_edge = re.search(self.label_pattern, line)

                    # Setting edge name
                    edge_name = line[label_edge.start(): label_edge.end()]

                    ###### Vertex handling #########
                    while len(stack_vertex) > lvl:
                        stack_vertex.pop()
                        stack_nodes.pop()

                    # Building vertex
                    vertex = {"name": line[label_edge.end() + 1:].strip(), "children_vertices": {}}

                    stack_vertex[-1]["children_vertices"][edge_name] = vertex
                    stack_vertex.append(vertex)

                    edge = stack_nodes[-1].replace(" ", "") + " " + vertex["name"].replace(" ", "") \
                           + " {'edge_name':'" + edge_name.replace(":", "") + "'}"

                    edge_list.append(edge)
                    stack_nodes.append(vertex["name"])
                    ###### Vertex handling #########

                    i = i + 1

            else:

                i = i + 1
                continue

        return graph_list

    @staticmethod
    def write_graph_list_in_file(write_path, graph_list):

        with open(write_path, "wb") as f:
            pkl.dump(graph_list, f)

    @staticmethod
    def read_graph_list_in_file(read_path):
        """

        :param read_path:
        :return:
        """

        with open(read_path, "rb") as f:
            data = pkl.load(f)

        return data

    @staticmethod
    def document_to_splitted_sentences(document__file_list, document__result_path):
        """

        :param document__file_list:
        :param document__result_path:
        :return:
        """
        files_path = []
        for file in document__file_list:

            if os.path.exists(file):
                document = Preprocessor().raw_document_splitter(file)

            else:
                raise Exception("")

            file_name = file.split("/")[-1].replace(".txt", "")
            out_path = document__result_path + file_name + ".txt"

            with open(out_path, "w") as f:

                f.write("\n".join(document))

            files_path.append(out_path)

        return files_path

    @staticmethod
    def parse_graph(parse__graph_str_list, graph_type=nx.DiGraph, vocabulary: dict = {}):
        """

        :param parse__graph_str_list:
        :param graph_type:
        :param vocabulary:
        :return:
        """

        parse__graph_list = []

        for graph_str in parse__graph_str_list:

            graph = nx.parse_edgelist(lines=graph_str, nodetype=str, create_using=graph_type)

            # Building dictionary TODO: Test this
            nodes = graph.nodes()

            size = len(vocabulary)

            # TODO: Using dictionary may cause excessive usage memory consumption
            for node in nodes:

                if node not in vocabulary.keys():
                    size = size + 1
                    vocabulary[node] = size

            parse__graph_list.append(graph)

        return parse__graph_list

    def generate_bag_of_concepts(self, generate__path_list, only_main_concept=False):

        generate__bOC = []

        for generate__path in generate__path_list:

            generate__graph_list = self.amr_graph_reader(generate__path)

            generate__graph_list = self.parse_graph(generate__graph_list, nx.DiGraph)

            for generate__graph in generate__graph_list:

                if only_main_concept:
                    generate__concepts = [node for node in generate__graph.nodes][0]
                else:
                    generate__concepts = [node for node in generate__graph.nodes]

                for generate__concept in generate__concepts:

                    if generate__concept not in generate__bOC:
                        generate__bOC.append(generate__concept)

        return generate__bOC

    def amr_parse(self, parse_amr__jamr_path, parse_amr__file_list, parse_amr__output_path):

        """
                From a text file the parser convert raw text to discourse tree structure

                This method need:
                    - A file with path of all texts will be parsed
                    - The out path where parsed files will be saved

        """

        print('Verifying dependencies...')

        if not os.path.exists(parse_amr__output_path):
            raise Exception("Output path: %s doesn't exists!!" % parse_amr__output_path)

        print('Starting parse...')

        length = len(parse_amr__file_list)

        for i in range(0, length):

            file = parse_amr__file_list[i]

            if file == '\n':
                break

            name = file.split("/")[-1].split('\n')[0]
            name_output = name[0: name.find('.txt')] + '.amr'
            name_output = os.path.join(parse_amr__output_path, name_output)

            if os.path.exists(name_output):
                print("The document %s was skipped because was processed!!!" % name)
                continue

            file = file.split('\n')[0]
            if not os.path.exists(file):
                raise Exception("The file %s doesn't exists!!" % file)

            print('Starting file processing %d of %d ...' % (i + 1, length))

            # Mounting command
            cmd = "./amr_generator.sh {jamr_path} {file_path} {out_path}".format(
                jamr_path=parse_amr__jamr_path,
                file_path=file,
                out_path=name_output
            )

            # Setting of process
            parser = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, universal_newlines=True)
            try:
                stdout, stderr = parser.communicate()

                if stderr != '':
                    raise OSError('Some error occurred in parser subprocess, error info:\n%s' % stderr)
                else:
                    print(stdout)

                print('File: {file} concluded'.format(file=name))

            except Exception as e:

                raise e

        del parse_amr__file_list
        print('Parse terminated!')


if __name__ == '__main__':
    path = "src/data/source-document00015.txt"
    result_path = "src/data/document_splitted/"

    tool = AMRTools()

    # print(tool.amr_graph_reader(
    #    "/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document02968.amr"))

    # graph = tool.amr_graph_reader("src/data/file_amr.txt")

    # graph = tool.amr_graph_reader("src/data/file_amr_3.txt")

    # tool.write_graph_list_in_file("src/data/documents_amr/file_amr_3", graph)

    # graph = tool.amr_graph_reader("/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/result_document00015.txt")

    # tool.document_to_sentences_splitted(path=path, result_path=result_path)

    # path_list = ["/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/documents_amr/file_amr_2",
    #             "/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/documents_amr/file_amr_3"]

    # boc = tool.generate_bag_of_concepts(path_list, True)

    file_list = [
        "/home/forrest/workspace/LINE/plagiarism_rae_rst/datasets/pan-plagiarism-corpus-2010/suspicious-document/part6/suspicious-document02968.txt",
        "/home/forrest/workspace/LINE/plagiarism_rae_rst/datasets/pan-plagiarism-corpus-2010/suspicious-document/part21/suspicious-document10403.txt",
        "/home/forrest/workspace/LINE/plagiarism_rae_rst/datasets/pan-plagiarism-corpus-2010/suspicious-document/part4/suspicious-document01501.txt",
        "/home/forrest/workspace/LINE/plagiarism_rae_rst/datasets/pan-plagiarism-corpus-2010/suspicious-document/part13/suspicious-document06001.txt"]

    # output_path_splitted = "/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/document_splitted/"

    output_path_amr = "/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results"

    # files_splitted = tool.document_to_splitted_sentences(document__file_list=file_list,
    #                                                     document__result_path=output_path_splitted)
    """
    files_splitted = [
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/document_splitted/source-document00015_splitted.txt',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/document_splitted/suspicious-document06001.txt',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/document_splitted/suspicious-document10403.txt',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/document_splitted/suspicious-document01501.txt',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/document_splitted/suspicious-document02968.txt']

    tool.amr_parse(parse_amr__jamr_path="/home/forrest/workspace/LINE/Baselines/AMR/jamr/scripts",
                   parse_amr__file_list=files_splitted,
                   parse_amr__output_path=output_path_amr)
    """
    files_amr = [
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document02968.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document10403.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document01501.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document06001.amr'
    ]

    print(tool.generate_bag_of_concepts(files_amr, False))
