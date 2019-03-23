import re
import subprocess
import os
import networkx as nx
from preprocessor.preprocessor import Preprocessor
import pickle as pkl
import matplotlib.pyplot as plt



class AMRTools(object):
    """
    . {scripts_path}PARSE.sh < source-document00015_splitted.txt > {output_path} 2> out.out
    """

    def __init__(self, jamr_root_path):

        self.tab = "      "  # tab no AMR Parser é igual a 6 espaços
        self.space = " "
        self.open_parentheses = "("
        self.close_parentheses = ")"
        self.octothorp = "#"
        self.label_pattern = re.compile(":[A-Z|a-z|\-|of|0-9]+")
        self.jamr_root_path = jamr_root_path
        self.script_folder = os.path.join(self.jamr_root_path, "scripts")
        self.parser_path = os.path.join(self.jamr_root_path, self.script_folder)
        self.jamr_parse_config_cmd = ". {scripts_path}/config.sh"

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
                    if lines[i][0] == "\n" or lines[i][0] == self.octothorp or len(lines[i]) == 0:
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

                    if not i < size:
                        graph_list.append(edge_list)

            else:

                i = i + 1
                continue

        return graph_list

    @staticmethod
    def write_graph_list_in_file(write_path, graph_list):

        with open(write_path, "wb") as f:
            pkl.dump(graph_list, f)

    @staticmethod
    def load_document_graph(read_path):
        """

        :param read_path:
        :return:
        """

        with open(read_path, "rb") as f:
            data = pkl.load(f)

        return data

    @staticmethod
    def document_to_splitted_sentences(document__file_list, document__output_path):
        """

        :param document__file_list:
        :param document__output_path:
        :return:
        """
        files_path = []
        for file in document__file_list:

            file_name = file.split("/")[-1].replace(".txt", "")
            out_path = document__output_path + file_name + ".txt"

            if os.path.exists(out_path):
                print("The document %s was skipped because was processed!!!" % file_name)
                files_path.append(out_path)
                continue

            if os.path.exists(file):

                print('Split processing %s ...' % file)
                document = Preprocessor().raw_document_splitter(file)

            else:
                raise Exception("Document doesn't exists!!!")

            file_name = file.split("/")[-1].replace(".txt", "")
            out_path = document__output_path + file_name + ".txt"

            with open(out_path, "w") as f:

                f.write("\n".join(document))

            files_path.append(out_path)

        return files_path

    @staticmethod
    def parse_graph(parse__graph_str_list, graph_type=nx.DiGraph):
        """

        :param parse__graph_str_list:
        :param graph_type:
        :param vocabulary:
        :return:
        """

        parse__graph_list = []

        for graph_str in parse__graph_str_list:
            graph = nx.parse_edgelist(lines=graph_str, nodetype=str, create_using=graph_type)

            parse__graph_list.append(graph)

            """
            # Building dictionary TODO: Test this
            nodes = graph.nodes()

            
            size = len(vocabulary)

            # TODO: Using dictionary may cause excessive usage memory consumption
            for node in nodes:

                if node not in vocabulary.keys():
                    size = size + 1
                    vocabulary[node] = size
            parse__graph_list.append(graph)
            """

        return parse__graph_list

    def generate_bag_of_concepts(self, generate__path_list, only_main_concept=False, with_prefix=False):

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

                    entity = generate__concept.split('/')[-1].strip()

                    if entity not in generate__bOC:

                        if with_prefix is False:

                            generate__bOC.append(entity)

                        else:
                            generate__bOC.append(generate__concept.strip())

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

        path_list = []

        for i in range(0, length):

            file = parse_amr__file_list[i]

            if file == '\n':
                break

            name = file.split("/")[-1].split('\n')[0]
            name_output = name[0: name.find('.txt')] + '.amr'
            name_output = os.path.join(parse_amr__output_path, name_output)

            if os.path.exists(name_output):
                print("The document %s was skipped because was processed!!!" % name)
                path_list.append(name_output)
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

            print(cmd)
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
                path_list.append(name_output)
            except Exception as e:

                raise e

        del parse_amr__file_list
        print('Parse terminated!')

        return path_list

    # TODO: Test -
    def amr_parse_corpus_in_one_file(self, parse_amr__jamr_path, parse_amr__file, parse_amr__output_path):
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

        file = parse_amr__file

        name = file.split("/")[-1].split('\n')[0]
        name_output = name[0: name.find('.txt')] + '.amr'
        name_output = os.path.join(parse_amr__output_path, name_output)

        if os.path.exists(name_output):

            print("The document %s was skipped because was processed!!!" % name)

            return self.__split_msr_paraphrase_corpus(split_msrpc__file_path=name_output,
                                                      split_msrpc__output_path=parse_amr__output_path
                                                      )

        file = file.split('\n')[0]

        if not os.path.exists(file):
            raise Exception("The file %s doesn't exists!!" % file)

        print('Starting %s file processing ...' % name)

        # Mounting command
        cmd = "./amr_generator.sh {jamr_path} {file_path} {out_path}".format(
            jamr_path=parse_amr__jamr_path,
            file_path=file,
            out_path=name_output
        )

        print(cmd)
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

        print('Parse terminated!')

        return self.__split_msr_paraphrase_corpus(split_msrpc__file_path=name_output,
                                                  split_msrpc__output_path=parse_amr__output_path
                                                  )

    # TODO: Test -
    def __split_msr_paraphrase_corpus(self, split_msrpc__file_path, split_msrpc__output_path):
        """
        """

        lines = None
        prefix_name = 'msrpc_'
        path_list = []

        with open(split_msrpc__file_path, "r") as f:

            lines = f.readlines()

        size = len(lines)
        
        i = 0
        j = 0
        k = 0

        while i < size:

            content_file = []

            if lines[i][0] == self.open_parentheses:

                # Getting content
                content_file.append(lines[i])
                
                i = i + 1

                # Iterating over graph
                while i < size:

                    # Verifying end of graph
                    if lines[i][0] == "\n" or lines[i][0] == self.octothorp or len(lines[i]) == 0:
                        
                        # Getting content
                        content_file.append(lines[i])

                        file_name = prefix_name + str(k) + '_' + str(j) + '.amr'
                        file_name = split_msrpc__output_path + '/' + file_name

                        with open(file_name, 'w') as f:

                            f.write(''.join(content_file))

                        path_list.append(file_name)

                        i = i + 1
                        j = j + 1

                        if j > 1:

                            j = 0
                            k = k + 1

                        break

                    # Getting content
                    content_file.append(lines[i])

                    i = i + 1
            else:

                i = i + 1
                continue

        return path_list

    # TODO: Test - DONE
    @staticmethod
    def collapse_graph(G):
        """

        :param G:
        :return:

        """

        # Generating node in queue
        node_list = [node for node in G.nodes]

        contract_graph = {}

        # Getting size graph nodes
        size_list = len(node_list)

        # Starting processing nodes equal 0
        processed = 0

        # Iterating over graph to collapse nodes
        while processed < size_list:

            # Getting node name
            node_name = node_list.pop(0)

            # Plus in processed counter
            processed = processed + 1

            # Getting node successors
            successors = G[node_name]

            # If successors is equal 1 then collapse may possible
            if len(successors) == 1:

                # Appending node name to collapse step
                names_to_join = [node_name]

                # Getting successor node name
                successor_equal_1 = list(dict(successors.items()).keys())

                # Getting edge labels in names to join
                edge_name = G.get_edge_data(node_name, successor_equal_1[0])
                # Adding edge label in name to join
                names_to_join.append(edge_name['edge_name'])

                # Iterating over successors of successor
                while len(successor_equal_1) > 0:

                    # Getting successor name
                    successor_name = successor_equal_1.pop(0)

                    # Getting successors of successors
                    successors_of_successor = G[successor_name]

                    # Verifying if successor is adequate to collapsed condition
                    if len(successors_of_successor) == 1:

                        # If yes, then add successor of successor to verified in collapsed condition
                        successor_of_successor = list(dict(successors_of_successor.items()).keys())[0]
                        successor_equal_1.append(successor_of_successor)

                        # Adding node name to collapse
                        names_to_join.append(successor_name)

                        # Getting edge labels in names to join
                        edge_name = G.get_edge_data(successor_name, successor_of_successor)
                        # Adding edge label in name to join
                        names_to_join.append(edge_name['edge_name'])

                    # Verifying if is leaf
                    elif len(successors_of_successor) == 0:

                        # If is leaf, add in list to collapse
                        names_to_join.append(successor_name)

                    else:
                        continue

                # If has more than 1, the nodes must collapsed
                if len(names_to_join) > 1:

                    # Building new name to collapsed nodes
                    new_name_node = "_".join(names_to_join)

                    # Adding collapsed nodes to graph list to contracted
                    contract_graph[new_name_node] = names_to_join

                # Removing node_names from node_list
                for node_name_i in names_to_join:

                    # Verifying if name_node is in node_list
                    if node_name_i in node_list:

                        # If in, remove node_name
                        processed = processed + 1
                        node_list.remove(node_name_i)

        # If has nodes to collapse, do contraction in G graph in iteration
        for key, node_name_list in contract_graph.items():

            # Getting first node in collapsed node
            first_node_in_collapse = node_name_list[0]

            # Get ancestor of first node in collapse
            ancestor = list(G.predecessors(first_node_in_collapse))
            ancestor = ancestor[0] if len(ancestor) > 0 else None

            # adding collapsed node in G graph
            G.add_node(key)

            # Create edge between ancestor and collapsed node
            if ancestor is not None:

                # Get edge name if have ancestor
                edge_name = G.get_edge_data(ancestor, first_node_in_collapse)

                # Creating new edge between ancestor and new node
                G.add_edge(ancestor, key, edge_name=edge_name['edge_name'])

            # Remove nodes in collapsed node from G graph
            for node_name_i in node_name_list:

                # If node name in graph, then delete
                if G.has_node(node_name_i):
                    G.remove_node(node_name_i)

        return G

    @staticmethod
    def collapse_merge_graphs(G, H):
        """

        :param G:
        :param H:
        :return:
        """
        # Searching for equal concepts
        has_equal = [node_name for node_name in G.nodes if H.has_node(node_name)]

        # Verifying exists nodes to collapse_merge
        if len(has_equal) < 1:

            return None

        else:

            return nx.compose(G, H)

    @staticmethod
    def concept_rank(G):
        """

        :param G:
        :return:
        """
        return nx.pagerank(G)


if __name__ == '__main__':
    """
    path = "src/data/source-document00015.txt"
    result_path = "src/data/document_splitted/"

    tool = AMRTools("/home/forrest/workspace/LINE/Baselines/AMR/jamr/scripts")

    R = tool.amr_graph_reader(
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/test_2.amr')

    graph_list = tool.parse_graph(R)

    
    for G in graph_list:

        nx.draw(G)

        plt.show()

        g_line = tool.collapse_graph(G)

        nx.draw(g_line)

        plt.show()

    

    merged_graph = tool.collapse_merge_graphs(graph_list[0], graph_list[1])

    exit()
    """
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
    """
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

    tool = AMRTools("/home/forrest/workspace/LINE/Baselines/AMR/jamr/scripts")

    tool.amr_parse(parse_amr__jamr_path="/home/forrest/workspace/LINE/Baselines/AMR/jamr/scripts",
                   parse_amr__file_list=files_splitted,
                   parse_amr__output_path='/home/forrest/workspace/LINE/Baselines/AMR/results/19-03-17__17-23-29__TestCorpus/amrs')

    """
    files_amr = [
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/document_splitted/source-document00015_splitted.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document02968.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document10403.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document01501.amr',
        '/home/forrest/workspace/LINE/Baselines/AMR/reader/amr_reader/src/data/amr_results/suspicious-document06001.amr'
    ]

    print(tool.generate_bag_of_concepts(files_amr, False))
    """