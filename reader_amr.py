import re
tab = "     "  # tab no AMR Parser é igual a 6 espaços
space = " "
open_clasula = "("
close_clasula = ")"
stack_vertices = []
label_pattern = re.compile(":[A-Z|a-z|\-|of|0-9]+")


with open("data/file_amr.txt", "r") as f:

    # concept_root
    line = f.readline()

    line = line.replace("(", "").replace(")", "").replace("\n", "")

    vertice = {"name": line.strip(), "children_vertices": {}}

    stack_vertices.append(vertice)

    for line in f:

        line = line.replace("(", "").replace(")", "").replace("\n", "")

        lvl = line.count(tab)

        label_edge = re.search(label_pattern, line)

        edge_name = line[label_edge.start(): label_edge.end()]

        ######vertice handling#########
        while len(stack_vertices) > lvl:
            stack_vertices.pop()

        vertice = {"name": line[label_edge.end()+1:].strip(), "children_vertices": {}}

        stack_vertices[-1]["children_vertices"][edge_name] = vertice
        stack_vertices.append(vertice)
        ######vertice handling#########

print(stack_vertices)
