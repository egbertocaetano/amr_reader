import nltk
tab = "      "# tab no AMR Parser é igual a 6 espaços
open_clasula = "("
close_clasula = ")"
stack_vertices = []
stack_open_parenthesis = []


with open("file_amr.txt", "r") as f:
    
    #concept_root
    line = f.readline()

    stack_open_parenthesis.append(open_clasula)
    
    limit = line.find(close_clasula)

    if limit > 0:
    	vertice = {"name": line[line.find(open_clasula): limit], "children_vertices":{}}
    else:
		vertice = {"name": line[line.find(open_clasula):], "children_vertices":{}}

    stack_vertices.append(vertice)

    for line in f: 
    
    	lvl = line.count(tab)

    	edge_name = line[line.find(":") - 1: line.find("(")].strip()
		
		######Parenthesis handling##########
    	stack_open_parenthesis.append(open_clasula)

    	count_close_parenthesis = line.count(close_clasula)

    	i == 0
    	while i < count_parenthesis:
    		stack_open_parenthesis.pop()
    		i++
    	######Parenthesis handling##########

    	######vertice handling#########
    	while len(stack_vertices) > lvl:
        	stack_vertices.pop()

    	limit = line.find(close_clasula)
	    if limit > 0:
	    	vertice = {"name": line[line.find(open_clasula): limit], "children_vertices":{}}
	    else:
			vertice = {"name": line[line.find(open_clasula):], "children_vertices":{}}

    	stack_vertices[-1]["children_vertices"][edge_name] = vertice
    	######vertice handling#########

    

print(data)
