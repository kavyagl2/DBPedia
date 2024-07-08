import networkx as nx

def PreProcess(path, lang):
    """
    Preprocesses a set of RDF triples into a graph dataset using reification scheme.

    Loads RDF triples from a file and creates a NetworkX MultiDiGraph where:
    - Nodes represent entities from the triples.
    - Edges between nodes represent connections between entities with labels 'A_ZERO' and 'A_ONE'.

    Args:
    - path (str): Path to the RDF triple source file.
    - lang (str): Language identifier for operations.

    Returns:
    - nodes (list): List of node lists for each set of triples.
    - labels (list): List of edge labels for each set of triples.
    - node1 (list): List of start nodes for each edge in each set of triples.
    - node2 (list): List of end nodes for each edge in each set of triples.
    """
    nodes = []
    labels = []
    node1 = []
    node2 = []

    with open(path, 'r', encoding='utf-8') as dest:
        lang = '<' + lang + '>'
        for line in dest:
            g = nx.MultiDiGraph()
            temp_label = []
            temp_node1 = []
            temp_node2 = []
            triple_list = line.split('<TSP>')
            for l in triple_list:
                l = l.strip().split(' | ')
                g.add_edge(l[0], l[1], label='A_ZERO')
                g.add_edge(l[1], l[2], label='A_ONE')
            node_list = list(g.nodes())
            node_list.insert(0, lang)
            nodes.append(node_list)
            edge_list = list(g.edges.data())
            for edge in edge_list:
                temp_node1.append(edge[0])
                temp_node2.append(edge[1])
                label = edge[2]['label']
                temp_label.append(label)
            node1.append(temp_node1)
            node2.append(temp_node2)
            labels.append(temp_label)

    return nodes, labels, node1, node2
