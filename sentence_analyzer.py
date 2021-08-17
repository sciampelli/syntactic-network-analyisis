import xml.etree.ElementTree as ET
from alpinoxml import get_xml_from_alpino, get_starting_node
from edgelist import generate_edge_list, print_edge_list, get_lcc_count
from cytoscape import CytoscapeProcessor

class SentenceAnalyzer:
    def __init__(self):
        print("Creating cytoscape connection")
        self.cyto_processor = CytoscapeProcessor()

    def analyze(self, sentence):
        print("="*35)
        print("[*] Getting xml from alpino")
        xml = get_xml_from_alpino(sentence)
        print("[*] Analyzing sentence:", sentence)
        tree = ET.ElementTree(ET.fromstring(xml))
        root_node = get_starting_node(tree)

        print("[*] Generating edge list")
        edgelist = generate_edge_list(root_node)
        statistics = self.cyto_processor.process_from_edge_list(edgelist)
        if not statistics:
            return False

        self.cyto_processor.print(statistics)
        statistics["lcc_count"] = get_lcc_count(edgelist)

        # getting some basic statistics
        # findall uses xpath syntax (https://docs.python.org/3/library/xml.etree.elementtree.html#supported-xpath-syntax)
        node_count = len(root_node.findall(".//"))
        word_count = len(root_node.findall(".//*[@word]"))
        structural_node_count = node_count - word_count

        max_depth = self.find_depth(root_node, 0)
        subtrees = self.calculate_subtrees(root_node, min_childs=2)
        complex_subtrees3 = self.calculate_subtrees(root_node, min_childs=3)
        complex_subtrees4 = self.calculate_subtrees(root_node, min_childs=4)

        statistics["node_count"] = node_count
        statistics["structural_node_count"] = structural_node_count
        statistics["depth_count"] = max_depth
        statistics["word_count"] = word_count
        statistics["subtrees"] = subtrees
        statistics["complex_subtrees3"] = complex_subtrees3
        statistics["complex_subtrees4"] = complex_subtrees4

        print("Total nodes:", node_count)
        print("Total words:", word_count)
        print("LCC count:", statistics["lcc_count"])
        print("Total structural nodes:", structural_node_count)
        print("Word/structural node ratio:", 0 if structural_node_count == 0 else word_count / structural_node_count)
        print("Depth of sentence is:", max_depth)
        print("Amount of subtrees:", subtrees)
        print("Amount of complex subtrees (3):", complex_subtrees3)
        print("Amount of complex subtrees (4):", complex_subtrees4)
        print("="*35)

        return statistics

    # finds the depth of the sentence, given the root node with a iter of 0
    def find_depth(self, node,iter):
        # print("Finding depth of",node,"with iteration",iter)
        depth_cache = [iter]
        for child in node:
            child_depth = self.find_depth(child, iter + 1)
            depth_cache.append(child_depth)

        return max(depth_cache)

    # calculates the amount of nodes with more than min_childs child(s)
    def calculate_subtrees(self, node, min_childs = 2):
        amount = 0
        children_count = 0

        for child in node:
            children_count += 1
            amount += self.calculate_subtrees(child, min_childs)

        if children_count >= min_childs:
            amount += 1

        return amount
