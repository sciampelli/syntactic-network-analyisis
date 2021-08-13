import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import glob
import random
import pprint
from sentence_analyzer import SentenceAnalyzer

MAX_SENTENCES_PER_INTERVIEW = 12
OUTPUT_CSV_FILE = "output.csv"
analyzer = SentenceAnalyzer()
global_stats = []

def extract_values(items, column_name):
    list = []
    for item in items:
        list.append(item[column_name])

    return list


def get_avg(items):
    return sum(items) / len(items)

#analyse sentences per participant or per group on a text file
for file in ["chadata/psychosis/vrouwen/P025_pt.cha_filter.txt"]:
# for file in glob.glob("chadata/psychosis/vrouwen/*.cha_filter.txt"):
    all_sentence_stats = []
    stats_per_subject = {"File":file}

    # reading from cleaned cha file
    with open(file) as fp:
        sentences = fp.readlines()

    random.shuffle(sentences)
    sentence_count = 0
    for sentence in sentences:
        # we want to take a maximum amount of sentences per subject
        if sentence_count >= MAX_SENTENCES_PER_INTERVIEW:
            break
        sentence_stats = analyzer.analyze(sentence)
        if sentence_stats:
            all_sentence_stats.append(sentence_stats)
            sentence_count = sentence_count + 1

    # Cytoscape stats
    diameter_graph = extract_values(all_sentence_stats, 'diameter')


    nodeCount_graph = extract_values(all_sentence_stats, 'nodeCount')
    edgeCount_graph = extract_values(all_sentence_stats, 'edgeCount')
    avNeighbors_graph = extract_values(all_sentence_stats, 'avNeighbors')
    radius_graph = extract_values(all_sentence_stats, 'radius')
    avSpl_graph = extract_values(all_sentence_stats, 'avSpl')
    density_graph = extract_values(all_sentence_stats, 'density')
    ncc_graph = extract_values(all_sentence_stats, 'ncc')

    # Cytoscape table stats
    leafcount_graph = extract_values(all_sentence_stats, 'leafcount')

    stress_graph = extract_values(all_sentence_stats, 'stress_max')
    stress_avg_graph = extract_values(all_sentence_stats, 'stress')
    stress_min_graph = extract_values(all_sentence_stats, 'stress_min')

    edgecount_graph = extract_values(all_sentence_stats, 'edgecount_pernode')
    edgecount_max_graph = extract_values(all_sentence_stats, 'edgecount_pernode_max')
    edgecount_min_graph = extract_values(all_sentence_stats, 'edgecount_pernode_min')

    betweennesscentrality_graph = extract_values(all_sentence_stats, 'betweennesscentrality')

    # all LCC cytoscape stats
    eccentricity_lcc_graph = extract_values(all_sentence_stats, 'lcc_eccentricity')
    stress_lcc_graph = extract_values(all_sentence_stats, 'lcc_stress_max')
    stress_lcc_avg_graph = extract_values(all_sentence_stats, 'lcc_stress')
    edgecount_lcc_avg_graph = extract_values(all_sentence_stats, 'lcc_edgecount_pernode')
    edgecount_lcc_graph = extract_values(all_sentence_stats, 'lcc_edgecount_pernode_max')
    betweennesscentrality_lcc_graph = extract_values(all_sentence_stats, 'lcc_betweennesscentrality')
    # our LCC count stat
    lcc_count_graph = extract_values(all_sentence_stats, "lcc_count")

    stats_per_subject["Average node count"] = get_avg(nodeCount_graph)
    stats_per_subject["Average edge count"] = get_avg(edgeCount_graph)
    stats_per_subject["Average N. of Neighbors"] = get_avg(avNeighbors_graph)
    stats_per_subject["Average diameter"] = get_avg(diameter_graph)
    stats_per_subject["Average Radius"] = get_avg(radius_graph)
    stats_per_subject["Characteristic path length"] = get_avg(avSpl_graph)
    stats_per_subject["Average density"] = get_avg(density_graph)
    stats_per_subject["Average N. of connected components"] = get_avg(ncc_graph)

    stats_per_subject["Maximum Stress"] = get_avg(stress_graph)
    stats_per_subject["Minimum Stress"] = get_avg(stress_min_graph)
    stats_per_subject["Average Stress"] = get_avg(stress_avg_graph)
    stats_per_subject["Average leaf count"] = get_avg(leafcount_graph)
    stats_per_subject["Maximum leaf count"] = max(leafcount_graph)
    stats_per_subject["Minimum leaf count"] = min(leafcount_graph)
    stats_per_subject["Average Edge Count Per Node"] = get_avg(edgecount_graph)
    stats_per_subject["Maximum Edge Count Per Node"] = get_avg(edgecount_max_graph)
    stats_per_subject["Minimum Edge Count Per Node"] = get_avg(edgecount_min_graph)
    stats_per_subject["Average Betweenness Centrality"] = get_avg(betweennesscentrality_graph)

    # stats_per_subject["Maximum LCC Eccentricity/Diameter"] = get_avg(eccentricity_lcc_graph)
    # stats_per_subject["Maximum LCC Stress"] = get_avg(stress_lcc_graph)
    # stats_per_subject["Average LCC Stress"] = get_avg(stress_lcc_avg_graph)
    # stats_per_subject["Average LCC Edge Count Per Node"] = get_avg(edgecount_lcc_avg_graph)
    # stats_per_subject["Maximum LCC Edge Count Per Node"] = get_avg(edgecount_lcc_graph)
    # stats_per_subject["Average LCC Betweenness Centrality"] = get_avg(betweennesscentrality_lcc_graph)

    # find max edgeCount sentence and get the LCC statistics on that sentence
    max_edgecount_index = np.argmax(edgeCount_graph)
    stats_per_subject["Longest Sentence LCC node count"] = lcc_count_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence Maximum LCC Eccentricity/Diameter"] = eccentricity_lcc_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence Maximum LCC Stress"] = stress_lcc_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence Average LCC Stress"] = stress_lcc_avg_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence Average LCC Edge Count Per Node"] = edgecount_lcc_avg_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence Maximum LCC Edge Count Per Node"] = edgecount_lcc_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence Average LCC Betweenness Centrality"] = betweennesscentrality_lcc_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence node count"] = nodeCount_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence edge count"] = edgeCount_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence leaf count"] = leafcount_graph[max_edgecount_index]
    stats_per_subject["Longest Sentence diameter"] = diameter_graph[max_edgecount_index]

    # # Silvia stats
    node_count = extract_values(all_sentence_stats, "node_count")
    structural_node_count = extract_values(all_sentence_stats, "structural_node_count")
    depth_count = extract_values(all_sentence_stats, "depth_count")
    word_count = extract_values(all_sentence_stats, "word_count")
    subtrees = extract_values(all_sentence_stats, "subtrees")
    complex_subtrees3 = extract_values(all_sentence_stats, "complex_subtrees3")
    complex_subtrees4 = extract_values(all_sentence_stats, "complex_subtrees4")

    stats_per_subject["Silvia Statistics:"] = 1
    stats_per_subject["Average Amount of Nodes"] = get_avg(node_count)
    stats_per_subject["Average Amount of Word Nodes"] = get_avg(word_count)
    stats_per_subject["Average Amount of Structural Nodes"] = get_avg(structural_node_count)
    stats_per_subject["Average Depth"] = get_avg(depth_count)
    stats_per_subject["Structural node/Word node"] = (get_avg(structural_node_count) / get_avg(word_count))
    stats_per_subject["Average Amount of Subtrees with 2 children nodes"] = get_avg(subtrees)
    stats_per_subject["Average amount of Subtrees with  3 children nodes"] = get_avg(complex_subtrees3)
    stats_per_subject["Average amount of Subtrees with 4 children nodes"] = get_avg(complex_subtrees4)

    global_stats.append(stats_per_subject)
    pprint.pprint(stats_per_subject)


csv_columns = global_stats[0].keys()
try:
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for row in global_stats:
            writer.writerow(row)
except IOError:
    print("I/O error")

# node_graph = np.array(node_graph)
# complex_subtrees4_graph = np.array(complex_subtrees4_graph)
# area = np.pi*3
# plt.scatter(node_graph, depth_count_graph, s=area, alpha=0.5, label="Depth")
# plt.scatter(node_graph, structural_node_count_graph, s=area, alpha=0.5, label="Structural nodes")
# plt.scatter(node_graph, word_count_graph, s=area, alpha=0.5, label="Word Nodes")
# plt.scatter(node_graph, subtrees_graph, s=area, alpha=0.5, label="Subtrees with 2 children nodes")
# plt.scatter(node_graph, complex_subtrees3_graph, s=area, alpha=0.5, label="Subtrees with 3 children nodes")
# plt.scatter(node_graph, complex_subtrees4_graph, s=area, alpha=0.5, label="Subtrees with 4 children nodes")
# plt.hist(complex_subtrees4_graph / node_graph)
# print("aaaaaaaaaaaaaaaaaaaaaaaazzzz",complex_subtrees4_graph / node_graph)
# # plt.legend()
# plt.xlim([0, 0.14])
# plt.ylim([0, 100])

# plt.plot(node_graph)
# plt.title("Average Statistics")
# plt.xlabel('Nodes')
# plt.show()
#
# plt.plot()
