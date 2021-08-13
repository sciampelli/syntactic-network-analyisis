import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import glob
import random
import pprint
from sentence_analyzer import SentenceAnalyzer

MAX_SENTENCES_PER_INTERVIEW = 120
OUTPUT_CSV_FILE = "output_per_sentence.csv"
analyzer = SentenceAnalyzer()
global_stats = []

def extract_values(items, column_name):
    list = []
    for item in items:
        list.append(item[column_name])

    return list


def get_avg(items):
    return sum(items) / len(items)


# for file in ["chadata/control/vrouwen/P303_pt.cha_filter.txt"]:
for file in glob.glob("chadata/control/vrouwen/*.cha_filter.txt"):
    all_sentence_stats = []

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

    # loop over all sentences and choose which statistics we want to use for our csv
    for sentence_stats in all_sentence_stats:
        stats_per_sentence = {"File":file}
        # Cytoscape stats
        diameter_graph = sentence_stats['diameter']

        nodeCount_graph = sentence_stats['nodeCount']
        edgeCount_graph = sentence_stats['edgeCount']
        avNeighbors_graph = sentence_stats['avNeighbors']
        radius_graph = sentence_stats['radius']
        avSpl_graph = sentence_stats['avSpl']
        density_graph = sentence_stats['density']
        ncc_graph = sentence_stats['ncc']

        # Cytoscape table stats
        leafcount_graph = sentence_stats['leafcount']

        stress_graph = sentence_stats['stress_max']
        stress_avg_graph = sentence_stats['stress']
        stress_min_graph = sentence_stats['stress_min']

        edgecount_graph = sentence_stats['edgecount_pernode']
        edgecount_max_graph = sentence_stats['edgecount_pernode_max']
        edgecount_min_graph = sentence_stats['edgecount_pernode_min']

        betweennesscentrality_graph = sentence_stats['betweennesscentrality']

        # all LCC cytoscape stats
        eccentricity_lcc_graph = sentence_stats['lcc_eccentricity']
        stress_lcc_graph = sentence_stats['lcc_stress_max']
        stress_lcc_avg_graph = sentence_stats['lcc_stress']
        edgecount_lcc_avg_graph = sentence_stats['lcc_edgecount_pernode']
        edgecount_lcc_graph = sentence_stats['lcc_edgecount_pernode_max']
        betweennesscentrality_lcc_graph = sentence_stats['lcc_betweennesscentrality']
        # our LCC count stat
        lcc_count_graph = sentence_stats["lcc_count"]

        stats_per_sentence["node count"] = nodeCount_graph
        stats_per_sentence["edge count"] = edgeCount_graph
        stats_per_sentence["N. of Neighbors"] = avNeighbors_graph
        stats_per_sentence["diameter"] = diameter_graph
        stats_per_sentence["Radius"] = radius_graph
        stats_per_sentence["Characteristic path length"] = avSpl_graph
        stats_per_sentence["density"] = density_graph
        stats_per_sentence["N. of connected components"] = ncc_graph

        stats_per_sentence["Maximum Stress"] = stress_graph
        stats_per_sentence["Minimum Stress"] = stress_min_graph
        stats_per_sentence["Stress"] = stress_avg_graph
        stats_per_sentence["leaf count"] = leafcount_graph
        stats_per_sentence["Edge Count Per Node"] = edgecount_graph
        stats_per_sentence["Maximum Edge Count Per Node"] = edgecount_max_graph
        stats_per_sentence["Minimum Edge Count Per Node"] = edgecount_min_graph
        stats_per_sentence["Betweenness Centrality"] = betweennesscentrality_graph

        # # Silvia stats
        node_count = sentence_stats["node_count"]
        structural_node_count = sentence_stats["structural_node_count"]
        depth_count = sentence_stats["depth_count"]
        word_count = sentence_stats["word_count"]
        subtrees = sentence_stats["subtrees"]
        complex_subtrees3 = sentence_stats["complex_subtrees3"]
        complex_subtrees4 = sentence_stats["complex_subtrees4"]

        stats_per_sentence["Silvia Statistics:"] = 1
        stats_per_sentence["Average Amount of Nodes"] = node_count
        stats_per_sentence["Average Amount of Word Nodes"] = word_count
        stats_per_sentence["Average Amount of Structural Nodes"] = structural_node_count
        stats_per_sentence["Average Depth"] = depth_count
        stats_per_sentence["Structural node/Word node"] = (structural_node_count / word_count)
        stats_per_sentence["Average Amount of Subtrees with 2 children nodes"] = subtrees
        stats_per_sentence["Average amount of Subtrees with  3 children nodes"] = complex_subtrees3
        stats_per_sentence["Average amount of Subtrees with 4 children nodes"] = complex_subtrees4

        global_stats.append(stats_per_sentence)
        pprint.pprint(stats_per_sentence)

csv_columns = global_stats[0].keys()
try:
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for row in global_stats:
            writer.writerow(row)
except IOError:
    print("I/O error")
