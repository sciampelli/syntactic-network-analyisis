from py2cytoscape import cyrest
import tempfile
import os
import time
import json

class CytoscapeProcessor:
    def __init__(self):
        self.client = cyrest.cyclient()

    def process_from_edge_list(self, edgelist):
        # we cant do a network analysis on less than 6 nodes
        if len(edgelist) < 5:
            return False

        csv_file = self.write_edge_to_csv(edgelist)

        network = self.client.network.import_file(
            afile=csv_file,
            firstRowAsColumnNames="1",
            indexColumnSourceInteraction="1",
            indexColumnTargetInteraction="2",
            startLoadRow="1",
            defaultInteraction="interacts with"
        )
        # wait until cytoscape loaded the sentence
        time.sleep(.5)
        print("[*] imported network in cytoscape:", network)
        network_id = network["networks"][0]

        try:
            self.run_analyzer()
            stats = self.process_network(network_id)
            table_stats = self.get_table_stats(network_id)
            stats.update(table_stats)

            return stats
        finally:
            self.cleanup_workspace(network)

    def process_network(self, network_id):
        response = self.client.networks.getTables(networkId=network_id)

        for table in response.json():
            if "rows" not in table:
                continue
            if len(table["rows"]) == 0:
                continue
            if "layoutAlgorithm" not in table["rows"][0]:
                continue
            if table["rows"][0]["layoutAlgorithm"] != "Prefuse Force Directed Layout":
                continue

            statistics = json.loads(table["rows"][0]["statistics"])
            return self.convert_stats_to_number(statistics)

    def get_table_stats(self, network_id):
        response = self.client.networks.getTable(networkId=network_id, tableType="defaultnode")
        all_rows = json.loads(response)["rows"]
        lcc_rows = []
        statistics = {}

        for row in all_rows:
            if "lcc" in row["name"]:
                lcc_rows.append(row)

        extracted_lcc_values = {}
        extracted_values = {}

        values_to_extract = ["Eccentricity", "EdgeCount", "Stress", "BetweennessCentrality"]
        for column in values_to_extract:
            extracted_values[column] = extract_table_values(all_rows, column)

        print(extracted_values)
        # extract values specifically for the lcc rows
        for column in values_to_extract:
            extracted_lcc_values[column] = extract_table_values(lcc_rows, column)

        # count the amount of nodes with only one edge
        leafcount = 0
        for node in all_rows:
            # determine if this node is a leaf
            if node["EdgeCount"] == 1:
                leafcount = leafcount + 1

        statistics["leafcount"] = leafcount
        statistics["edgecount_pernode_max"] = max(extracted_values["EdgeCount"])
        statistics["edgecount_pernode_min"] = min(extracted_values["EdgeCount"])
        statistics["edgecount_pernode"] = sum(extracted_values["EdgeCount"]) / len(extracted_values["EdgeCount"])
        statistics["stress"] = sum(extracted_values["Stress"]) / len(extracted_values["Stress"])
        statistics["stress_max"] = max(extracted_values["Stress"])
        statistics["stress_min"] = min(extracted_values["Stress"])
        statistics["betweennesscentrality"] = sum(extracted_values["BetweennessCentrality"]) / len(extracted_values["BetweennessCentrality"])

        # gather lcc statistics
        statistics["lcc_eccentricity"] = max(extracted_lcc_values["Eccentricity"])
        statistics["lcc_edgecount_pernode_max"] = max(extracted_lcc_values["EdgeCount"])
        statistics["lcc_edgecount_pernode"] = sum(extracted_lcc_values["EdgeCount"]) / len(extracted_lcc_values["EdgeCount"])
        statistics["lcc_stress"] = sum(extracted_lcc_values["Stress"]) / len(extracted_lcc_values["Stress"])
        statistics["lcc_stress_max"] = max(extracted_lcc_values["Stress"])
        statistics["lcc_betweennesscentrality"] = sum(extracted_lcc_values["BetweennessCentrality"]) / len(extracted_lcc_values["BetweennessCentrality"])

        return statistics

    def get_lcc_stats_from_table(self, rows):
        statistics = {}

        return statistics

    def convert_stats_to_number(self, statistics):
        converted_statistics = {}
        for key in statistics:
            try:
                converted_statistics[key] = float(statistics[key])
            except:
                converted_statistics[key] = statistics[key]

        return converted_statistics

    def run_analyzer(self):
        script_file = os.path.dirname(__file__) + "//scripts//analyzer.cyto"
        self.client.command.run(script_file=script_file)

    def write_edge_to_csv(self, edgelist):
        f = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)

        for line in edgelist:
            f.write((line[0] + "," + line[1] + "\r\n").encode())

        print("[*] created edgelist csv file:", f.name)
        f.close()

        return f.name

    def cleanup_workspace(self, network):
        network_id = network["networks"][0]

        while True:
            try:
                self.cleanup(network_id)
                break
            except:
                time.sleep(1)
        self.client.session.runGarbageCollection()

    def cleanup(self, network_id):
        self.client.view.destroy()
        self.client.network.destroy(network="SUID:" + str(network_id))

    def print(self, statistics):
        print("[*] printing cytoscape statistics")
        print(json.dumps(statistics, indent = 4))


def extract_table_values(items, column_name):
    list = []
    for item in items:
        if column_name in item:
            list.append(item[column_name])

    return list
