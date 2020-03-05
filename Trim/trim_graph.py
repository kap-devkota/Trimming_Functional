import argparse
import json

parser = argparse.ArgumentParser();

parser.add_argument("graph_file", help = "Graph file")
parser.add_argument("state_file", help = "State file")
parser.add_argument("-j", "--json_file", help = "JSON file")
