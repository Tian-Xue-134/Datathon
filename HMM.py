from cgi import test
from itertools import count
import json
from nbformat import read
from numpy import average
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import copy
from math import exp

def read_file(filename):
    f = open(filename)
    data = json.load(f)
    f.close()
    return data

def read_graph():
    f = pd.read_csv('training_graph.csv')
    graph = defaultdict(list)
    for i in range(len(f)):
        graph[str(f['node1'][i])].append(str(f['node2'][i]))
        graph[str(f['node2'][i])].append(str(f['node1'][i]))
    return graph

def read_classification():
    cata_dict = defaultdict(list)
    data = pd.read_csv('node_classification.csv')['page_type']
    for i in range(len(data)):
        cata_dict[data[i]].append(i)
    return cata_dict

def generate_test_graph(graph):
    test_graph = copy.deepcopy(graph)
    removed_lst = []
    for i in range(100):
        node = random.randint(0, len(graph))
        if str(node) in graph and node not in removed_lst:
            removed_lst.append(node)
        else:
            i -= 1
    for node in removed_lst:
        test_graph.pop(str(node))
    for node in test_graph:
        neighbors = test_graph[node]
        for node in removed_lst:
            if str(node) in neighbors:
                neighbors.remove(str(node))
    return test_graph, removed_lst


def find_average_scatter():
    start_factor = 0.8
    end_factor = 0.9
    graph = read_graph()
    average_dict = {}
    random_node_lst = []
    for i in range(200):
        random_node = random.randint(int(len(graph) * start_factor), int(len(graph) * end_factor))
        if (random_node) in graph and random_node_lst not in random_node_lst:
            random_node_lst.append(random_node)
        else:
            i -= 1
    for node in random_node_lst:
        average_dict[(node)] = sum(graph[(node)])/len(graph[(node)])
    return average_dict

def find_average_individual_point():
    graph = read_graph()
    average_dict = {}
    random_node_lst = []
    for node in random_node_lst:
        average_dict[(node)] = sum(graph[(node)])/len(graph[(node)])
    return average_dict

def plot_average():
    return

def generate_first_order_HMM(graph):
    count_dict = defaultdict(lambda: defaultdict(float))
    class_dict = pd.read_csv('node_classification.csv')['page_type']
    for node in graph:
        start_type = str(class_dict[(int(node))])
        for destination in graph[node]:
            end_type = str(class_dict[int(destination)])
            count_dict[start_type][end_type] += 1
            count_dict[end_type][start_type] += 1
    hmm = copy.deepcopy(count_dict)
    for start_type in hmm:
        cur_summation = sum(hmm[start_type].values())
        for end_type in hmm[start_type]:
            hmm[start_type][end_type] /= cur_summation
    print(hmm)
    return hmm

def check_directness(graph):
    for start_node in graph:
        for end_node in graph[start_node]:
            if end_node not in graph:
                return end_node
            if (start_node) not in graph[(end_node)]:
                return start_node, end_node
    return 'undirected graph'

def generate_second_order_HMM(graph):
    # set the default value to be 0 to avoid bug
    count_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    class_dict = pd.read_csv('node_classification.csv')['page_type']
    for node1 in graph:
        type1 = str(class_dict[int(node1)])
        if node1 in graph:
            for node2 in graph[(node1)]:
                type2 = str(class_dict[int(node2)])
                if node2 in graph:
                    for node3 in graph[node2]:
                        type3 = str(class_dict[int(node3)])
                        count_dict[type1][type2][type3] += 1
    hmm = count_dict.copy()
    for type1 in hmm:
        for type2 in hmm[type1]:
            summation = sum(hmm[type1][type2].values())
            for type3 in hmm[type1][type2]:
                hmm[type1][type2][type3] /= summation
    return hmm

def read_feature():
    f = open('node_features_text.json')
    feature_dict = json.load(f)
    f.close()
    return feature_dict

def count_feature_num(feature_graph):
    count_dict = defaultdict(int)
    for node in feature_graph:
        count_dict[len(feature_graph[node])] += 1
    return count_dict

def plot_num_of_features_type(type, cata_dict, feature_graph):
    type_lst = cata_dict[type]
    count_dict = defaultdict(int)
    for node in type_lst:
        count_dict[len(feature_graph[str(node)])] += 1
    plt.bar(list(count_dict.keys()), count_dict.values())
    plt.show()
    return count_dict

def find_all_features_tag(feature_graph):
    tags = []
    for node in feature_graph:
        for tag in feature_graph[node]:
            if tag not in tags:
                tags.append(tag)
    return tags

def generate_feature_hmm(graph, feature_graph):
    count_dict = defaultdict(lambda: defaultdict(int))
    for node1 in graph:
        for node2 in graph[node1]:
            intersection = []
            for feature1 in feature_graph[str(node1)]:
                for feature2 in feature_graph[str(node2)]:
                    count_dict[str(feature1)][str(feature2)] += 1
                    count_dict[str(feature2)][str(feature1)] += 1
    for feature1 in count_dict:
        summation = sum(count_dict[feature1].values())
        for feature2 in count_dict[feature1]:
            count_dict[feature1][feature2] /= summation
    return count_dict

def count_features_in_same_type(cata_dict, feature_graph):
    color = ['c','b','g','r']
    marker_lst = ['o','^','*',"."]
    count_dict = defaultdict(int)
    for type in range(1, 5):
        node_lst = cata_dict[type]
        for node in node_lst:
            for feature in feature_graph[str(node)]:
                count_dict[feature] += 1
        plt.scatter(list(count_dict.keys()), count_dict.values(), c = color[type - 1], marker = marker_lst[type - 1])
    plt.legend(['1','2','3','4'])
    plt.show()

def plot_degree_and_features(graph, feature_graph):
    count_dict = defaultdict(int)
    feature_num_dict = defaultdict(int)
    for node in graph:
        num_of_features = len(feature_graph[str(node)])
        degree = len(graph[node])
        count_dict[num_of_features] += degree
        feature_num_dict[len(feature_graph[str(node)])] += 1        
    for num in count_dict:
        count_dict[num] /= feature_num_dict[num]
    plt.scatter(list(count_dict.keys()), count_dict.values())
    plt.show()
    return count_dict

def write_file(count_dict,filename):
    jsonString = json.dumps(count_dict)
    jsonFile = open(filename, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def enhance_function(p, beta):
    if p == 0:
        return 0
    return 1 / (1 + (p/(1 - p)) ** (-beta))

def weighted_probability(p):
    cur_p = random.random()
    if cur_p < p:
        return 1
    return 0

def sigmoid(x, a, b):
    if x == 0:
        return 0
    if 1-a*x == 0:
        return 0
    y = 1/(1+(a*x/(1-a*x))**(-b))
    return y

def generate_probability(first_order_hmm, feature_hmm, node1, node2, graph, feature_graph, type_dict):
    #node 1 in graph, node 2 is isolated
    p_type_lst = []
    p_feature_lst = []
    type2 = str(type_dict[int(node1)])
    type3 = str(type_dict[int(node2)])
    cur_p = first_order_hmm[type2][type3]
    p_type_lst.append(enhance_function(cur_p, 4))
    p_type = average(p_type_lst)
    for feature1 in feature_graph[(node1)]:
        for feature2 in feature_graph[(node2)]:
            cur_p = 0
            if str(feature1) not in feature_hmm:
                cur_p = 0
            elif str(feature2) in feature_hmm[str(feature1)]:
                cur_p = (feature_hmm[str(feature1)][str(feature2)])
            p_feature_lst.append(sigmoid(cur_p, 170, 3))
    p_feature = average(p_feature_lst)
    p_connect = enhance_function(p_type * p_feature, 30)
    return p_connect

def graph_feature_probability(feature_graph):
    count_dict = defaultdict(int)
    size = len(feature_graph)
    for node in feature_graph:
        for feature in feature_graph[node]:
            count_dict[feature] += 1
    for feature in count_dict:
        count_dict[feature] /= size
    return count_dict

# graph = read_file('test_model\graph_test.json')
# feature_graph = read_file('node_features_text.json')
# second_hmm = generate_second_order_HMM(graph)
# write_file(second_hmm,'test_model\second_hmm_test.json')
# second_order_hmm = read_file('complete_model\second_hmm.json')
# feature_hmm = read_file('feature_hmm.json')
# feature_graph = read_file('node_features_text.json')
# plot_data = {}
# for i in range(1, 5):
#     plot_data[i] = len(cata_dict[i])
# plt.bar(list(plot_data.keys()), plot_data.values())
# plt.show()
# data = read_graph()
# print(len(data))
# average_dict = find_average_scatter()
# plt.scatter(list(average_dict.keys()), average_dict.values())
# plt.show()
# feature_graph = read_feature()
# print(len(feature_graph.keys()))
# print(len(graph.keys()))
# count_dict = count_feature_num(feature_graph)
# plt.show()
# print(check_directness(graph))

# cata_dict = read_classification()
# plot_num_of_features_type(2, cata_dict, feature_graph)
# print(len(find_all_features_tag(feature_graph)))
# print(feature_hmm(graph, feature_graph))
# plot_degree_and_features(graph, feature_graph)
# p_connect = generate_probability(second_order_hmm, feature_hmm, node1, node2, graph, feature_graph)



type_dict = pd.read_csv('node_classification.csv')['page_type']
first_order_hmm = read_file('first_hmm.json')
feature_hmm = read_file('feature_hmm.json')
graph = read_file('complete_model\graph.json')
test_nodes = pd.read_csv('test_edges.csv')
answer = pd.read_csv('test_labels.csv')
feature_graph = read_file('graph_feature.json')
count_num = 0
my_answers = []
for i in range(len(test_nodes)):
    cur_answer = answer['label'][i]
    node2 = str(test_nodes['node1'][i])
    node1 = str(test_nodes['node2'][i])
    p = generate_probability(first_order_hmm, feature_hmm, node1, node2, graph, feature_graph, type_dict)
    my_answer = weighted_probability(p)
    my_answers.append(my_answer)
    if cur_answer == my_answer:
        count_num += 1

print(count_num/len(test_nodes))

# print(generate_probability(first_order_hmm, feature_hmm, node1, node2, graph, feature_graph, type_dict))