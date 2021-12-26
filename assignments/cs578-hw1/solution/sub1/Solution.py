import numpy as np
import pandas as pd
import sys 
from numpy import log2 as log
import random
eps = np.finfo(float).eps


# defining the column labels
def define_column_labels():
	attributes = {
    'A1': ['b','a'],
    'A2': 'continuous',
    'A3': 'continuous',
    'A4': ['u','y','l','t'],
    'A5': ['g','p','gg'],
    'A6': ['c','d','cc','i','j','k','m','r','q','w','x','e','aa','ff'],
    'A7': ['v','h','bb','j','n','z','dd','ff','o'],
    'A8': 'continuous',
    'A9': ['t','f'],
    'A10': ['t','f'],
    'A11': 'continuous',
    'A12': ['t','f'],
    'A13': ['g','p','s'],
    'A14': 'continuous',
    'A15': 'continuous',
    'A16': ['+','-'],
	}
	attributes_copy = attributes.copy()
	attributes_copy.pop('A16')
	return attributes, attributes_copy

def fill_missing_values(df, attributes):
	# filling in Missing Values
	for i in range(1, 16):
		attribute = 'A'+str(i)
		if type(attributes.get(attribute))!=list:
			# replacing with mean value for continuous variable
			mean_value = pd.to_numeric(df[df[attribute] != '?'][attribute]).mean()
			df[attribute] = pd.to_numeric(df[attribute].replace('?', mean_value))
		else:
			# replacing with mode value for categorical variable
			value_counts = df[attribute].value_counts()
			mode_value = None
			for index, value in value_counts.items():
				if index != '?':
					mode_value = index
					break
			if mode_value == None:
				print('Unable to find values other than \'?\'')
			df[attribute] = df[attribute].replace('?', mode_value)
	return df

# to determine the type of the attribute whether continuous or categorical
def determine_type_of_feature(df):
	feature_types = []
	n_unique_values_treshold = 14
	for feature in df.columns:
		if feature != "A16":
			unique_values = df[feature].unique()
			if (len(unique_values) <= n_unique_values_treshold):
				feature_types.append("categorical")
			else:
				feature_types.append("continuous")
	return feature_types

# for finding the entropy in a given dataframe
def find_entropy(df):
    Class = df.keys()[-1]  
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy

# for finding the entropy of continuous attributes
def entropy_continuous(df,data_portion):
    data_portion_count = len(data_portion)
    output = {}
    entropy = 0
    for x,y in data_portion:
        if y in output:
            output[y].append((x,y))
        else:
            output[y] = [(x,y)]
    if ('+' in output.keys()):
        pos_labels_count = len(output.get('+'))
    else:
        pos_labels_count = 0
    neg_labels_count = data_portion_count - pos_labels_count
    pos_fraction = pos_labels_count/(data_portion_count+eps)
    pos_entropy = -pos_fraction*log(pos_fraction+eps)
    neg_fraction = neg_labels_count/(data_portion_count+eps)
    neg_entropy = -neg_fraction*log(neg_fraction+eps)
    entropy = pos_entropy + neg_entropy
    return entropy

def find_entropy_attribute(df,key, attributes, threshold_dict):
    # for categorical attributes
    if(type(attributes.get(key))==list):
        Class = df.keys()[-1]   
        target_variables = df[Class].unique()  
        variables = df[key].unique()   
        entropy2 = 0
        threshold = None
        threshold_dict[key] = threshold
        for variable in variables:
            entropy = 0
            for target_variable in target_variables:
                num = len(df[key][df[key]==variable][df[Class] == target_variable])
                den = len(df[key][df[key]==variable])
                fraction = num/(den+eps)
                entropy += -fraction*log(fraction+eps)
            fraction2 = den/len(df)
            entropy2 += fraction2*entropy
    else:
    # for continuous attributes
        threshold = 0
        min_entropy = sys.maxsize
        entropy2 = 0
        threshold = None
        zipped = list(zip(df[key], df[df.keys()[-1]]))
        result = sorted(zipped, key = lambda x: x[0])
        # finding the entropy for every split and then taking the min out of them to maximize the information gain
        for i in range(len(result)):
            entropy2 = 0
            if(i!=0 and result[i][0]!=result[i-1][0]):     
                mid_value = (float(result[i][0])+float(result[i-1][0]))/2.0
                left_part = result[0:i]
                left_part_count = len(left_part)
                if(left_part_count == 0):
                    continue
                entropy_left = entropy_continuous(df,left_part)
                left_fraction = len(left_part)/len(df)
                weighted_entropy_left = left_fraction*entropy_left
                entropy2+= weighted_entropy_left
                right_part = result[i:]
                right_part_count = len(right_part)
                if(right_part_count != 0):
                    entropy_right = entropy_continuous(df,right_part)
                    right_fraction = len(right_part)/len(df)
                    weighted_entropy_right = right_fraction*entropy_right
                    entropy2+= weighted_entropy_right
                    if(entropy2 < min_entropy):
                        min_entropy = entropy2
                        threshold = mid_value
        entropy2 = min_entropy
        threshold_dict[key] = threshold
    return (entropy2,threshold)
    
# for finding the node with max information gain along with its threshold value
def find_winner(df, attributes_copy, attributes, threshold_dict):
    Entropy_att = []
    IG = []
    for key in attributes_copy.keys():
        entropy_attribute, threshold = find_entropy_attribute(df, key, attributes, threshold_dict)
        IG.append(find_entropy(df)-entropy_attribute)
    return list(attributes_copy)[np.argmax(IG)], threshold_dict.get(list(attributes_copy)[np.argmax(IG)])

# to check if data belongs to single class
def check_label_purity(data):
    label_column = data.keys()[-1]
    unique_classes = np.unique(data[label_column])
    if len(unique_classes) == 1:
        return True
    else:
        return False

# to find the majority class of the given dataset
def majority_class(data):
    label_column = data.keys()[-1]
    unique_classes, counts_unique_classes = np.unique(data[label_column], return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification

# structure of the node
class Node():
    def __init__(self):
        self.val = []
        self.children = {}
        self.target_val = -1

    def add_child(self, cat, node):
        self.children[cat] = node

    def __repr__(self) -> str:
        return str(self.val) + " " + str(self.target_val)


def printTree(node, key, parent):
    print(parent, key, node)

    for key, child in node.children.items():
        printTree(child, key, node)

# algorithm for building the decision tree
def build_decision_tree(df, attribute_features, depth, maxDepth, feature_types, attributes, threshold_dict):
    node = Node()
    
    target = len(df.columns)-1
    if (depth == maxDepth) or (len(attribute_features) == 0) or (len(df) <= 1) or (check_label_purity(df)) :
        tmp = df.iloc[:,target].mode()
        
        if (len(tmp) == 0):
            node.target_val = 0
        else:
            classification = majority_class(df)
            node.target_val = classification
        return node

    attr, thresh = find_winner(df, attribute_features, attributes, threshold_dict)
    type_of_feature = feature_types[int(attr[1:]) - 1]
    attribute_features.pop(attr)

    if type_of_feature == 'categorical':
        thresh = attributes.get(attr)
        node.val = [attr, thresh]
        for val in thresh:
            node.add_child(val, build_decision_tree(df[df[attr] == val], attribute_features, depth+1, maxDepth, feature_types, attributes, threshold_dict))
    else:
        node.val = [attr, thresh]
        node.add_child('leq', build_decision_tree(df[df[attr] <= thresh], attribute_features, depth+1, maxDepth, feature_types, attributes, threshold_dict))
        node.add_child('ge', build_decision_tree(df[df[attr] > thresh], attribute_features, depth+1, maxDepth, feature_types, attributes, threshold_dict))

    return node

# for getting the predicted labels
def get_predicted_labels(df, root):
    labels = []
    node = root
    for _, row in df.iterrows():
        # leaf node condition
        while node.target_val == -1:
            b_attr = node.val[0]
            b_val = node.val[1]
            # for categorical attribute
            if isinstance(b_val, list):
                t_val = row[b_attr]
                node = node.children[t_val]
            else :
            # for continuous attribute
                t_val = row[b_attr]
                if (t_val <= b_val):
                    node = node.children['leq']
                else :
                    node = node.children['ge']
        labels.append(node.target_val)
        node = root
    return labels

# for getting the accuracy on the predicted labels
def accuracy(orig, pred):
	num = len(pred)
	if(num != len(pred)):
		print('Error!! Num of labels are not equal.')
		return
	match = 0
	for i in range(len(orig)):
		o_label = orig[i]
		p_label = pred[i]
		if(o_label == p_label):
			match += 1
	return float(match)/num

# for reading the given file
def read_file(filename, attributes):
	df = pd.read_csv(filename,sep = '\t', header = None, names = attributes.keys())
	return df

# for splitting the training set
def train_validation_split(df):
    df = df.sample(frac=1)
    size = int(0.8 * len(df))
    train_data = df[:size]
    val_data = df[size:]
    return train_data, val_data

def DecisionTree():
    attributes, attributes_copy = define_column_labels()
    training_file = 'train.txt'
    testing_file = 'test.txt'
    validation_file = 'validation.txt'
    train_df = read_file(training_file, attributes)
    train_df = fill_missing_values(train_df, attributes)
    validation_df = read_file(validation_file, attributes)
    validation_df = fill_missing_values(validation_df, attributes)
    test_df = read_file(testing_file, attributes)
    test_df = fill_missing_values(test_df, attributes)
    feature_type = determine_type_of_feature(train_df)
    threshold_dict = {}
    node = build_decision_tree(train_df, attributes_copy, 0, 20, feature_type, attributes, threshold_dict)
    predicted_validation_labels = get_predicted_labels(validation_df, node)
    original_validation_labels = validation_df['A16'].values.tolist()
    predicted_test_labels = get_predicted_labels(test_df, node)
    original_test_labels = test_df['A16'].values.tolist()
    return original_validation_labels, predicted_validation_labels, original_test_labels, predicted_test_labels


def DecisionTreeBounded(maxDepth):
    attributes, attributes_copy = define_column_labels()
    training_file = 'train.txt'
    testing_file = 'test.txt'
    validation_file = 'validation.txt'
    train_df = read_file(training_file, attributes)
    train_df = fill_missing_values(train_df, attributes)
    validation_df = read_file(validation_file, attributes)
    validation_df = fill_missing_values(validation_df, attributes)
    test_df = read_file(testing_file, attributes)
    test_df = fill_missing_values(test_df, attributes)
    feature_type = determine_type_of_feature(train_df)
    threshold_dict = {}
    node = build_decision_tree(train_df, attributes_copy, 0, maxDepth, feature_type, attributes, threshold_dict)
    predicted_validation_labels = get_predicted_labels(validation_df, node)
    original_validation_labels = validation_df['A16'].values.tolist()
    predicted_test_labels = get_predicted_labels(test_df, node)
    original_test_labels = test_df['A16'].values.tolist()
    return original_validation_labels, predicted_validation_labels, original_test_labels, predicted_test_labels
