#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys 
eps = np.finfo(float).eps
from numpy import log2 as log
import random
from pprint import pprint


# In[2]:


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
len(attributes_copy)


# In[3]:


train_df= pd.read_csv("train.txt",sep = '\t', header = None, names = attributes.keys())
train_df


# In[4]:


def fill_missing_values(df):
    # Filling in Missing Values
    for i in range(1, 16):
        attribute = 'A'+str(i)
        if type(attributes.get(attribute))!=list:
            mean_value = pd.to_numeric(df[df[attribute] != '?'][attribute]).mean()
            df[attribute] = pd.to_numeric(df[attribute].replace('?', mean_value))
        else:
            value_counts = df[attribute].value_counts()
            mode_value = None
            for index, value in value_counts.items():
                if index != '?':
                    mode_value = index
                    break
            if mode_value == None:
                print('Unable to find values other than \'?\'')
            df[attribute] = df[attribute].replace('?', mode_value)
fill_missing_values(train_df)


# In[5]:


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 14
    for feature in df.columns:
#         print(feature)
        if feature != "A16":
            unique_values = df[feature].unique()
#             example_value = unique_values[0]

            if (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
#                 print('c')
            else:
                feature_types.append("continuous")
#                 print('conti')
    
    return feature_types
feature_types = determine_type_of_feature(train_df)
feature_types
# feature_types[int(find_winner(train_df)[0][1:])]


# In[12]:


threshold_dict = {}
def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy

def entropy_continuous(df,data_portion):
    data_portion_count = len(data_portion)
#     print("In entropy continuous:" , data_portion_count)
    output = {}
    entropy = 0
    for x,y in data_portion:
        if y in output:
            output[y].append((x,y))
        else:
            output[y] = [(x,y)]
#     print("In entropy continuous:", output)
#     print("In entropy continuous:", len(output.get('+')))
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

def find_entropy_attribute(df,key):
    if(type(attributes.get(key))==list):
        Class = df.keys()[-1]   #To make the code generic, changing target variable class name
        target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
        variables = df[key].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
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
#             entropy2 += -fraction2*entropy
    else:
        threshold = 0
        min_entropy = sys.maxsize
        entropy2 = 0
        threshold = None
        zipped = list(zip(df[key], df[df.keys()[-1]]))
        result = sorted(zipped, key = lambda x: x[0])
#         print(result)
        for i in range(len(result)):
            entropy2 = 0
            if(i!=0 and result[i][0]!=result[i-1][0]):      #check for all possible cases
                mid_value = (float(result[i][0])+float(result[i-1][0]))/2.0
#                 print(mid_value)
                left_part = result[0:i]
                left_part_count = len(left_part)
                if(left_part_count == 0):
                    continue
#                 print(left_part)
                entropy_left = entropy_continuous(df,left_part)
                left_fraction = len(left_part)/len(df)
                weighted_entropy_left = left_fraction*entropy_left
                entropy2+= weighted_entropy_left
#                 print(left_part_count)
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
        #         print(right_part_count)
                
#             else:
#                 print("Do nothing:", i)
    return (entropy2,threshold)
#     return abs(entropy2)
    

def find_winner(df, attributes_copy):
    Entropy_att = []
    IG = []
    for key in attributes_copy.keys():
#         print(key)
#         Entropy_att.append(find_entropy_attribute(df,key))
        entropy_attribute, threshold = find_entropy_attribute(df, key)
#         print(entropy_attribute, threshold)
        IG.append(find_entropy(df)-entropy_attribute)
    return list(attributes_copy)[np.argmax(IG)], threshold_dict.get(list(attributes_copy)[np.argmax(IG)])

# entropy_target = find_entropy(df)-find_entropy_attribute(df,'A6')
# entropy_target
# train_df.keys()[:-1]


# In[13]:


def check_label_purity(data):
    label_column = data.keys()[-1]
    unique_classes = np.unique(data[label_column])

    if len(unique_classes) == 1:
        return True
    else:
        return False


# In[14]:


def majority_class(data):
    
    label_column = data.keys()[-1]
    unique_classes, counts_unique_classes = np.unique(data[label_column], return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


# In[15]:


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


# In[16]:


def build_tree(df, attribute_features, depth, maxDepth, feature_types):
    node = Node()
    
    target = len(df.columns)-1
    if depth == maxDepth or (len(attribute_features) == 0) or (len(df) <= 1) or check_label_purity(df) :
        tmp = df.iloc[:,target].mode()
#         classification = majority_class(df)
        if (len(tmp) == 0):
            node.target_val = 0
        else:
            node.target_val = tmp.iloc[0]
#         node.target_val = classification
        return node

    attr, thresh = find_winner(df, attribute_features)
#     print(attr)
    type_of_feature = feature_types[int(attr[1:]) - 1]
#     # categorial attribute
#     if thresh == 'None':
#         thresh = attributes.get(attr)

    
#     at_orig = [x for x in attributes if x != attr]
    attribute_features.pop(attr)
    # at_orig.remove(attr)

    if type_of_feature == 'categorical':
        thresh = attributes.get(attr)
        node.val = [attr, thresh]
        for val in thresh:
            node.add_child(val, build_tree(df[df[attr] == val], attribute_features, depth+1, maxDepth, feature_types))
    else:
        node.val = [attr, thresh]
        node.add_child('leq', build_tree(df[df[attr] <= thresh], attribute_features, depth+1, maxDepth, feature_types))
        node.add_child('ge', build_tree(df[df[attr] > thresh], attribute_features, depth+1, maxDepth, feature_types))

    return node


# In[17]:


attributes_copy = attributes.copy()
attributes_copy.pop('A16')
node = build_tree(train_df,attributes_copy,0,20,feature_types)
printTree(node,"root","none")


# In[19]:


def evaluate(df, root):
    labels = []
    # tar = len(df.columns) - 1
    node = root
    # acc = 0
    for _, row in df.iterrows():
        while node.target_val == -1:
            b_attr = node.val[0]
            b_val = node.val[1]
            if isinstance(b_val, list):
                t_val = row[b_attr]
                node = node.children[t_val]
            else :
                t_val = row[b_attr]
                if (t_val <= b_val):
                    node = node.children['leq']
                else :
                    node = node.children['ge']
        # print(row[tar], node.target_val)
        # if(row[tar] == node.target_val):
        # 	acc += 1
        labels.append(node.target_val)
        node = root
    # print(acc/len(df))
    return labels
predicted_labels = evaluate(train_df,node)
# predicted_labels


# In[20]:


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
    print('***************\nAccuracy: ' + str(float(match)/num) + '\n***************')
original_labels = train_df['A16'].values
accuracy_score = accuracy(original_labels, predicted_labels)
accuracy_score


# In[21]:


test_df= pd.read_csv("test.txt",sep = '\t', header = None, names = attributes.keys())
test_df
fill_missing_values(test_df)
test_prediction_label = evaluate(test_df,node)
test_original_label = test_df['A16'].values
test_accuracy_score = accuracy(test_original_label, test_prediction_label)
test_accuracy_score


# In[ ]:




