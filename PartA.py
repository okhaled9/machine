import numpy as np
import pandas as pd


def replace_absent(dataset):
    num_cols = dataset.shape[1]
    
    for i in range(1,num_cols):
        col_name = dataset.columns[i]
        votes_sorted = dataset[col_name].value_counts().keys().tolist() #get most common vote
        dataset[col_name]=dataset[col_name].replace(['?'],votes_sorted[0])        
        
def split_dataset(dataset, size):          #split dataset randomly given size by percentage
    train = dataset.sample(frac=size/100)
    test = dataset.drop(train.index)
    return train , test

def split_by_column(dataset, split_column):   #splits accoring to column and removes it
    yes = dataset[dataset[split_column] == 'y']
    no = dataset[dataset[split_column] == 'n']
    
    yes.drop(split_column, inplace=True, axis=1)
    no.drop(split_column, inplace=True, axis=1)
    
    return yes, no

def check_purity(dataset):                 #check if we have to split again
    label_column = dataset.iloc[:,0].tolist()
    unique_classes = np.unique(label_column)
    return len(unique_classes) == 1
        
def best_classifier(dataset):     #returns name of the column with  highest information gain
    party = dataset.iloc[:,0].tolist()
    
    maxgain=0
    maxind=0
    
    rows = dataset.shape[0]
    cols = dataset.shape[1]
    
    for i in range(1,cols):     #go over columns
        
        y_demo = 0     #democrats who voted yes
        n_demo = 0     #democrats who voted no
    
        y_rep = 0      #republicans who voted yes
        n_rep = 0      #republicans who voted no
        
        vote = dataset.iloc[:,i].tolist()
        for j in range(0,rows):
            if (vote[j] == "y") and (party[j] == "democrat"):
                y_demo +=1
            if (vote[j] == "n") and (party[j] == "democrat"):
                n_demo +=1
            if (vote[j] == "y") and (party[j] == "republican"):
                y_rep +=1
            if (vote[j] == "n") and (party[j] == "republican"):
                n_rep +=1

        rep = y_rep + n_rep     #total republicans
        dem = y_demo + n_demo   #total democrats
        
        yes = y_demo+y_rep      #total yes
        no = n_demo + n_rep     #total no
            
        parent_entropy=0
        yes_entropy=0
        no_entropy=0
        
        if rows>0:
            parent_entropy = -(rep/rows)*np.log2(rep/rows) -(dem/rows)*np.log2(dem/rows)
        
        if yes>0:
            yes_entropy = (yes/rows)*((-y_rep/yes) * np.log2(y_rep/yes)    -(y_demo/yes) * np.log2(y_demo/yes))
        
        if no>0:
            no_entropy  = (no/rows)*((-n_rep/no ) * np.log2(n_rep/no )    -(n_demo/no ) * np.log2(n_demo/no ))
        
        igain = parent_entropy - yes_entropy - no_entropy
        if igain > maxgain:
            maxgain = igain
            maxind = i
        
        # print(dataset.head(),"\n")
        # print("split on:",dataset.columns[maxind],"\n")
        # print("rows:",rows,"\n")
        
    return dataset.columns[maxind]

def build_tree(node,counter=1):
    if check_purity(node.dataset) or (node.dataset.shape[1] <= 2):
        return counter
    
    split_col = best_classifier(node.dataset)
    print(split_col,"\n\n")
    node.split_col = split_col
    left_data, right_data = split_by_column(node.dataset, split_col)
    
    left_node = node_(left_data,None,None,None)
    right_node = node_(right_data,None,None,None)
    
    node.add_left(left_node)
    node.add_right(right_node)
    
    counter += 1
    
    return max(build_tree(node.left,counter) , build_tree(node.right,counter))
    
        
class node_:
    def __init__(self,dataset,left,right,split_col):
        self.dataset = dataset
        self.left  = left
        self.right = right
        self.split_col = split_col
    
    def add_left(self,node_):
        self.left = node_
    def add_right(self,node_):
        self.right = node_
        
################################################################################################ 
        
dataset = pd.read_csv("house-votes-84.data.txt")
train , test = split_dataset(dataset, 25)

replace_absent(train)
replace_absent(test)

root = node_(train,None,None,None)

print(build_tree(root,1))   #tree depth



