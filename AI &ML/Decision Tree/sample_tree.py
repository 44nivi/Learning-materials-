import pandas as pd # type: ignore

df = pd.read_csv("D:\\study material\\AI instructions\\ML workspace\\Decision Tree\\newdata.csv")
features = [ 'Other online courses', 'Student background','Working Status']
target_feature = "Exam Result"
target_values = ["pass","fail"]
class DecisionTreeNode:
    def __init__(self,subtrees=None, value=None,right=None,left=None):
        self.subtrees = subtrees  # Subtrees for different feature values
        self.value = value  # Value to return if this node is a leaf node
        self.right=right
        self.left=left
class Tree_classifier:

    def __init__(self, depth=None):
        self.depth = depth  # Maximum depth of the decision tree
        self.tree = None  # Root node of the deci
        
    def gini(feature_data, target_values, target_feature):
        print(target_feature)
        g_value = 0
        denom = len(feature_data)
        for tv  in target_values:
            num = len(feature_data[feature_data[target_feature]==tv])

            g_value += (num/denom)**2
        g_value = 1 - g_value
        return g_value
    def feature_wise_gini(self,df, features, target_values, target_feature):
            feature_name=[]
            low_gini=[]
            subtrees={}
            minimum_gini=None
            if self.depth == 0 :
                    return DecisionTreeNode(subtrees=None)
            for ft in features:
                
                ft_gini = 0
            
                for feature_value,group in df.groupby(ft):
                    g_value = self.gini(group, target_values,target_feature)
                    ft_gini += (len(group)/len(df)*g_value)
                    print(f"{ft} - {feature_value} = {g_value}")        
                    
                low_gini.append(ft_gini)
                feature_name.append(ft)
                # print("hi nivi this", ft)
                print(f"Feature column: {ft} gini value = {ft_gini}\n")

                

            if len(low_gini)==0:
                return None
            else:
                minimum_gini=min(low_gini)
                index=low_gini.index(minimum_gini)
                low_feature=feature_name[index]
                features.remove(low_feature)
                self.depth-=1
                for feature_value,group in df.groupby(low_feature):
                    print("*****************",feature_value)
                    if feature_value=="maths":
                        DecisionTreeNode.left=feature_wise_gini(self,df[df[low_feature]=="maths"], features, target_values, target_feature)
                    elif feature_value=="cs":
                        DecisionTreeNode.left=feature_wise_gini(self,df[df[low_feature]=="cs"], features, target_values, target_feature) 
                    elif feature_value=="other":
                        DecisionTreeNode.left=feature_wise_gini(self,df[df[low_feature]=="other"], features, target_values, target_feature) 
                    else :
                        DecisionTreeNode.right=feature_wise_gini(self,df, features, target_values, target_feature) 

    '''def visualize_tree(node, depth=0):
        if node is None:
            return
        print("  " * depth, end="")
        if node.feature_index is None:
            print("Leaf, Predicted Class:", node.value)
        else:
            print("Feature:", features[node.feature_index], ", Threshold:", node.threshold)
            print("  " * depth, "Subtrees:")
            for value, subtree in node.subtrees.items():
                print("  " * (depth + 1), f"Value: {value}")
                visualize_tree(subtree, depth + 2) '''


low_gini=[]
feature_name=[]
my_clf = Tree_classifier(depth=4)
my_clf.feature_wise_gini(df, features, target_values, target_feature)
            