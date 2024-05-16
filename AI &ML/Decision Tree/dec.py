import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Define a class for a decision tree node
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, subtrees=None, value=None):
        self.feature_index = feature_index  # Index of the feature used for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.subtrees = subtrees  # Subtrees for different feature values
        self.value = value  # Value to return if this node is a leaf node

# Define a class for the decision tree classifier
class MyDecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the decision tree
        self.tree = None  # Root node of the decision tree

    # Function to find the best split for a dataset
    def find_best_split(self, X, y):
        n_features = X.shape[1]
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        # Calculate Gini impurity for each feature and threshold
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                gini_left = self.calculate_gini(y[left_indices])
                gini_right = self.calculate_gini(y[right_indices])

                gini = (len(left_indices) / len(y)) * gini_left + (len(right_indices) / len(y)) * gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    # Function to calculate Gini impurity
    def calculate_gini(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        gini = 1
        for class_count in class_counts:
            proportion = class_count / len(y)
            gini -= proportion ** 2
        return gini

    # Function to build the decision tree
    def build_tree(self, X, y, depth=0):
        # Stopping criteria: if maximum depth is reached or all instances belong to the same class
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))

        best_feature_index, best_threshold = self.find_best_split(X, y)

        # Stopping criteria: if no best split is found
        if best_feature_index is None:
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))

        # Split the dataset based on the best split found
        feature_values = np.unique(X[:, best_feature_index])
        subtrees = {}
        for value in feature_values:
            indices = np.where(X[:, best_feature_index] == value)[0]
            subtrees[value] = self.build_tree(X[indices], y[indices], depth + 1)

        return DecisionTreeNode(feature_index=best_feature_index, threshold=best_threshold,
                                subtrees=subtrees)

    # Function to fit the decision tree to the training data
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    # Function to predict class labels for instances in the test data
    def predict(self, X):
        predictions = []
        for instance in X:
            node = self.tree
            while node.subtrees:
                value = instance[node.feature_index]
                if value in node.subtrees:
                    node = node.subtrees[value]
                else:
                    break
            predictions.append(node.value)
        return np.array(predictions)

# Function to visualize the decision tree
def visualize_tree(node, depth=0):
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
            visualize_tree(subtree, depth + 2)

# Read Data
df = pd.read_csv('D:\\study material\\AI instructions\\ML workspace\\Decision Tree\\newdata.csv')

# Define features and target variable
features = ['Other online courses', 'Student background', 'Working Status']
X = df[features].values
y = df['Exam Result'].values

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, y_train = X,y

# Use the custom DecisionTreeClassifier
my_clf = MyDecisionTreeClassifier(max_depth=4)
my_clf.fit(X_train, y_train)

# Make predictions
y_pred = my_clf.predict(X_train)

# Calculate accuracy
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy:", accuracy)

# Visualize the decision tree
visualize_tree(my_clf.tree)
