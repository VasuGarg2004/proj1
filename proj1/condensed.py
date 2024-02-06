import pandas as pd
from sklearn.model_selection import train_test_split

mushrooms = pd.read_csv('mushrooms.csv')

mushrooms_train, mushrooms_rest = train_test_split(mushrooms, test_size=0.3)
mushrooms_test, mushrooms_validate = train_test_split(mushrooms_rest, test_size=0.33)

def impurity(data):
    try:
        count_p = data['class'].value_counts()['p']
    except KeyError:
        count_p = 0
    try:
        count_e = data['class'].value_counts()['e']
    except KeyError:
        count_e = 0
    gini = 1 - (count_p/len(data))**2 - (count_e/len(data))**2
    return gini

attribute_values = {}
for column in mushrooms_train.columns:
    if column != 'class':
        attribute_values[column] = mushrooms_train[column].unique()

def attribute_choice(data):
    min_gini = 1
    for data_column in data.columns:
        if data_column != 'class':
            gini = 0
            for attribute_value in attribute_values[data_column]:
                if len(data[data[data_column] == attribute_value]) == 0:
                    continue
                weight = len(data[data[data_column] == attribute_value]) / len(data)
                gini += impurity(data[data[data_column] == attribute_value]) * weight
            if gini < min_gini:
                min_gini = gini
                min_gini_column = data_column
    return min_gini_column, min_gini

decision_tree = {}

def build_tree(data):
    if impurity(data) == 0:
        return {'class': data['class'].iloc[0]}
    attribute_name = attribute_choice(data)[0]
    return_dict = {'attribute': attribute_name}
    for attribute_value in attribute_values[attribute_name]:
        if len(data[data[attribute_name] == attribute_value]) == 0:
            continue
        return_dict[attribute_value] = build_tree(data[data[attribute_name] == attribute_value])
    return return_dict

decision_tree = build_tree(mushrooms_train)

def classify(tree, data):
    if 'class' in tree:
        return tree['class']
    attribute_name = tree['attribute']
    data_value = data[attribute_name]
    return classify(tree[data_value], data)

i = 0
for index, row in mushrooms_validate.iterrows():
    val = classify(decision_tree, row.drop('class'))
    if row['class'] == val:
        i += 1

i = 0
for index, row in mushrooms_test.iterrows():
    val = classify(decision_tree, row.drop('class'))
    if row['class'] == val:
        i += 1

print('accuracy:', i, len(mushrooms_test))
