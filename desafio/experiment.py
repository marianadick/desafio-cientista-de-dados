import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

def replace_invalid_values(dataset):
    missing_value_cols = ['workclass', 'occupation',  'native_country']

    for col in missing_value_cols:
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)

    return dataset

def feature_scaling(dataset, columns_category):
    le = preprocessing.LabelEncoder()
    for col in columns_category:
        le.fit(dataset[col])
        dataset[col] = le.fit_transform(dataset[col])

    dataset_normalize = dataset.copy()
    dataset_normalize[dataset.columns] = preprocessing.MinMaxScaler().fit_transform(dataset[dataset.columns])

    dataset = dataset_normalize

    return dataset

def main():
    # Carregamento
    training_set = pd.read_csv("wage_train.csv", skipinitialspace = True)
    test_set = pd.read_csv("wage_test.csv", skipinitialspace = True)
    training_set = training_set.drop(columns=['Unnamed: 0'])
    test_set = test_set.drop(columns=['Unnamed: 0'])

    # Preprocessamento
    training_set = replace_invalid_values(training_set)
    test_set = replace_invalid_values(test_set)
    columns_category = ['workclass', 'education', 'marital_status', 'occupation', 
                    'relationship', 'race', 'sex', 'native_country', 'yearly_wage']
    training_set = feature_scaling(training_set, columns_category)
    columns_category = ['workclass', 'education', 'marital_status', 'occupation', 
                        'relationship', 'race', 'sex', 'native_country']
    test_set = feature_scaling(test_set, columns_category)
    
    # Separando target de trainamento
    X_train = training_set.iloc[:,0:14].values
    Y_train = training_set.iloc[:,14].values
    
    # Implementando Random Forest
    classifier = RandomForestClassifier()
    classifier.fit(X_train,Y_train)
    prediction = classifier.predict(test_set)
    
    # Convertendo para csv
    results = pd.DataFrame(prediction, columns = ['predictedValues'])
    results.to_csv('predicted.csv')
    
main()