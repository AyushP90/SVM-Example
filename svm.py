import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.svm import SVC


def load_file(filename):
    dataset = pd.read_csv(filename)
    dataset = shuffle(dataset)
    dataset = preprocessing(dataset)
    label = dataset['class']
    dataset = dataset.drop(['class', 'veil-type'], 1)
    return label, dataset


def preprocessing(dataset):
    for col in dataset.columns:
        le = LE()
        dataset[col] = le.fit_transform(dataset[col])
    return dataset


def shuffle(dataset):
    dataset = dataset.sample(frac=1)
    return dataset


def split_dataset(dataset):
    train_count = int(dataset.shape[0]*0.6)
    cv_count = train_count + int(dataset.shape[0]*0.2)
    train_set, cv_set, test_set = dataset[:train_count], dataset[train_count:cv_count], dataset[cv_count:]
    return train_set, cv_set, test_set


if __name__ == "__main__":
    labels, data = load_file('mushrooms.csv')
    train, validation, test = split_dataset(data)
    train_label, cv_label, test_label = split_dataset(labels)
    svm = SVC(verbose=0)
    svm.fit(train, train_label)
    prediction = svm.predict(test)
    misses, hits = 0, 0
    for i in range(prediction.shape[0]):
        if prediction[i] != test_label.iloc[i]:
            misses += 1
        else:
            hits += 1

    print "Misses: ", misses
    print "Hits: ", hits
