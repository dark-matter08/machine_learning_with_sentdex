import numpy as np
from collections import Counter
import warnings


# euclidean_distance = sqrt( (X[0] - Y[0])**2 + (X[1] - Y[1])**2 )

class CustomKNN:
    def __init__(self):
        pass

    def k_nearest_neighbors(self, data, predict, k=3):
        if len(data) >= k:
            warnings.warn('K is set to a value less than total voting groups')
        predict = predict.astype(float).tolist()
        distances = []
        for group in data:
            for features in data[group]:
                features = features.astype(float).tolist()
                # euclidean_distance = sqrt( (features[0] - predict[0])**2 + (features[1] - predict[1])**2 )
                # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
                distances.append([euclidean_distance, group])

        votes = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1] / k

        return vote_result, confidence

    def split_data(self, data_x, data_y, test_size=0.2):
        train_data_x = data_x[:-int(test_size*len(data_x))]
        train_data_y = data_y[:-int(test_size*len(data_y))]
        test_data_x = data_x[-int(test_size*len(data_x)):]
        test_data_y = data_y[-int(test_size*len(data_y)):]

        return train_data_x, test_data_x, train_data_y, test_data_y

    def fit(self, X_train, y_train):
        classes = set(y_train)
        train_set = {}
        
        for class_ in classes:
            train_set[class_] = []

        for x, y in zip(X_train, y_train):
            train_set[y].append(x)

        self.train_set = train_set

    def score(self, X_test, y_test):
        correct = 0
        total = 0
        confidences = []

        for data, group in zip(X_test, y_test):
            vote, confidence = self.k_nearest_neighbors(self.train_set, data, k=5)
            if group == vote:
                correct += 1
                confidences.append(confidence)

            total += 1

        accuracy = correct/total
        final_confidence = sum(confidences)/len(confidences)

        return f"================================\nAccuracy: {accuracy} \n================================\nConfidence: {final_confidence} \n================================"

    def predict(self, predict):
        result = []
        for data in predict:
            vote, confidence = self.k_nearest_neighbors(self.train_set, data, k=5)
            result.append(vote)

        return result
