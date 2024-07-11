from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class Evaluator:
    def __init__(self, true_labels, predicted_labels, num_classes):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.num_classes = num_classes
        self.true_binary = self._binarize_labels(true_labels)
        self.predicted_binary = self._binarize_labels(predicted_labels)
    
    def _binarize_labels(self, labels):
        """
        Convert multilabel classification labels into a binary matrix format.
        
        Each row corresponds to an instance and each column corresponds to a class.
        A value of 1 indicates the presence of a label, and 0 indicates its absence.

        Args:
            labels (list of list of int): List of lists where each sublist contains the labels for an instance.
        
        Returns:
            np.ndarray: Binary matrix representation of the labels.
        """
        binary = np.zeros((len(labels), self.num_classes))
        for i, label_list in enumerate(labels):
            for label in label_list:
                if label <= self.num_classes:
                    binary[i, label - 1] = 1
        return binary

    def calculate_accuracy(self):
        """
        Calculate the accuracy of the predicted labels.
        
        Accuracy:
        - Proportion of correct predictions over all predictions.
        
        Returns:
            float: Accuracy score.
        """
        return accuracy_score(self.true_binary, self.predicted_binary)

    def calculate_precision(self, average='macro'):
        """
        Calculate the precision of the predicted labels.
        
        Precision:
        - Proportion of true positive results over all predicted positive results.
        
        Returns:
            float: Precision score.
        """
        return precision_score(self.true_binary, self.predicted_binary, average=average, zero_division=0)

    def calculate_recall(self, average='macro'):
        """
        Calculate the recall of the predicted labels.
        
        Recall:
        - Proportion of true positive results over all actual positive results.
        
        Returns:
            float: Recall score.
        """
        return recall_score(self.true_binary, self.predicted_binary, average=average, zero_division=0)

    def calculate_f1(self, average='macro'):
        """
        Calculate the F1 score of the predicted labels.
        
        F1 Score:
        - The harmonic mean of precision and recall.
        
        Returns:
            float: F1 score.
        """
        return f1_score(self.true_binary, self.predicted_binary, average=average, zero_division=0)

    def evaluate(self):
        """
        Evaluate the predicted labels against the true labels.
        
        Returns:
            dict: A dictionary containing various evaluation metrics.
        """
        results = {
            'accuracy': self.calculate_accuracy(),
            'precision_macro': self.calculate_precision(average='macro'),
            'recall_macro': self.calculate_recall(average='macro'),
            'f1_macro': self.calculate_f1(average='macro'),
            'precision_micro': self.calculate_precision(average='micro'),
            'recall_micro': self.calculate_recall(average='micro'),
            'f1_micro': self.calculate_f1(average='micro')
        }
        return results

# # Example usage with dummy data
# true_labels = [
#     [1, 3, 11],
#     [10],
#     [2, 5, 7],
#     [6, 8, 11]
# ]

# predicted_labels = [
#     [1, 3],
#     [10, 12],
#     [2, 5],
#     [6, 8, 11, 13]
# ]

# # Instantiate evaluator
# num_classes = 12
# evaluator = Evaluator(true_labels, predicted_labels, num_classes)

# # Get evaluation results
# results = evaluator.evaluate()

# # Print results
# for metric, value in results.items():
#     print(f"{metric}: {value}")
