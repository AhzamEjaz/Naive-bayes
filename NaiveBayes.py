import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.probs_list = {}
        self.prob_col_dict = {}
    def train(self):
        ind_col_counts = {}
        col_dict = {}
        for ele in self.train_df[self.test_df.iloc[0].shape[0] - 1].unique():
            ind_count = np.asarray(self.train_df[self.train_df[self.test_df.iloc[0].shape[0] - 1]==ele][self.test_df.iloc[0].shape[0] - 1]).size
            ind_col_counts[ele] = ind_count
        col_dict = ind_col_counts

        prob_col_dict = {}
        for key in col_dict.keys():
            prob_col_dict[key] = (col_dict[key]/self.train_df[self.test_df.iloc[0].shape[0] - 1].shape[0])
        self.prob_col_dict = prob_col_dict

        probs_list = {}
        for header in range(self.test_df.iloc[0].shape[0] - 1):
            ind_df = pd.DataFrame(index = self.train_df[self.test_df.iloc[0].shape[0] - 1].unique(), columns = self.train_df[header].unique())
            for row in self.train_df[self.test_df.iloc[0].shape[0] - 1].unique():
                total_vals = self.train_df[self.train_df[self.test_df.iloc[0].shape[0] - 1] == row].shape[0]
                for ele in self.train_df[header].unique():
                    unacc_med = self.train_df[(self.train_df[self.test_df.iloc[0].shape[0] - 1] == row) & (self.train_df[header] == ele)].shape[0]
                    ind_df[ele][row] = unacc_med / total_vals
            probs_list[header] = ind_df
        self.probs_list = probs_list

    def getProbabilityItems(self, feature_list, pred_var):
        selection_lst = []
        for item_no in range(self.test_df.iloc[0].shape[0] - 1):
            selection_lst.append(self.probs_list[item_no][feature_list[item_no]][pred_var])
        selection_lst.append(self.prob_col_dict[pred_var])
        return selection_lst

    def productOfItems(self, item_list):
        product = 1
        for item in item_list:
            product = product * item
        return product

    def getMaxIndex(self, productsOfClass):
        return productsOfClass.index(max(productsOfClass))

    def predict(self, feature_list):
        productsOfClass = []
        for pred_item in self.prob_col_dict.keys():
            probabs_lst = self.getProbabilityItems(feature_list = feature_list, pred_var = pred_item)
            productsOfClass.append(self.productOfItems(item_list = probabs_lst))
        prediction = list(self.prob_col_dict.keys())[self.getMaxIndex(productsOfClass)]
        return prediction

    def test(self):

        predictions = []
        for row in range(self.test_df.shape[0]):
            feature_list = self.test_df.iloc[row]
            feature_list = list(feature_list[:6])
            predictions.append(self.predict(feature_list = feature_list))
        no_of_right_predictions = self.test_df[self.test_df[self.test_df.iloc[0].shape[0] - 1] == predictions].shape[0]
        total_class = self.test_df.shape[0]
        return (no_of_right_predictions/ total_class)

    def setTrainingData(self, train_df):
        self.train_df = train_df
    def setTestingData(self, test_df):
        self.test_df = test_df