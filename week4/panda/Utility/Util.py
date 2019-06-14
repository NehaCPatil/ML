import pandas as pd
import numpy as np


class Utility:
    
    def __init__(self):
        self.exam_data = {
            'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin',
                     'Jonas'],
            'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
            'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
            'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
        self.labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        self.data_frame = pd.DataFrame(self.exam_data, index=self.labels)

    data = pd.Series()
    data_series = pd.Series()

    def series_data(self, n):
        for x in range(n):
            element = int(input("Enter element"))
            self.data.at[x] = element
        return self.data

    def convert_series_to_list(self, data):
        print("Convert Pandas Series to Python list")
        my_list = data.tolist()
        print(my_list)
        print(type(my_list))

    def operation_on_series(self, data, data1):
        ds = data + data1
        print("\n Add two Series:")
        print(ds)
        print("\n Subtract two Series:")
        ds = data - data1
        print(ds)
        print("\n Multiply two Series:")
        ds = data * data1
        print(ds)
        print("\n Divide Series1 by Series2:")
        ds = data / data1
        print(ds)

    def powers_of_an_array(self, data):
        print(np.power(data, 3))

    def dataFrame_from_dictionary(self):
        print(self.data_frame)

    def basic_Information_About_Data_Frame(self):
        print("describing:\n{}\nshape:{}\naxes:{}\nmissing vals:\n{}\ninfo:{}".format(self.data_frame.describe(),
                                                                                      self.data_frame.shape,
                                                                                      self.data_frame.axes,
                                                                                      self.data_frame.isna().sum(),
                                                                                      self.data_frame.info()))

    def first_3_rows_of_DataFrame(self):
        print(self.data_frame.head(3))

    def select_name_and_score_columns(self):
        print(self.data_frame[['name', 'score']])

    def specified_Columns_and_rows(self):
        print(self.data_frame.loc[['a', 'c', 'e', 'f'], ['name', 'score']])


    def select_rows_with_attempt_more_than_2(self):
        print(self.data_frame[(self.data_frame.attempts > 2)])
    #

    def number_of_rows_and_columns(self):
        print(self.data_frame, "\nnumber of rows and columns\n", self.data_frame.shape)

    def select_rows_where_score_is_missing(self):
        print(self.data_frame[self.data_frame.score.isna()])

    def number_attempts_less_than_2_and_score_greater_than_15(self):
        print(self.data_frame.dtypes)
        print(self.data_frame[(self.data_frame.score > 15.0) & (self.data_frame.attempts > 2)])

    def change_score_in_row_d(self):
        print("before changjng row d :\n", self.data_frame.loc[['d', 'e'],], "\n")
        self.data_frame.loc['d', 'score'] = 11.5
        print("After change: ", self.data_frame.head())

    def sum_of_attempts(self):
        print("Sum of attempts is ", self.data_frame.attempts.sum())

    def mean_score(self):
        print("mean of score is ", self.data_frame.score.mean())

    def append_row_k_(self):
        self.data_frame.loc['k'] = {'name': "Suresh", 'score': 15.5, 'attempts': 1, 'qualify': "yes"}
        print(self.data_frame)

    def sort_DataFrame_by_name_in_descending_and_ascending_order(self):
        print("Sorting in descending\n{}\n\nSorting in ascending\n{}".format(
            self.data_frame.sort_values(by=['name'], ascending=False)
            , self.data_frame.sort_values(by=['name'])))

    def replace_qualify_column_values(self):
        # inplace is to set the values permanently
        self.data_frame.qualify.replace(['yes', 'no'], ['True', 'False'], inplace=True)
        print(self.data_frame)

    def delete_attempts_column(self):
        print(self.data_frame.drop('attempts', axis=1))

    def insert_new_Column_in_existing_DataFrame(self):
        colour = pd.DataFrame(['red', 2, 'black', 'green', 'red', 'blu', 'black', 'green', 'red', 'voilet'])
        self.data_frame['colour'] = colour
        print(self.data_frame.info())

    def iterate_over_rows_in_DataFrame(self):
        for rows in self.data_frame.iterrows():
            print('\n', rows)

    def list_from_DataFrame_column_headers(self):
        df = pd.DataFrame(self.exam_data , index=self.labels)
        print(list(df.columns.values))



