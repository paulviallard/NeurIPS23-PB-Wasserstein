import re
import numpy as np
import pandas as pd

from core.lock import Lock

###############################################################################


class NDData(Lock):

    def _load(self):

        try:
            # We try to read the csv file (in the lock file)
            self.data = pd.read_csv(
                self._lock_file,
                index_col=0)
        except pd.errors.EmptyDataError:
            # If the file is empty, we create a new data frame
            self.data = pd.DataFrame()
        except FileNotFoundError:
            # If we do not found the lock file, we keep lock flag to False
            self.got_lock = False

        # We initialize the data
        self._init_hidden_data()
        self._init_index()

    def _save(self):
        # We save the (new) data in the lock file
        self.data.to_csv(self._lock_file)
        return True

    # ----------------------------------------------------------------------- #
    # Data

    def _init_hidden_data(self):
        # We first copy the data
        self._hidden_data = self.data.copy(deep=True)

        # For each index,
        for i, index_ in enumerate(list(self.data.index)):

            # We get all the key and values associated with an index
            index = self.__interpret_index(index_)

            # For each key, we will add a new column in the data
            for col, val in index:
                # If the column does not exist, we create it
                if(col not in self._hidden_data):
                    self._hidden_data[col] = np.nan
                # We add the value (associated with the key) to the data
                self._hidden_data.loc[index_, col] = val

    def _init_index(self):

        # We create a dictionary where we store the values
        self.index_dict = {}

        # For each index,
        for i, index_ in enumerate(list(self._hidden_data.index)):

            # We get all the key and values associated with an index
            index = self.__interpret_index(index_)

            # For each key/value
            for key, val in index:

                # If the key is not in the dictionay
                if(key not in self.index_dict):
                    # We add the value and the key in the dictionay
                    self.index_dict[key] = {val: [i]}
                # If the value is not in the dictionary, we add it
                elif(val not in self.index_dict[key]):
                    self.index_dict[key][val] = [i]
                # Otherwise, we add the value to the list
                else:
                    self.index_dict[key][val].append(i)

    def __rename_dash(self, var):
        # For a given variable, we replace "-" by "_" (for the indices)
        var = var.replace("-", "_")
        return var

    def __interpret_index(self, index):

        key_val_dict = []

        # For split the index to get all the values
        for key_val in index.split("/"):
            # For each value, we split it again to obtain the "key" and "val"
            # from "key=val"
            key_val = re.split("[=,]", key_val)

            # If we have "key=val"
            if(len(key_val) == 2):
                # We get the "key"
                key = key_val[0]
                key = self.__rename_dash(key)
                # We get the "val"
                val = key_val[1]
                # and we save in the dictionary
                key_val_dict.append((key, val))
            else:
                # Otherwise, we raise an error
                raise RuntimeError("We must have the key=val in the index")

        # We return the dictionary
        return tuple(key_val_dict)

    def __create_index(self, index_dict):
        # We initialize the index
        index = ""
        # For each key,
        for key in sorted(list(index_dict.keys())):
            # we add in the index the key and the values
            index += "{}={}/".format(
                key, index_dict[key])
        # We remove the last character "/"
        index = index[:-1]
        return index

    # ----------------------------------------------------------------------- #
    # Functions

    def index_keys(self, key=None, sort=None):
        return self.do(self.__index_keys, key=key, sort=sort)

    def __index_keys(self, key=None, sort=None):

        # We load the data
        self._load()

        # If "key" is None, we get the keys of the data
        if(key is None):
            key_list = list(self.index_dict.keys())
        # If "key" is not None, we get the values for a given "key"
        else:
            key_list = list(self.index_dict[key].keys())

        # We sort the keys
        key_list.sort(key=sort)

        # We return the keys
        return key_list

    def col_keys(self, sort=None):
        return self.do(self.__col_keys, sort=sort)

    def __col_keys(self, sort=None):
        # We load the data
        self._load()

        # We get the columns of the data
        key_list = list(self.data.columns)
        # We sort the columns
        key_list.sort(key=sort)
        return key_list

    def get(self, *args, **kwargs):
        return self.do(self.__get, *args, **kwargs)

    def __get(self, *args, **kwargs):
        # We load the data
        self._load()

        # We get the data with the columns in "args"
        data = self._hidden_data[list(args)].copy(deep=True)

        # We initialize the set of rows to select
        row_set = set(range(len(data)))

        # For each key and value(s) in "kwargs"
        for key, val in kwargs.items():

            # If we have only one value, we convert it in a list (with one
            # element)
            if(not(isinstance(val, list))):
                val = [val]

            # We initialize a set of rows to select for a given key/value(s)
            new_row_set = set()

            val_list = val
            # For each value in the list of values
            for val in val_list:
                val = str(val)

                # We add the rows in "new_row_set" for has the "value"
                # associated with "key"
                try:
                    new_row_set = (
                        set(self.index_dict[key][val]) | new_row_set)
                except KeyError:
                    pass

            # We select the rows that are in "row_set" and "new_row_set"
            row_set = row_set & new_row_set

        # We get the data if the selected rows
        data = data.iloc[list(row_set)]
        # We sort the rows by indices
        data = data.sort_index()
        # We remove the index column
        data = data.reset_index()
        data = data.drop(columns=["index"])
        return data

    def set(self, col_dict, index_dict, erase=False):
        return self.do(self.__set, col_dict, index_dict, erase=erase)

    def __set(self, col_dict, index_dict, erase=False):

        # We load the data
        self._load()

        # For each key,
        for key in self.index_dict.keys():
            # if it does not exist in the dictionary, we create it
            if(key not in index_dict):
                index_dict[key] = None
        # We create the name of the index
        index_name = self.__create_index(index_dict)

        # For each index in the data
        for old_index in self.data.index:
            # We get the (old) dictionary associated with the index
            old_index_dict = dict(self.__interpret_index(old_index))
            # For each key in the new dictionary
            for key in index_dict:
                # We insert the new key if it's not in the (old) dictionary
                if(key not in old_index_dict):
                    old_index_dict[key] = None
            # We create the new name of the index
            new_index = self.__create_index(old_index_dict)
            # and we rename it in the data
            self.data = self.data.rename(index={old_index: new_index})

        # If the index exists
        if(index_name in self.data.index):

            # If erase = "full", we erase the line associated to the index
            if(erase == "full"):
                for column in self.data.columns:
                    self.data.at[index_name, column] = np.nan

            # for all columns in the dictionary
            for column, value in col_dict.items():

                # We create the column in the data if it does not exist
                if(column not in self.data.columns):
                    # In order to do
                    # self.data[column] = np.nan,
                    # we execute the following lines
                    # to remove a PerformanceWarning
                    data_copy = self.data.copy()
                    data_copy[column] = np.nan
                    self.data = data_copy

                # We write the value in the right column and index
                if(np.isnan(self.data.loc[index_name][column])
                   or (not(np.isnan(self.data.loc[index_name][column]))
                       and (erase == "partial" or erase == "full"))):
                    self.data.at[index_name, column] = value

        else:
            # Otherwise, we create the row and insert the values
            old_set = set(self.data.columns)
            new_set = set(col_dict.keys())
            new_column = sorted(list(new_set))
            new_set = sorted(list(old_set.difference(new_set)))

            # We create the new columns
            for column in new_set:
                col_dict[column] = np.nan
            for column in new_column:
                if column not in self.data:
                    # In order to do
                    # self.data[column] = np.nan,
                    # we execute the following lines
                    # to remove a PerformanceWarning
                    data_copy = self.data.copy()
                    data_copy[column] = np.nan
                    self.data = data_copy

            # We add the values
            self.data.loc[index_name] = col_dict

        # We save the data
        self._save()

    def __str__(self):
        return self.do(self.__str)

    def __str(self):
        # We print the data
        return str(self.data)

    # ----------------------------------------------------------------------- #

    @staticmethod
    def to_latex(data, fun, col_name_list=None, col_format="l"):
        # We copy the data
        data = data.copy(deep=True)
        copy_data = data.copy(deep=True)

        # For each value in the data
        for i in list(data.index):
            for col in list(data.columns):
                # we run the function
                data.loc[i, col] = fun(data.loc[i, col], i, col, copy_data)
        # We change the name in the columns
        if(col_name_list is not None):
            data.columns = col_name_list

        # We return the LaTeX of the table
        s = data.style
        s = s.hide(axis=0)
        return s.to_latex(
            column_format=col_format, hrules=True)
