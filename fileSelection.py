import pandas as pd
from tkinter import filedialog
from tkinter import Tk

class fileSelect:
    def __init__(self):
        self.win = Tk()
        self.win.wm_attributes('-topmost', 1)
        self.win.withdraw()
    def importFilesNamesList(self):
        file_types = (('text files', '*.txt'), ('CSV files', '*.csv'))
        
        self.train_dir_name = filedialog.askopenfile(filetypes=file_types, title= 'Select Training dataset')
        self.test_dir_name = filedialog.askopenfile(filetypes=file_types, title='Select Testing Dataset')


    def importFiles(self):
        
        self.importFilesNamesList()
        train_df = pd.read_csv(self.train_dir_name, header = None)
        test_df = pd.read_csv(self.test_dir_name, header = None)
        return train_df, test_df

    def importTestFileName(self):
        
        file_types = (('text files', '*.txt'), ('CSV files', '*.csv'))
        self.test_dir_name = filedialog.askopenfile(parent=self.win, filetypes=file_types, title='Select Testing Dataset')
        print('checkpoint 2 complete')


    def importTrainFileName(self):
        
        file_types = (('text files', '*.txt'), ('CSV files', '*.csv'))
        self.train_dir_name = filedialog.askopenfile(parent=self.win, filetypes=file_types, title= 'Select Training dataset')

    def importTrainFile(self):
        
        self.importTrainFileName()
        train_df = pd.read_csv(self.train_dir_name, header = None)
        return train_df

    def importTestFile(self):
        
        self.importTestFileName()
        test_df = pd.read_csv(self.test_dir_name, header = None)
        return test_df