
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
# Re-write the code using PyQt6 instead of PySide6

import pandas as pd
import numpy as np

from PyQt6.QtWidgets import * #import (QApplication, QWidget, QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, \
                            #QPushButton, QItemDelegate, QVBoxLayout)
from PyQt6.QtCore import QAbstractTableModel, Qt, QModelIndex, QItemSelectionModel
from PyQt6.QtGui import *

from PyQt6 import QtWidgets
from PyQt6 import QtCore

import os
import time
from .helperclasses import *

class FloatDelegate(QItemDelegate):
    # adapted from https://learndataanalysis.org/create-a-pandas-dataframe-editor-with-pyqt5/
    def __init__(self, parent=None):
        super().__init__()

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setValidator(QDoubleValidator())
        return editor

class TableWidget(QTableWidget):
    def __init__(self, df, *args):
        super().__init__()
        self.df = df
        #self.setStyleSheet('font-size: 35px;')

        # set table dimension
        nRows, nColumns = self.df.shape
        self.setColumnCount(nColumns)
        self.setRowCount(nRows)

        self.setHorizontalHeaderLabels(self.df.columns)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents) 
        #self.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.setVerticalHeaderLabels(self.df.index.astype(str))
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setItemDelegateForColumn(1, FloatDelegate())

        # data insertion
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                self.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))

        self.cellChanged[int, int].connect(self.updateDF)

    def updateDF(self, row, column):
        text = self.item(row, column).text()
        self.df.iloc[row, column] = text

class platemapWindow(QDialog):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe, meta = None, parent = None, first = True):
        if first:
            #QAbstractTableModel.__init__(self, parent)
            super().__init__()
            self.resize(1200, 800)
            mainLayout = QGridLayout()
            tableBox =QVBoxLayout()
            # if dataframe is a dictionary, then we have multiple dataframes to display in tabs
            metapd = pd.read_csv(meta.GetMetadataFilename(), sep="\t")
            if type(dataframe) == dict:    # i.e. if Excel file is passed...
                for i in dataframe.keys(): # each key is a sheet name in the xlsx file.
                    tabwidget = QTabWidget()
                    df = dataframe[i].set_index(dataframe[i].columns[0])
                    # replace all nan with empty string
                    df = df.replace(np.nan, '', regex=True)
                    metapd[i] = metapd.apply(lambda x: df.iloc[int(x['Row'])-1, int(x['Column'])-1], axis = 1)
                    table = TableWidget(df)
                    tabwidget.addTab(table, i)
                    tableBox.addWidget(tabwidget)
                    tabwidget.setTabPosition(QTabWidget.TabPosition.West)
                
            elif type(dataframe) == pd.DataFrame:
                self._dataframe = dataframe
                name = self._dataframe.iloc[0, 0]
                self.table = TableWidget(self._dataframe)
                tabwidget = QTabWidget()
                tabwidget.addTab(self.table, name)
                tableBox.addWidget(tabwidget)
                df = dataframe.set_index(dataframe.columns[0])
                df = df.replace(np.nan, '', regex=True)
                tabwidget.setTabPosition(QTabWidget.TabPosition.West)
                metapd[name] = metapd.apply(lambda x: df.iloc[int(x['Row']), int(x['Column'])], axis = 1)
            
            self._metadata = metapd
            buttonBox = QVBoxLayout()
            button_export = QPushButton('Save Updated Metadata File')
            button_export.clicked.connect(self.updateMetadata)
            cancel_button = QPushButton('Cancel')
            cancel_button.clicked.connect(self.close)
            buttonBox.addWidget(button_export)
            buttonBox.addWidget(cancel_button)
            mainLayout.addLayout(tableBox, 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
            mainLayout.addLayout(buttonBox, 0, 1, alignment=QtCore.Qt.AlignmentFlag.AlignBottom)
            self.setLayout(mainLayout)
            self.platemapDataframe = dataframe.copy()
        
        else:       # this is the data selection window.
            #QAbstractTableModel.__init__(self, parent)
            super().__init__()
            self.resize(1200, 800)
            mainLayout = QGridLayout()
            tableBox =QVBoxLayout()
            self.classesDict = {}
            self.samplesDict = {}
            self.selectedDataframe = None

            metapd = pd.read_csv(meta, sep="\t", na_values='        NaN')   # this time the meta is a file path.

            #dataframe = pd.read_excel('/Users/work/Desktop/SplitImages/platemap_sample.xlsx', sheet_name = None)
            # if dataframe (i.e. platemap) is none, print error message and return
            
            if dataframe is None:
                print('Error: PLATEMAP is None. Please load a platemap in the previous window.')
                return
            
            # if dataframe is a dictionary, then we have multiple dataframes to display in tabs
            if type(dataframe) == dict:    # i.e. if Excel file is passed...
                tables = []
                for i in dataframe.keys(): # each key is a sheet name in the xlsx file.
                    tabwidget = QTabWidget()
                    df = dataframe[i].set_index(dataframe[i].columns[0])
                    # replace all nan with empty string
                    df = df.replace(np.nan, '', regex=True)
                    metapd[i] = metapd.apply(lambda x: df.iloc[int(x['Row']) - 1, int(x['Column']) - 1], axis = 1)
                    table = TableWidget(df)
                    tables.append(table)
                    tabwidget.addTab(table, i)
                    tableBox.addWidget(tabwidget)
                    tabwidget.setTabPosition(QTabWidget.TabPosition.West)
                
            elif type(dataframe) == pd.DataFrame:
                self._dataframe = dataframe
                name = self._dataframe.iloc[0, 0]
                self.table = TableWidget(self._dataframe)
                tabwidget = QTabWidget()
                tabwidget.addTab(self.table, name)
                tableBox.addWidget(tabwidget)
                df = dataframe.set_index(dataframe.columns[0])
                df = df.replace(np.nan, '', regex=True)
                tabwidget.setTabPosition(QTabWidget.TabPosition.West)
                metapd[name] = metapd.apply(lambda x: df.iloc[int(x['Row']), int(x['Column'])], axis = 1)
            
            self.tables = tables
            self._featurefile = metapd

            button_export = QPushButton('Add to Selection')
            button_export.clicked.connect(self.syncSelection)

            clear_button = QPushButton('Clear Selections')
            clear_button.clicked.connect(self.clearSelection)

            classSelect = QPushButton('Select as Label/Class')
            classSelect.clicked.connect(self.addClass)

            sampleSelect = QPushButton('Select as Sample')
            sampleSelect.clicked.connect(self.addSample)

            selectionButtonsBox = QVBoxLayout()
            selectionButtonsBox.addWidget(button_export)
            selectionButtonsBox.addWidget(clear_button)
            
            nameInput = QLineEdit()
            self.labelInput = nameInput


            nameInput.setPlaceholderText('Enter a name for the selection')
            nameInput.setFixedWidth(150)
            selectionButtonsBox.addWidget(nameInput)
            selectionButtonsBox.addWidget(classSelect)
            selectionButtonsBox.addWidget(sampleSelect)

            displayBox = QVBoxLayout()

            selectionButtonsBox.addLayout(displayBox)

            submit_button = QPushButton('Submit')
            submit_button.clicked.connect(self.submit)

            mainLayout.addLayout(tableBox, 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
            mainLayout.addLayout(selectionButtonsBox, 0, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
            mainLayout.addWidget(submit_button, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignBottom)
            self.setLayout(mainLayout)
            
    def updateMetadata(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', '', "TSV files (*.tsv)")
        # now, save metapd
        try:
            self._metadata.to_csv(fname[0], sep = '\t', index = False)
            print('Metadata file exported.')
            # show new dialog box to confirm
            msg = QMessageBox()
            msg.setWindowTitle("Metadata file exported")
            msg.setText("Updated metadata file successfully exported!")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.exec()
            # close the window
            self.close()

        except:
            print("Error at updateMetadata")
            pass
    
    def syncSelection(self):
        try:
            tables = self.tables.copy()
            index_list = []
            for table in tables:
                index = table.selectedIndexes()
                index_list.append(index)

            for indexes in index_list:
                ranges = []
                for index in indexes:
                    row = index.row()
                    col = index.column()
                    range1 = QTableWidgetSelectionRange(row, col, row, col)
                    ranges.append(range1)
                for table in tables:
                    for range in ranges:
                        table.setRangeSelected(range, True)
        except:
            print("Error at syncSelection")
            pass

    def clearSelection(self):
        try:
            for table in self.tables:
                table.clearSelection()
        except:
            print("Error at clearSelection")
            pass
    
    def addClass(self):
        '''Get the [row, col] of the selected cells in the table with the most selections and add them to the classDict'''
        try:
            tables = self.tables.copy()
            index_list = []
            for table in tables:
                index = table.selectedIndexes()
                index_list.append(index)
            # find the table with the most selections
            max_index = 0
            for i in range(len(index_list)):
                if len(index_list[i]) > len(index_list[max_index]):
                    max_index = i
            # get the [row, col] of the selected cells in the table with the most selections
            index = index_list[max_index]

            # now convert to [row, col] for every index in index
            index2 = []
            for i in range(len(index)):
                row = index[i].row()
                row = int(row) + 1
                col = index[i].column()
                col = int(col) + 1
                index2.append([row, col])

            # append the [row, col] to the classDict
            name = self.labelInput.text()
            self.classesDict[name] = index2

            # now, get just the rows of phindfeature that correspond to the selected cells
            for key in self.classesDict.keys():
                selections = self.classesDict[key]
                for selection in selections:
                    try:
                        row = selection[0]
                        col = selection[1]
                        # get the rows of phindfeature that corresponds to this row, col - phindfeature has columns called 'Row' and 'Column'
                        rows = self._featurefile.loc[(self._featurefile['Row'] == row)]
                        rows = rows.loc[(rows['Column'] == col)]
                        rows['Type'] = key # this is the class name
                        # now, add this row to the selectedDataframe
                        if self.selectedDataframe is None:
                            self.selectedDataframe = rows
                        else:
                            self.selectedDataframe = pd.concat([self.selectedDataframe, rows])
                    except:
                        print("Error at addClass")
                        pass

            for table in self.tables:
                table.clearSelection()
            # clear the input box
            self.labelInput.clear()

            for table in self.tables:
                table.clearSelection()
            # clear the input box
            self.labelInput.clear()
            # self.selectedDataframe.to_csv('tmp1.tsv', sep = '\t', index = False)

        except:
            print("Error at addClass")
            pass

    def addSample(self):
        '''Get the [row, col] of the selected cells in the table with the most selections and add them to the sampleDict'''
        try:
            tables = self.tables.copy()
            index_list = []
            for table in tables:
                index = table.selectedIndexes()
                index_list.append(index)
            # find the table with the most selections
            max_index = 0
            for i in range(len(index_list)):
                if len(index_list[i]) > len(index_list[max_index]):
                    max_index = i
            # get the [row, col] of the selected cells in the table with the most selections
            index = index_list[max_index]

            # now convert to [row, col] for every index in index
            index2 = []
            for i in range(len(index)):
                row = index[i].row()
                row = int(row) + 1
                col = index[i].column()
                col = int(col) + 1
                index2.append([row, col])

            # add the [row, col] to the classDict
            name = self.labelInput.text()
            self.samplesDict[name] = index2

            # now, get just the rows of phindfeature that correspond to the selected cells
            for key in self.samplesDict.keys():
                selections = self.samplesDict[key]
                for selection in selections:
                    try:
                        row = selection[0]
                        col = selection[1]
                        # get the rows of phindfeature that corresponds to this row, col - phindfeature has columns called 'Row' and 'Column'
                        rows = self._featurefile.loc[(self._featurefile['Row'] == row)]
                        rows = rows.loc[(rows['Column'] == col)]
                        rows['Type'] =  str(key) + "_SAMPLE"  # this is the class name
                        # now, add this row to the selectedDataframe
                        if self.selectedDataframe is None:
                            self.selectedDataframe = rows
                        else:
                            self.selectedDataframe = pd.concat([self.selectedDataframe, rows])
                    except:
                        print("Error at addSample")
                        pass
            
            # clear the selections
            for table in self.tables:
                table.clearSelection()

            # clear the input box
            self.labelInput.clear()
            #self.selectedDataframe.to_csv('tmp1.tsv', sep = '\t', index = False)

        except:
            print("Error at addSample")
            pass
    
    # a function for the submit button
    def submit(self):
        # get cwd, save the selectedDataframe to a file in cwd as tmp.tsv, and return the selectedDataframe path.
        cwd = os.getcwd()
        # path = cwd + '/tmp.tsv'
        self.selectedDataframe.to_csv(cwd + '/tmp-selectedClassSample.tsv', sep = '\t', index = False)
        self.close()
