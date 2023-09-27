# Copyright (C) 2022 Sunnybrook Research Institute
# This file is part of Phindr3D <https://github.com/DWALab/Phindr3D>.
#
# Phindr3D is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Phindr3D is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Phindr3D.  If not, see <http://www.gnu.org/licenses/>.

import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from ..GUI.windows.helperclasses import *
from ..GUI.windows.platemapWindow import *

class TrainingFunctions:
    """Static methods for training.

    Referenced from
    https://github.com/DWALab/Phindr3D/tree/9b95aebbd2a62c41d3c87a36f1122a78a21019c8/Lib
    and
    https://github.com/SRI-RSST/Phindr3D-python/blob/ba588bc925ef72c72103738d17ea922d20771064/phindr_functions.py
    """

    @staticmethod
    def partition_data(mv_data, grps, select_grps):
        """Split selected class into 60/40 (train/test)."""
        train_mv=[]
        train_lbl=[]
        train_ind=[]
        for grp in select_grps:
            idx=np.array(np.where(grps==grp)[0], dtype=int)
            X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
                mv_data[idx], grps[idx],idx,test_size = 0.4)
            train_mv.extend(X_train)
            train_lbl.extend(y_train)
            train_ind.extend(ind_train)
        #test data & indices
        test_mv = np.delete(mv_data[:], train_ind, axis= 0)
        test_ind= np.delete(
            np.linspace(0,np.shape(mv_data)[0]-1, num=np.shape(mv_data)[0], dtype=int),
            train_ind, axis=0)
        return(train_mv, train_lbl, test_mv, test_ind)

    @staticmethod
    def random_forest_model(X_train, y_train, X_test, test_classes):
        """Get classifier predictions of selected classes."""
        clf = RandomForestClassifier(n_estimators=500, bootstrap=True)
        clf.fit(X_train, y_train)
        labels=clf.predict(X_test)
        class_table = pd.crosstab(index=test_classes,columns=labels)
        class_table.index.name = None
        return(class_table)
    # end random_forest_model

    def selectclasses(self, featureDF, platemap):
        """Open platemapWindow's data selection to select training classes, and pass to random forest model."""
        '''Remember: X = self.filtered_data. Now, partition_data takes in mv data as np.array, not pd.DataFrame.
        So, we need to convert self.filtered_data to np.array - taking just the columns that are of the format "MV#". 
        We can do this using df.columns.str.contains("MV") to get the columns that contain "MV" in their name. Labels 
        are just the titles of the treatment classes (i.e. what we named them) of all rows while select_grps are the 
        treatments selected as training classes.
        '''

        try:
            featureDF = pd.read_csv(featureDF, sep='\t')
            try:
                view = platemapWindow(dataframe = platemap, meta = featureDF, first = False)
                view.show()
                view.exec()
            except:
                print("Error in opening platemapWindow")
                pass
            mv = pd.read_csv('tmp1.tsv', sep='\t',  na_values='        NaN')
            # delete tmp1.tsv
            os.remove('tmp1.tsv')
            # drop rows with NaN values.
            mv = mv.fillna(0)
            lbls = mv.iloc[:, -1]
            select_grps = lbls[~lbls.str.contains("_SAMPLE")]
            mv = mv.loc[:, mv.columns.str.contains("MV")]
            mv = mv.loc[:, ~mv.columns.str.contains("NumMV")]
            mv = mv.to_numpy()
            lbls = lbls.to_numpy()
            select_grps = select_grps.to_numpy()
            select_grps = np.unique(select_grps)

            X_train, y_train, X_test, y_test= self.partition_data(mv, lbls, select_grps)
            class_tbl=self.random_forest_model(X_train, y_train, X_test, lbls[y_test])

            # get column names from class_tbl and add a column for each column name as "<name_percentage>"
            class_colnames = class_tbl.columns
            for col in class_colnames:
                class_tbl[col + "_percentage"] = 100 * class_tbl[col] / class_tbl[class_colnames].sum(axis=1)

            #export classification table
            name = QFileDialog.getSaveFileName(None, 'Save File', filter=".csv")
            if name[0]!= '':
                class_tbl.to_csv("".join(name), sep=',', mode='w')
            #except: # print error message
        except Exception as e:
            print(e)
    # end selectclasses
# end TrainingFunctions


