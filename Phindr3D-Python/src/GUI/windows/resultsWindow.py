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

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import numpy as np
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from scipy.optimize import curve_fit

try:
    from .interactive_click import interactive_points
    from .plot_functions import *
    from ...Training import *
    from .platemapWindow import *
except ImportError:
    from src.GUI.windows.interactive_click import interactive_points
    from src.GUI.windows.plot_functions import *
    from src.GUI.windows.platemapWindow import *
    from src.Training import *

class resultsWindow(QDialog):
    """Build a GUI window for the user to analyze data and view plots."""  
    def __init__(self, color, metadata, platemap = None):
        """Construct the GUI window for users to analyze data and view plots."""
        super(resultsWindow, self).__init__()
        self.platemap = platemap
        self.setWindowTitle("Results")
        self.feature_file=[]
        self.imageIDs=[]
        self.plots=[]
        self.filtered_data=0
        self.numcluster=None
        self.metadata=metadata
        self.bounds=0
        self.color=color
        #menu tabs
        menubar = QMenuBar()
        file = menubar.addMenu("File")
        inputfile = file.addAction("Input Feature File")
        data = menubar.addMenu("Data Analysis")
        classification = data.addMenu("Classification")
        selectclasses = classification.addAction("Select Classes")
        doseResponse = classification.addAction("Plot Dose Response")
        clustering = data.addMenu("Clustering")
        estimate = clustering.addAction("Estimate Clusters")
        setnumber = clustering.addAction("Set Number of Clusters")
        piemaps = clustering.addAction("Pie Maps")
        export = clustering.addAction("Export Cluster Results")
        plotproperties = menubar.addMenu("Plot Properties")
        rotation_enable = plotproperties.addAction("3D Rotation Enable")
        rotation_disable = plotproperties.addAction("3D Rotation Disable")
        pointSize = plotproperties.addAction("Point Size By ...")
        resetview = plotproperties.addAction("Reset Plot View")
        selectData = data.addAction("Select Data")

        # defining widgets
        box = QGroupBox()
        boxlayout = QGridLayout()
        selectfile = QPushButton("Select Feature File")
        prevdata = QPushButton("Import Previous Plot Data (.JSON)")
        exportdata = QPushButton("Export Current Plot Data (.JSON)")
        cmap=QPushButton("Legend Colours")
        map_type = QComboBox()
        map_type.addItems(["PCA","t-SNE","Sammon", "UMAP"])
        twod = QRadioButton("2D")
        threed = QRadioButton("3D")
        dimensionbox = QGroupBox()
        dimensionboxlayout = QHBoxLayout()
        dimensionboxlayout.addWidget(twod)
        dimensionboxlayout.addWidget(threed)
        dimensionbox.setLayout(dimensionboxlayout)
        colordropdown = QComboBox()
        boxlayout.addWidget(QLabel("File Options"), 0, 0, 1, 1)
        boxlayout.addWidget(selectfile, 1, 0, 1, 1)
        boxlayout.addWidget(exportdata, 2, 0, 1, 1)
        boxlayout.addWidget(prevdata, 3, 0, 1, 1)
        boxlayout.addWidget(QLabel("Plot Type"), 0, 1, 1, 1)
        boxlayout.addWidget(map_type, 1, 1, 1, 1)
        boxlayout.addWidget(dimensionbox, 2, 1, 1, 1)
        boxlayout.addWidget(cmap, 2, 2, 1, 1)
        boxlayout.addWidget(QLabel("Color By"), 0, 2, 1, 1)
        boxlayout.addWidget(colordropdown, 1, 2, 1, 1)
        box.setLayout(boxlayout)
        #menu actions activated
        inputfile.triggered.connect(
            lambda: self.loadFeaturefile(colordropdown, map_type.currentText(), True))
        selectclasses.triggered.connect(                    # classificationRF
            lambda: self.classificationRF())                                                                          
        doseResponse.triggered.connect(
            lambda: self.plotDoseResponse())
        estimate.triggered.connect(
            lambda: Clustering.Clustering().cluster_est(self.filtered_data)
                if len(self.plot_data) > 0
                else errorWindow("Error Dialog","Please Select Feature File. No data is currently displayed"))
        setnumber.triggered.connect(
            lambda: self.setnumcluster(colordropdown.currentText())
                if len(self.plot_data) > 0
                else errorWindow("Error Dialog","Please Select Feature File. No data is currently displayed"))
        piemaps.triggered.connect(
            lambda: Clustering.piechart(self.plot_data, self.filtered_data,
                    self.numcluster, np.array(self.labels),
                    [np.array(plot.get_facecolor()[0][0:3]) for plot in self.plots])
                if len(self.plot_data) > 0
                else errorWindow("Error Dialog","Please Select Feature File. No data is currently displayed"))
        export.triggered.connect(
            lambda: Clustering.export_cluster(self.plot_data, self.filtered_data,
                    self.numcluster, self.feature_file[0])
                if len(self.plot_data) > 0
                else errorWindow("Error Dialog","Please Select Feature File. No data is currently displayed"))
        rotation_enable.triggered.connect(lambda: self.main_plot.axes.mouse_init())
        rotation_disable.triggered.connect(lambda: self.main_plot.axes.disable_mouse_rotation())
        resetview.triggered.connect(lambda: reset_view(self))
        pointSize.triggered.connect(lambda: self.pointSizeFeature())
        exportdata.clicked.connect(
            lambda: save_file(self, map_type.currentText())
                if len(self.plot_data) > 0
                else errorWindow("Error Dialog","Please Select Feature File. No data is currently displayed"))
        prevdata.clicked.connect(
            lambda: import_file(self, map_type, colordropdown, twod, threed))
        selectData.triggered.connect(lambda: self.chooseDataSubset())
        #setup Matplotlib
        matplotlib.use('qtagg')
        self.plot_data = []
        self.labels = []
        self.main_plot = MplCanvas(self, width=10, height=10, dpi=100, projection="3d")

        # get size of points
        self.point_size = 10
        self.alpha = 0.1

        # plot points
        sc_plot = self.main_plot.axes.scatter3D(
            [], [], [], s=self.point_size, alpha=self.alpha, depthshade=False)  # , picker=True)
        self.main_plot.axes.set_position([-0.2, -0.05, 1, 1])
        self.original_xlim = sc_plot.axes.get_xlim3d()
        self.original_ylim = sc_plot.axes.get_ylim3d()
        self.original_zlim = sc_plot.axes.get_zlim3d()
        self.projection = "2d"  # update from radiobutton

        def check_projection(dim, plot):
            """Set configuration based on user selection of 2d versus 3d."""
            if dim == "2d":
                self.projection = dim
                self.main_plot.axes.mouse_init()
                self.main_plot.axes.view_init(azim=-90, elev=90)
                self.main_plot.axes.get_zaxis().line.set_linewidth(0)
                self.main_plot.axes.tick_params(axis='z', labelsize=0)
                self.main_plot.draw()
                self.main_plot.axes.disable_mouse_rotation()
            elif dim == "3d":
                self.projection = dim
                self.main_plot.axes.get_zaxis().line.set_linewidth(1)
                self.main_plot.axes.tick_params(axis='z', labelsize=10)
                self.main_plot.draw()
                #rotate left click, disabled zoom right click
                self.main_plot.axes.mouse_init(rotate_btn=1, zoom_btn=[])
            if self.feature_file and colordropdown.count() > 0 and len(self.plot_data)>0:
                self.data_filt(colordropdown, self.projection, plot, True)

        # button features and callbacks
        selectfile.clicked.connect(
            lambda: self.loadFeaturefile(colordropdown, map_type.currentText(), True))
        cmap.clicked.connect(
            lambda: legend_colors(self)
                if len(self.labels)>0
                else errorWindow("Error Dialog","Please Select Feature File. No data is currently displayed"))
        twod.toggled.connect(
            lambda: check_projection("2d", map_type.currentText()) if twod.isChecked() else None)
        threed.toggled.connect(
            lambda: check_projection("3d", map_type.currentText()) if threed.isChecked() else None)
        threed.setChecked(True)
        picked_pt = interactive_points(
            self.main_plot, self.projection, self.plot_data, self.labels,
            self.feature_file, self.color, self.imageIDs)
        self.main_plot.fig.canvas.mpl_connect('pick_event', picked_pt)
        colordropdown.currentIndexChanged.connect(
            lambda: self.data_filt(colordropdown, self.projection, map_type.currentText(), False)
                if self.feature_file and colordropdown.count() > 0 else None)
        map_type.currentIndexChanged.connect(
            lambda: self.data_filt(colordropdown, self.projection, map_type.currentText(),True)
                if self.feature_file and colordropdown.count() > 0 else None)
        # building layout
        layout = QGridLayout()
        toolbar = NavigationToolbar(self.main_plot, self)
        layout.addWidget(toolbar, 0, 0, 1, 1)
        layout.addWidget(self.main_plot, 1, 0, 1, 1)
        layout.addWidget(box, 2, 0, 1, 1)
        layout.setMenuBar(menubar)
        self.setLayout(layout)
        minsize = self.minimumSizeHint()
        minsize.setHeight(self.minimumSizeHint().height() + 700)
        minsize.setWidth(self.minimumSizeHint().width() + 700)
        self.setFixedSize(minsize)
    # end constructor

    def loadFeaturefile(self, grouping, plot, new_plot, prevfile=None):
        """Read from a feature file in response to user selection."""
        filename=''
        if new_plot:
            filename, dump = QFileDialog.getOpenFileName(
                self, 'Open Feature File', '', 'Text files (*.txt *.tsv)')
        if filename != '' or (not isinstance(prevfile, type(None)) and os.path.exists(prevfile)):
            try:
                self.feature_file.clear()
                if new_plot:
                    self.feature_file.append(filename)
                else:
                    self.feature_file.append(prevfile)
                grouping, cancel=self.color_groupings(grouping)
                if not cancel:
                    reset_view(self)
                    self.data_filt(grouping, self.projection, plot, new_plot)
                    self.numcluster=None
            except Exception as ex:
                if len(self.plot_data)==0:
                    grouping.clear()
                errorWindow("Feature File Error",
                    "Check Validity of Feature File (.txt). \nPython Exception Error: {}".format(ex))
    # end loadFeatureFile

    def color_groupings(self, grouping):
        """Set plot point colors based on metadata field values."""
        #read feature file
        feature_data = pd.read_csv(self.feature_file[0], sep='\t', na_values='        NaN')
        grouping.blockSignals(True)
        grps=[]
        #get labels
        chk_lbl = list(filter(
            lambda col: (col[:2].find("MV")==-1 and col!='bounds'
                and col!='intensity_thresholds' and col[:5]!='text_'
                and col.find("Channel_")==-1), feature_data.columns))
        #Get Channels
        meta_col=pd.read_csv(
            feature_data["MetadataFile"].str.replace(r'\\', '/', regex=True).iloc[0],
            nrows=1,  sep="\t", na_values='NaN').columns.tolist()
        col_lbl=list(filter(lambda col: (col.find("Channel")>-1), meta_col))
        #Get MV and Texturefeatures labels
        self.filt=[]
        filt_lbl=np.array(["MV"])
        if max(feature_data.columns.str.contains("text_", case=False)):
            filt_lbl=np.concatenate((filt_lbl, ["Texture_Features"]))

        #select features window
        win=selectWindow(chk_lbl, col_lbl, "Filter Feature File Groups and Channels", "Grouping", "Channels", grps, filt_lbl, self.filt)
        if not win.x_press:
            #change colorby window
            grouping.clear()
            grouping.addItem("No Grouping")
            for col in grps:
                grouping.addItem(col)
        grouping.blockSignals(False)
        return(grouping, win.x_press)
    # end color_groupings

    def pointSizeFeature(self):
        '''Opens a window to select a feature to determine point size'''
        #read feature file
        feature_data = pd.read_csv(self.feature_file[0], sep='\t', na_values='        NaN')
        
        # get column names for features
        columns = feature_data.columns
        # only columns that do not start with MV
        columns = columns[columns.map(lambda col: not col.startswith('MV'))]
        columns = ['Default: Constant Size'] + columns.tolist()
        # prompt user to select a feature
        feature, ok = QInputDialog.getItem(self, "Feature Selection", "Select a feature to determine point size:", columns, 0, False)

        # if user selects a feature, return the feature name
        if ok:
            self.pointSize_feature = feature
        else:
            self.pointSize_feature = 'Default: Constant Size'

        # update point size
        self.updatePointSize()
    # end pointSizeFeature

    def updatePointSize(self):
        '''Updates the point size based on the selected feature'''
        # if the selected feature is not default, update the point size
        if self.pointSize_feature != 'Default: Constant Size':
            # get the feature data
            feature_data = pd.read_csv(self.feature_file[0], sep='\t', na_values='        NaN')
            # get the feature values
            feature_values = feature_data[self.pointSize_feature].to_numpy().astype(np.float64)
            # get the min and max values
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            # get the point size
            self.point_size = 10 + 90 * (feature_values - min_val) / (max_val - min_val)
            self.alpha = 0.1
        else:
            print('Error in updatePointSize')
            pass
            #self.point_size = 10
            #self.alpha = 1

        # update the point size
        for plot in self.plots:
            plot.set_sizes(self.point_size)
            plot.set_alpha(self.alpha)


        # update the plot
        self.main_plot.draw()

    def chooseDataSubset(self):
        """Choose a subset of data to display."""
        try:
            view = platemapWindow(dataframe = self.platemap, meta = self.feature_file[0], first = False) 
            view.show()
            view.exec()
        except:
            print('Error in chooseDataSubset')
            pass


    def data_filt(self, grouping, projection, plot, new_plot):
        """Choose dataset to use for clustering.

        Choices:
        'MV' -> megavoxel frequencies,
        'Texture_Features' -> 4 haralick texture features,
        'Combined' -> both together
        """
        filter_data = grouping.currentText()
        image_feature_data = pd.read_csv(self.feature_file[0], sep='\t', na_values='        NaN')
        # Identify columns
        columns = image_feature_data.columns
        mv_cols = columns[columns.map(lambda col: col.startswith('MV'))]
        # all columns corresponding to megavoxel categories
        # should usually be -4 since contrast is still included here.
        texture_cols = columns[columns.map(lambda col: col.startswith('text_'))]
        featurecols = columns[columns.map(lambda col: col.startswith('MV') or col.startswith('text_'))]
        mdatacols = columns.drop(featurecols)
        # drop duplicate data rows:
        image_feature_data.drop_duplicates(subset=featurecols, inplace=True)

        # remove non-finite/ non-scalar valued rows in both
        image_feature_data = image_feature_data[np.isfinite(image_feature_data[featurecols]).all(1)]
        image_feature_data.sort_values(list(featurecols), axis=0, inplace=True)

        # min-max scale all data and split to feature and metadata
        mind = np.min(image_feature_data[featurecols], axis=0)
        maxd = np.max(image_feature_data[featurecols], axis=0)
        #check that mind, maxd doesn't return zero division
        if np.array_equal(mind, maxd) == False:
            featuredf = (image_feature_data[featurecols] - mind) / (maxd - mind)
            mdatadf = image_feature_data[mdatacols]
            #drop cols with nan
            featuredf.dropna(axis=1, inplace=True)
            mv_cols = featuredf.columns[featuredf.columns.map(lambda col: col.startswith('MV'))]
            # select data
            if len(self.filt) == 1:
                if self.filt[0] == 'MV':
                    X = featuredf[mv_cols].to_numpy().astype(np.float64)
                elif self.filt[0] == 'Texture_Features':
                    X = featuredf[texture_cols].to_numpy().astype(np.float64)
            elif self.filt == ['MV', 'Texture_Features']:
                X = featuredf.to_numpy().astype(np.float64)
            else:
                X = featuredf[mv_cols].to_numpy().astype(np.float64)
            self.filtered_data = X

            # reset imageIDs
            self.imageIDs.clear()
            self.imageIDs.extend(np.array(mdatadf['ImageID'], dtype='object').astype(int))
            # reset labels
            z = np.ones(X.shape[0]).astype(int)
            if filter_data != "No Grouping":
                z = np.array(mdatadf[filter_data], dtype='object')

            self.labels.clear()
            self.labels.extend(list(map(str, z)))
            # misc info
            numMVperImg = np.array(image_feature_data['NumMV']).astype(np.float64)
            num_images_kept = X.shape[0]
            result_plot(self, X, projection, plot, new_plot)
        else:
            errorWindow('Feature File Data Error',
                'Check if have more than 1 row of data and that min and max values of MV or texture columns are not the same')
    # end data_filt

    def setnumcluster(self, group):
        """Set the number of clusters in response to user input."""
        clustnum = Clustering.setcluster(
            self.numcluster, self.filtered_data, self.plot_data, np.array(self.labels), group)
        self.numcluster = clustnum.clust
    # end setnumcluster

    def classificationRF(self):
        """Open a platemap window for the user to select training classes, and pass to random forest model."""
        try:
            view = platemapWindow(dataframe = self.platemap, meta = self.feature_file[0], first = False)
            view.show()
            view.exec()
        except:
            print('Error in classificationRF')
            pass

        try:
            # first, open the tmp-selectedClassSample.tsv file and get the classes
            cwd = os.getcwd()
            mapOnlyDF = pd.read_csv(cwd + '/tmp-selectedClassSample.tsv', sep='\t')


            # find columns where all numbers are 0
            zeroCols = []
            for col in mapOnlyDF.columns:
                if mapOnlyDF[col].sum() == 0:
                    zeroCols.append(col)

            # drop columns where all numbers are 0
            mapOnlyDF = mapOnlyDF.drop(zeroCols, axis=1)

            # drop rows where the column "NumMV" is 0
            mapOnlyDF = mapOnlyDF[mapOnlyDF['NumMV'] != 0]

            # drop rows with any NaN values
            mapOnlyDF = mapOnlyDF.dropna()

            # features are columns that start with MV
            features = mapOnlyDF.columns[mapOnlyDF.columns.map(lambda col: col.startswith('MV'))]

            # standardize the features
            mapOnlyDF[features] = (mapOnlyDF[features] - mapOnlyDF[features].mean()) / mapOnlyDF[features].std()

            # also include the Type column
            dataTypes = pd.Index(['Type'])
            features = features.append(pd.Index(['Type']))

            # get names of columns that contain the string 'Dose'
            DoseCols = mapOnlyDF.columns[mapOnlyDF.columns.str.contains('Dose')]

            # append the dose column to the features
            features = features.append(DoseCols)

            # create a new dataframe with only the features
            mapOnlyDF = mapOnlyDF[features]

            mapOnlyDF.to_csv('tmp-check.tsv', sep='\t', index=False)

            # separate into two dataframes: one where the 'Type' contains 'SAMPLE' and one where it doesn't
            samplesDF = mapOnlyDF[mapOnlyDF['Type'].str.contains('SAMPLE')]
            classesDF = mapOnlyDF[~mapOnlyDF['Type'].str.contains('SAMPLE')]
            mvs = mapOnlyDF.columns[mapOnlyDF.columns.map(lambda col: col.startswith('MV'))]
            Xmv = classesDF[mvs]
            ytype = classesDF['Type']

            # export the samples dataframe
            samplesDF.to_csv('tmp-samples.tsv', sep='\t', index=False)

            # export the classes dataframe
            classesDF.to_csv('tmp-classes.tsv', sep='\t', index=False)

            # We will be training a Random Forest model to classify the samples into the classes. First, we need to
            # train the model using the classes and we will use the model to predict the classes of the samples.
            # We will use the megavoxel frequencies as features and the classes as labels.

            # First, determine the number of features to use in the model.
            # We will use cross-validation to determine the optimal number of features.
            cv_inner = KFold(n_splits=10, shuffle=True)
            initialRF = RandomForestClassifier(n_estimators=1000)

            # search space is...
            space = dict()
            space['max_features'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, len(mvs)]
            search = GridSearchCV(initialRF, space, scoring='accuracy', cv=cv_inner, refit=True)
            # fit the model
            result = search.fit(Xmv, ytype)
            # the best model is...
            RF = result.best_estimator_

            
            #RF.fit(classesDF[features[:-1]], classesDF['Type'])

            # predict the classes of the samples
            predictions = RF.predict(samplesDF[mvs])

            # add the Dose columns to a predictions dataframe
            predictionsDF = samplesDF[DoseCols]
            # rename the columns to 'Dose'
            predictionsDF.rename(columns={DoseCols[0]: 'Dose'}, inplace=True)

            # add the predictions to the predictions dataframe
            predictionsDF['Predicted Class'] = predictions

            # For each unique dose, get the percentage of samples in each class
            # and then put into a dataframe - make it such that for each dose (row),
            # there is a column for each class and the value is the percentage of
            # samples in that class
            dfOutput = pd.DataFrame()
            for dose in predictionsDF['Dose'].unique():
                # get the dataframe for each dose
                doseDF = predictionsDF[predictionsDF['Dose'] == dose]
                # get the percentage of samples in each class
                doseDF = doseDF['Predicted Class'].value_counts(normalize=True)
                # add the dose as a column
                doseDF = pd.DataFrame(doseDF).transpose()
                doseDF['Dose'] = dose
                # concat to the output dataframe
                dfOutput = pd.concat([dfOutput, doseDF])

            # set the index to the dose
            dfOutput.set_index('Dose', inplace=True)

            #export classification table
            name = QFileDialog.getSaveFileName(None, 'Save File', filter=".tsv")
            if name[0]!= '':
                dfOutput.to_csv("".join(name), sep='\t', mode='w')

            # finally, delete the tmp-selectedClassSample.tsv file
            os.remove(cwd + '/tmp-selectedClassSample.tsv')
        except:
            # throw an error if the user has not selected a feature file
            errorWindow("Error Dialog", "Please select Feature file and load platemap in the Main window. No data is currently displayed.")
            pass
    # end classificationRF
        
    def plotDoseResponse(self):
        '''Plot a dose-response curve from classification results saved in a .tsv file'''
        # open the file
        filename, dump = QFileDialog.getOpenFileName(
            self, 'Open Classification Results File', '', 'Text files (*.tsv)')
        if filename != '':
            # read the file
            df = pd.read_csv(filename, sep='\t').set_index('Dose')

            # get the doses
            doses = df.index.to_numpy()
            print(doses)

            # get the classes
            classes = df.columns.to_numpy()

            # ask the user to select a class
            class_, ok = QInputDialog.getItem(self, "Class Selection", "Select a class to plot:", classes, 0, False)

            # now, fit a 4-parameter logistic function to the data
            # first, get the class data
            classData = df[class_].to_numpy()

            # fit the function to the data
            popt = self.fit_ll4(doses, classData)

            # plot the function
            if ok:
                fig, ax = plt.subplots()
                ax.scatter(doses, classData)
                # use np.linspace to get a smooth curve
                x = np.linspace(doses[0], doses[-1], 100)
                ax.plot(x, self.ll4(x, *popt), 'r-')
                # x-axis is log scale
                ax.set_xscale('log')
                ax.set_xlabel('Dose')
                # write the x-ticks as numbers, not in scientific notation
                # also, include marking the x-ticks at half of each log interval
                ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(0.25,0.5,1.0)))
                ax.set_ylabel('Percentage of Samples in Class')
                # y axis is percentage - multiply by 100
                ax.set_ylim([0, 1.05])
                ax.set_yticks(np.arange(0, 1.1, 0.1))
                ax.set_yticklabels(np.arange(0, 110, 10))
                ax.set_title('Dose-Response Curve for ' + class_ + '\n EC50 = ' + str(np.round(popt[2], 2)))
                plt.show()
        else:
            pass
    # end plotDoseResponse
        
    def fit_ll4(self, x, y):
        """Fit a 4-parameter logistic function to the data."""
        # define the function
        def ll4(x, A, B, C, D):
            return ((A - D) / (1 + ((x / C) ** B))) + D

        # fit the function to the data
        try:
            popt, pcov = curve_fit(ll4, x, y, maxfev=100000)
        except:
            print('Error in fit_ll4')
            pass

        # return the parameters
        return popt
    # end fit_ll4

    def ll4(self, x, A, B, C, D):
        """4-parameter logistic function."""
        return ((A - D) / (1 + ((x / C) ** B))) + D
    # end ll4
        
# end resultsWindow
