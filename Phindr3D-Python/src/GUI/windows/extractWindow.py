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

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

try:
    from .regexWindow import regexWindow
    from ...Data import *
except ImportError:
    from src.GUI.windows.regexWindow import *
    from src.Data import *

class extractWindow(QDialog):
    """Build window to take input from user to create a metadata file."""
    def __init__(self):
        """Construct the window, its widgets, and button actions."""
        super(extractWindow, self).__init__()
        largetext = QFont("Arial", 12, 1)
        self.setWindowTitle("Create Metadatafile")
        directory = "Image Directory"
        self.samplefilename = "Sample File Name"
        layout = QGridLayout()

        selectimage = QPushButton("Select Image Directory")
        selectimage.setFixedSize(selectimage.minimumSizeHint())
        selectimage.setFixedHeight(40)

        imagerootbox = QTextEdit()
        imagerootbox.setReadOnly(True)
        imagerootbox.setPlaceholderText(directory)
        imagerootbox.setFixedSize(450, 50)
        imagerootbox.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        imagerootbox.setFont(largetext)

        samplelabel = QLabel()
        samplelabel.setText("File in the Selected Directory")

        samplefilebox = QTextEdit()
        samplefilebox.setReadOnly(True)
        samplefilebox.setPlaceholderText(self.samplefilename)
        samplefilebox.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        samplefilebox.setFont(largetext)
        samplefilebox.setFixedSize(450, 30)

        instructionslabel = QLabel()
        instructionslabel.setFixedHeight(40)
        instructionslabel.setWordWrap(True)
        instructionslabel.setText("Identify key values in the file name, "
            "either by clicking Build Regular Expression or by "
            "manually entering a regular expression. Check your work by clicking "
            "Evaluate Regular Expression.")

        expressionbox = QLineEdit()
        expressionbox.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        expressionbox.setFont(largetext)
        expressionbox.setFixedSize(450, 30)
        expressionbox.setPlaceholderText("Enter Regular Expression Here")

        evaluateexpression = QPushButton("Evaluate Regular Expression")
        evaluateexpression.setFixedSize(evaluateexpression.minimumSizeHint())
        evaluateexpression.setFixedHeight(30)

        createregex = QPushButton("Build Regular Expression")
        createregex.setFixedSize(evaluateexpression.minimumSizeHint())
        createregex.setFixedHeight(30)

        outlabel = QLabel()
        outlabel.setText("Enter output file name")

        outputfilebox = QLineEdit()
        outputfilebox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outputfilebox.setFont(largetext)
        outputfilebox.setPlaceholderText("Output Metadata File Name")
        outputfilebox.setFixedSize(450, 30)

        createfile = QPushButton("Create Metadatafile")
        createfile.setFixedSize(createfile.minimumSizeHint())
        createfile.setFixedHeight(40)

        cancel = QPushButton("Cancel")
        cancel.setFixedSize(cancel.minimumSizeHint())
        cancel.setFixedHeight(40)

        # button functions
        def selectImageDir():
            """Act on user clicking Select Image button."""
            imagedir = QFileDialog.getExistingDirectory()
            if not os.path.exists(imagedir):
                return
            imagerootbox.setText(imagedir)
            # select first '.tif' or '.tiff' file to be sample file
            for file in os.listdir(imagedir):
                if file.endswith('.tiff') or file.endswith('.tif'):
                    samplefilebox.setText(file)
                    break
        # end selectImageDir

        def createFile():
            """Act on user clicking Create File button."""
            imagedir = imagerootbox.toPlainText()
            regex = expressionbox.text()
            outputname = outputfilebox.text()
            # replace '?<' patterns with '?P<' to make compatible with re.fullmatch function
            # first checks if '?<' corresponds to a '?<=' or '?<!' pattern first before replacing
            # part of Python specific regular expression syntax
            regex = DataFunctions.regexPatternCompatibility(regex)
            try:
                alert = QMessageBox()
                try:
                    if outputname != "":
                        if not outputname.endswith(".tsv"):
                            outputname = outputname + ".tsv"
                        created = DataFunctions.createMetadata(imagedir, regex, outputname)
                    else:
                        created = DataFunctions.createMetadata(imagedir, regex)
                    if created:
                        alert.setText("Metadatafile creation success.")
                        alert.setIcon(QMessageBox.Icon.Information)
                        alert.setWindowTitle("Notice")
                        self.close()
                    else:
                        alert.setText("Error: No Regex matches found in selected folder.")
                        alert.setIcon(QMessageBox.Icon.Critical)
                except MissingChannelStackError:
                    alert.setText("Error: No Channel and/or Stack groups found in regex.")
                    alert.setIcon(QMessageBox.Icon.Critical)
                alert.show()
                alert.exec()
            except WindowsError:
                alert = QMessageBox()
                alert.setWindowTitle("Error")
                alert.setText("No such image directory exists.")
                alert.setIcon(QMessageBox.Icon.Critical)
                alert.show()
                alert.exec()
        # end createFile

        def evalRegex():
            """Act on user clicking Evaluate Regular Expression button."""
            regex = expressionbox.text()
            samplefile = samplefilebox.toPlainText()
            if regex == "":
                alert = QMessageBox()
                alert.setWindowTitle("Error")
                alert.setText("Please enter a regular expression to evaluate")
                alert.setIcon(QMessageBox.Icon.Critical)
                alert.show()
                alert.exec()
                return
            if samplefile == "":
                alert = QMessageBox()
                alert.setWindowTitle("Error")
                alert.setText("No sample file was found. Please check the selected image directory.")
                alert.setIcon(QMessageBox.Icon.Critical)
                alert.show()
                alert.exec()
                return
            # replace '?<' patterns with '?P<' to make compatible with re.fullmatch function
            # first checks if '?<' corresponds to a '?<=' or '?<!' pattern first before replacing
            # part of Python specific regular expression syntax
            regex = DataFunctions.regexPatternCompatibility(regex)
            # parse the sample file with the regular expression to find field values
            reout = DataFunctions.parseAndCompareRegex(samplefile, regex)
            # Create the GUI that displays the output
            winex = QDialog()
            winex.setWindowTitle("Evaluate Regular Expression")
            winlayout = QGridLayout()
            labelText = "Regular Expression Match"
            # List of regex keys and values
            relist = QListWidget()
            if len(reout) == 0:
                relist.addItem("No results")
            else:
                for rekey in reout.keys():
                    nextline = str(rekey)+" ::: "+str(reout[rekey])
                    relist.addItem(nextline)
            # Ok button closes the window
            reok = QPushButton("Ok")
            # button behaviour
            def okPressed():
                winex.close()
            reok.clicked.connect(okPressed)

            # add the widgets to the layout
            winlayout.addWidget(QLabel(labelText))
            winlayout.addWidget(relist)
            winlayout.addWidget(reok)
            # add the layout and show the window
            winex.setLayout(winlayout)
            winex.show()
            winex.exec()
        #end evalRegex

        def regexCreation():
            """Open regex creation window in response to button click."""
            creationWindow = regexWindow()
            creationWindow.samplefile = samplefilebox.toPlainText()
            creationWindow.inputSampleFile()
            creationWindow.show()
            result = creationWindow.exec()
            if result:
                expressionbox.setText(creationWindow.regex)
                creationWindow.close()
        # end regexCreation

        cancel.clicked.connect(self.close)
        selectimage.clicked.connect(selectImageDir)
        createfile.clicked.connect(createFile)
        createregex.clicked.connect(regexCreation)
        evaluateexpression.clicked.connect(evalRegex)

        layout.addWidget(selectimage, 0, 0, 1, 1)
        layout.addWidget(imagerootbox, 0, 1, 1, 3)
        layout.addWidget(samplelabel, 1, 0, 1, 1)
        layout.addWidget(samplefilebox, 1, 1, 1, 3)
        layout.addWidget(instructionslabel, 2, 0, 1, 4)
        layout.addWidget(createregex, 3, 0, 1, 1)
        layout.addWidget(expressionbox, 3, 1, 1, 3)
        layout.addWidget(evaluateexpression, 4, 0, 1, 1)
        layout.addWidget(outlabel, 5, 0, 1, 1)
        layout.addWidget(outputfilebox, 5, 1, 1, 3)
        layout.addWidget(createfile, 6, 0, 1, 1)
        layout.addWidget(cancel, 6, 1, 1, 1)

        layout.setSpacing(10)
        self.setLayout(layout)
        self.setFixedSize(self.minimumSizeHint())
    # end __init__
# end extractWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = extractWindow()
    window.show()
    app.exec()
# end main