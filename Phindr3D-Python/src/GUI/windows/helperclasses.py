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

"""These classes are not windows, but help build other features in windows."""

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

class MplCanvas(FigureCanvasQTAgg):
    """matplotlib figure-plot creation"""
    def __init__(self, parent=None, width=5, height=5, dpi=100, projection="3d"):
        """Build canvas widget derived from matplotlib.FigureCanvasQTAgg."""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if projection=="3d":
            self.axes = self.fig.add_subplot(111, projection=projection)
        else:
            self.axes = self.fig.add_subplot(111, projection=None)
        super(MplCanvas, self).__init__(self.fig)
        self.cmap = None
    
    def setNearFull(self):
        """Set the size of the axes within the canvas."""
        self.axes.set_aspect('auto')
        self.axes.set_position([0.01, 0.01, 0.98, 0.98])
# end MplCanvas

class NavigationToolbar(NavigationToolbar):
    """Create a custom toolbar derived from the base NavigationToolbar class."""
    NavigationToolbar.toolitems = (
        (None, None, None, None),
        (None, None, None, None),
        (None, None, None, None),
        (None, None, None, None),
        (None, None, None, None),
        ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        (None, None, None, None),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure')
    )
# end NavigationToolbar

class selectWindow(object):
    """Create a window for the user to select classes using checkboxes."""
    def __init__(
            self, chk_lbl, col_lbl, win_title, grp_title, col_title,
            groupings, filtby='False', filterlist=[],
            filt_title="Filter By Feature(s)"):
        """Construct select window GUI."""
        win = QDialog()
        win.setWindowTitle(win_title)
        win.setLayout(QGridLayout())
        self.x_press=True
        ok_button = QPushButton("OK")
        # setup checkbox for groups
        if len(chk_lbl) > 0:
            lbl_box=self.checkbox(grp_title, chk_lbl, win, [0, 0], True)
            if 'False' not in filtby:
                filt_box=self.checkbox(filt_title, filtby, win, [0, 1], False)
                if len(filtby)==1:
                    filt_box.grp_list[0].setChecked(True)
                ok_button.clicked.connect(
                    lambda: self.selected(lbl_box.grp_checkbox,
                        filt_box.grp_checkbox, win, groupings, filterlist))
            else:
                ok_button.clicked.connect(
                    lambda: self.selected(lbl_box.grp_checkbox,
                        filtby, win, groupings, filterlist))
        # setup Column box
        if len(col_lbl) > 0:
            ch_title = QLabel(col_title)
            ch_title.setFont(QFont('Arial', 10))
            ch_box = QGroupBox()
            ch_box.setFlat(True)
            ch_vbox = QVBoxLayout()
            ch_vbox.addWidget(ch_title)
            # add columns to layout
            for lbl in col_lbl:
                ch_label = QLabel(lbl)
                ch_vbox.addWidget(ch_label)
            ch_vbox.addStretch(1)
            ch_box.setLayout(ch_vbox)
            win.layout().addWidget(ch_box, 0, 2)
            #Scroll if labels exceed window size
            scrollArea = QScrollArea()
            scrollArea.setWidget(ch_box)
            scrollArea.setWidgetResizable(True)
            win.layout().addWidget(scrollArea, 0, 2)
            win.layout().addWidget(ok_button, 1, 2)
            if len(chk_lbl) == 0:
                ok_button.clicked.connect(lambda: win.close())
        else:
            win.layout().addWidget(ok_button, 2, 0)
        #size window to fit all elements
        minsize = win.minimumSizeHint()
        minsize.setHeight(win.minimumSizeHint().height() + 100)
        minsize.setWidth(win.minimumSizeHint().width() + 100)
        win.setFixedSize(minsize)
        win.show()
        win.setWindowFlags(
            win.windowFlags() | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        win.exec()
    # end constructor

    class checkbox(object):
        """Define a box consisting of a list of text options with checkboxes."""
        def __init__(self, grp_title, chk_lbl, win, chk_loc, all_btn):
            """Construct the textbox widget."""
            grp_title = QLabel(grp_title)
            grp_title.setFont(QFont('Arial', 10))
            self.grp_checkbox = QGroupBox()
            self.grp_checkbox.setFlat(True)
            self.grp_list = []
            grp_vbox = QVBoxLayout()
            grp_vbox.addWidget(grp_title)

            # add checkboxes to layout
            for lbl in chk_lbl:
                self.grp_list.append(QCheckBox(lbl))
                grp_vbox.addWidget(self.grp_list[-1])
            grp_vbox.addStretch(1)
            self.grp_checkbox.setLayout(grp_vbox)
            win.layout().addWidget(self.grp_checkbox, chk_loc[0], chk_loc[1])
            #scroll if labels exceed window size
            scrollArea = QScrollArea()
            scrollArea.setWidget(self.grp_checkbox)
            scrollArea.setWidgetResizable(True)
            win.layout().addWidget(scrollArea, chk_loc[0], chk_loc[1])
            if all_btn:
                # select all button
                all_button = QPushButton("Select All")
                win.layout().addWidget(all_button, 1, 0)
                all_button.clicked.connect(
                    lambda: [box.setChecked(True) for box in self.grp_list])
        # end constructor
    # end checkbox

    def selected(self, grp_checkbox, filt_checkbox, win, groupings, filtlist):
        """Compose lists of the checked options in the select window."""
        close=True
        if ('False' != filt_checkbox):
            for checkbox in filt_checkbox.findChildren(QCheckBox):
                if checkbox.isChecked():
                    filtlist.append(checkbox.text())
            if len(filtlist)==0:
                close=False
                errorWindow("Filter by Feature Error", "Must Select At Least One Feature")
        if close:
            for checkbox in grp_checkbox.findChildren(QCheckBox):
                if checkbox.isChecked():
                    groupings.append(checkbox.text())
        self.x_press=False
        if close:
            win.close()
    # end selected
# end selectWindow

class errorWindow(object):
    """Show an error window with a specified title and error text."""
    def __init__(self, win_title, text):
        """Construct and show the error window with the given title and error text."""
        alert = QMessageBox()
        alert.setWindowTitle(win_title)
        alert.setText(text)
        alert.setIcon(QMessageBox.Icon.Critical)
        alert.show()
        alert.setWindowFlags(
            alert.windowFlags() | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        alert.exec()
    # end constructor
# end errorWindow