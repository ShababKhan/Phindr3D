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

from .windows import *
from .windows import platemapWindow

class external_windows():
    """Abstract class inherited by MainGUI with build methods for other windows."""
    def buildExtractWindow(self):
        """Create an instance of class extractWindow."""
        return extractWindow()

    def buildResultsWindow(self, color, metadata = None, platemap = None):
        """Create an instance of class resultsWindow."""
        return resultsWindow(color, metadata, platemap)

    def buildParamWindow(
            self, metaheader, supercoords, svcategories, megacoords,
            mvcategories, voxelnum, trainingnum, bg, norm, conditiontrain,
            trainingcol, treatcol):
        """Create an instance of class paramWindow."""
        return paramWindow(
            metaheader, supercoords, svcategories, megacoords,
            mvcategories, voxelnum, trainingnum, bg, norm, conditiontrain,
            trainingcol, treatcol)

    def buildSegmentationWindow(self, metadata):
        """Create an instance of class segmentationWindow."""
        return segmentationWindow(metadata)

    def buildColorchannelWindow(self):
        """Create an instance of class colorchannelWindow."""
        return colorchannelWindow()

    def buildRegexWindow(self):
        """Create an instance of class regexWindow."""
        return regexWindow()

