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

from os import path
import cProfile
import pstats

from src import *
import threading

class Phindr3D(threading.Thread):  # this will allow us to run the GUI in a separate thread so that we can implement a progress bar.
    def __init__(self, iconFile):
        self.iconFile = iconFile
        threading.Thread.__init__(self)

    def run_mainGUI(self):
        """Create an instance of MainGUI and run the application."""
        app = QApplication(sys.argv)
        window = MainGUI(self.iconFile)
        window.show()
        app.exec()

    def run_with_profiling(self):
        """Run the main GUI with profiling."""
        with cProfile.Profile() as pr:
            self.run_mainGUI()

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        # Now you have two options, either print the data or save it as a file
        stats.print_stats()  # Print The Stats
        stats.dump_stats("phindrstats7.prof")
        pstats.Stats("phindrstats7.prof").print_stats()  # Saves the data in a file, can be used to see the data visually

# end class Phindr3D

if __name__ == '__main__':
    """Phindr3D is designed for automated cell phenotype analysis."""
    iconFile = path.abspath(
        path.join(path.dirname(__file__), 'phindr3d_icon.png'))

    phindr3d = Phindr3D(iconFile)

    phindr3d.run_mainGUI()                # <= this is the normal way to run the GUI.
    #phindr3d.run_with_profiling()

# end main
