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
import json
import numpy as np
import tifffile as tf
from scipy import ndimage
import time 

try:
    from .SegmentationFunctions import *
except ImportError:
    from SegmentationFunctions import *

class Segmentation:
    """Class associated with segmenting organoids from a 3D image."""

    def __init__(self):
        """Segmentation class constructor"""
        self.defaultSettings = {
            'min_area_spheroid':200.0,
            'intensity_threshold':1000.0,
            'radius_spheroid':75.0,
            'smoothin_param':0.01,
            'scale_spheroid':1.0,
            'entropy_threshold':1.0,
            'max_img_fraction':0.25,
            'seg_Channel':'All Channels',
            'remove_border_objects':True
            }
        self.settings = self.defaultSettings
        self.metadata = None
        self.outputDir = None
        self.labDir = None
        self.segDir = None
        self.segmentationSuccess = False
        self.focusIms = {} #key = image ID, val = imagepath
        self.labelIms = {} #same same.
        self.allIDs = [] 
        self.IDidx = None
    # end constructor
    
    def saveSettings(self, outputpath):
        """Save the settings to a json file for later use."""
        with open(outputpath, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=4)
    # end saveSettings

    def loadSettings(self, settingJsonPath):
        """Restore saved settings from a json file."""
        with open(settingJsonPath, 'r') as f:
            newsettings = json.load(f)
        self.settings = newsettings
    # end loadSettings
    
    def createSubfolders(self):
        """Create folders for labelled and segmented image outputs."""
        self.labDir = os.path.join(self.outputDir, 'LabelledImages')
        self.segDir = os.path.join(self.outputDir, 'SegmentedImages')
        os.makedirs(self.labDir, exist_ok=True)
        os.makedirs(self.segDir, exist_ok=True)
    # end createSubfolders

    def getCurrentIMs(self):
        """Open, read, and return image objects for the current image."""
        if self.allIDs == []:
            return None, None
        else:
            return tf.imread(self.focusIms[self.allIDs[self.IDidx]]), tf.imread(self.labelIms[self.allIDs[self.IDidx]])
    # end getCurrentIMs

    def getNextIMs(self):
        """Open, read, and return image objects for the next image."""
        if self.allIDs == []:
            return None, None
        else:
            if self.IDidx == len(self.allIDs)-1:
                self.IDidx = 0
            else:
                self.IDidx += 1
            return tf.imread(self.focusIms[self.allIDs[self.IDidx]]), tf.imread(self.labelIms[self.allIDs[self.IDidx]])
    # end getNextIMs

    def getPrevIMs(self):
        """Open, read, and return image objects for the previous image."""
        if self.allIDs == []:
            return None, None
        else:
            if self.IDidx == 0:
                self.IDidx = len(self.allIDs)-1
            else:
                self.IDidx -= 1
            return tf.imread(self.focusIms[self.allIDs[self.IDidx]]), tf.imread(self.labelIms[self.allIDs[self.IDidx]])
    # end getPrevIMs

    def RunSegmentation(self, mdata):
        """
        Segment a 3D image stack into component organoids.
        
        It iterates over each image in the provided metadata.
        
        Depending on the segmentation settings, it either processes all channels 
        of the image or a specific channel.
        
        It then obtains a binary image of the segmented objects using the 
        getSegmentedOverlayImage function from SegmentationFunctions.py.
        
        It identifies unique labels (representing different objects) in the binary 
        image and counts the number of objects.
        
        For each object, it checks if the object's size exceeds a maximum fraction 
        of the image size specified in the settings. If it does, the object is removed 
        from the binary image.
        
        Otherwise, it appends the focus planes per object to a list using the 
        getFocusplanesPerObjectMod function from SegmentationFunctions.py.
        
        It then retrieves the keys of the stack layers and channels from the image 
        stack.
        
        For each object, it reads the image from the corresponding channel and plane, 
        segments the object, and writes the segmented image to a file in the segmentation 
        directory.
        
        It also writes the labelled image to a file in the labelled images directory.
        
        Finally, it writes all objects and the focus image to separate files in the 
        labelled images directory.
        
        If the function encounters any errors during execution, it sets the 
        segmentationSuccess attribute to False and clears the allIDs and IDidx attributes. 
        If the function completes successfully, it sets segmentationSuccess to True.
        """

        def write_file(dir, filenameParts, suffix, data):
            filenameParts.append(suffix)
            filename = os.path.join(dir, ('__'.join(filenameParts) + '.tiff'))
            tf.imwrite(filename, data)
            return filename

        try: 
            for id in mdata.images: # for each well:
                print(f'Segmenting image of well {id}')
                start = time.time()
                imstack = mdata.images[id]
                if self.settings['seg_Channel'] == 'All Channels':
                    IM, focusIndex = getfsimage_multichannel(imstack)
                else:
                    chanIndx = int(self.settings['seg_Channel'])
                    IM, focusIndex = getfsimage_scharr(imstack, chanIndx)
                print(f'getfsimage time: {time.time()-start}') 

                L = getSegmentedOverlayImage(IM, self.settings) 
                uLabels = np.unique(L)
                uLabels = uLabels[uLabels != 0] 
                ll = [getFocusplanesPerObjectMod(L == label, focusIndex) for label in uLabels if np.sum(L == label) <= (L.size * self.settings['max_img_fraction'])]
                ll = np.array(ll)

                zVals = list(imstack.stackLayers.keys())
                channels = list(imstack.stackLayers[zVals[0]].channels.keys())
                otherparams = imstack.GetOtherParams()
                filenameParts = [f'{param[0]}{otherparams[param]}' for param in otherparams]

                if len(ll) > 0:
                    print(f'Number of objects: {len(ll)}')
                    L = cv.dilate(L, morph.disk(25))
                    fstruct = ndimage.find_objects(L.astype(int))
                    for iObjects in range(len(ll)):
                        for iPlanes in range(int(ll[iObjects, 0]), int(ll[iObjects, 1]+1)):
                            for chan in channels:
                                IM1 = tf.imread( imstack.stackLayers[iPlanes].channels[chan].channelpath )
                                IM2 = IM1[fstruct[iObjects]]
                                tmpparts = filenameParts + [f'Z{iPlanes}', f'CH{chan}', f'ID{id}', f'OB{iObjects+1}']
                                write_file(self.segDir, tmpparts, '.tiff', IM2)
                        tmpparts = filenameParts + [f'ID{id}', f'OB{iObjects+1}']
                        IML = L[fstruct[iObjects]]
                        write_file(self.labDir, tmpparts, '.tiff', IML)
                allobsname = filenameParts.copy()
                focusname = filenameParts.copy()
                allobsname.append(f'ID{id}')
                focusname.append(f'ID{id}')
                allobsname.append(f'All_{len(ll)}_Objects')
                focusname.append('FocusIM')
                IML = L
                objFileName = write_file(self.labDir, allobsname, '.tiff', IML)
                focFileName = write_file(self.labDir, focusname, '.tiff', IM)
                self.focusIms[id] = focFileName
                self.labelIms[id] = objFileName
            self.allIDs = list(self.focusIms.keys())
            self.IDidx = 0
            self.segmentationSuccess = True
            print(f'Segmentation time in total for this well: {time.time()-start}')
        except:
            self.allIDs = []
            self.IDidx = None
            self.segmentationSuccess = False
    # End RunSegmentation
# end class Segmentation

if __name__ == '__main__':
    """Tests of the Segmentation class that can be run directly."""
    from Data import Metadata, Generator
    deterministic = Generator(1234)
    mdata = Metadata(deterministic)
    segtest = Segmentation()

    segtest.outputDir = 'testdata\\segmentation_tests\\check_results'
    mdatafile = 'testdata\\segmentation_tests\\segtestmdata.txt'

    mdata.loadMetadataFile(mdatafile)
    print(f'Loading segmentation test images metadata success? ... {mdata.metadataLoadSuccess}')

    segtest.createSubfolders()
    segtest.RunSegmentation(mdata)
    print(f'Segmentation success? ... {segtest.segmentationSuccess}')
    for c, e in zip(os.listdir(segtest.labDir), os.listdir('testdata\\segmentation_tests\\expect_results\\LabelledImages')):
        check = os.path.abspath(segtest.labDir + '\\' + c)
        expect = os.path.abspath('testdata\\segmentation_tests\\expect_results\\LabelledImages\\' + e)
        same = (tf.imread(check) == tf.imread(expect)).all()
        if not same:
            break 
    print(f'Expected label image results? ... {same}')
    for c, e in zip(os.listdir(segtest.segDir), os.listdir('testdata\\segmentation_tests\\expect_results\\SegmentedImages')):
        check = os.path.abspath(segtest.segDir + '\\' + c)
        expect = os.path.abspath('testdata\\segmentation_tests\\expect_results\\SegmentedImages\\' + e)
        same = (tf.imread(check) == tf.imread(expect)).all()
        if not same:
            break
    print(f'Expected segmented image results? ... {same}')
    for f in os.listdir(segtest.segDir):
        os.remove(os.path.abspath(segtest.segDir + '\\' + f))
    for f in os.listdir(segtest.labDir):
        os.remove(os.path.abspath(segtest.labDir + '\\' + f))
    os.rmdir(segtest.segDir)
    os.rmdir(segtest.labDir)
# end main

