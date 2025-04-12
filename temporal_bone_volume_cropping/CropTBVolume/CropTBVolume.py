# Standard library imports
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union, Annotated

# Third-party imports
import numpy as np
import vtk
from vtk.util import numpy_support as nps

# Slicer imports
import slicer
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLNode
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
    Choice
)

# Qt imports
from qt import QPushButton, QTimer

RENAMED_EVENT = vtk.vtkCommand.UserEvent + 1  # Typically vtkCommand.UserEvent + 1 is used for renamed events

#
# CropTBVolumeParameterNode
#

@parameterNodeWrapper
class CropTBVolumeParameterNode:
    """
    Parameters for volume cropping:
    - inputVolume: Input volume node to crop
    - outputVolume: Output cropped volume node
    - roiNode: ROI node defining crop region
    - fillValue: Value for voxels outside input volume
    """
    inputVolume: slicer.vtkMRMLScalarVolumeNode
    outputVolume: slicer.vtkMRMLScalarVolumeNode
    roiNode: slicer.vtkMRMLMarkupsROINode
    fillValue: float = 0.0

#
# CropTBVolumeWidget
#

class CropTBVolumeWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class"""

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self.roiObservers = []  # Initialize observers list here
        self.ui = None  # Initialize ui here
        self._roiUpdateTimer = QTimer()
        self._roiUpdateTimer.setSingleShot(True)
        self._roiUpdateTimer.timeout.connect(self.updateROISizeWidget)
        self.inputVolumeObserverTag = None
        self.outputVolumeObserverTag = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        
        # Load UI file
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/CropTBVolume.ui'))
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Initialize logic after UI is loaded
        self.logic = CropTBVolumeLogic()
        if not self.logic:
            raise ValueError("Logic initialization failed")
        
        # Initialize parameter node
        self._parameterNode = None
        self.setParameterNode(self.logic.wrappedParameterNode)
        
        # Configure widgets
        if hasattr(self.ui, 'fillValueSpinBox'):
            self.ui.fillValueSpinBox.decimals = 2
            self.ui.fillValueSpinBox.minimum = -10000.0
            self.ui.fillValueSpinBox.maximum = 10000.0
        
        # Add ROI visibility toggle button
        if hasattr(self.ui, 'roiVisibilityButton'):
            self.ui.roiVisibilityButton.setCheckable(True)
            self.ui.roiVisibilityButton.toggled.connect(self.onROIVisibilityToggled)
        
        # Set MRML Scene for selectors
        selectors = [self.ui.inputSelector, self.ui.outputSelector, self.ui.roiSelector]
        for selector in selectors:
            selector.setMRMLScene(slicer.mrmlScene)
            selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNode)
    
        self.ui.roiSelector.connect("nodeAddedByUser(vtkMRMLNode*)", self.onROISelectorNodeAdded)  
        
        self.ui.sizeXSpinBox.connect('valueChanged(double)', self.onROISizeChanged)
        self.ui.sizeYSpinBox.connect('valueChanged(double)', self.onROISizeChanged)
        self.ui.sizeZSpinBox.connect('valueChanged(double)', self.onROISizeChanged)
        
        self.ui.fitToVolumeButton.connect('clicked()', self.onFitToVolume)
        self.ui.applyButton.connect('clicked(bool)', self.onApply)
            
        # Add observers for ROI size changes
        self.updateROISizeWidget()
        
        # Connect fitToVolume signal to update display
        self.ui.fitToVolumeButton.clicked.connect(self.updateROISizeWidget)
        
        # Disable ROI and output selectors initially
        self.ui.roiSelector.setEnabled(False)
        self.ui.outputSelector.setEnabled(False)
        
        # Modify the selector connections to include info updates
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda: self.updateParameterNodeAndInfo())
        # Modify output selector connection
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputVolumeChanged)
        
        self.updateVolumeInfo()  # This is the proper place to call it
        
    def updateParameterNodeAndInfo(self):
        """Update parameter node and only input volume information"""
        self.updateParameterNode()

        # Handle input volume info
        if self._parameterNode:
            if self._parameterNode.inputVolume is not None:  # Explicit None check
                try:
                    inputSpacing = self._parameterNode.inputVolume.GetSpacing()
                    inputImageData = self._parameterNode.inputVolume.GetImageData()
                    if inputImageData:
                        inputDims = inputImageData.GetDimensions()
                        self.ui.inputInfoLabel.setText(
                            f"Input: {inputDims[0]}x{inputDims[1]}x{inputDims[2]} "
                            f"({inputSpacing[0]:.2f}x{inputSpacing[1]:.2f}x{inputSpacing[2]:.2f} mm)")
                    else:
                        self.ui.inputInfoLabel.setText("Input: (no image data)")
                except Exception as e:
                    logging.error(f"Error updating input info: {str(e)}")
                    self.ui.inputInfoLabel.setText("Input: (error)")
            else:
                # Explicitly handle None case
                self.ui.inputInfoLabel.setText("Input: (none)")
        else:
            self.ui.inputInfoLabel.setText("Input: ")
        
        # Handle output volume info
        if self._parameterNode and self._parameterNode.outputVolume is not None:
            try:
                outputSpacing = self._parameterNode.outputVolume.GetSpacing()
                outputImageData = self._parameterNode.outputVolume.GetImageData()
                if outputImageData:
                    outputDims = outputImageData.GetDimensions()
                    self.ui.outputInfoLabel.setText(
                        f"Output: {outputDims[0]}x{outputDims[1]}x{outputDims[2]} "
                        f"({outputSpacing[0]:.2f}x{outputSpacing[1]:.2f}x{outputSpacing[2]:.2f} mm)")
                else:
                    self.ui.outputInfoLabel.setText("Output: (no image data)")
            except Exception as e:
                logging.error(f"Error updating output info: {str(e)}")
                self.ui.outputInfoLabel.setText("Output: (error)")
        else:
            self.ui.outputInfoLabel.setText("Output: ")
    
    def onNodeAdded(self, caller, event, callData):
        node = callData
        if isinstance(node, slicer.vtkMRMLMarkupsROINode):
            logging.debug(f"New ROI added: {node.GetName()}")
            self.onROISelectorNodeAdded(node)
    
    def updateParameterNode(self):
        """Update parameter node with additional validation"""
        if not hasattr(self, '_parameterNode') or not self._parameterNode:
            return
            
        inputNode = self.ui.inputSelector.currentNode()
        
        # Store current state before changes
        hadInputVolume = self._parameterNode.inputVolume is not None
        hadOutputVolume = self._parameterNode.outputVolume is not None
        
        # Update parameter node based on UI
        if inputNode:
            self.ui.roiSelector.setEnabled(True)
            self.ui.outputSelector.setEnabled(True)
            
            # Update input volume reference
            self._parameterNode.inputVolume = inputNode
            
            # Update ROI reference if selector has a node
            if self.ui.roiSelector.currentNode():
                self._parameterNode.roiNode = self.ui.roiSelector.currentNode()
                
            # Update output volume reference if selector has a node
            if self.ui.outputSelector.currentNode():
                self._parameterNode.outputVolume = self.ui.outputSelector.currentNode()
                
            # If we had an output volume but input was cleared, remove it
            if not inputNode and hadOutputVolume:
                slicer.mrmlScene.RemoveNode(self._parameterNode.outputVolume)
                self._parameterNode.outputVolume = None
        else:
            self.ui.roiSelector.setEnabled(False)
            self.ui.outputSelector.setEnabled(False)
            
            # Clear references when no input is selected
            if hadInputVolume:
                self._parameterNode.inputVolume = None
            if hadOutputVolume:
                if self._parameterNode.outputVolume:
                    slicer.mrmlScene.RemoveNode(self._parameterNode.outputVolume)
                self._parameterNode.outputVolume = None
            self._parameterNode.roiNode = None
    
        # Force UI update
        self.updateVolumeInfo()
    
    def cleanup(self) -> None:
        # Clean up rename observers
        if hasattr(self, '_parameterNode') and self._parameterNode:
            if hasattr(self, 'inputVolumeObserverTag') and self.inputVolumeObserverTag:
                try:
                    self._parameterNode.inputVolume.RemoveObserver(self.inputVolumeObserverTag)
                except:
                    pass
                self.inputVolumeObserverTag = None
            if hasattr(self, 'outputVolumeObserverTag') and self.outputVolumeObserverTag:
                try:
                    self._parameterNode.outputVolume.RemoveObserver(self.outputVolumeObserverTag)
                except:
                    pass
                self.outputVolumeObserverTag = None
            
        # Clean up timer
        if hasattr(self, '_roiUpdateTimer'):
            self._roiUpdateTimer.stop()
            self._roiUpdateTimer.timeout.disconnect()
            
        # Remove ROI observers properly
        for observer in self.roiObservers:
            if isinstance(observer, tuple):
                caller, tag = observer
                caller.RemoveObserver(tag)
        self.roiObservers = []
        
        # Remove other observers and disconnect signals
        if hasattr(self, 'ui') and self.ui:
            if hasattr(self.ui, 'roiVisibilityButton'):
                try:
                    self.ui.roiVisibilityButton.toggled.disconnect()
                except:
                    pass
        ScriptedLoadableModuleWidget.cleanup(self)

    def enter(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

    def onOutputVolumeChanged(self, node):
        """Handle output volume changes - update info display"""
        if self._parameterNode:
            self._parameterNode.outputVolume = node
            self.updateVolumeInfo()  # Update display with new output volume info
        
    def setParameterNode(self, parameterNode: Union[vtkMRMLNode, CropTBVolumeParameterNode]):
        if isinstance(parameterNode, CropTBVolumeParameterNode):
            rawNode = parameterNode.parameterNode
        else:
            rawNode = parameterNode

        # Disconnect from previous parameter node
        if self._parameterNode:
            self._parameterNode.disconnectGui(self.ui)
            # Remove observers
            if hasattr(self, 'inputVolumeObserverTag') and self.inputVolumeObserverTag:
                try:
                    self._parameterNode.inputVolume.RemoveObserver(self.inputVolumeObserverTag)
                except:
                    pass
                self.inputVolumeObserverTag = None
            if hasattr(self, 'outputVolumeObserverTag') and self.outputVolumeObserverTag:
                try:
                    self._parameterNode.outputVolume.RemoveObserver(self.outputVolumeObserverTag)
                except:
                    pass
                self.outputVolumeObserverTag = None
        
        # Wrap the raw node
        self._parameterNode = CropTBVolumeParameterNode(rawNode) if rawNode else None

        if self._parameterNode:
            # Connect widgets manually with proper null checks
            if hasattr(self.ui, 'fillValueSpinBox'):
                self.ui.fillValueSpinBox.value = self._parameterNode.fillValue
                self.ui.fillValueSpinBox.valueChanged.connect(
                    lambda v: setattr(self._parameterNode, 'fillValue', v))

            # Observe input volume renames if it exists
            if self._parameterNode.inputVolume:
                self.inputVolumeObserverTag = self._parameterNode.inputVolume.AddObserver(
                    RENAMED_EVENT, self.onInputVolumeRenamed)
            
            # Observe output volume renames if it exists
            if self._parameterNode.outputVolume:
                self.outputVolumeObserverTag = self._parameterNode.outputVolume.AddObserver(
                    RENAMED_EVENT, self.onOutputVolumeRenamed)
            
            # Update all UI elements
            self.updateVolumeInfo()
            self.updateROISizeWidget()

    def onROIModified(self, caller, event):
        """Handle ROI modification events"""
        logging.debug(f"ROI modified - caller: {caller.GetClassName() if caller else 'None'}")
        try:
            if not self._parameterNode or not self._parameterNode.roiNode:
                return
                
            current_roi = self._parameterNode.roiNode
            
            # Filter spurious MRML node events
            if caller == current_roi or caller == current_roi.GetDisplayNode():
                # Restart the debounce timer
                self._roiUpdateTimer.start(100)  # 100ms delay
            
        except Exception as e:
            logging.error(f"Error in onROIModified: {str(e)}")
    
    def onSceneClosing(self, caller, event) -> None:
        self.setParameterNode(None)

    def onApply(self) -> None:
        with slicer.util.tryWithErrorDisplay("Failed to crop volume.", waitCursor=True):
            if not self._parameterNode or not self._parameterNode.inputVolume:
                raise ValueError("Input volume not selected")
            
            # Auto-name output volume if none exists
            if not self._parameterNode.outputVolume:
                self._autoCreateOutputVolume()
                
            if not self._parameterNode.outputVolume:
                raise ValueError("Output volume not created")
                
            self.logic.cropVolume()
            slicer.util.resetSliceViews()
            self.updateVolumeInfo()

    def onROISizeChanged(self) -> None:
        """Update ROI size while maintaining center position"""
        if not self._parameterNode or not self._parameterNode.roiNode:
            return
            
        roi = self._parameterNode.roiNode
        new_size = [self.ui.sizeXSpinBox.value, 
                    self.ui.sizeYSpinBox.value, 
                    self.ui.sizeZSpinBox.value]
        center = roi.GetCenter()
        
        # Only update if size actually changed
        current_size = roi.GetSize()
        if (abs(new_size[0]-current_size[0]) > 0.01 or
            abs(new_size[1]-current_size[1]) > 0.01 or
            abs(new_size[2]-current_size[2]) > 0.01):
            roi.SetSize(new_size)
            roi.SetCenter(center)

    def onROISelectorNodeAdded(self, node):
        """Handle new ROI creation with automatic naming and visibility control"""
        try:
            # Create display nodes if they don't exist
            if not node.GetDisplayNode():
                node.CreateDefaultDisplayNodes()
            
            # Set up ROI properties
            displayNode = node.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(True)  # Make sure ROI is initially visible
            
            # Generate unique name
            base_name = "CropROI"
            existing_names = [n.GetName() for n in slicer.util.getNodesByClass("vtkMRMLMarkupsROINode") 
                            if n.GetName().startswith(base_name)]
            
            # Find next available number
            i = 1
            while f"{base_name}_{i}" in existing_names:
                i += 1
            node.SetName(f"{base_name}_{i}")
            
            # Store reference to current ROI
            self._parameterNode.roiNode = node
            self.ui.roiSelector.setCurrentNode(node)
            
            # Initialize visibility button state
            if hasattr(self.ui, 'roiVisibilityButton'):
                self.ui.roiVisibilityButton.setChecked(False)
                self.ui.roiVisibilityButton.setText("Hide ROI")
            
            # Set up observers for this ROI
            self.setupROIObservers(node)
            
            # Auto-create output volume if input exists
            if self._parameterNode.inputVolume and not self._parameterNode.outputVolume:
                self._autoCreateOutputVolume()

            self.onFitToVolume()
        except Exception as e:
            logging.error(f"Error in onROISelectorNodeAdded: {str(e)}")
    
    def _autoCreateOutputVolume(self):
        """Automatically create and name output volume based on input name"""
        if not self._parameterNode.inputVolume:
            return
        
        try:
            input_name = self._parameterNode.inputVolume.GetName()
            output_name = self._generateUniqueOutputName(input_name)
            
            # Remove existing output volume if it follows our naming pattern
            if (self._parameterNode.outputVolume and 
                self._parameterNode.outputVolume.GetName().startswith("Cropped_")):
                slicer.mrmlScene.RemoveNode(self._parameterNode.outputVolume)
            
            # Initialize with empty image data to prevent NoneType issues
            output_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_name)
            imageData = vtk.vtkImageData()
            output_node.SetAndObserveImageData(imageData)
            
            self._parameterNode.outputVolume = output_node
            if hasattr(self, 'ui') and self.ui and hasattr(self.ui, 'outputSelector'):
                self.ui.outputSelector.setCurrentNode(output_node)
            
            # Update volume info after creation
            self.updateVolumeInfo()
        except Exception as e:
            logging.error(f"Error in _autoCreateOutputVolume: {str(e)}")
        
    def _generateUniqueOutputName(self, base_name):
        """Generate unique output volume name with numbering"""
        existing_names = [n.GetName() for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode") 
                        if n.GetName().startswith(f"Cropped_{base_name}")]
        
        # Find next available number
        i = 1
        while f"Cropped_{base_name}_{i}" in existing_names:
            i += 1
        return f"Cropped_{base_name}_{i}"

    def setupROIObservers(self, roiNode):
        """Set up observers for ROI modifications and renames"""  
        # Remove existing observers
        for observer in self.roiObservers:
            if isinstance(observer, tuple):
                caller, tag = observer
                caller.RemoveObserver(tag)
        self.roiObservers = []
        
        if roiNode:
            # Observe ROI modifications
            tag = roiNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onROIModified)
            self.roiObservers.append((roiNode, tag))
            
            # Observe ROI renames
            tag = roiNode.AddObserver(RENAMED_EVENT, self.onROIRenamed)
            self.roiObservers.append((roiNode, tag))
            
            displayNode = roiNode.GetDisplayNode()
            if displayNode:
                tag = displayNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onROIModified)
                self.roiObservers.append((displayNode, tag))
        
        self.updateROISizeWidget()
            
    def onROIVisibilityToggled(self, checked):
        """Toggle ROI visibility with better error handling"""
        try:
            if not self._parameterNode or not self._parameterNode.roiNode:
                return
                
            roiNode = self._parameterNode.roiNode
            displayNode = roiNode.GetDisplayNode()
            
            # Create display node if it doesn't exist
            if not displayNode:
                roiNode.CreateDefaultDisplayNodes()
                displayNode = roiNode.GetDisplayNode()
                
            if displayNode:
                # Toggle both ROI and handles visibility
                displayNode.SetVisibility(not checked)
                
                # Update button text
                self.ui.roiVisibilityButton.setText("Show ROI" if checked else "Hide ROI")
                
                # Force view update
                slicer.util.forceRenderAllViews()
        except Exception as e:
            logging.error(f"Error in onROIVisibilityToggled: {str(e)}")
        
    def onFitToVolume(self) -> None:
        """Set ROI to match input volume bounds"""
        vol = self._parameterNode.inputVolume
        roi = self._parameterNode.roiNode
        if vol and roi:
            bounds = [0]*6
            vol.GetRASBounds(bounds)
            center = [(bounds[1]+bounds[0])/2, (bounds[3]+bounds[2])/2, (bounds[5]+bounds[4])/2]
            size = [(bounds[1]-bounds[0]), (bounds[3]-bounds[2]), (bounds[5]-bounds[4])]
            roi.SetCenter(center)
            roi.SetSize(size)
        self.updateVolumeInfo()  # Make sure to update info after fitting

    def updateROISizeWidget(self) -> None:
        """Update UI size widget from ROI node"""
        try:
            if (not hasattr(self, 'ui') or not self.ui or 
                not self._parameterNode or not self._parameterNode.roiNode):
                return
                
            roi = self._parameterNode.roiNode
            if not roi:
                return
            size = roi.GetSize()
            
            # Block signals to prevent infinite loops
            self.ui.sizeXSpinBox.blockSignals(True)
            self.ui.sizeYSpinBox.blockSignals(True)
            self.ui.sizeZSpinBox.blockSignals(True)
            
            # Update spin boxes with formatted values
            self.ui.sizeXSpinBox.setValue(round(size[0], 2))
            self.ui.sizeYSpinBox.setValue(round(size[1], 2))
            self.ui.sizeZSpinBox.setValue(round(size[2], 2))
            
            # Force immediate update
            self.ui.sizeXSpinBox.repaint()
            self.ui.sizeYSpinBox.repaint()
            self.ui.sizeZSpinBox.repaint()
            
        except Exception as e:
            logging.error(f"Error in updateROISizeWidget: {str(e)}")
        finally:
            # Always unblock signals
            self.ui.sizeXSpinBox.blockSignals(False)
            self.ui.sizeYSpinBox.blockSignals(False)
            self.ui.sizeZSpinBox.blockSignals(False)

    def updateVolumeInfo(self) -> None:
        """Update volume information display - only shows output info when output volume exists"""
        # Clear both labels first
        self.ui.inputInfoLabel.setText("Input: ")
        self.ui.outputInfoLabel.setText("Output: ")
        
        # Handle input volume information
        if self._parameterNode:
            if self._parameterNode.inputVolume is not None:  # Explicit None check
                try:
                    inputSpacing = self._parameterNode.inputVolume.GetSpacing()
                    inputImageData = self._parameterNode.inputVolume.GetImageData()
                    if inputImageData:  # Check if image data exists
                        inputDims = inputImageData.GetDimensions()
                        self.ui.inputInfoLabel.setText(
                            f"Input: {inputDims[0]}x{inputDims[1]}x{inputDims[2]} "
                            f"({inputSpacing[0]:.2f}x{inputSpacing[1]:.2f}x{inputSpacing[2]:.2f} mm)")
                    else:
                        self.ui.inputInfoLabel.setText("Input: (no image data)")
                except Exception as e:
                    logging.error(f"Error updating input info: {str(e)}")
                    self.ui.inputInfoLabel.setText("Input: (error)")
            else:
                # Explicitly handle None input volume case
                self.ui.inputInfoLabel.setText("Input: (none)")
                self.ui.outputInfoLabel.setText("Output: (none)")
        
        # Handle output volume info
        if self._parameterNode:
            if self._parameterNode.outputVolume:
                try:
                    outputSpacing = self._parameterNode.outputVolume.GetSpacing()
                    outputImageData = self._parameterNode.outputVolume.GetImageData()
                    if outputImageData:  # Check if image data exists
                        outputDims = outputImageData.GetDimensions()
                        self.ui.outputInfoLabel.setText(
                            f"Output: {outputDims[0]}x{outputDims[1]}x{outputDims[2]} "
                            f"({outputSpacing[0]:.2f}x{outputSpacing[1]:.2f}x{outputSpacing[2]:.2f} mm)")
                    else:
                        self.ui.outputInfoLabel.setText("Output: (no image data)")
                except Exception as e:
                    logging.error(f"Error updating output info: {str(e)}")
                    self.ui.outputInfoLabel.setText("Output: (error)")
            else:
                # Explicitly handle None output volume case
                self.ui.outputInfoLabel.setText("Output: (none)")

    def onInputVolumeRenamed(self, node):
        """Handle input volume rename and update output volume name if needed"""
        if node and node == self._parameterNode.inputVolume:
            # Only update output name if it follows our auto-naming pattern
            if (self._parameterNode.outputVolume and 
                self._parameterNode.outputVolume.GetName().startswith("Cropped_")):
                self._autoCreateOutputVolume()

    def onROIRenamed(self, node):
        """Handle ROI rename"""
        if node and node == self._parameterNode.roiNode:
            # No special handling needed for ROI rename
            pass
        
    def onOutputVolumeRenamed(self, node):
        """Handle output volume rename"""
        if node and node == self._parameterNode.outputVolume:
            # No special handling needed for output rename
            pass
    
#
# CropTBVolume
#
           
class CropTBVolume(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Temporal Bone Volume Cropping")
        self.parent.categories = ["Volume"]  # Matches CMake category
        self.parent.dependencies = ["Markups"]  # Required modules
        self.parent.contributors = ["Jonathan Wang (JHUSOM)", "Andy Ding (JHU Hospital)"]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = _("""Prepares temporal bone CT scans for nnUNet processing through automated cropping and resampling.""")
        self.parent.acknowledgementText = _("""This file was originally developed by Jonathan Wang under Andy Ding and Francis Creighton at Johns Hopkins Hospital Department of Otolarynology.""")
    
#
# CropTBVolumeLogic
#

class CropTBVolumeLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._parameterNode = None  # Initialize first
        self._parameterNode = self.getParameterNode()

    @property
    def parameterNode(self) -> CropTBVolumeParameterNode:
        """Return the wrapped parameter node (preferred access method)"""
        return CropTBVolumeParameterNode(self.getParameterNode())
    
    @property
    def wrappedParameterNode(self) -> CropTBVolumeParameterNode:
        """Alias for parameterNode (for backward compatibility)"""
        return self.parameterNode
    
    def getParameterNode(self):
        # Get the raw parameter node from the base class
        paramNode = super().getParameterNode()
        if not paramNode:
            # Create new node if none exists
            paramNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "CropTBVolumeParams")
            super().setParameterNode(paramNode)
            # Initialize defaults using the wrapper
            wrappedNode = CropTBVolumeParameterNode(paramNode)
            wrappedNode.interpolatorType = 1
            wrappedNode.spacingScale = 1.0
            wrappedNode.fillValue = 0.0
            wrappedNode.isotropicSpacing = False
        return paramNode
    
    def cropVolume(self) -> None:
        """Perform the volume cropping operation"""
        p = self.parameterNode
        if not p.inputVolume or not p.inputVolume.GetImageData():
            raise ValueError("Input volume has no image data")
        if not p.outputVolume:
            raise ValueError("Output volume not specified")
        if not p.roiNode:
            raise ValueError("ROI not specified")

        try:
            # Voxel-based cropping (no resampling)
            # Get input volume properties
            input_spacing = p.inputVolume.GetSpacing()
            input_ijk_to_ras = vtk.vtkMatrix4x4()
            p.inputVolume.GetIJKToRASMatrix(input_ijk_to_ras)
            
            # Get ROI bounds in RAS
            ras_bounds = np.zeros(6)
            p.roiNode.GetBounds(ras_bounds)
            
            # Calculate voxel-aligned extent
            extent = self._calculateVoxelBasedOutputExtent(ras_bounds, 
                                                        p.inputVolume.GetOrigin(),
                                                        input_spacing,
                                                        input_ijk_to_ras,
                                                        p.inputVolume)
            
            # Extract VOI from input volume
            extract = vtk.vtkExtractVOI()
            extract.SetInputData(p.inputVolume.GetImageData())
            extract.SetVOI(extent)
            extract.Update()
            
            # Get the extracted region's bounds in RAS
            ijk_min = [extent[0], extent[2], extent[4]]
            ijk_max = [extent[1], extent[3], extent[5]]
            
            # Calculate new origin using the input's IJKToRAS matrix
            transform = vtk.vtkTransform()
            transform.SetMatrix(input_ijk_to_ras)
            new_origin = transform.TransformPoint(ijk_min)
            
            # Set output properties
            p.outputVolume.SetAndObserveImageData(extract.GetOutput())
            p.outputVolume.SetSpacing(input_spacing)
            p.outputVolume.SetOrigin(new_origin)
            p.outputVolume.SetIJKToRASMatrix(input_ijk_to_ras)
            
            extracted_dims = extract.GetOutput().GetDimensions()
            
            logging.info(
                f"Voxel-based crop applied. "
                f"Extent: {extent}, "
                f"Dimensions: {extracted_dims}, "
                f"Spacing: {input_spacing}, "
                f"Origin: {new_origin}"
            )
        except Exception as e:
            logging.error(f"Error in cropVolume: {str(e)}")
            raise

    def _calculateVoxelBasedOutputExtent(self, roiBounds, inputOrigin, inputSpacing, ijkToRas, inputVolume):
        """Calculate voxel-aligned extent with proper coordinate handling"""
        rasToIjk = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)
        
        # Transform all ROI corners to IJK
        boundsCorners = [
            [roiBounds[0], roiBounds[2], roiBounds[4], 1],
            [roiBounds[1], roiBounds[2], roiBounds[4], 1],
            [roiBounds[0], roiBounds[3], roiBounds[4], 1],
            [roiBounds[1], roiBounds[3], roiBounds[4], 1],
            [roiBounds[0], roiBounds[2], roiBounds[5], 1],
            [roiBounds[1], roiBounds[2], roiBounds[5], 1],
            [roiBounds[0], roiBounds[3], roiBounds[5], 1],
            [roiBounds[1], roiBounds[3], roiBounds[5], 1]
        ]
        ijkCorners = []
        for corner in boundsCorners:
            pt = [0.0, 0.0, 0.0, 0.0]
            rasToIjk.MultiplyPoint(corner, pt)
            ijkCorners.append(pt[:3])
        
        ijkCorners = np.array(ijkCorners)
        ijkMin = np.floor(ijkCorners.min(axis=0)).astype(int)
        ijkMax = np.ceil(ijkCorners.max(axis=0)).astype(int)
        
        # Clamp to valid input dimensions
        inputDims = inputVolume.GetImageData().GetDimensions()
        ijkMin = np.clip(ijkMin, [0, 0, 0], [inputDims[0]-1, inputDims[1]-1, inputDims[2]-1])
        ijkMax = np.clip(ijkMax, [0, 0, 0], [inputDims[0]-1, inputDims[1]-1, inputDims[2]-1])
        
        return [ijkMin[0], ijkMax[0], ijkMin[1], ijkMax[1], ijkMin[2], ijkMax[2]]
        
#
# CropTBVolumeTest
#

class CropTBVolumeTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear"""
        ScriptedLoadableModuleTest.setUp(self)
        slicer.mrmlScene.Clear(0)
        
        # Initialize logic first
        self.logic = CropTBVolumeLogic()
        if not self.logic:
            self.fail("Logic creation failed")
            
        # Create test nodes first
        self.inputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "TestInput")
        self.roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "TestROI")
        self.outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "TestOutput")
        
        # Create test volume data
        arr = np.random.rand(50, 50, 50) * 100
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(50, 50, 50)
        imageData.AllocateScalars(vtk.VTK_FLOAT, 1)
        nps.numpy_to_vtk(arr.ravel(), deep=1, array_type=vtk.VTK_FLOAT).SetName("TEST")
        self.inputVolume.SetAndObserveImageData(imageData)
        
        # Set up ROI 
        self.roi.SetCenter([25, 25, 25])
        self.roi.SetSize([30, 30, 30])
        
        # Create widget and logic
        self.widget = CropTBVolumeWidget()
        self.widget.setup()
        self.logic = self.widget.logic
        
        # Connect ROI to widget before testing
        self.widget.ui.roiSelector.setCurrentNode(self.roi)
        slicer.app.processEvents()  # Allow UI to update

    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'widget') and self.widget:
            self.widget.cleanup()
        slicer.mrmlScene.Clear(0)
        
    def cleanup(self):
        """Clean up after each test"""
        self.widget.cleanup()
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run all tests"""
        self.setUp()
        try:
            self.test_VoxelBasedCropping()
            self.test_ROIInteraction()
        finally:
            self.tearDown()

    def test_VoxelBasedCropping(self):
        # Set parameters
        self.logic.parameterNode.inputVolume = self.inputVolume
        self.logic.parameterNode.roiNode = self.roi
        self.logic.parameterNode.outputVolume = self.outputVolume
        
        # Execute cropping
        self.logic.cropVolume()
        
        # Verify output
        self.assertIsNotNone(self.outputVolume.GetImageData())
        outputDims = self.outputVolume.GetImageData().GetDimensions()
        self.assertTrue(all(d > 0 for d in outputDims))
        print(f"Output dimensions: {outputDims}")

    def test_ROIInteraction(self):
        """Test ROI modification updates UI correctly"""
        # Make sure ROI is properly connected to widget
        self.assertEqual(self.widget.ui.roiSelector.currentNode(), self.roi)
        
        # First force an update of the spinboxes from current ROI size
        self.widget.updateROISizeWidget()
    
        # Set new size and wait for updates
        new_size = [20, 20, 20]
        self.roi.SetSize(new_size)
        
        # Explicitly trigger UI update since automatic updates may not happen in tests
        self.widget.updateROISizeWidget()
        
        # Verify UI reflects changes
        epsilon = 0.01  # Tight tolerance
        self.assertAlmostEqual(self.widget.ui.sizeXSpinBox.value, new_size[0], delta=epsilon,
                            msg=f"X size mismatch: {self.widget.ui.sizeXSpinBox.value} vs {new_size[0]}")
        self.assertAlmostEqual(self.widget.ui.sizeYSpinBox.value, new_size[1], delta=epsilon,
                            msg=f"Y size mismatch: {self.widget.ui.sizeYSpinBox.value} vs {new_size[1]}")
        self.assertAlmostEqual(self.widget.ui.sizeZSpinBox.value, new_size[2], delta=epsilon,
                            msg=f"Z size mismatch: {self.widget.ui.sizeZSpinBox.value} vs {new_size[2]}")