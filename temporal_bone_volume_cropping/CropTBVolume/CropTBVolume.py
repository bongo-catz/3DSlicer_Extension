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
from qt import QPushButton, QTimer, QMessageBox
import qt

RENAMED_EVENT = vtk.vtkCommand.UserEvent + 1  # Typically vtkCommand.UserEvent + 1 is used for renamed events
DEFAULT_ROI_SIZE = [51.2, 51.2, 51.2] # mm
ROI_LOCKED = True # Default state

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
        self.roiLocked = ROI_LOCKED # Track ROI Lock state
        self.tempMarkupNode = None

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
        self.ui.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.roiSelector.setMRMLScene(slicer.mrmlScene)
        
        # Connect signals
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNode)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputVolumeChanged)
        # self.ui.roiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNode)
        
        self.ui.roiSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", 
            lambda node: [self.updateParameterNode(), self.checkApplyButtonEnabled()]
        )
        
        # Add ROI observers after UI is set up
        self.addROIObservers()
        
        # Connect ROI modified signal to update function
        self.ui.roiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onROISelectionChanged)
    
        # Configure save volume selector
        self.ui.saveVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.saveVolumeSelector.setCurrentNode(None)  # Start with no selection
        self.ui.saveVolumeSelector.removeEnabled = False  # Disable the remove button
        self.ui.saveVolumeSelector.noneEnabled = True  # Allow deselecting
        self.ui.saveVolumeSelector.addEnabled = False  # Disable adding new volumes

        # Save button
        self.ui.saveButton.connect('clicked()', self.onSaveClicked)
        self.updateVolumeInfo()
    
        # Configure ROI selector
        self.ui.roiSelector.noneEnabled = True
        self.ui.roiSelector.addEnabled = False  # Disable the "+" button to create new ROIs
        self.ui.roiSelector.removeEnabled = True  # Keep remove functionality
        
        # Connect ROI controls
        self.ui.sizeXSpinBox.connect('valueChanged(double)', self.onROISizeChanged)
        self.ui.sizeYSpinBox.connect('valueChanged(double)', self.onROISizeChanged)
        self.ui.sizeZSpinBox.connect('valueChanged(double)', self.onROISizeChanged)
        
        # Connect buttons
        self.ui.fitToVolumeButton.connect('clicked()', self.onFitToVolume)
        self.ui.applyButton.connect('clicked(bool)', self.onApply)
        self.ui.createROIFromPointButton.connect('clicked(bool)', self.onCreateROIFromPoint)
        self.ui.roiLockButton.toggled.connect(self.onROILockToggled)
    
        # Set initial ROI size to default
        self.ui.sizeXSpinBox.value = DEFAULT_ROI_SIZE[0]
        self.ui.sizeYSpinBox.value = DEFAULT_ROI_SIZE[1]
        self.ui.sizeZSpinBox.value = DEFAULT_ROI_SIZE[2]
        
        # Set initial locked state
        self.setROIControlsEnabled(False)
        self.ui.roiLockButton.setChecked(False)
        
        # Initial UI state - disable controls until ROI is created
        self.setControlsEnabled(False)
        self.updateVolumeInfo()

    def addROIObservers(self):
        """Add observers to current ROI node"""
        self.removeROIObservers()  # Clean up any existing observers first
        
        roiNode = self.ui.roiSelector.currentNode()
        if roiNode:
            # Observe ROI modified events
            tag = roiNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onROIModified)
            self.roiObservers.append((roiNode, tag))
            
            # Also observe display node if it exists
            displayNode = roiNode.GetDisplayNode()
            if displayNode:
                tag = displayNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onROIModified)
                self.roiObservers.append((displayNode, tag))

    def removeROIObservers(self):
        """Remove all ROI observers"""
        for caller, tag in self.roiObservers:
            caller.RemoveObserver(tag)
        self.roiObservers = []

    def onROISelectionChanged(self, node):
        """Handle ROI node selection changes"""
        self.removeROIObservers()
        self.addROIObservers()
        self.updateROISizeWidget()
    
    def onSaveClicked(self):
        """Handle save button click with proper volume selection and error handling"""
        try:
            # Get selected volume - use output volume if available, otherwise allow any selection
            volumeNode = self.ui.saveVolumeSelector.currentNode()
            if volumeNode is None:
                if hasattr(self, '_parameterNode') and self._parameterNode and self._parameterNode.outputVolume:
                    volumeNode = self._parameterNode.outputVolume
                    self.ui.saveVolumeSelector.setCurrentNode(volumeNode)
                else:
                    QMessageBox.warning(self.parent, "Save Error", "Please select a volume to save.")
                    return

            # Generate default filename
            defaultFilename = f"{volumeNode.GetName()}.nrrd"

            # Show save file dialog
            result = qt.QFileDialog.getSaveFileName(
                slicer.util.mainWindow(),
                "Save Volume As...",
                defaultFilename,
                "NRRD files (*.nrrd);;NIfTI files (*.nii *.nii.gz);;All files (*)"
            )

            # Handle different return types
            fileName = result if isinstance(result, str) else result[0]
            if not fileName:
                return  # User canceled

            # Append extension if missing
            if not fileName.lower().endswith(('.nrrd', '.nii', '.nii.gz')):
                if "nii" in fileName.lower():
                    fileName += ".nii"
                else:
                    fileName += ".nrrd"

            # Save the volume
            success = slicer.util.saveNode(volumeNode, fileName)
            if success:
                QMessageBox.information(
                    self.parent,
                    "Save Successful",
                    f"Volume saved successfully to:\n{fileName}"
                )
            else:
                QMessageBox.warning(
                    self.parent,
                    "Save Failed",
                    "Failed to save volume. Unknown error occurred."
                )

        except Exception as e:
            logging.error(f"Unexpected error in save operation: {str(e)}")
            QMessageBox.critical(
                self.parent,
                "Error",
                f"An unexpected error occurred:\n{str(e)}"
            )

    
    def updateSaveVolumeSelector(self):
        """Update the save volume selector with current output volume"""
        if hasattr(self, '_parameterNode') and self._parameterNode and self._parameterNode.outputVolume:
            self.ui.saveVolumeSelector.setCurrentNode(self._parameterNode.outputVolume)
        else:
            self.ui.saveVolumeSelector.setCurrentNode(None)
        
    def checkApplyButtonEnabled(self):
        """Enable Apply button only when both input volume and ROI are selected"""
        inputNode = self.ui.inputSelector.currentNode()
        roiNode = self.ui.roiSelector.currentNode()
        if inputNode is not None and roiNode is not None:
            self.ui.applyButton.setEnabled(True)
        else:
            self.ui.applyButton.setEnabled(False)
        
    def setControlsEnabled(self, enabled):
        """Enable/disable all controls except input selector"""
        self.ui.outputSelector.setEnabled(enabled)
        self.ui.roiSelector.setEnabled(enabled)
        self.ui.fitToVolumeButton.setEnabled(enabled)
        self.ui.applyButton.setEnabled(enabled)
        self.setROIControlsEnabled(enabled and not self.roiLocked)
    
    def setROIControlsEnabled(self, enabled):
        """Enable/disable ROI controls"""
        self.ui.sizeXSpinBox.setEnabled(enabled)
        self.ui.sizeYSpinBox.setEnabled(enabled)
        self.ui.sizeZSpinBox.setEnabled(enabled)
        self.ui.fitToVolumeButton.setEnabled(enabled)
        
        # Update ROI interactive state
        self.updateROILockState()
        
        # Invalidate views if disabling to indicate ROI is locked
        if not enabled:
            slicer.util.forceRenderAllViews()
            
    def onCreateROIFromPoint(self):
        """Stable point selection that works across Slicer versions"""
        selectedInputNode = self.ui.inputSelector.currentNode()
        if selectedInputNode is None:
            QMessageBox.warning(None, "Error", "Please select an input volume first")
            return
        
        logging.info(f"The selected input volume name is: {self._parameterNode.inputVolume.GetName()}")
        
        # Confirm user wants to enter point selection mode
        result = QMessageBox.question(
            None, "Create ROI from Point\n", 
            "Left click in ANY SLICE VIEW (axial, saggital, coronal) to place the ROI center point.\n"
            "\nPress [Esc] key to cancel.",
            QMessageBox.Yes | QMessageBox.No)
        
        logging.info(f"User selected: {'Yes' if result == QMessageBox.Yes else 'No'} (1st Prompt)")
        
        if result == QMessageBox.No:
            return
        
        # Create temporary markup node
        self.tempMarkupNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'TempROIPoint')
        self.tempMarkupNode.CreateDefaultDisplayNodes()
        displayNode = self.tempMarkupNode.GetDisplayNode()
        
        # Version-compatible display settings
        try:
            # Slicer 5.0+ style
            displayNode.SetGlyphType(displayNode.Sphere3D)
        except:
            # Fallback for older versions
            displayNode.SetGlyphTypeFromString('Sphere')
        displayNode.SetGlyphScale(3.0)
        displayNode.SetSelectedColor(1,1,0) # Yellow
        displayNode.SetTextScale(0) # Hide text
        
        # Clear any pending mouse events
        slicer.app.processEvents()
        
        # Set up mouse click observers
        self.observerTags = []
        self.sliceWidgets = []
        layoutManager = slicer.app.layoutManager()
        
        # Observe all slice views
        for sliceViewName in ['Red', 'Yellow', 'Green']:
            sliceWidget = layoutManager.sliceWidget(sliceViewName)
            if sliceWidget:
                self.sliceWidgets.append(sliceWidget)
                interactor = sliceWidget.sliceView().interactor()
                # Observe left mouse clicks
                tag = interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.onSliceClick)
                self.observerTags.append((interactor, tag))
                # Observe Escape key
                tag_esc = interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.onKeyPress)
                self.observerTags.append((interactor, tag_esc))

        # Initialize status message
        slicer.util.showStatusMessage("Click in any slice view to place ROI center. Press Esc to cancel.")

    def onSliceClick(self, interactor, event):
        """Handle slice view mouse clicks"""
        try:
            # Find which slice widget was clicked
            for sliceWidget in self.sliceWidgets:
                if sliceWidget.sliceView().interactor() == interactor:
                    x, y = interactor.GetEventPosition()
                    
                    # Convert click to RAS coordinates
                    sliceView = sliceWidget.sliceView()
                    xy = [x, y]
                    xyz = list(sliceView.convertDeviceToXYZ(xy))  # Returns QList<double> [x,y,z]
                    ras = list(sliceView.convertXYZToRAS(xyz))    # Returns QList<double> [R,A,S]
                    
                    # Add visual feedback
                    self.tempMarkupNode.RemoveAllControlPoints()
                    self.tempMarkupNode.AddControlPoint(ras[0], ras[1], ras[2])
                    
                    # Confirm placement after short delay
                    QTimer.singleShot(100, lambda: self.confirmAndCreateROI(ras))
                    self.cleanupObservers()
                    return 1  # Prevent default handling
                    
        except Exception as e:
            logging.error(f"Error in onSliceClick: {str(e)}")
            self.cleanupObservers()
            return 1

    def onKeyPress(self, interactor, event):
        """Handle Escape key cancellation"""
        key = interactor.GetKeySym()
        if key == 'Escape':
            logging.info("User clicked [Esc] on placing ROI center point")
            self.cleanupObservers()
            slicer.mrmlScene.RemoveNode(self.tempMarkupNode)
            slicer.util.showStatusMessage("ROI placement cancelled.")
        return 1
    
    def convertClickToRAS(self, sliceWidget, x, y):
        """Convert mouse click position to RAS coordinates"""
        try:
            sliceView = sliceWidget.sliceView()
            ras = [0,0,0]
            # Convert mouse position to RAS using slice view's coordinate system
            sliceView.convertDeviceToXYZ(x, y, ras)
            return ras
        except Exception as e:
            logging.error(f"Coordinate conversion failed: {str(e)}")
            return None
    
    def cleanupObservers(self):
        """Remove all temporary observers"""
        for interactor, tag in self.observerTags:
            interactor.RemoveObserver(tag)
        self.observerTags = []
        self.sliceWidgets = []
        slicer.util.showStatusMessage("")

    def confirmAndCreateROI(self, rasPoint):
        """Handle confirmation after event processing completes"""
        try:
            # Show confirmation dialog
            confirm = QMessageBox.question(
                None, "Confirm Position",
                f"Create ROI at:\nX: {rasPoint[0]:.1f}\nY: {rasPoint[1]:.1f}\nZ: {rasPoint[2]:.1f}",
                QMessageBox.Yes|QMessageBox.No)
            
            logging.info(f"User selected: {'Yes' if confirm == QMessageBox.Yes else 'No'} (2nd Prompt)")
            
            if confirm == QMessageBox.Yes:
                self.createROIAtPoint(rasPoint)
            else:
                # Restart placement if user wants to try again
                self.onCreateROIFromPoint()
                
        finally:
            # Ensure cleanup happens in all cases
            self.exitPlacementMode()
 
    def createROIAtPoint(self, rasPoint):
        """Create ROI at specified point"""
        logging.info("Creating ROI at Point!")
        if not self._parameterNode:
            return
        
         # Generate unique name
        base_name = "CropROI"
        existing_names = [n.GetName() for n in slicer.util.getNodesByClass("vtkMRMLMarkupsROINode") 
                        if n.GetName().startswith(base_name)]
        
        # Find next available number
        i = 1
        new_name = f"{base_name}_{i}"
        while f"{base_name}_{i}" in existing_names:
            i += 1
            new_name = f"{base_name}_{i}"
            
        # Create new ROI
        roiNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode', new_name)
        roiNode.CreateDefaultDisplayNodes()
        roiNode.SetCenter(rasPoint)
        roiNode.SetSize(DEFAULT_ROI_SIZE)
        
        # Configure appearance
        displayNode = roiNode.GetDisplayNode()
        if displayNode:
            # Always show outline in all views
            displayNode.SetVisibility(True)
            try:
                # Newer Slicer versions use SetVisibility2D
                displayNode.SetVisibility2D(True)  # Show in slice views
            except AttributeError:
                # Fallback for older versions
                displayNode.SetSliceIntersectionVisibility(True)
            displayNode.SetVisibility3D(True)  # Show in 3D view
            
            # Set outline properties
            displayNode.SetSelectedColor(0, 1, 0)  # Green when selected/interacting
            displayNode.SetColor(0.5, 0.5, 0.5)   # Gray when not selected
            displayNode.SetOpacity(1.0)            # Fully visible outline
            displayNode.SetFillOpacity(0.0)        # Completely transparent fill
            displayNode.SetFillVisibility(False)    # Hide fill
            displayNode.SetOutlineVisibility(True)  # Show outline
                
        # Update UI
        self._parameterNode.roiNode = roiNode
        # Force the selector to update its list of available ROIs
        self.ui.roiSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.roiSelector.setCurrentNode(roiNode)
        self.updateROISizeWidget()
        self.setControlsEnabled(True)  # Enable other controls now that we have ROI
        slicer.util.resetSliceViews()
            
        # Update lock state for new ROI
        self.updateROILockState()
        
    def exitPlacementMode(self):
        """Safely exit placement mode"""
        # Exit placement mode
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode and interactionNode.GetCurrentInteractionMode() == interactionNode.Place:
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        
        # Clean up observers
        if hasattr(self, 'pointPlacementObserverTag') and hasattr(self, 'tempMarkupNode'):
            try:
                self.tempMarkupNode.RemoveObserver(self.pointPlacementObserverTag)
            except:
                pass
            del self.pointPlacementObserverTag
        
        # Remove temp node
        if hasattr(self, 'tempMarkupNode') and self.tempMarkupNode:
            try:
                slicer.mrmlScene.RemoveNode(self.tempMarkupNode)
            except:
                pass
            self.tempMarkupNode = None
            
    def onROILockToggled(self, unlocked):
        """Handle ROI lock/unlock toggle using global ROI_LOCKED default"""
        if unlocked:
            # Show confirmation dialog when unlocking
            result = QMessageBox.question(
                None, "Confirm ROI Edit", 
                "Are you sure you want to edit the ROI?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result == QMessageBox.No:
                self.ui.roiLockButton.setChecked(False)
                unlocked = False
            logging.info(f"User selected: {'Yes' if result == QMessageBox.Yes else 'No'} (ROI Locking)")
        self.roiLocked = not unlocked
        self.setROIControlsEnabled(unlocked)
        self.ui.roiLockButton.setText("Lock ROI" if unlocked else "Unlock ROI")
        self.updateROILockState()
    
    def updateROILockState(self):
        """Update the ROI's interactive state based on lock status"""
        roiNode = self.ui.roiSelector.currentNode()
        if not hasattr(self, '_parameterNode') or not self._parameterNode or roiNode is None:
            return
        
        displayNode = roiNode.GetDisplayNode()
        if not displayNode:
            return
        
         # Safely update interaction for ROI
        try:
            displayNode.SetHandlesInteractive(not self.roiLocked)
        except AttributeError:
            logging.warning("ROI display node does not support SetHandlesInteractive")

        # Lock ROI geometry to disable movement
        try:
            roiNode.SetLocked(self.roiLocked)
        except AttributeError:
            logging.warning("ROI node does not support SetLocked")

        # Force view update
        slicer.util.forceRenderAllViews()
    
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
            
            logging.info(f"The input volume is: {self._parameterNode.inputVolume.GetName()}")
            
            # Focus slice views on the new input volume
            self.focusSliceViewsOnVolume(inputNode)
        
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
        self.checkApplyButtonEnabled()
        self.updateSaveVolumeSelector()
    
    def focusSliceViewsOnVolume(self, volumeNode):
        """Center 2D slice views on the specified volume and display it as the background"""
        if not volumeNode:
            return

        try:
            # Ensure volume has display node and is visible in slice views
            displayNode = volumeNode.GetDisplayNode()
            if not displayNode:
                volumeNode.CreateDefaultDisplayNodes()
                displayNode = volumeNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(True)

            # Get RAS bounds of the volume
            bounds = [0] * 6
            volumeNode.GetRASBounds(bounds)
            center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2,
            ]

            # Access layout manager and update each slice view
            layoutManager = slicer.app.layoutManager()

            # Get volume dimensions and spacing
            imageData = volumeNode.GetImageData()
            spacing = volumeNode.GetSpacing()
            dimensions = imageData.GetDimensions()

            # Calculate full physical size of the volume in mm
            size_mm = [dimensions[i] * spacing[i] for i in range(3)]

            for sliceViewName in ['Red', 'Yellow', 'Green']:
                sliceWidget = layoutManager.sliceWidget(sliceViewName)
                if sliceWidget:
                    sliceLogic = sliceWidget.sliceLogic()
                    if sliceLogic:
                        # Set background volume in this slice view
                        compositeNode = sliceLogic.GetSliceCompositeNode()
                        compositeNode.SetBackgroundVolumeID(volumeNode.GetID())

                        # Center view on the volume
                        sliceNode = sliceLogic.GetSliceNode()
                        sliceNode.JumpSliceByCentering(*center)

                        # Calculate and apply appropriate FOV
                        viewAxis = sliceNode.GetOrientationString()
                        if viewAxis == 'Axial':
                            sliceNode.SetFieldOfView(size_mm[0], size_mm[1], 1)
                        elif viewAxis == 'Sagittal':
                            sliceNode.SetFieldOfView(size_mm[1], size_mm[2], 1)
                        elif viewAxis == 'Coronal':
                            sliceNode.SetFieldOfView(size_mm[0], size_mm[2], 1)

                        # Refresh the view
                        sliceLogic.FitSliceToAll()

            slicer.util.forceRenderAllViews()

        except Exception as e:
            logging.error(f"Error focusing slice views: {str(e)}")
        
    def cleanup(self) -> None:
        """Clean up when module is closed"""
        self.removeROIObservers()
        
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
                # self._roiUpdateTimer.start(100)  # 100ms delay
                # Update immediately without debouncing
                self.updateROISizeWidget()
            
        except Exception as e:
            logging.error(f"Error in onROIModified: {str(e)}")

    def onApply(self) -> None:
        with slicer.util.tryWithErrorDisplay("Failed to crop volume.", waitCursor=True):
            selectedInputNode = self.ui.inputSelector.currentNode()
            if selectedInputNode is None:
                QMessageBox.warning(None, "Error", "Please select an input volume first")
                raise ValueError("Input volume not selected")
            
            # Auto-name output volume if none exists
            selectedOutputNode = self.ui.outputSelector.currentNode()
            if selectedOutputNode is None:
                self._autoCreateOutputVolume()
                
            if not self._parameterNode.outputVolume:
                raise ValueError("Output volume not created")
            
            logging.info(f"The created output volume name is: {self._parameterNode.outputVolume.GetName()}")
            self.logic.cropVolume()
            slicer.util.resetSliceViews()
            self.updateVolumeInfo()
            
            # Show success message
            QMessageBox.information(
                None,
                "Cropping Completed",
                f"Cropping Succesful!\n\nOutput Volume Name: {self._parameterNode.outputVolume.GetName()}\n\nClick OK to continue.",
                QMessageBox.Ok
            )
            
            # Auto-select the output volume in the save selector
            self.ui.saveVolumeSelector.setCurrentNode(self._parameterNode.outputVolume)
            
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
            
            # Create default display nodes immediately
            output_node.CreateDefaultDisplayNodes()
        
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
            
            # Only update if values actually changed
            current_x = self.ui.sizeXSpinBox.value
            current_y = self.ui.sizeYSpinBox.value
            current_z = self.ui.sizeZSpinBox.value
            
            if (abs(size[0] - current_x) > 0.01 or
                abs(size[1] - current_y) > 0.01 or
                abs(size[2] - current_z) > 0.01):
                
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
            
            # Verify extent is valid
            input_dims = p.inputVolume.GetImageData().GetDimensions()
            if (extent[0] < 0 or extent[1] >= input_dims[0] or
                extent[2] < 0 or extent[3] >= input_dims[1] or
                extent[4] < 0 or extent[5] >= input_dims[2]):
                raise ValueError("Calculated extent is outside input volume bounds")
            
            # Extract VOI from input volume
            extract = vtk.vtkExtractVOI()
            extract.SetInputData(p.inputVolume.GetImageData())
            extract.SetVOI(extent)
            extract.Update()
            
            outputImage = extract.GetOutput()
            # Get dimensions before modifying extent
            dims = outputImage.GetDimensions()
            
            # Reset the extent to 0-based for Slicer compatibility
            outputImage.SetExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
            
            # Get the extracted region's bounds in RAS
            ijk_min = [extent[0], extent[2], extent[4]]
            
            # Calculate new origin using the input's IJKToRAS matrix
            transform = vtk.vtkTransform()
            transform.SetMatrix(input_ijk_to_ras)
            new_origin = transform.TransformPoint(ijk_min)
            
            # Create a new IJKToRAS matrix for the cropped volume
            cropped_ijk_to_ras = vtk.vtkMatrix4x4()
            cropped_ijk_to_ras.DeepCopy(input_ijk_to_ras)
            cropped_ijk_to_ras.SetElement(0, 3, new_origin[0])
            cropped_ijk_to_ras.SetElement(1, 3, new_origin[1])
            cropped_ijk_to_ras.SetElement(2, 3, new_origin[2])
            
            # Set output properties
            p.outputVolume.SetAndObserveImageData(outputImage)
            p.outputVolume.SetSpacing(input_spacing)
            p.outputVolume.SetIJKToRASMatrix(cropped_ijk_to_ras)
            
            # IMPORTANT: Reset slice views BEFORE setting the new volume
            slicer.util.resetSliceViews()
            
            # Set the new volume as background in all slice views
            layoutManager = slicer.app.layoutManager()
            for sliceViewName in ['Red', 'Yellow', 'Green']:
                sliceWidget = layoutManager.sliceWidget(sliceViewName)
                if sliceWidget:
                    sliceLogic = sliceWidget.sliceLogic()
                    if sliceLogic:
                        compositeNode = sliceLogic.GetSliceCompositeNode()
                        compositeNode.SetBackgroundVolumeID(p.outputVolume.GetID())
            
            # Calculate center of the cropped volume in RAS
            p.outputVolume.GetRASBounds(ras_bounds)
            center = [
                (ras_bounds[0] + ras_bounds[1]) / 2,
                (ras_bounds[2] + ras_bounds[3]) / 2,
                (ras_bounds[4] + ras_bounds[5]) / 2,
            ]
            
            # Center all slice views using the proper method
            for sliceViewName in ['Red', 'Yellow', 'Green']:
                sliceWidget = layoutManager.sliceWidget(sliceViewName)
                if sliceWidget:
                    sliceNode = sliceWidget.sliceLogic().GetSliceNode()
                    sliceNode.JumpSliceByCentering(center[0], center[1], center[2])

            slicer.util.forceRenderAllViews()
            
            logging.info(
                f"Voxel-based crop applied. "
                f"Extent: {extent}, "
                f"Dimensions: {dims}, "
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