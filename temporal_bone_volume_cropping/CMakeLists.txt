cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(temporal_bone_volume_cropping)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/temporal_bone_volume_cropping")
set(EXTENSION_CATEGORY "Volume")
set(EXTENSION_CONTRIBUTORS "Jonathan Wang (Johns Hopkins School of Medicine), Andy Ding (Johns Hopkins Medicine)")
set(EXTENSION_DESCRIPTION "Integrated pipeline for preparing temporal bone CT scans for nnUNet models")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/temporal_bone_volume_cropping.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/temporal_bone_volume_cropping/Screenshots/1.png")
set(EXTENSION_DEPENDS "Markup")  # Added required dependency

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CropTBVolume)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})