# 2023-04-04 Trinity Poplar Island Dataset

## Overview
This dataset contains aerial imagery collected over Poplar Island, Maryland on April 4, 2023 by the Trinity UAV platform. The images capture various terrain features including vegetation, water bodies, and constructed surfaces.

## Dataset Specifications
- **Collection Date**: April 4, 2023
- **Location**: Poplar Island, Maryland
- **Platform**: Trinity UAV
- **Image Format**: Multi-band GeoTIFF
- **Spatial Resolution**: 10cm
- **Coordinate System**: WGS 84 / UTM Zone 18N

## Band Information
The dataset contains multi-spectral imagery with the following bands:
1. Blue (450-510nm)
2. Green (520-590nm)
3. Red (630-690nm)
4. Near Infrared (760-850nm)

## Dataset Structure
The processed dataset contains multiband GeoTIFF files created by merging individual band images. Each multiband file represents one complete image capture.

## Processing
The raw single-band images have been processed using the `merge_bands.py` utility to create multiband GeoTIFF files. This processing preserves all metadata and organizes the data for easier analysis.

## Usage
This dataset is suitable for:
- Land cover classification
- Vegetation health assessment
- Wetland monitoring
- Temporal change analysis
- Environmental impact studies

## Contact Information
For questions or additional information regarding this dataset, please contact the project administrator.