# Smart Urban Analyzer
A comprehensive tool for urban planning analysis with multiple data collection methods.
A comprehensive tool for urban planning analysis with multiple data collection methods for automatic detection of regulation violations and compliance analysis.
## Features
- **PDF Document Analysis**: Upload urban planning regulations PDFs and extract setback requirements automatically. Ask questions about the regulations using AI.
- **PDF Document Analysis**: Upload urban planning regulations PDFs and extract setback requirements, height restrictions, allowed building types, and coverage ratio limits automatically. Ask questions about the regulations using AI.
- **Multiple Survey Methods**:
  - **Satellite Image Analysis**: Upload and analyze satellite imagery to detect building structures.
  - **Drone Survey**: Process drone imagery with metadata about flight altitude and resolution.
  - **Satellite Image Analysis**: Upload and analyze satellite imagery to detect building structures, measure coverage area, and identify building types.
  - **Drone Survey**: Process drone imagery with metadata about flight altitude and resolution to create 3D models for height analysis.
  - **Phone GPS Coordinates**: Input GPS data from smartphone apps with interactive map visualization.
  - **Manual Measurements**: Enter plot and building dimensions with interactive visualization.
  - **Leica Precision Instruments**: Import and process data from professional survey equipment.
  - **Manual Measurements**: Enter plot and building dimensions with interactive visualization including height and floor information.
  - **Leica Precision Instruments**: Import and process data from professional survey equipment for high-precision measurements.
  - **BIM Integration**: Import BIM (Building Information Modeling) files to analyze detailed 3D building models directly.
- **Compliance Checking**: Automatically compare building layouts against setback regulations to identify potential violations.
- **Comprehensive Compliance Checking**:
  - **Setback Violations**: Automatically compare building boundaries against setback regulations.
  - **Height Violations**: Analyze building height against zoning height restrictions using 3D models.
  - **Building Type Classification**: Identify building types (residential, commercial, industrial) and verify against zoning permissions.
  - **Coverage Area Ratio (CAR)**: Calculate the ratio of building footprint to total plot area and check against maximum allowed coverage percentages.
  - **Floor Area Ratio (FAR)**: Calculate total floor area relative to plot size and verify compliance with density regulations.
## Advanced BIM Integration
The Smart Urban Analyzer supports Building Information Modeling (BIM) file import for comprehensive 3D analysis:
- **IFC File Processing**: Import Industry Foundation Classes (IFC) files containing complete building information.
- **Automatic Element Classification**: Identify structural elements, rooms, floors, and spaces.
- **3D Visualization**: View the building model with highlighted compliance issues.
- **Volumetric Analysis**: Calculate precise building volumes, floor areas, and heights.
- **Regulation Mapping**: Map zoning requirements directly onto the BIM model.
## Height Violation Detection
The application can detect height violations through several methods:
1. **Drone Photogrammetry**: Create 3D point clouds from drone imagery to measure precise building heights.
2. **BIM Model Analysis**: Extract exact height information from BIM models.
3. **Shadow Analysis**: Use satellite imagery to analyze shadows and calculate approximate building heights.
4. **Manual Input Verification**: Compare user-provided height measurements against zoning restrictions.
## Building Type Classification
The system classifies buildings into types using multiple approaches:
- **Visual Pattern Recognition**: AI analysis of imagery to identify residential, commercial, industrial, or mixed-use structures.
- **Floor Plan Analysis**: Examine interior layouts to determine building usage.
- **BIM Metadata**: Extract building type information from BIM model properties.
- **Regulation Cross-Reference**: Compare building characteristics against zoning definitions of different building categories.
## Coverage Area Ratio Analysis
Coverage Area Ratio (CAR) is automatically calculated to ensure compliance:
- **Building Footprint Detection**: Automatically trace building outlines from satellite or drone imagery.
- **Plot Boundary Detection**: Define legal property boundaries from survey data.
- **Ratio Calculation**: Calculate percentage of plot area covered by buildings.
- **Multi-Building Analysis**: Account for all structures on a single plot.
- **Compliance Verification**: Compare calculated ratio against maximum allowed coverage in zoning regulations.
