# Smart Urban Analyzer

A sophisticated urban planning analysis platform that automates construction setback compliance checks using AI, geospatial data processing, and multi-source data integration.

<p align="center">
  <img src="generated-icon.png" alt="Smart Urban Analyzer" width="250"/>
</p>

## Overview

Smart Urban Analyzer is a comprehensive GUI application designed for urban planners, architects, civil engineers, municipalities, and property owners worldwide. It imports regulations from technical documents, analyzes land plots using multiple data collection methods, and alerts about violations of construction permits and zoning regulations, with specific emphasis on setback requirements.

The application is built for global use with flexible data integration capabilities. The Qatar GIS Portal integration (detailed below) serves as a case study demonstrating the application's ability to connect with region-specific geospatial platforms.

## Key Features

- **AI-Powered Document Analysis**:
  - Extract regulations and setback requirements from PDF documents
  - Natural language Q&A about complex urban regulations
  - Multiple AI providers with fallback mechanisms (OpenAI, DeepSeek, Anthropic)
  - Automatic dimension detection from technical drawings

- **Multi-Source Data Integration**:
  - PDF document parsing and data extraction
  - CAD file processing (DWG, DXF) with dimension detection
  - BIM (IFC) model analysis for 3D building compliance
  - Satellite and drone imagery processing
  - GPS coordinates mapping and visualization
  - Survey equipment data import (Leica, Trimble, GPS tools)
  - Regional GIS Portal integration (Qatar GIS as implementation example)

- **Compliance Analysis**:
  - Automated setback requirement verification
  - Building height violation detection
  - Plot coverage area ratio calculation
  - Comprehensive compliance reporting with visualizations
  - Adaptable to any region's zoning regulations

- **Advanced Visualization**:
  - Interactive 2D and 3D plot visualizations
  - Regulatory vs. actual setback comparisons
  - Folium-based interactive maps
  - Technical drawing visualization and annotation

- **Document Processing**:
  - OCR for text extraction from images and PDFs
  - Drawing extraction from technical documents
  - Dimension detection from CAD files and technical drawings

## Demo Screenshots

<p align="center">
  <img src="screenshots/screenshot1.png" alt="Main Interface" width="45%"/>
  <img src="screenshots/screenshot2.png" alt="Setback Analysis" width="45%"/>
</p>

<p align="center">
  <img src="screenshots/screenshot3.png" alt="GIS Integration" width="45%"/>
  <img src="screenshots/screenshot4.png" alt="Compliance Report" width="45%"/>
</p>

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-urban-analyzer.git
cd smart-urban-analyzer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional packages for BIM functionality
pip install ifcopenshell
```

## Running the Application

```bash
streamlit run pdf_analyzer.py
```

## System Requirements

- Python 3.11+
- Streamlit 1.30.0+
- OpenCV and NumPy for image processing
- Matplotlib for data visualization
- AI APIs (optional, for enhanced features):
  - OpenAI API key (set as environment variable `OPENAI_API_KEY`)
  - DeepSeek API key (set as environment variable `DEEPSEEK_API_KEY`)
  - Anthropic API key (set as environment variable `ANTHROPIC_API_KEY`)

## Data Collection Methods

### GIS Portal Integration (Qatar Case Study)
- Integration with regional GIS portals (Qatar GIS Portal as case study)
- Interactive map with project coordinates
- GIS screenshot analysis for automatic setback detection
- Compliance checking against regional regulations
- Framework adaptable to other GIS platforms worldwide

### PDF Document Analysis
- Extract urban regulations and setback requirements
- Identify and process technical drawings
- Dimension detection from plans and elevations
- AI-powered question-answering for complex documents

### CAD and BIM Processing
- Import DWG and DXF files for dimension extraction
- Process IFC models for 3D building analysis
- Calculate building height, footprint, and volume
- Extract setbacks and measurements from technical files

### Satellite and Drone Imagery
- Process satellite imagery to detect structures
- Analyze drone survey data with geospatial coordinates
- Automatic building and plot boundary detection
- Setback measurement and verification

### Survey Equipment Data
- Import data from Leica precision instruments
- Process standard surveying equipment coordinates
- Convert technical measurements to usable dimensions
- Integrate with other data sources for comprehensive analysis

### Manual Measurements
- Input precise plot and building measurements
- Interactive visualization of setbacks and boundaries
- Coverage ratio calculation and compliance checking
- Export results for reports and documentation

## Bilingual Support
The application supports both English and Arabic interfaces, with comprehensive translations for all features and functionality.

## Developer Documentation

### Project Structure
- `pdf_analyzer.py`: Main application file with Streamlit interface
- `utils/`: Utility modules for different functionalities
  - `gis_tools.py`: GIS processing and visualization (Qatar implementation as reference)
  - `pdf_processor.py`: PDF document handling and text extraction
  - `ocr_tools.py`: Optical character recognition for images and documents
  - `dimension_detector.py`: Automatic dimension detection from drawings
  - `compliance_checker.py`: Regulatory compliance verification
  - `deepseek_tools.py`: DeepSeek AI integration
  - `plot_handler.py`: Plot visualization and rendering

### AI Integration
The application supports multiple AI providers with automatic fallback mechanisms:
- Primary: DeepSeek API (preferred for cost efficiency)
- Secondary: OpenAI API (fallback option)
- Optional: Anthropic Claude (for specialized tasks)
- Fallback: Rule-based processing without AI when API keys are unavailable

### Extension Points
- Add new data collection methods by extending the appropriate tab
- Implement additional compliance checks in `compliance_checker.py`
- Add support for new file formats in the relevant processor modules

## Future Development Roadmap

- [ ] Add support for more regional GIS portals
- [ ] Implement 3D visualization of building models
- [ ] Add support for more CAD/BIM file formats
- [ ] Enhance machine learning models for boundary detection
- [ ] Create API endpoints for integration with other systems
- [ ] Add support for additional languages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the interactive web framework
- OpenAI, DeepSeek, and Anthropic for AI capabilities
- LangChain for document processing and embeddings
- OpenCV and NumPy for image processing
- Folium for interactive maps
- PaddleOCR and Tesseract for optical character recognition

## Contact

Project Link: [https://github.com/yourusername/smart-urban-analyzer](https://github.com/MAIDINASSIM84/smart-urban-analyzer)
Email:mohamednassimmaidi@gmail.com
