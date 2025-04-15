# Smart Urban Analyzer

A sophisticated urban planning analysis platform that automates construction setback compliance checks using AI, geospatial data processing, and multi-source data integration.



## Overview

Smart Urban Analyzer is a comprehensive GUI application designed for urban planners, architects, civil engineers, municipalities, and property owners worldwide. It imports regulations from technical documents, analyzes land plots using multiple data collection methods, and alerts about violations of construction permits and zoning regulations, with specific emphasis on setback requirements.

The application is built for global use with flexible data integration capabilities. The Qatar GIS Portal integration (detailed below) serves as a case study demonstrating the application's ability to connect with region-specific geospatial platforms.

## Key Features

- **AI-Powered Document Analysis**:
  - Extract regulations and setback requirements from PDF documents
  - Natural language Q&A about complex urban regulations=
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
  
- **Interactive User Guide**:
  - Guided tours for key application features
  - Contextual tooltips for enhanced usability
  - Step-by-step instructions for complex workflows
  - Comprehensive help documentation
  - Interactive FAQ and reference materials

- **Compliance Analysis**:
  - Automated setback requirement verification
  - Building height violation detection
  - Plot coverage area ratio calculation
  - Comprehensive compliance reporting with visualizations
  - Adaptable to any region's zoning regulations
  - Monitoring and violation detection system
  - Scheduled property compliance checks
  - Email and SMS alerts for violations

- **Advanced Visualization**:
  - Interactive 2D and 3D plot visualizations
  - Regulatory vs. actual setback comparisons
  - Folium-based interactive maps
  - Technical drawing visualization and annotation

- **Document Processing**:
  - OCR for text extraction from images and PDFs
  - Drawing extraction from technical documents
  - Dimension detection from CAD files and technical drawings
   # Advanced Features
### Offline Mode
The Smart Urban Analyzer includes a sophisticated caching system that:
1. Stores API responses for future use
2. Learns patterns from prior questions and answers
3. Enables offline usage after initial training
This makes it ideal for field work where internet connectivity may be limited.
### Pattern Learning
The application learns from interactions to improve over time:
- Recognizes common question types
- Identifies setback patterns
- Stores term definitions
- Builds a knowledge base of document sections

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install additional packages for BIM functionality
pip install ifcopenshell
```

## Running the Application

```bash
# Launch the Smart Urban Analyzer application
streamlit run Smart_Urban_Analyzer.py
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
  -  Notification Services (optional, for alerts and reports):
  - SMTP Settings (for email notifications):
    - `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SENDER_EMAIL`
  - Twilio Settings (for SMS notifications):
    - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`

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

### Urban Planning Chatbot
- Interactive conversational AI for urban planning questions
- Multiple AI provider support (OpenAI, DeepSeek, Anthropic)
- Context-aware responses based on uploaded documents
- Guidance on regulations, setbacks, and compliance requirements
- Example prompts for common urban planning questions

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

### Interactive Help & Guide System
- Guided tours for application features (Setback Analysis, GIS Analysis, Chatbot Usage)
- Step-by-step instructions with animated highlights
- Context-sensitive tooltips and explanations
- Comprehensive documentation with example workflows
- Tour navigation controls (Next/Previous/End tour)
### Monitoring & Notification System
- Property registration and tracking
- Scheduled monitoring with customizable frequency (daily, weekly, monthly, quarterly)
- Automated setback violation detection
- Configurable violation threshold settings
- Multi-channel notifications (email, SMS, in-app)
- Comprehensive violation reports with visual evidence
- Violation history and status tracking
- Integration with email (SMTP) and SMS (Twilio) services


## Bilingual Support
The application supports both English and Arabic interfaces, with comprehensive translations for all features and functionality.

## Developer Documentation

### Project Structure
- `Smart_Urban_Analyzer.py`: Main application file with Streamlit interface
- `utils/`: Utility modules for different functionalities
  - `gis_tools.py`: GIS processing and visualization (Qatar implementation as reference)
  - `pdf_processor.py`: PDF document handling and text extraction
  - `ocr_tools.py`: Optical character recognition for images and documents
  - `dimension_detector.py`: Automatic dimension detection from drawings
  - `compliance_checker.py`: Regulatory compliance verification
  - `deepseek_tools.py`: DeepSeek AI integration
  - `plot_handler.py`: Plot visualization and rendering
  - `guide_system.py`: Interactive tour and help system
  - `satellite_imagery.py`: Satellite imagery fetching and processing
  - `notification_system.py`: Email and SMS notification services
  - `monitoring_scheduler.py`: Scheduled monitoring and violation detection
  - `cache_system.py`: Response caching and pattern learning for AI optimization

### AI Integration
The application supports multiple AI providers with automatic fallback mechanisms:
- Primary: OpenAI API (fallback option)
- Optional: Anthropic Claude (for specialized tasks)
- Fallback: Rule-based processing without AI when API keys are unavailable

### Extension Points
- Add new data collection methods by extending the appropriate tab
- Implement additional compliance checks in `compliance_checker.py`
- Add support for new file formats in the relevant processor modules
- Extend the notification system with new channels in `notification_system.py`
- Customize monitoring schedules and thresholds in `monitoring_scheduler.py`
- Add new satellite imagery providers in `satellite_imagery.py`
- Implement additional AI service providers in the AI integration modules

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the interactive web framework
- OpenAI and Anthropic for AI capabilities
- LangChain for document processing and embeddings
- OpenCV and NumPy for image processing
- Folium for interactive maps
- PaddleOCR and Tesseract for optical character recognition
