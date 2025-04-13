# Smart Urban Analyzer

A comprehensive tool for urban planning analysis with multiple data collection methods.

## Features

- **PDF Document Analysis**: Upload urban planning regulations PDFs and extract setback requirements automatically. Ask questions about the regulations using AI.

- **Multiple Survey Methods**:
  - **Satellite Image Analysis**: Upload and analyze satellite imagery to detect building structures.
  - **Drone Survey**: Process drone imagery with metadata about flight altitude and resolution.
  - **Phone GPS Coordinates**: Input GPS data from smartphone apps with interactive map visualization.
  - **Manual Measurements**: Enter plot and building dimensions with interactive visualization.
  - **Leica Precision Instruments**: Import and process data from professional survey equipment.

- **Compliance Checking**: Automatically compare building layouts against setback regulations to identify potential violations.

## Requirements

- Python 3.9+
- Streamlit
- OpenAI API key
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set your OpenAI API key in environment variables
4. Run the application: `streamlit run Smart_Urban_Analyzer_enhanced.py`

## Usage

1. Upload a PDF containing urban regulations
2. Select a site survey method in the sidebar
3. Provide the required input for your selected method
4. View the analysis results and compliance checks

## License

This code is provided for non-commercial use only. See LICENSE file for details.
