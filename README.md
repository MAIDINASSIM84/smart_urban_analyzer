# Smart Urban Analyzer

*Smart Urban Analyzer* is a semi-automated, AI-powered application that analyzes urban planning regulations, building setbacks, heights, and land-use from PDF regulations and compares them to geospatial data from multiple sources. It's tailored for authorities, urban planners, and developers, offering compliance checks in real-time.

---

## Key Features

- Extracts urban planning rules from PDF regulations using NLP.
- Detects and evaluates site plots from:
  - Satellite imagery
  - Drone captures (uploaded manually)
  - Mobile phone GPS with image input
  - Manual uploads of CAD/IFC files
- Compares extracted site data against regulatory constraints (e.g., setbacks).
- Alerts on violations visually and via reports.
- Compatible with IFC, AutoCAD, and PDF documents.

---

## Supported Inputs

- *PDF regulations* (extracted using LangChain + PyPDF2)
- *Satellite imagery* via APIs
- *Drone images* (manual upload)
- *GPS & photos* from mobile devices
- *AutoCAD (.dwg/.dxf)* and *IFC (BIM)* file upload

---

## Use Cases

- Urban Planning Departments
- Municipalities (e.g., MME Qatar)
- Architecture & Engineering Firms
- Construction Compliance Checks
- Smart City Platforms

---

## Getting Started

1. Clone the repo
2. Set up virtual env & install requirements
3. Run the Streamlit app
4. Upload your site data (PDF + CAD + images)
5. View alerts and generate reports

---

## License

MIT License – See [LICENSE](LICENSE)

---

## Contact

For collaboration, licensing, or research use:  
*Mohamed Nassim Maidi*  
Email: mohamednassimmaidi@gmail.com
