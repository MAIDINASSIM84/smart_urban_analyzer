# Smart Urban Analyzer

*Smart Urban Analyzer* is an AI-powered application that transforms how urban planners, architects, and municipalities assess compliance with urban planning regulations. It extracts rules from official documents and automatically evaluates proposed building designs using AI, computer vision, and BIM data.

---

### Key Features

- *Automated Rule Extraction* from PDF urban planning codes
- *Compliance Checking* for setbacks, height limits, floor area ratios, and more
- *Upload Support*: PDF documents, IFC files (BIM), AutoCAD files (DWG/DXF), and satellite images
- *Satellite Structure Detection* using AI-based image analysis
- *Visual and Textual Compliance Reports*
- *User-Friendly Interface* for professionals with no programming background

---

### How It Works

1. Upload PDF regulations or urban code documents.
2. Upload your building model (IFC, DWG, DXF) and optionally a satellite image.
3. The system extracts rules and analyzes your project.
4. It returns an intelligent report showing:
   - Which rules are satisfied
   - Where compliance issues exist
   - Visual illustrations (where available)

---

### Technologies Used

- *Python, **LangChain, **OpenAI API*
- *PyMuPDF, **ifcopenshell, **AutoCAD file parser*
- *Satellite image processing*
- *Streamlit for the interface*
- *PDF + BIM + Geospatial integration*

---

### Use Cases

- Urban development authorities
- Architects and civil engineers
- Construction permit approval systems
- Academic and smart city research projects

---

### License

This project is licensed under a *customized MIT License*:

- *Permitted*: Research, educational, personal, and non-commercial use
- *Prohibited: Any commercial use, paid redistribution, or publication in commercial journals (e.g., Elsevier) without **explicit written permission from the author*

For licensing inquiries: [mohamednassimmaidi@gmail.com]

---

### Citation

If you use this software in your research, please cite:

```bibtex
@software{maidi_Smart_Urban_Analyzer_2025,
  author       = {Mohamed Nassim Maidi},
  title        = {Smart Urban Analyzer: AI for Urban Regulation and BIM Compliance},
  year         = 2025,
  url          = {https://github.com/your-repo},
  note         = {Preprint published, under review}
}
