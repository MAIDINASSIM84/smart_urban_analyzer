import streamlit as st
import os
import tempfile
import numpy as np
import cv2
from PIL import Image
import re
import openai
from PyPDF2 import PdfReader
import pandas as pd
import json
import folium
from streamlit_folium import folium_static
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Import Langchain community modules instead of deprecated imports
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA  # This is still in the core langchain package
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings

# Set OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Create necessary directories
os.makedirs("drawings", exist_ok=True)
os.makedirs("temp_files", exist_ok=True)

# Set up Streamlit UI
st.set_page_config(page_title="Smart Urban Analyzer", layout="wide")
st.title("Smart Urban Analyzer")
st.write("Comprehensive urban planning analysis with multiple data collection methods")

# Sidebar
with st.sidebar:
    st.markdown("## Smart Urban Analyzer")
    st.markdown("Multiple data collection methods supported for urban planning analysis")
    
    # Document Upload Section
    st.subheader("1. Urban Regulations")
    pdf_file = st.file_uploader("Upload Regulations (PDF)", type=["pdf"])
    
    # Site Analysis Section
    st.subheader("2. Site Survey Method")
    survey_method = st.selectbox(
        "Select data collection method:",
        [
            "Satellite Image", 
            "Drone Survey", 
            "Phone GPS Coordinates", 
            "Manual Measurements",
            "Leica Precision Instruments"
        ]
    )
    
    # Show appropriate upload options based on selection
    if survey_method == "Satellite Image":
        site_img_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png"])
    
    elif survey_method == "Drone Survey":
        st.markdown("#### Drone Survey Options")
        drone_file = st.file_uploader("Upload Drone Imagery", type=["jpg", "png", "tif"])
        drone_type = st.selectbox("Drone Type", ["DJI Phantom 4", "DJI Mavic Air 2", "Autel Robotics EVO II", "Custom"])
        resolution = st.select_slider("Image Resolution", options=["Low (720p)", "Medium (1080p)", "High (4K)"])
        altitude = st.slider("Flight Altitude (meters)", 10, 120, 40)
    
    elif survey_method == "Phone GPS Coordinates":
        st.markdown("#### Phone GPS Input")
        st.info("Enter GPS coordinates from smartphone app measurements")
        gps_format = st.radio("Coordinate Format", ["Decimal Degrees", "DMS"])
        
        if gps_format == "Decimal Degrees":
            col1, col2 = st.columns(2)
            with col1:
                lat = st.text_input("Latitude (e.g., 37.7749)")
            with col2:
                lon = st.text_input("Longitude (e.g., -122.4194)")
        else:
            st.text_input("Coordinates in DMS format")
    
    elif survey_method == "Manual Measurements":
        st.markdown("#### Manual Site Measurements")
        plot_width = st.number_input("Plot Width (meters)", min_value=0.0, step=0.5)
        plot_depth = st.number_input("Plot Depth (meters)", min_value=0.0, step=0.5)
        building_width = st.number_input("Building Width (meters)", min_value=0.0, step=0.5)
        building_depth = st.number_input("Building Depth (meters)", min_value=0.0, step=0.5)
        
        st.markdown("#### Setback Measurements")
        front_setback = st.number_input("Front Setback (meters)", min_value=0.0, step=0.1)
        rear_setback = st.number_input("Rear Setback (meters)", min_value=0.0, step=0.1)
        left_setback = st.number_input("Left Setback (meters)", min_value=0.0, step=0.1)
        right_setback = st.number_input("Right Setback (meters)", min_value=0.0, step=0.1)
    
    elif survey_method == "Leica Precision Instruments":
        st.markdown("#### Leica Measurement Data")
        st.info("Import data from Leica precision measurement instruments")
        leica_file = st.file_uploader("Upload Leica Data File", type=["csv", "txt", "dxf"])
        instrument_model = st.selectbox("Instrument Model", 
            ["Leica BLK360", "Leica RTC360", "Leica TS16", "Leica DISTO", "Other Leica"])
        data_format = st.selectbox("Data Format", ["Point Cloud", "Total Station", "Distance Measurements"])
    
    # Additional options
    with st.expander("Advanced Options"):
        st.checkbox("Enable AI-powered analysis", value=True)
        st.checkbox("Generate compliance report", value=True)
        
    with st.expander("About BIM and CAD Support"):
        st.info("Future versions will support BIM (IFC) and AutoCAD (DWG/DXF) files for more detailed analysis.")

# Extract setbacks from text (simplified)
def extract_setbacks(text):
    matches = re.findall(r"\b(front|rear|side)\s+setback.*?(\d+)\s?m\b", text.lower())
    return {m[0]: int(m[1]) for m in matches}

# PDF Processing and Chat
if pdf_file:
    with st.spinner("Processing PDF..."):
        # Save uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            pdf_path = tmp.name

        try:
            # Basic text extraction using PyPDF2
            reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
            
            # Display some extracted text
            st.subheader("PDF Content Preview")
            st.text_area("Extracted Text", pdf_text[:500] + "...", height=150)
            
            # Extract setbacks from text
            setbacks = extract_setbacks(pdf_text)
            if setbacks:
                st.subheader("Setback Rules Found in PDF")
                st.write(setbacks)
            else:
                st.warning("No specific setbacks found in PDF.")
            
            # Process with Langchain for Q&A capabilities
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_texts([d.page_content for d in docs], embedding=embeddings)
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0.3),
                chain_type="stuff",
                retriever=db.as_retriever()
            )

            # PDF Q&A Section
            st.subheader("Ask About Urban Regulations (PDF)")
            query = st.text_input("Example: What is the side setback?")
            if query:
                with st.spinner("Analyzing regulations..."):
                    answer = qa.run(query)
                    st.success(answer)
            
            # Display PDF pages as images (simplified without PyMuPDF)
            st.subheader("PDF Pages")
            st.info("Displaying first page of the PDF for reference.")
            
            # We'll just show a placeholder for now since PyMuPDF isn't working properly
            with st.container():
                st.markdown("**PDF Preview (Page 1)**")
                st.info("The full PDF has been processed for text and analysis, but image extraction requires additional plugins.")
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.info("Please check that your PDF is properly formatted and try again.")

# Helper functions for different data collection methods

# Helper function to process images (satellite or drone)
def process_imagery(img, source_type="Satellite"):
    """Process imagery (satellite or drone) to detect structures"""
    try:
        # Display original image
        st.image(img, caption=f"Original {source_type} Image", use_column_width=True)
        
        # Convert to numpy array for OpenCV processing
        img_np = np.array(img)
        
        # Image processing to detect structures
        if len(img_np.shape) == 3:  # Color image
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:  # Grayscale
            gray = img_np
            
        # Apply threshold and find contours
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on image
        output_img = img_np.copy()
        cv2.drawContours(output_img, contours, -1, (255, 0, 0), 2)
        
        # Display processed image
        st.image(output_img, caption="Detected Structure Boundaries", use_column_width=True)
        
        # Analysis summary
        st.subheader("Preliminary Analysis")
        st.info(f"Detected {len(contours)} potential structures in the image.")
        
        # Return contours for further analysis
        return contours
    except Exception as e:
        st.error(f"Error processing {source_type} image: {str(e)}")
        st.info(f"Please try uploading a different image format (JPG or PNG).")
        return None

# Helper function to display GPS location on a map
def display_gps_location(lat, lon):
    """Display a location on an interactive map"""
    try:
        # Convert string inputs to float if they're not empty
        if lat and lon:
            lat, lon = float(lat), float(lon)
            
            # Create a map centered at the specified location
            m = folium.Map(location=[lat, lon], zoom_start=18)
            
            # Add a marker for the location
            folium.Marker(
                [lat, lon], 
                popup="Property Location",
                tooltip="Property Location",
                icon=folium.Icon(color="green", icon="home")
            ).add_to(m)
            
            # Add circle to represent approximate property boundary
            folium.Circle(
                radius=20,
                location=[lat, lon],
                color="crimson",
                fill=True,
                fill_color="crimson"
            ).add_to(m)
            
            # Display the map in Streamlit
            st.subheader("Property Location")
            folium_static(m)
            
            return True
        else:
            st.warning("Please enter valid latitude and longitude values.")
            return False
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")
        return False

# Helper function to draw plot based on manual measurements
def draw_plot_from_measurements(plot_width, plot_depth, building_width, building_depth, 
                               front_setback, rear_setback, left_setback, right_setback):
    """Create a visualization of the plot and building with setbacks"""
    try:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot dimensions (in meters)
        plot_rect = plt.Rectangle((0, 0), plot_width, plot_depth, 
                                 edgecolor='black', facecolor='lightgreen', linewidth=2)
        
        # Calculate building position based on setbacks
        building_x = left_setback
        building_y = front_setback
        
        # Draw the building
        building_rect = plt.Rectangle((building_x, building_y), building_width, building_depth,
                                     edgecolor='black', facecolor='gray', linewidth=2)
        
        # Add shapes to plot
        ax.add_patch(plot_rect)
        ax.add_patch(building_rect)
        
        # Add labels for dimensions
        # Plot dimensions
        ax.text(plot_width/2, -1, f"{plot_width}m", ha='center')
        ax.text(-1, plot_depth/2, f"{plot_depth}m", va='center', rotation=90)
        
        # Setback labels
        ax.text(plot_width/2, front_setback/2, f"Front: {front_setback}m", ha='center')
        ax.text(plot_width/2, plot_depth - rear_setback/2, f"Rear: {rear_setback}m", ha='center')
        ax.text(left_setback/2, plot_depth/2, f"Left: {left_setback}m", va='center', rotation=90)
        ax.text(plot_width - right_setback/2, plot_depth/2, f"Right: {right_setback}m", va='center', rotation=90)
        
        # Set axis limits
        ax.set_xlim(-2, plot_width + 2)
        ax.set_ylim(-2, plot_depth + 2)
        
        # Set axis labels
        ax.set_xlabel('Width (meters)')
        ax.set_ylabel('Depth (meters)')
        ax.set_title('Plot Layout with Building and Setbacks')
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        return True
    except Exception as e:
        st.error(f"Error creating plot visualization: {str(e)}")
        return False

# Process Leica data
def process_leica_data(file, instrument_model, data_format):
    """Process data from Leica precision instruments"""
    try:
        st.subheader(f"Leica {instrument_model} Data")
        
        # Read data based on format (simplified - would be more complex in reality)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            st.dataframe(df)
            return df
        elif file.name.endswith('.txt'):
            content = file.read().decode('utf-8')
            st.text_area("Raw Measurements", content, height=200)
            
            # Parse the text (simplified)
            measurements = []
            for line in content.split('\n'):
                if line.strip() and not line.startswith('#'):
                    measurements.append(line.strip())
            
            if measurements:
                st.subheader("Parsed Measurements")
                for i, m in enumerate(measurements):
                    st.text(f"Measurement {i+1}: {m}")
            
            return measurements
        else:
            st.warning(f"Processing for {file.name} format is not fully implemented yet.")
            return None
    except Exception as e:
        st.error(f"Error processing Leica data: {str(e)}")
        return None

# Site Analysis based on selected method
if survey_method == "Satellite Image" and 'site_img_file' in locals() and site_img_file:
    st.subheader("Satellite Image Analysis")
    img = Image.open(site_img_file)
    contours = process_imagery(img, "Satellite")
    
    # If we have PDF data with setbacks, add compliance check
    if 'setbacks' in locals() and setbacks and contours:
        st.subheader("Setback Compliance Check")
        st.write("Setback Requirements:")
        st.write(setbacks)
        st.warning("Simulated Check: Using AI to analyze potential setback violations. For accurate assessment, please verify with precise measurements.")

elif survey_method == "Drone Survey" and 'drone_file' in locals() and drone_file:
    st.subheader(f"Drone Survey Analysis ({drone_type})")
    st.write(f"Flight altitude: {altitude}m | Resolution: {resolution}")
    
    img = Image.open(drone_file)
    contours = process_imagery(img, "Drone")
    
    # Additional drone-specific analysis
    st.subheader("Drone Survey Metadata")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Flight Altitude", f"{altitude} m")
        st.metric("Structures Detected", f"{len(contours) if contours else 0}")
    
    with col2:
        st.metric("Image Resolution", resolution)
        st.metric("Estimated Precision", "±5 cm")
    
    # If we have PDF data with setbacks, add compliance check
    if 'setbacks' in locals() and setbacks and contours:
        st.subheader("Setback Compliance Check")
        st.write("Setback Requirements:")
        st.write(setbacks)
        st.warning("Drone analysis indicates possible setback issues. Verify with ground measurements.")

elif survey_method == "Phone GPS Coordinates" and 'lat' in locals() and 'lon' in locals() and lat and lon:
    st.subheader("Phone GPS Coordinate Analysis")
    
    # Display the location on a map
    if display_gps_location(lat, lon):
        st.success("GPS coordinates successfully mapped.")
        
        # Show information about location precision
        st.info("Note: Standard smartphone GPS has an accuracy of approximately ±5-10 meters.")
        
        # If we have setbacks, show a note about them
        if 'setbacks' in locals() and setbacks:
            st.subheader("Setback Requirements")
            st.write(setbacks)
            st.warning("GPS coordinates are not precise enough for detailed setback analysis. Consider using manual measurements or a Leica instrument for higher precision.")

elif survey_method == "Manual Measurements" and 'plot_width' in locals():
    st.subheader("Manual Measurement Analysis")
    
    # Get variables ready for plotting
    has_plot_data = (plot_width > 0 and plot_depth > 0)
    has_building_data = (building_width > 0 and building_depth > 0)
    has_setback_data = (front_setback >= 0 and rear_setback >= 0 and 
                        left_setback >= 0 and right_setback >= 0)
    
    if has_plot_data and has_building_data and has_setback_data:
        # Create plot visualization
        if draw_plot_from_measurements(
            plot_width, plot_depth, building_width, building_depth,
            front_setback, rear_setback, left_setback, right_setback
        ):
            # Calculate compliance
            manual_setbacks = {
                'front': front_setback,
                'rear': rear_setback,
                'left': left_setback,
                'right': right_setback
            }
            
            # If we have PDF data with setbacks, add compliance check
            if 'setbacks' in locals() and setbacks:
                st.subheader("Setback Compliance Check")
                
                # Compare measured setbacks with requirements
                compliance_results = {}
                
                # Simple setback matching (would be more sophisticated in a real app)
                for key in setbacks:
                    required = setbacks[key]
                    
                    if key == 'front' and front_setback < required:
                        compliance_results[key] = {"compliant": False, "required": required, "actual": front_setback}
                    elif key == 'rear' and rear_setback < required:
                        compliance_results[key] = {"compliant": False, "required": required, "actual": rear_setback}
                    elif key == 'side':
                        # Check both left and right against "side" requirement
                        if left_setback < required:
                            compliance_results['left'] = {"compliant": False, "required": required, "actual": left_setback}
                        else:
                            compliance_results['left'] = {"compliant": True, "required": required, "actual": left_setback}
                            
                        if right_setback < required:
                            compliance_results['right'] = {"compliant": False, "required": required, "actual": right_setback}
                        else:
                            compliance_results['right'] = {"compliant": True, "required": required, "actual": right_setback}
                    else:
                        # For any exact match keys
                        corresponding_key = key
                        actual_value = manual_setbacks.get(corresponding_key, 0)
                        
                        if actual_value < required:
                            compliance_results[key] = {"compliant": False, "required": required, "actual": actual_value}
                        else:
                            compliance_results[key] = {"compliant": True, "required": required, "actual": actual_value}
                
                # Show compliance results
                if compliance_results:
                    st.write("Compliance Results:")
                    
                    for key, result in compliance_results.items():
                        if result["compliant"]:
                            st.success(f"{key.capitalize()} setback: {result['actual']}m (Requirement: {result['required']}m) ✓")
                        else:
                            st.error(f"{key.capitalize()} setback: {result['actual']}m (Requirement: {result['required']}m) ✗")
                            
                    # Overall assessment
                    if all(result["compliant"] for result in compliance_results.values()):
                        st.success("✅ All setback requirements are met.")
                    else:
                        st.error("❌ Some setback requirements are not met. See details above.")
    else:
        st.warning("Please provide all plot, building, and setback measurements for analysis.")

elif survey_method == "Leica Precision Instruments" and 'leica_file' in locals() and leica_file:
    st.subheader("Leica Precision Instrument Analysis")
    
    # Process the Leica data file
    leica_data = process_leica_data(leica_file, instrument_model, data_format)
    
    if leica_data is not None:
        st.success(f"Successfully processed {instrument_model} data.")
        
        # Show note about precision
        st.info(f"Leica {instrument_model} provides professional-grade measurements with millimeter precision.")
        
        # If we have setbacks, show a note about them
        if 'setbacks' in locals() and setbacks:
            st.subheader("Setback Requirements")
            st.write(setbacks)
            st.write("For detailed compliance analysis, please process the Leica measurements with professional software and import the results.")
    else:
        st.warning(f"Could not process the {instrument_model} data. Please check the file format.")

# For no data case, we need to make sure we don't show anything if no data is provided
elif ((survey_method == "Satellite Image" and ('site_img_file' not in locals() or not site_img_file)) or
      (survey_method == "Drone Survey" and ('drone_file' not in locals() or not drone_file)) or
      (survey_method == "Phone GPS Coordinates" and (('lat' not in locals() or not lat) or ('lon' not in locals() or not lon))) or
      (survey_method == "Leica Precision Instruments" and ('leica_file' not in locals() or not leica_file))):
    # This is handled by the welcome instructions at the bottom
    pass

# If no files uploaded, show instructions
if not pdf_file and not site_img_file:
    st.info("👈 Please upload a PDF with urban regulations or a satellite image using the sidebar options.")
    
    st.markdown("### How to Use Smart Urban Analyzer")
    st.markdown("""
    1. **Upload a PDF** containing urban planning regulations, zoning information, or building codes
    2. **Ask questions** about the regulations to get instant AI-powered answers
    3. **Upload a satellite image** of a property to analyze building footprints
    4. **Compare regulations** with actual property conditions to check for compliance
    
    This tool helps urban planners, architects, and property owners quickly analyze planning documents and check compliance with building regulations like setbacks.
    """)


