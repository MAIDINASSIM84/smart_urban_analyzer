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
import math
import hashlib
from datetime import datetime
from shapely.geometry import Polygon, Point, LineString
import uuid

# Create necessary directories
os.makedirs("drawings", exist_ok=True)
os.makedirs("cached_responses", exist_ok=True)
os.makedirs("utils", exist_ok=True)
os.makedirs("temp_files", exist_ok=True)

# Import Langchain community modules
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Try to import BIM processing libraries
try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as Element
    HAS_IFC = True
except ImportError:
    HAS_IFC = False

# Cache System Implementation
class ResponseCache:
    def __init__(self, cache_dir="cached_responses"):
        """Initialize the response caching system
        
        Args:
            cache_dir (str): Directory to store cached responses
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, query, context=""):
        """Generate a unique key for caching based on query and context"""
        combined = (query + context).encode('utf-8')
        return hashlib.md5(combined).hexdigest()
    
    def get_cached_response(self, query, context=""):
        """Get a cached response if it exists"""
        cache_key = self._generate_cache_key(query, context)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    print(f"Cache hit for query: {query[:50]}...")
                    return cached_data
            except Exception as e:
                print(f"Error reading cache: {str(e)}")
        
        return None
    
    def cache_response(self, query, response, context=""):
        """Cache an API response for future offline use"""
        cache_key = self._generate_cache_key(query, context)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            cache_data = {
                "query": query,
                "context": context,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error caching response: {str(e)}")
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        try:
            all_files = os.listdir(self.cache_dir)
            cache_files = [f for f in all_files if f.endswith('.json')]
            
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
            
            # Get the 5 most recent cache entries
            entries = []
            for f in cache_files[:5]:
                try:
                    with open(os.path.join(self.cache_dir, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        entries.append({
                            "query": data["query"][:50] + "..." if len(data["query"]) > 50 else data["query"],
                            "timestamp": data["timestamp"]
                        })
                except:
                    pass
            
            return {
                "total_entries": len(cache_files),
                "total_size_kb": round(total_size / 1024, 2),
                "recent_entries": entries
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear all cached responses"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            for f in cache_files:
                os.remove(os.path.join(self.cache_dir, f))
            return len(cache_files)
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            return 0

# Pattern learning system that improves over time
class PatternLearner:
    def __init__(self, patterns_file="utils/learned_patterns.json"):
        """Initialize the pattern learning system"""
        self.patterns_file = patterns_file
        self.patterns = self._load_patterns()
    
    def _load_patterns(self):
        """Load learned patterns from file"""
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading patterns: {str(e)}")
        
        # Default patterns structure
        return {
            "setback_patterns": {},
            "term_definitions": {},
            "common_answers": {},
            "document_sections": {}
        }
    
    def _save_patterns(self):
        """Save learned patterns to file"""
        try:
            os.makedirs(os.path.dirname(self.patterns_file), exist_ok=True)
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self.patterns, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {str(e)}")
    
    def learn_from_response(self, query, response, document_context=""):
        """Learn patterns from API response"""
        query_lower = query.lower()
        
        # Learn setback patterns
        if "setback" in query_lower:
            setback_match = re.search(r"(\d+)\s*(?:meter|m|feet|ft)", response, re.IGNORECASE)
            if setback_match:
                setback_value = setback_match.group(1)
                setback_type = "unknown"
                for s_type in ["front", "rear", "side", "left", "right"]:
                    if s_type in query_lower:
                        setback_type = s_type
                        break
                
                if setback_type != "unknown":
                    self.patterns["setback_patterns"][setback_type] = setback_value
        
        # Learn common term definitions
        for term in ["setback", "coverage ratio", "height restriction", "zoning", "plot ratio"]:
            if term in query_lower and "what is" in query_lower:
                definition = response[:200] + "..."
                self.patterns["term_definitions"][term] = definition
        
        # Store common answers
        question_keywords = ["how", "what", "where", "when", "why"]
        if any(kw in query_lower for kw in question_keywords):
            first_word = query_lower.split()[0]
            if first_word in question_keywords:
                question_key = " ".join(query_lower.split()[:5])
                self.patterns["common_answers"][question_key] = response[:200] + "..."
        
        # Save the learned patterns
        self._save_patterns()
    
    def get_offline_response(self, query, context=""):
        """Get a response based on learned patterns without API"""
        query_lower = query.lower()
        
        # Check for setback query
        for setback_type, value in self.patterns["setback_patterns"].items():
            if setback_type in query_lower and "setback" in query_lower:
                return f"Based on previously learned patterns, the {setback_type} setback requirement is {value} meters."
        
        # Check for term definition
        for term, definition in self.patterns["term_definitions"].items():
            if term in query_lower and any(prefix in query_lower for prefix in ["what is", "define", "meaning of"]):
                return f"Based on previously learned patterns: {definition}"
        
        # Check for common answers
        for question_key, answer in self.patterns["common_answers"].items():
            words = query_lower.split()
            first_words = " ".join(words[:min(5, len(words))])
            if question_key.startswith(first_words[:10]):
                return f"Based on previously learned patterns: {answer}"
        
        return None

# Function to get an answer from cache, patterns or API
def get_answer(query, context="", force_api=False):
    """Get an answer either from cache, learned patterns, or API"""
    cache = ResponseCache()
    pattern_learner = PatternLearner()
    
    # Try to get response from cache first
    if not force_api:
        cached_response = cache.get_cached_response(query, context)
        if cached_response:
            return cached_response["response"], False
    
    # If not in cache, try to generate from learned patterns
    if not force_api:
        pattern_response = pattern_learner.get_offline_response(query, context)
        if pattern_response:
            return pattern_response, False
    
    # If we need to use the API
    if os.environ.get("OPENAI_API_KEY"):
        try:
            # Set up OpenAI client
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            # Create system message with context
            system_message = "You are an urban planning assistant. Provide helpful, accurate information about urban planning regulations."
            if context:
                system_message += f"\n\nAnalyze this context to answer the user's question: {context}"
            
            # Call OpenAI API
            completion = client.chat.completions.create(
                model="gpt-4o",  # Use the latest model
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0.2
            )
            
            response = completion.choices[0].message.content
            
            # Cache the response
            cache.cache_response(query, response, context)
            
            # Learn from the response
            pattern_learner.learn_from_response(query, response, context)
            
            return response, True
        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            return error_msg, True
    else:
        return "No API keys available and no matching cached responses or patterns found. Please provide an API key in the Settings tab.", False

# If we have OpenAI key in environment, set it up
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

if st.session_state.openai_api_key:
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    openai.api_key = st.session_state.openai_api_key

# Helper function to extract setbacks from text
def extract_setbacks(text):
    """Extract setback values from text using pattern matching"""
    setbacks = {}
    
    # Look for front setback
    front_pattern = re.compile(r'(?:front\s+setback|setback\s+from\s+front).*?(\d+(?:\.\d+)?)\s*m', re.IGNORECASE)
    front_match = front_pattern.search(text)
    if front_match:
        setbacks['front'] = float(front_match.group(1))
    
    # Look for rear setback
    rear_pattern = re.compile(r'(?:rear\s+setback|setback\s+from\s+rear).*?(\d+(?:\.\d+)?)\s*m', re.IGNORECASE)
    rear_match = rear_pattern.search(text)
    if rear_match:
        setbacks['rear'] = float(rear_match.group(1))
    
    # Look for side setbacks
    side_pattern = re.compile(r'(?:side\s+setback|setback\s+from\s+side).*?(\d+(?:\.\d+)?)\s*m', re.IGNORECASE)
    side_match = side_pattern.search(text)
    if side_match:
        setbacks['side'] = float(side_match.group(1))
    
    # Look for left/right specific setbacks
    left_pattern = re.compile(r'(?:left\s+setback|setback\s+from\s+left).*?(\d+(?:\.\d+)?)\s*m', re.IGNORECASE)
    left_match = left_pattern.search(text)
    if left_match:
        setbacks['left'] = float(left_match.group(1))
    
    right_pattern = re.compile(r'(?:right\s+setback|setback\s+from\s+right).*?(\d+(?:\.\d+)?)\s*m', re.IGNORECASE)
    right_match = right_pattern.search(text)
    if right_match:
        setbacks['right'] = float(right_match.group(1))
    
    return setbacks

# Function to process imagery (satellite or drone)
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

# Function to display GPS location on a map
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

# Function to draw plot based on manual measurements
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
        plt.text(plot_width/2, -1, f"{plot_width}m", ha='center')
        plt.text(-1, plot_depth/2, f"{plot_depth}m", va='center', rotation=90)
        
        # Label setbacks
        plt.text(plot_width/2, front_setback/2, f"Front: {front_setback}m", ha='center')
        plt.text(plot_width/2, plot_depth - rear_setback/2, f"Rear: {rear_setback}m", ha='center')
        plt.text(left_setback/2, plot_depth/2, f"Left: {left_setback}m", va='center', rotation=90)
        plt.text(plot_width - right_setback/2, plot_depth/2, f"Right: {right_setback}m", va='center', rotation=90)
        
        # Set axis properties
        plt.xlim(-2, plot_width + 2)
        plt.ylim(-2, plot_depth + 2)
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add title and labels
        plt.title('Plot and Building Measurements with Setbacks')
        plt.xlabel('Width (meters)')
        plt.ylabel('Depth (meters)')
        
        # Return the figure for display
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# Function to check coverage ratio
def check_coverage_ratio(building_area, plot_area, max_allowed=0.6):
    """Check if building coverage ratio complies with regulations"""
    if plot_area == 0:
        return 0, False, "Invalid plot area (zero)"
    
    ratio = building_area / plot_area
    complies = ratio <= max_allowed
    
    return ratio, complies, f"Building covers {ratio:.2%} of plot (max allowed: {max_allowed:.0%})"

# Function to check setback violations
def check_setback_requirements(actual_setbacks, required_setbacks):
    """Check if actual setbacks comply with required minimum setbacks"""
    violations = {}
    complies = True
    
    for side, required in required_setbacks.items():
        if side in actual_setbacks:
            actual = actual_setbacks[side]
            if actual < required:
                violations[side] = f"{side.capitalize()} setback: {actual}m (required: {required}m)"
                complies = False
    
    return complies, violations

# Set up the Streamlit UI
st.set_page_config(
    page_title="Smart Urban Analyzer",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .st-eb {
        border-radius: 8px;
    }
    .st-emotion-cache-16txtl3 h1 {
        color: #1c4b82;
    }
    .st-emotion-cache-16txtl3 h2 {
        color: #5390d9;
    }
    .arabic-text {
        font-family: 'Cairo', 'Arial', sans-serif;
        direction: rtl;
        text-align: right;
        margin: 0;
        color: #666;
        font-weight: normal;
        font-style: normal;
    }
    .arabic-title {
        font-family: 'Cairo', 'Arial', sans-serif;
        font-weight: bold;
        direction: rtl;
        text-align: right;
        font-size: 1.2em;
        margin: 5px 0;
        color: #5E8B7E;
    }
</style>
""", unsafe_allow_html=True)

# Title and intro
st.title("Smart Urban Analyzer")
st.write("Comprehensive urban planning analysis with multiple data collection methods")

# Create tabs for different sections of the application
tab1, tab2, tab3 = st.tabs(["Main", "Help", "Settings"])

with tab1:
    # Main content will be displayed here
    st.info("Please select data collection method from the sidebar")

with tab2:
    st.header("Help & Documentation")
    st.markdown("""
    ### Smart Urban Analyzer User Guide
    
    This application helps urban planners and architects analyze building regulations and check compliance
    with setback requirements, height restrictions, and other urban planning rules.
    
    #### Main Features:
    
    1. **PDF Analysis**: Upload urban planning regulations in PDF format and extract rules
    2. **Multiple Survey Methods**: Choose from satellite, drone, GPS, manual measurements, and more
    3. **Interactive Visualization**: View plots, maps, and visual representations of buildings and setbacks
    4. **AI-Powered Q&A**: Ask questions about regulations in plain language
    5. **Offline Mode**: Use the app without an internet connection after initial setup
    
    #### Getting Started:
    
    1. Upload a PDF containing urban regulations using the sidebar
    2. Select a survey method from the dropdown menu
    3. Input the required data for your chosen method
    4. View analysis results and compliance checks
    
    #### Offline Usage:
    
    The application includes a smart caching system that:
    
    - Saves API responses for future offline use
    - Learns patterns from prior questions and responses
    - Works without an internet connection once you've built up a cache
    
    To use offline mode effectively:
    1. First use the app with an API key connected
    2. Ask common questions you'll need answers to later
    3. The app will store these for later use even without an API key
    """)
    
    # Display common questions
    st.subheader("Common Questions")
    
    with st.expander("What are setbacks?"):
        st.markdown("""
        **Setbacks** are the minimum distances required between a building and property lines or other structures.
        
        They typically include:
        - **Front setback**: Distance from the front property line to the building
        - **Rear setback**: Distance from the rear property line to the building
        - **Side setbacks**: Distance from side property lines to the building
        
        Setbacks are mandatory in most urban planning codes to ensure proper spacing, ventilation, 
        and aesthetic continuity in neighborhoods.
        """)
    
    with st.expander("How do I check for setback compliance?"):
        st.markdown("""
        To check setback compliance:
        
        1. Upload a PDF with regulations first to extract required setbacks
        2. Then select a survey method to input actual measurements
        3. The application will automatically compare your measurements with requirements
        4. A compliance report will highlight any violations
        
        For most accurate results, use precise measurements from Leica instruments
        or detailed manual measurements.
        """)
    
    with st.expander("Can I use the app without an internet connection?"):
        st.markdown("""
        Yes! The Smart Urban Analyzer is designed to work offline after initial setup:
        
        1. First use the app with an API key connected
        2. Ask the common questions about your regulations
        3. The app caches responses and learns patterns
        4. Later, you can use it offline with previously cached responses
        
        This is especially useful for field work where internet connectivity may be limited.
        """)

with tab3:
    st.header("Application Settings")
    st.write("Manage API settings and cached responses")
    
    # API Key Settings
    st.subheader("API Settings")
    api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
    
    # Save API key to session state
    if api_key:
        st.session_state.openai_api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
        st.success("API key saved!")
    
    # Display current API key status
    key_available = os.environ.get("OPENAI_API_KEY") is not None
    
    if key_available:
        st.success("OpenAI API key is configured and ready to use")
    else:
        st.warning("OpenAI API key is not configured. Some features will use offline mode.")
    
    # Cache Management
    st.subheader("Response Cache")
    cache = ResponseCache()
    stats = cache.get_cache_stats()
    
    # Show cache statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cached Responses", stats.get("total_entries", 0))
    with col2:
        st.metric("Cache Size", f"{stats.get('total_size_kb', 0)} KB")
    
    # Cache actions
    if st.button("Clear Cache"):
        deleted = cache.clear_cache()
        st.success(f"Cleared {deleted} cached responses")
    
    # Recent entries
    if "recent_entries" in stats and stats["recent_entries"]:
        st.subheader("Recent Cache Entries")
        for entry in stats["recent_entries"]:
            st.markdown(f"**Query:** {entry['query']}")
            st.markdown(f"*Cached on: {entry['timestamp']}*")
            st.markdown("---")
    
    # Learn about offline mode
    with st.expander("About Offline Mode"):
        st.info("""
        The Smart Urban Analyzer implements an offline response system that:
        
        1. Caches every API response for future use
        2. Learns patterns from responses to identify setbacks, terms, and common answers
        3. Can generate responses without API calls based on learned patterns
        
        This allows the application to function without an internet connection or API key
        after you've used it with an API key initially.
        """)

# Sidebar for inputs
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
            "Leica Precision Instruments",
            "BIM Model (IFC)"
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
    
    elif survey_method == "BIM Model (IFC)":
        st.markdown("#### BIM Model Analysis")
        if HAS_IFC:
            st.info("Import Building Information Model (IFC) files for comprehensive 3D analysis")
            ifc_file = st.file_uploader("Upload IFC File", type=["ifc"])
            
            # Additional BIM input options
            st.markdown("#### Building Details")
            building_height = st.number_input("Building Height (meters)", min_value=0.0, step=0.5, value=10.0)
            floors = st.number_input("Number of Floors", min_value=1, step=1, value=2)
            building_type = st.selectbox("Building Type", 
                ["Residential", "Commercial", "Industrial", "Mixed-Use", "Institutional", "Other"])
            
            # Plot information for coverage calculation
            st.markdown("#### Plot Information")
            plot_area = st.number_input("Total Plot Area (square meters)", min_value=0.0, step=10.0, value=500.0)
            max_coverage = st.slider("Maximum Allowed Coverage (%)", min_value=0, max_value=100, value=60)
            max_height = st.number_input("Maximum Allowed Height (meters)", min_value=0.0, step=0.5, value=12.0)
        else:
            st.error("IFC processing library (ifcopenshell) is not installed. Please install it to use BIM features.")
            st.info("To install ifcopenshell, run: pip install ifcopenshell")
    
    # Additional options
    with st.expander("Advanced Options"):
        st.checkbox("Enable AI-powered analysis", value=True)
        st.checkbox("Generate compliance report", value=True)
        
    with st.expander("About BIM and CAD Support"):
        st.info("This version supports BIM (IFC) files for 3D building analysis. Future versions will add support for AutoCAD (DWG/DXF) files.")

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
            
            # Process with Langchain for Q&A capabilities if available
            vectordb = None
            if LANGCHAIN_AVAILABLE and st.session_state.openai_api_key:
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    embeddings = OpenAIEmbeddings()
                    vectordb = FAISS.from_texts([d.page_content for d in docs], embedding=embeddings)
                except Exception as e:
                    st.warning(f"Could not set up PDF Q&A: {str(e)}")
            
            # PDF Q&A Section
            st.subheader("Ask About Urban Regulations (PDF)")
            query = st.text_input("Example: What is the side setback?")
            
            # Add a checkbox to allow forcing API use
            force_api = st.checkbox("Force using API (ignore cache)", value=False)
            
            if query:
                with st.spinner("Analyzing regulations..."):
                    # Check if we can use cached response
                    if st.session_state.openai_api_key and not force_api:
                        # Use our caching system for the response
                        response, used_api = get_answer(query, pdf_text, force_api)
                        
                        # Show source information
                        if used_api:
                            st.success(response)
                            st.info("Response generated using API and cached for future use")
                        else:
                            st.success(response)
                            st.info("Response retrieved from cache - no API call needed")
                    else:
                        # Use the standard Langchain approach if caching system not available
                        if vectordb:
                            qa = RetrievalQA.from_chain_type(
                                llm=ChatOpenAI(temperature=0.3),
                                chain_type="stuff",
                                retriever=vectordb.as_retriever()
                            )
                            answer = qa.run(query)
                            st.success(answer)
                        else:
                            # Fallback keyword search if no API key or vector DB
                            keywords = query.lower().split()
                            context_size = 100
                            excerpts = []
                            
                            for keyword in keywords:
                                if len(keyword) < 4:
                                    continue
                                
                                for match in re.finditer(rf'\b{re.escape(keyword)}\b', pdf_text.lower()):
                                    start = max(0, match.start() - context_size)
                                    end = min(len(pdf_text), match.end() + context_size)
                                    excerpt = pdf_text[start:end]
                                    excerpts.append(excerpt)
                            
                            if excerpts:
                                st.info("Found the following relevant passages (AI analysis not available without API key):")
                                for i, excerpt in enumerate(excerpts[:3]):
                                    st.text_area(f"Excerpt {i+1}", excerpt, height=100)
                            else:
                                st.warning("No relevant information found in the document.")
            
            # Display PDF pages as images
            st.subheader("PDF Pages")
            st.info("Displaying first page of the PDF for reference.")
            
            with st.container():
                st.markdown("**PDF Preview (Page 1)**")
                st.info("The full PDF has been processed for text and analysis, but image extraction requires additional plugins.")
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.info("Please check that your PDF is properly formatted and try again.")
            
# Handle the Manual Measurements survey method
if survey_method == "Manual Measurements" and plot_width > 0 and plot_depth > 0:
    st.header("Manual Measurements Analysis")
    
    # Check if building dimensions are provided
    if building_width > 0 and building_depth > 0:
        # Calculate building area
        building_area = building_width * building_depth
        plot_area = plot_width * plot_depth
        
        # Display the measurements
        st.subheader("Plot and Building Dimensions")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Plot Width", f"{plot_width}m")
            st.metric("Plot Depth", f"{plot_depth}m")
            st.metric("Plot Area", f"{plot_area}m²")
        with cols[1]:
            st.metric("Building Width", f"{building_width}m")
            st.metric("Building Depth", f"{building_depth}m")
            st.metric("Building Area", f"{building_area}m²")
        
        # Create visual representation
        st.subheader("Plot Visualization")
        fig = draw_plot_from_measurements(
            plot_width, plot_depth, building_width, building_depth,
            front_setback, rear_setback, left_setback, right_setback
        )
        if fig:
            st.pyplot(fig)
            
            # Download option
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            st.download_button(
                label="Download Plot Diagram",
                data=buf,
                file_name=f"plot_diagram_{uuid.uuid4().hex[:8]}.png",
                mime="image/png"
            )
        
        # Check compliance with setbacks
        if 'setbacks' in locals() and setbacks:
            st.subheader("Setback Compliance Check")
            
            # Create a dictionary of actual setbacks
            actual_setbacks = {
                'front': front_setback,
                'rear': rear_setback,
                'left': left_setback,
                'right': right_setback
            }
            
            # Check compliance
            complies, violations = check_setback_requirements(actual_setbacks, setbacks)
            
            if complies:
                st.success("✅ All setbacks comply with regulations")
            else:
                st.error("❌ Setback violations detected:")
                for violation in violations.values():
                    st.warning(violation)
            
            # Check coverage ratio if max_coverage is defined
            if 'max_coverage' in locals():
                ratio, ratio_complies, message = check_coverage_ratio(
                    building_area, plot_area, max_coverage / 100)
                
                st.subheader("Coverage Ratio Check")
                
                if ratio_complies:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        else:
            st.info("Upload a PDF with regulations first to check setback compliance")
    else:
        st.warning("Please enter building dimensions to see analysis")

# Handle Satellite or Drone Image processing
if (survey_method == "Satellite Image" and 'site_img_file' in locals() and site_img_file) or \
   (survey_method == "Drone Survey" and 'drone_file' in locals() and drone_file):
    
    st.header(f"{survey_method} Analysis")
    
    # Process either satellite or drone image
    img_file = site_img_file if survey_method == "Satellite Image" else drone_file
    source_type = "Satellite" if survey_method == "Satellite Image" else "Drone"
    
    # Open image with PIL
    img = Image.open(img_file)
    
    # Process image to detect structures
    contours = process_imagery(img, source_type)
    
    # Display additional drone info if applicable
    if survey_method == "Drone Survey" and 'drone_type' in locals() and 'altitude' in locals():
        st.subheader("Drone Survey Parameters")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Drone Model", drone_type)
        with cols[1]:
            st.metric("Flight Altitude", f"{altitude}m")
        with cols[2]:
            st.metric("Resolution", resolution)
        
        # Calculate approximate GSD (Ground Sample Distance)
        if drone_type == "DJI Phantom 4":
            # Simple GSD calculation (cm/pixel)
            gsd = altitude * 0.24  # Approx for Phantom 4
            st.info(f"Estimated Ground Sample Distance: {gsd:.2f} cm/pixel")

# Handle GPS coordinates
if survey_method == "Phone GPS Coordinates" and 'lat' in locals() and 'lon' in locals() and lat and lon:
    st.header("GPS Location Analysis")
    # Display the map with the provided coordinates
    display_gps_location(lat, lon)

# Handle Leica Precision Instruments data
if survey_method == "Leica Precision Instruments" and 'leica_file' in locals() and leica_file:
    st.header("Leica Precision Instrument Data Analysis")
    
    st.info(f"Processing {instrument_model} data in {data_format} format")
    
    # For demonstration, we'll just show the file info
    st.subheader("File Information")
    st.json({
        "filename": leica_file.name,
        "size": f"{leica_file.size/1024:.2f} KB",
        "instrument": instrument_model,
        "format": data_format
    })
    
    st.warning("Full Leica data processing requires specialized libraries. Basic file information shown.")

# Handle BIM Model (IFC) analysis
if survey_method == "BIM Model (IFC)" and HAS_IFC and 'ifc_file' in locals() and ifc_file:
    st.header("BIM Model Analysis")
    
    with st.spinner("Processing IFC file..."):
        # Save the IFC file
        temp_ifc_path = os.path.join("temp_files", f"temp_model_{uuid.uuid4().hex[:8]}.ifc")
        with open(temp_ifc_path, "wb") as f:
            f.write(ifc_file.read())
        
        try:
            # Load the IFC model
            ifc_model = ifcopenshell.open(temp_ifc_path)
            
            # Get basic model information
            spaces = ifc_model.by_type("IfcSpace")
            walls = ifc_model.by_type("IfcWall")
            slabs = ifc_model.by_type("IfcSlab")
            
            # Display model statistics
            st.subheader("Model Information")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Spaces", len(spaces))
            with cols[1]:
                st.metric("Walls", len(walls))
            with cols[2]:
                st.metric("Slabs/Floors", len(slabs))
            with cols[3]:
                st.metric("Total Elements", len(list(ifc_model.by_type("IfcElement"))))
            
            # Simplified BIM analysis - in a real app, this would be more detailed
            st.subheader("Building Compliance Analysis")
            
            # Calculate footprint area from BIM - simplified approach
            # In a real application, this would properly calculate the building footprint from geometry
            footprint_area = plot_area * 0.5  # Simplified estimation
            for space in spaces:
                if hasattr(space, "NetFloorArea") and space.NetFloorArea:
                    footprint_area += float(space.NetFloorArea)
            
            # Check coverage ratio
            ratio, complies, message = check_coverage_ratio(footprint_area, plot_area, max_coverage/100)
            
            if complies:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
            
            # Check height compliance
            if building_height > max_height:
                st.error(f"❌ Building height ({building_height}m) exceeds maximum allowed ({max_height}m)")
            else:
                st.success(f"✅ Building height ({building_height}m) complies with maximum allowed ({max_height}m)")
            
        except Exception as e:
            st.error(f"Error processing IFC file: {str(e)}")
            st.info("Please check that your IFC file is valid and try again.")
