import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from geopy.geocoders import Nominatim
import folium
from shapely.geometry import Polygon


# === PDF TEXT PROCESSING ===
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)


def ask_question(vectordb, query):
    docs = vectordb.similarity_search(query)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=query)


# === DRAWING EXTRACTION ===
def extract_drawings_from_pdf(pdf_path, output_folder="drawings"):
    os.makedirs(output_folder, exist_ok=True)
    drawings = []
    pdf = fitz.open(pdf_path)

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            filename = f"{output_folder}/page{page_index+1}_img{img_index+1}.{ext}"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            drawings.append(filename)
    return drawings


# === SETBACK EXTRACTION ===
def extract_setbacks(text):
    matches = re.findall(r"\b(front|rear|side)\s+setback.*?(\d+)\s?m\b", text.lower())
    return {m[0]: int(m[1]) for m in matches}


# === GEOLOCATION + PLOT MAP ===
def get_coordinates_from_address(address):
    geolocator = Nominatim(user_agent="setback_bot")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    return None


def generate_satellite_map(lat, lon, zoom=20):
    map_ = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")
    folium.Marker([lat, lon], tooltip="Site").add_to(map_)
    map_.save("site_map.html")
    return "site_map.html"


# === CHECK SETBACK VIOLATIONS ===
def check_setback_violation(plot_coords, setbacks):
    poly = Polygon(plot_coords)
    min_setback = min(setbacks.values()) if setbacks else 0
    buffer_distance = min_setback / 111  # Convert meters to degrees
    inner_poly = poly.buffer(-buffer_distance)  # Negative buffer for inner boundary
    return inner_poly.area <= 0


# === STREAMLIT APP ===
st.set_page_config(page_title="PDF Site Chatbot", layout="wide")
st.title("PDF Chatbot + Site Drawing & Setback Checker")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    # Save PDF locally
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    # Load text & chunks
    text = load_pdf("temp.pdf")
    chunks = split_text(text)
    vectordb = create_vectorstore(chunks)

    # Show extracted setback info
    setbacks = extract_setbacks(text)
    if setbacks:
        st.subheader("Setback Rules Found in PDF")
        st.write(setbacks)
    else:
        st.warning("No setbacks found in PDF.")

    # Drawing Extraction
    st.subheader("Extracted Drawings")
    drawings = extract_drawings_from_pdf("temp.pdf")
    if drawings:
        for img_path in drawings:
            st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
    else:
        st.info("No drawings/images found in the PDF.")

    # Ask Chatbot
    st.subheader("Ask Something About the PDF")
    query = st.text_input("Your question")
    if query:
        answer = ask_question(vectordb, query)
        st.write("Answer:", answer)

    # Site Check
    st.subheader("Site Address for Map and Setback Check")
    address = st.text_input("Enter site address (manual)")
    if address:
        coords = get_coordinates_from_address(address)
        if coords:
            map_file = generate_satellite_map(*coords)
            with open(map_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=500)

            # Dummy rectangular plot for demo
            lat, lon = coords
            plot_coords = [
                (lat, lon),
                (lat + 0.0001, lon),
                (lat + 0.0001, lon + 0.0001),
                (lat, lon + 0.0001)
            ]
            violation = check_setback_violation(plot_coords, setbacks)
            if violation:
                st.error("Setback Violation Detected!")
            else:
                st.success("Setback Compliant.")
        else:
            st.warning("Could not geolocate the address.")
