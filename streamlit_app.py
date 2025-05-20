import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image, ImageEnhance
import io
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict
import os
from dotenv import load_dotenv
import logging
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded from .env file")

# Set page configuration
st.set_page_config(
    page_title="🕊️Marriage Bio Analyser Buddy",
    page_icon="💕",
    layout="wide"
)

# Define directories for storing inputs
PDF_DIR = "pdfs"
URL_DIR = "urls"
IMAGE_DIR = "images"
TEXT_DIR = "texts"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(URL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Define the state for LangGraph
class BiodataState(TypedDict):
    pdf_texts: List[Dict[str, Optional[str]]]
    web_texts: List[Dict[str, Optional[str]]]
    image_texts: List[Dict[str, Optional[str]]]
    text_inputs: List[Dict[str, Optional[str]]]
    user_query: str
    llm_response: Optional[str]
    api_key: Optional[str]
    llm_model: Optional[str]

# Function to sanitize filenames
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-\.]', '_', name)

# Function to load Gemini credentials
def load_credentials():
    """Load Gemini API key and model from credentials.json with environment variable fallback"""
    try:
        credentials_file = "credentials.json"
        if not os.path.exists(credentials_file):
            logger.info("credentials.json file not found. Creating with default values.")
            default_credentials = {
                "gemini_api_key": os.getenv("GOOGLE_API_KEY", ""),
                "llm_model": "gemini-2.0-flash"
            }
            with open(credentials_file, 'w') as f:
                json.dump(default_credentials, f, indent=2)
            return default_credentials["gemini_api_key"], default_credentials["llm_model"]

        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
            api_key = credentials.get("gemini_api_key", os.getenv("GOOGLE_API_KEY", ""))
            llm_model = credentials.get("llm_model", "gemini-2.0-flash")
            if not api_key:
                logger.warning("Gemini API key not found in credentials.json or environment variable")
            return api_key, llm_model
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
        return "", "gemini-2.0-flash"

# Function to initialize configuration
def initialize_config():
    """Initialize API key and LLM model, storing in session state"""
    if not hasattr(st.session_state, 'config_initialized') or not st.session_state.config_initialized:
        api_key, llm_model = load_credentials()
        st.session_state.api_key = api_key
        st.session_state.llm_model = llm_model
        st.session_state.config_initialized = True
        logger.info(f"Configuration initialized: API key={'set' if api_key else 'not set'}, model={llm_model}")
    return st.session_state.api_key, st.session_state.llm_model

# Function to save PDF and return metadata
def save_pdf(pdf_file, index: int) -> Dict[str, Optional[str]]:
    try:
        original_name = pdf_file.name
        filename = sanitize_filename(f"pdf_{index}_{original_name}")
        path = os.path.join(PDF_DIR, filename)
        with open(path, "wb") as f:
            f.write(pdf_file.getbuffer())
        logger.info(f"PDF {index} saved to {path} (overwritten if existed)")
        return {"path": path, "original_name": original_name}
    except Exception as e:
        logger.error(f"Error saving PDF {index}: {e}")
        st.error(f"Error saving PDF {index}: {e}")
        return {"path": None, "original_name": original_name}

# Function to save URL and scraped content, return metadata
def save_url(url: str, index: int) -> Dict[str, Optional[str]]:
    try:
        filename = sanitize_filename(f"url_{index}_{url.replace('https://', '').replace('http://', '')}")
        url_path = os.path.join(URL_DIR, f"{filename}.txt")
        content_path = os.path.join(URL_DIR, f"{filename}_content.txt")
        with open(url_path, "w", encoding="utf-8") as f:
            f.write(url)
        logger.info(f"URL {index} saved to {url_path} (overwritten if existed)")
        metadata = {"path": url_path, "original_name": url, "content_path": content_path}
        return metadata
    except Exception as e:
        logger.error(f"Error saving URL {index}: {e}")
        st.error(f"Error saving URL {index}: {e}")
        return {"path": None, "original_name": url, "content_path": None}

# Function to save image and return metadata
def save_image(image_file, index: int) -> Dict[str, Optional[str]]:
    try:
        original_name = image_file.name
        filename = sanitize_filename(f"image_{index}_{original_name}")
        path = os.path.join(IMAGE_DIR, filename)
        with open(path, "wb") as f:
            f.write(image_file.getbuffer())
        logger.info(f"Image {index} saved to {path} (overwritten if existed)")
        return {"path": path, "original_name": original_name}
    except Exception as e:
        logger.error(f"Error saving image {index}: {e}")
        st.error(f"Error saving image {index}: {e}")
        return {"path": None, "original_name": original_name}

# Function to save text input and return metadata
def save_text(text_input: str, index: int) -> Dict[str, Optional[str]]:
    try:
        original_name = f"text_input_{index}"
        filename = sanitize_filename(f"text_{index}_{original_name}.txt")
        path = os.path.join(TEXT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text_input)
        logger.info(f"Text input {index} saved to {path} (overwritten if existed)")
        return {"path": path, "original_name": original_name, "text": text_input}
    except Exception as e:
        logger.error(f"Error saving text input {index}: {e}")
        st.error(f"Error saving text input {index}: {e}")
        return {"path": None, "original_name": original_name, "text": None}

# Function to extract text from PDF
def extract_pdf_text(pdf_file, metadata: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    try:
        logger.info(f"Extracting text from PDF: {metadata['path']}")
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        logger.info(f"PDF text extracted successfully from {metadata['path']}")
        metadata["text"] = text if text.strip() else None
        return metadata
    except Exception as e:
        logger.error(f"Error extracting PDF text from {metadata['path']}: {e}")
        st.error(f"Error extracting PDF text from {metadata['original_name']}: {e}")
        metadata["text"] = None
        return metadata

# Function to scrape text from a website and save content
def scrape_website(url: str, metadata: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    try:
        logger.info(f"Scraping website: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if text.strip():
            if metadata["content_path"]:
                with open(metadata["content_path"], "w", encoding="utf-8") as f:
                    f.write(text)
                logger.info(f"Scraped content saved to {metadata['content_path']} (overwritten if existed)")
            metadata["text"] = text
        else:
            metadata["text"] = None
        logger.info(f"Website text scraped successfully from {url}")
        return metadata
    except Exception as e:
        logger.error(f"Error scraping website {url}: {e}")
        st.error(f"Error scraping website {url}: {e}")
        metadata["text"] = None
        return metadata

# Function to perform OCR on images
def extract_image_text(image_file, metadata: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    try:
        logger.info(f"Extracting text from image: {metadata['path']}")
        image = Image.open(image_file)
        # Preprocess image for better OCR
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = image.convert('L')  # Convert to grayscale
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increase contrast
        text = pytesseract.image_to_string(image, lang='eng+hin+mar')
        logger.info(f"Image text extracted successfully from {metadata['path']}")
        metadata["text"] = text if text.strip() else None
        return metadata
    except Exception as e:
        logger.error(f"Error extracting image text from {metadata['path']}: {e}")
        st.error(f"Error extracting image text from {metadata['original_name']}: {e}")
        metadata["text"] = None
        return metadata

# Function to process inputs
def process_inputs(state: BiodataState) -> BiodataState:
    logger.info("Processing inputs")
    pdf_texts = state.get("pdf_texts", [])
    web_texts = state.get("web_texts", [])
    image_texts = state.get("image_texts", [])
    text_inputs = state.get("text_inputs", [])
    user_query = state.get("user_query", "")

    # Combine all available texts with labels
    combined_text = ""
    biodata_count = 1
    for i, pdf in enumerate(pdf_texts, 1):
        if pdf["text"]:
            combined_text += f"📄 Biodata {biodata_count} (PDF - {pdf['original_name']}):\n{pdf['text']}\n\n"
            biodata_count += 1
    for i, web in enumerate(web_texts, 1):
        if web["text"]:
            combined_text += f"🌐 Biodata {biodata_count} (Website - {web['original_name']}):\n{web['text']}\n\n"
            biodata_count += 1
    for i, img in enumerate(image_texts, 1):
        if img["text"]:
            combined_text += f"🖼️ Biodata {biodata_count} (Image - {img['original_name']}):\n{img['text']}\n\n"
            biodata_count += 1
    for i, txt in enumerate(text_inputs, 1):
        if txt["text"]:
            combined_text += f"📝 Biodata {biodata_count} (Text - {txt['original_name']}):\n{txt['text']}\n\n"
            biodata_count += 1

    if not combined_text.strip():
        combined_text = "No content extracted from provided inputs."

    # Prepare prompt for LLM
    prompt_template = ChatPromptTemplate.from_template(
        """You are Marriage Bio Analyser Buddy, a friendly and insightful assistant for analyzing marriage biodata. Your task is to process biodata from PDFs, URLs, images, and text inputs (in English, Hindi, or Marathi) and respond to the user's query. Provide clear, concise, and culturally sensitive answers based on the provided data.

**Instructions**:
1. Analyze the biodata to extract relevant details (e.g., name, age, education, occupation, family details).
2. Answer the user's query directly, using bullet points or headings for clarity.
3. If the query involves comparison (e.g., multiple biodatas), highlight key differences and similarities.
4. If data is missing or unclear, note it and provide a partial answer or suggest clarification.
5. Keep the tone friendly, professional, and respectful.

**User Query**:
{user_query}

**Biodata Content**:
{biodata_text}

**Output**:
Provide a clear, structured response in markdown format (use headings, bullet points, or tables as needed).
"""
    )
    prompt = prompt_template.format(
        user_query=user_query,
        biodata_text=combined_text
    )
    state["llm_response"] = prompt
    logger.info("Inputs processed and prompt prepared")
    return state

# Function to query the LLM
def query_llm(state: BiodataState) -> BiodataState:
    try:
        logger.info("Querying Google Gemini LLM")
        api_key = state.get("api_key")
        llm_model = state.get("llm_model")
        if not api_key or not llm_model:
            logger.error("API key or LLM model missing in state")
            state["llm_response"] = "Error: API key or LLM model not configured. Please set them in Admin Settings."
            return state
        llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=api_key,
            temperature=0.25
        )
        message = HumanMessage(content=state["llm_response"])
        response = llm.invoke([message])
        state["llm_response"] = response.content
        logger.info("LLM query successful")
        return state
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        state["llm_response"] = f"Error querying LLM: {e}"
        return state

# Define the LangGraph workflow
def create_workflow():
    workflow = StateGraph(BiodataState)
    workflow.add_node("process_inputs", process_inputs)
    workflow.add_node("query_llm", query_llm)
    workflow.set_entry_point("process_inputs")
    workflow.add_edge("process_inputs", "query_llm")
    workflow.add_edge("query_llm", END)
    return workflow.compile()

# Streamlit app
def main():
    # Initialize configuration
    api_key, llm_model = initialize_config()

    st.title("💕⃝🕊️ Marriage Bio Analyser Buddy")
    st.write(
        "📤 Upload multiple PDFs, provide website URLs, upload images, or enter text to analyze and compare marriage biodatas, then enter your query.")

    # Admin settings for Gemini configuration
    with st.sidebar:
        st.header("Admin Settings")
        if 'is_admin' not in st.session_state:
            st.session_state.is_admin = False
        if not st.session_state.is_admin:
            admin_password = st.text_input("Admin Password", type="password")
            if st.button("Login"):
                if admin_password == "SuperAdmin123!":  # Replace with secure password
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
        else:
            st.success("Logged in as Admin")
            if st.button("Logout"):
                st.session_state.is_admin = False
                st.rerun()
            with st.expander("Configure Gemini API"):
                current_api_key = getattr(st.session_state, 'api_key', '')
                current_model = getattr(st.session_state, 'llm_model', 'gemini-2.0-flash')
                available_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]
                new_api_key = st.text_input("Gemini API Key", value=current_api_key, type="password")
                new_model = st.selectbox("Select LLM Model", options=available_models, index=available_models.index(current_model) if current_model in available_models else 0)
                if st.button("Save Gemini Configuration"):
                    try:
                        credentials_file = "credentials.json"
                        credentials = {}
                        if os.path.exists(credentials_file):
                            with open(credentials_file, 'r') as f:
                                credentials = json.load(f)
                        credentials["gemini_api_key"] = new_api_key
                        credentials["llm_model"] = new_model
                        with open(credentials_file, 'w') as f:
                            json.dump(credentials, f, indent=2)
                        st.session_state.api_key = new_api_key
                        st.session_state.llm_model = new_model
                        st.session_state.config_initialized = False  # Force reinitialization
                        st.success("Gemini configuration updated!")
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error saving Gemini configuration: {e}")
                        st.error(f"Failed to save configuration: {e}")

    # Input fields
    st.subheader("📂 Upload Biodata")
    pdf_files = st.file_uploader("📄 Upload PDF Biodatas", type=["pdf"], accept_multiple_files=True)
    urls = st.text_area("🌐 Enter Website URLs (one per line, optional)",
                        placeholder="https://example.com\nhttps://another.com")
    image_files = st.file_uploader("🖼️ Upload Image Biodatas", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    text_inputs = st.text_area("📝 Enter Text Biodatas (one per line, optional)",
                               placeholder="Name: John, Age: 30\nName: Jane, Age: 28")
    user_query = st.text_area("❓ Enter Your Query",
                              placeholder="e.g., Analyse the candidate profile. Or let's compare the profile details of all candidates.")

    if st.button("🚀 Analyze"):
        if not user_query:
            st.error("Please provide a query ❓")
            logger.warning("Analysis attempted without a query")
            return
        if not api_key:
            st.error("Gemini API key not configured. Please set it in Admin Settings.")
            logger.warning("Analysis attempted without API key")
            return

        with st.spinner("Analyzing..."):
            # Initialize state with API key and model
            state = BiodataState(
                pdf_texts=[],
                web_texts=[],
                image_texts=[],
                text_inputs=[],
                user_query=user_query,
                llm_response=None,
                api_key=api_key,
                llm_model=llm_model
            )

            # Process and save PDFs
            if pdf_files:
                for i, pdf_file in enumerate(pdf_files, 1):
                    metadata = save_pdf(pdf_file, i)
                    if metadata["path"]:
                        metadata = extract_pdf_text(pdf_file, metadata)
                    state["pdf_texts"].append(metadata)

            # Process and save URLs
            if urls:
                url_list = [url.strip() for url in urls.split("\n") if url.strip()]
                for i, url in enumerate(url_list, 1):
                    metadata = save_url(url, i)
                    if metadata["path"]:
                        metadata = scrape_website(url, metadata)
                    state["web_texts"].append(metadata)

            # Process and save images
            if image_files:
                for i, image_file in enumerate(image_files, 1):
                    metadata = save_image(image_file, i)
                    if metadata["path"]:
                        metadata = extract_image_text(image_file, metadata)
                    state["image_texts"].append(metadata)

            # Process and save text inputs
            if text_inputs:
                text_list = [text.strip() for text in text_inputs.split("\n") if text.strip()]
                for i, text in enumerate(text_list, 1):
                    metadata = save_text(text, i)
                    state["text_inputs"].append(metadata)

            if not any([state["pdf_texts"], state["web_texts"], state["image_texts"], state["text_inputs"]]):
                st.error("No valid input provided. Please upload PDFs, provide URLs, upload images, or enter text 📤")
                logger.warning("No valid inputs provided for analysis")
                return

            # Execute workflow
            logger.info("Executing LangGraph workflow")
            workflow = create_workflow()
            result = workflow.invoke(state)

            # Display result
            st.subheader("📊 Analysis Result")
            if result["llm_response"]:
                st.markdown(result["llm_response"])
                logger.info("Analysis result displayed")
            else:
                st.error("No response from LLM 🚫")
                logger.error("No LLM response received")

if __name__ == "__main__":
    main()