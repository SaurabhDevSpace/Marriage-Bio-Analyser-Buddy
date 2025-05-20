import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import io
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict
import os
from dotenv import load_dotenv
import logging
from langchain_openai import AzureChatOpenAI
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded from .env file")
deployment_name = os.getenv("deployment_name")
api_version = os.getenv("api_version")
openai_api_base = os.getenv("openai_api_base")
openai_api_key = os.getenv("openai_api_key")

# Define directories for storing inputs
PDF_DIR = "pdfs"
URL_DIR = "urls"
IMAGE_DIR = "images"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(URL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# Define the state for LangGraph
class BiodataState(TypedDict):
    pdf_texts: List[Dict[str, Optional[str]]]
    web_texts: List[Dict[str, Optional[str]]]
    image_texts: List[Dict[str, Optional[str]]]
    user_query: str
    llm_response: Optional[str]


# Function to sanitize filenames
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-\.]', '_', name)


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
        # Sanitize URL for filename
        filename = sanitize_filename(f"url_{index}_{url.replace('https://', '').replace('http://', '')}")
        url_path = os.path.join(URL_DIR, f"{filename}.txt")
        content_path = os.path.join(URL_DIR, f"{filename}_content.txt")

        # Save URL
        with open(url_path, "w", encoding="utf-8") as f:
            f.write(url)
        logger.info(f"URL {index} saved to {url_path} (overwritten if existed)")

        # Save scraped content (will be updated after scraping)
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
            # Save scraped content
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
        text = pytesseract.image_to_string(image, lang='eng+hin+mar')
        logger.info(f"Image text extracted successfully from {metadata['path']}")
        metadata["text"] = text if text.strip() else None
        return metadata
    except Exception as e:
        logger.error(f"Error extracting image text from {metadata['path']}: {e}")
        st.error(f"Error extracting image text from {metadata['original_name']}: {e}")
        metadata["text"] = None
        return metadata


# Function to setup Azure OpenAI LLM
def setup_llm():
    """Initialize and setup the LLM model."""
    logger.info("Setting up LLM with Azure OpenAI")
    try:
        llm = AzureChatOpenAI(
            azure_deployment=deployment_name,
            azure_endpoint=openai_api_base,
            api_key=openai_api_key,
            openai_api_version=api_version,
            temperature=1
        )
        logger.info("LLM setup successful")
        return llm
    except Exception as e:
        logger.error(f"Error setting up LLM: {str(e)}")
        st.error(f"Error setting up LLM: {str(e)}")
        return None


# Function to process inputs
def process_inputs(state: BiodataState) -> BiodataState:
    logger.info("Processing inputs")
    pdf_texts = state.get("pdf_texts", [])
    web_texts = state.get("web_texts", [])
    image_texts = state.get("image_texts", [])
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

    if not combined_text.strip():
        combined_text = "No content extracted from provided inputs."

    # Prepare prompt for LLM
    prompt = f"""Analyze the following marriage biodata information and respond to the user's query. Each biodata is labeled for clarity.

Biodata Information:
{combined_text}

User Query:
{user_query}

Provide a concise and relevant response based on the biodata and query. If the query involves comparison, compare the biodatas accordingly.
"""
    state["llm_response"] = prompt
    logger.info("Inputs processed and prompt prepared")
    return state


# Function to query the LLM
def query_llm(state: BiodataState) -> BiodataState:
    try:
        logger.info("Querying LLM")
        llm = setup_llm()
        if llm is None:
            state["llm_response"] = "Failed to initialize LLM."
            logger.error("LLM initialization failed")
            return state
        response = llm.invoke(state["llm_response"]).content
        state["llm_response"] = response
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
    st.title("💕⃝🕊️ Marriage Bio Analyser Buddy")
    st.write(
        "📤 Upload multiple PDFs, provide website URLs, or upload images to analyze and compare marriage biodatas, then enter your query.")

    # Input fields
    st.subheader("📂 Upload Biodata")
    pdf_files = st.file_uploader("📄 Upload PDF Biodatas", type=["pdf"], accept_multiple_files=True)
    urls = st.text_area("🌐 Enter Website URLs (one per line, optional)",
                        placeholder="https://example.com\nhttps://another.com")
    image_files = st.file_uploader("🖼️ Upload Image Biodatas", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    user_query = st.text_area("❓ Enter Your Query",
                              placeholder="e.g., Analyse the candidate profile. Or let's compare the profile details of all candidates.")

    if st.button("🚀 Analyze"):
        if not user_query:
            st.error("Please provide a query ❓")
            logger.warning("Analysis attempted without a query")
            return

        with st.spinner("Analyzing..."):
            # Initialize state
            state = BiodataState(
                pdf_texts=[],
                web_texts=[],
                image_texts=[],
                user_query=user_query,
                llm_response=None
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

            if not any([state["pdf_texts"], state["web_texts"], state["image_texts"]]):
                st.error("No valid input provided. Please upload PDFs, provide URLs, or upload images 📤")
                logger.warning("No valid inputs provided for analysis")
                return

            # Execute workflow
            logger.info("Executing LangGraph workflow")
            workflow = create_workflow()
            result = workflow.invoke(state)

            # Display result
            st.subheader("📊 Analysis Result")
            if result["llm_response"]:
                st.write(result["llm_response"])
                logger.info("Analysis result displayed")
            else:
                st.error("No response from LLM 🚫")
                logger.error("No LLM response received")


if __name__ == "__main__":
    main()