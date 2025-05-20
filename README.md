# 💕⃝🕊️ Marriage Bio Analyser Buddy

Marriage Biodata Analyser Buddy is a Streamlit-based web application that leverages Azure OpenAI to analyze marriage biodata from PDFs, websites, and images. It supports text extraction in English, Hindi, and Marathi, enabling users to summarize or compare biodata details based on custom queries.

## Features
- **Input Types**: Analyze biodata from PDFs, website URLs, and images (PNG, JPG, JPEG).
- **Multilingual Support**: Extract text from images in English, Hindi, and Marathi using OCR.
- **Flexible Queries**: Summarize details for a single biodata or compare multiple biodatas with user-defined queries.
- **User-Friendly Interface**: Simple UI with emoji-enhanced inputs and a loading spinner for analysis.

## Prerequisites
- Python 3.8+
- Tesseract OCR installed with language packs for English (`eng`), Hindi (`hin`), and Marathi (`mar`).
- Azure OpenAI account with valid credentials (`deployment_name`, `api_version`, `openai_api_base`, `openai_api_key`).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/saurabh1131/marriage-biodata-analyser.git
   cd marriage-biodata-analyser
   ```
2. Install system dependencies (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-mar
   ```
   For other OS, see [Tesseract installation instructions](https://github.com/tesseract-ocr/tesseract/wiki).
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with Azure OpenAI credentials:
   ```env
   deployment_name=your_azure_deployment_name
   api_version=your_api_version
   openai_api_base=your_azure_endpoint
   openai_api_key=your_azure_api_key
   ```

## Running the App Locally
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to `http://localhost:8501`.

## Deploying on Streamlit Cloud
1. Push the repository to GitHub, including `app.py`, `requirements.txt`, and `packages.txt`.
2. Log in to [Streamlit Cloud](https://share.streamlit.io/).
3. Create a new app, select your repository, and set `app.py` as the main script.
4. Add Azure OpenAI credentials in Streamlit Cloud’s "Secrets" settings:
   ```toml
   deployment_name = "your_azure_deployment_name"
   api_version = "your_api_version"
   openai_api_base = "your_azure_endpoint"
   openai_api_key = "your_azure_api_key"
   ```
5. Deploy the app and access it via the provided URL.

## How to Interact with the App
Marriage Biodata Analyser Buddy allows flexible analysis of biodata through a web interface. Below are the ways to interact with the app:

### 1. **Analyze a Single Biodata**
- **Purpose**: Summarize or extract specific details from one biodata.
- **How to Use**:
  - Upload one PDF, enter one website URL, or upload one image in the respective input fields.
  - Example inputs:
    - PDF: `biodata.pdf`
    - URL: `https://example.com/biodata`
    - Image: `bio_photo.jpg` (containing text in English, Hindi, or Marathi)
  - Enter a query, e.g., "Summarize the candidate's education and family details."
  - Click "🚀 Analyze" to get a response based on the single input.

### 2. **Compare Multiple Biodatas for a Single Person**
- **Purpose**: Analyze multiple sources (e.g., PDF, website, image) for one person to get a comprehensive summary.
- **How to Use**:
  - Upload multiple bio inputs for the same person, e.g., a PDF, a URL, and an image.
  - Example inputs:
    - PDF: `bio_candidate1.pdf`
    - URL: `https://candidate1-profile.com`
    - Image: `candidate1_bio_photo.jpg`
  - Enter a query, e.g., "Summarize all details for this candidate."
  - Click "🚀 Analyze" to get a combined analysis from all sources.

### 3. **Compare Multiple Biodatas for Multiple Candidates**
- **Purpose**: Compare details across different candidates (e.g., education, family, profession).
- **How to Use**:
  - Upload multiple PDFs, enter multiple URLs (one per line), and/or upload multiple images.
  - Example Bio inputs:
    - PDFs: `bio_candidate1.pdf`, `bio_candidate2.pdf`
    - URLs: `https://candidate1_info.com`, `https://candidate2_info.com`
    - Images: `bio_photo1.jpg`, `bio_photo2.jpg`
  - Enter a comparative query, e.g., "Compare the education and family details of all candidates."
  - Click "🚀 Analyze" to get a comparison across all biodatas.

### Example Queries
- **Single Biodata**: "What is the candidate's occupation and marital status?"
- **Single Person, Multiple Inputs**: "Combine details from all sources to describe the candidate’s background."
- **Multiple Candidates**: "Compare the educational qualifications and family backgrounds of all candidates."

## Notes
- Ensure Tesseract is installed with `eng`, `hin`, and `mar` language packs for image OCR.
- Azure OpenAI credentials must be valid and configured correctly.
- For Streamlit Cloud, use `packages.txt` to install Tesseract and language packs.
- The app supports multilingual text extraction but may require clear images for accurate OCR.

## Troubleshooting
- **Tesseract Error**: If you see "tesseract is not installed or it's not in your PATH," ensure Tesseract is installed and in the system PATH. For Streamlit Cloud, verify `packages.txt` is included.
- **Azure OpenAI Errors**: Check credentials in `.env` (local) or Streamlit Cloud secrets.
- **Logs**: Check console or Streamlit Cloud logs for detailed error messages.

## License
MIT License
