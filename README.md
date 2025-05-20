# 💕⃝🕊️ Marriage Bio Analyser Buddy

Marriage Bio Analyser Buddy is a Streamlit-based web application that leverages Google Gemini to analyze marriage biodata from PDFs, websites, images, and plain text inputs. It supports text extraction in English, Hindi, and Marathi, enabling users to summarize or compare biodata details based on custom queries.

## Features
- **Multiple Input Types**: Analyze biodata from PDFs, website URLs, images (PNG, JPG, JPEG), and plain text inputs.
- **Multilingual Support**: Extract text from images and text inputs in English, Hindi, and Marathi using OCR and direct processing.
- **Flexible Queries**: Summarize details for a single biodata or compare multiple biodatas with user-defined queries.
- **User-Friendly Interface**: Intuitive UI with emoji-enhanced inputs, a loading spinner, and admin settings for Gemini API configuration.
- **LangGraph Workflow**: Modular processing pipeline using LangGraph for robust input handling and LLM queries.

## Prerequisites
- Python 3.8+
- Tesseract OCR installed with language packs for English (`eng`), Hindi (`hin`), and Marathi (`mar`).
- Google Gemini API key (obtainable from [Google AI Studio](https://aistudio.google.com/)).
- `.env` file or Streamlit Cloud secrets for storing the Gemini API key.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/marriage-biodata-analyser.git
   cd marriage-biodata-analyser
   ```
2. Install system dependencies (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-mar
   ```
   For macOS:
   ```bash
   brew install tesseract tesseract-lang
   ```
   For other OS, see [Tesseract installation instructions](https://github.com/tesseract-ocr/tesseract/wiki).
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
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
4. Add the Gemini API key in Streamlit Cloud’s "Secrets" settings:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key"
   ```
5. Ensure `packages.txt` includes:
   ```
   tesseract-ocr
   tesseract-ocr-eng
   tesseract-ocr-hin
   tesseract-ocr-mar
   ```
6. Deploy the app and access it via the provided URL.

## How to Interact with the App
Marriage Bio Analyser Buddy allows flexible analysis of biodata through a web interface. Below are the ways to interact with the app:

### 1. **Analyze a Single Biodata**
- **Purpose**: Summarize or extract specific details from one biodata.
- **How to Use**:
  - Upload one PDF, enter one website URL, upload one image, or enter one text biodata in the respective input fields.
  - Example inputs:
    - PDF: `biodata.pdf`
    - URL: `https://example.com/biodata`
    - Image: `bio_photo.jpg` (containing text in English, Hindi, or Marathi)
    - Text: `Name: John, Age: 30, Education: B.Tech`
  - Enter a query, e.g., "Summarize the candidate's education and family details."
  - Click "🚀 Analyze" to get a response based on the single input.

### 2. **Compare Multiple Biodatas for a Single Person**
- **Purpose**: Analyze multiple sources (e.g., PDF, website, image, text) for one person to get a comprehensive summary.
- **How to Use**:
  - Upload multiple bio inputs for the same person, e.g., a PDF, a URL, an image, and a text input.
  - Example inputs:
    - PDF: `bio_candidate1.pdf`
    - URL: `https://candidate1-profile.com`
    - Image: `candidate1_bio_photo.jpg`
    - Text: `Name: John, Age: 30, Occupation: Engineer`
  - Enter a query, e.g., "Summarize all details for this candidate."
  - Click "🚀 Analyze" to get a combined analysis from all sources.

### 3. **Compare Multiple Biodatas for Multiple Candidates**
- **Purpose**: Compare details across different candidates (e.g., education, family, profession).
- **How to Use**:
  - Upload multiple PDFs, enter multiple URLs (one per line), upload multiple images, and/or enter multiple text biodatas (one per line).
  - Example inputs:
    - PDFs: `bio_candidate1.pdf`, `bio_candidate2.pdf`
    - URLs: `https://candidate1_info.com`, `https://candidate2_info.com`
    - Images: `bio_photo1.jpg`, `bio_photo2.jpg`
    - Text: `Name: John, Age: 30\nName: Jane, Age: 28`
  - Enter a comparative query, e.g., "Compare the education and family details of all candidates."
  - Click "🚀 Analyze" to get a comparison across all biodatas.

### Example Queries
- **Single Biodata**: "What is the candidate's occupation and marital status?"
- **Single Person, Multiple Inputs**: "Combine details from all sources to describe the candidate’s background."
- **Multiple Candidates**: "Compare the educational qualifications and family backgrounds of all candidates."

## Notes
- Ensure Tesseract is installed with `eng`, `hin`, and `mar` language packs for image OCR.
- The Gemini API key must be valid; configure it via `.env`, Streamlit Cloud secrets, or admin settings.
- Text inputs are processed as-is; ensure they are well-formatted for best results.
- Mac screenshot images (PNG) are preprocessed for OCR (grayscale, contrast enhancement) to improve accuracy.

## License
MIT License
