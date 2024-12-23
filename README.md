# PDF Chat Application with Llama 3.3 and Groq

This application allows users to chat with their PDF documents using Llama 3.3 (70B) model via Groq API. It provides an intuitive interface for uploading PDFs and engaging in conversational Q&A about their contents.

## Features

- PDF document upload and processing
- Text extraction and chunking
- Semantic search using embeddings
- Conversational interface with context preservation
- Real-time responses using Groq's fast inference
- Multi-turn conversation support
- Caching for improved performance

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```
2. Open your web browser and navigate to the provided local URL
3. Upload your PDF documents using the sidebar
4. Click "Process PDFs" to extract and index the content
5. Start asking questions in the chat interface

## Requirements

- Python 3.8+
- Groq API key
- Internet connection for API calls

## Note

Ensure you have sufficient API credits and a stable internet connection for optimal performance. The application uses caching to improve response times for repeated queries. 