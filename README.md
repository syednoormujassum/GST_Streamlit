# 🎁 Dedicated to My Beloved Father

*"With deepest gratitude, I dedicate this project to my father. May this  ChatBot serve as your trusted companion, effortlessly delivering the information you seek without the toil of searching through countless pages."*

— Syed Noor Mujassum

---

# GST ChatBot 🤖

A Retrieval-Augmented Generation (RAG) powered chatbot for Goods and Services Tax (GST) queries, built with Streamlit and modern AI technologies.

## 📋 Overview

This application allows users to ask questions about GST (Goods and Services Tax) regulations and get accurate answers based on official GST documents. The system uses advanced natural language processing to understand queries and retrieve relevant information from Karnataka GST Act and Rules documents.

## ✨ Features

- **Document Processing**: Automatic PDF text extraction and processing
- **Intelligent Chunking**: Smart text splitting with overlap for better context retention
- **Vector Embeddings**: Sentence transformer-based embeddings for semantic search
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate, context-aware responses
- **Web Interface**: Clean Streamlit UI for easy interaction
- **Source Attribution**: Shows document sources and confidence scores for transparency

## 🏗️ Architecture

```
PDF Documents → Text Extraction → Chunking → Embeddings → Vector Store → RAG Retrieval → LLM Response
```

### Core Components

1. **Data Loader** (`src/data_loader.py`): Loads and processes PDF documents
2. **Chunking Manager** (`src/chunking.py`): Splits documents into manageable chunks with embeddings
3. **Vector Store** (`src/vectorstore.py`): Manages ChromaDB for vector storage and retrieval
4. **RAG Retriever** (`src/search_retreiver.py`): Handles query processing and response generation
5. **Streamlit App** (`streamlit.py`): Web interface for user interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.14+
- Groq API key (for LLM responses)
- OpenAI API key (optional, for alternative LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd streamlit-gst
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  # Optional
   ```

4. **Run the application**
   ```bash
   # Process documents and create vector store
   python main.py

   # Launch Streamlit app
   streamlit run streamlit.py
   ```

## 📁 Project Structure

```
streamlit-gst/
├── data/                          # PDF documents directory
│   ├── Karnataka-GST-Act-2017.pdf
│   └── Karnataka-GST-rules-2017.pdf
├── src/                           # Source code
│   ├── data_loader.py            # PDF processing
│   ├── chunking.py              # Text chunking & embeddings
│   ├── vectorstore.py           # ChromaDB management
│   └── search_retreiver.py      # RAG retrieval logic
├── .env                          # Environment variables
├── main.py                       # Document processing pipeline
├── streamlit.py                  # Web application
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Required for LLM responses (get from [Groq Console](https://console.groq.com/))
- `OPENAI_API_KEY`: Optional, for OpenAI GPT models

### Model Configuration

The system uses:
- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **LLM**: `llama-3.1-8b-instant` (via Groq)
- **Vector Store**: ChromaDB with persistent storage

## 💡 Usage Examples

### Sample Queries

- "What is the definition of taxable person under GST?"
- "Explain the GST registration process"
- "What are the penalties for late GST filing?"
- "How to calculate GST on services?"

### Response Format

Each response includes:
- **Answer**: AI-generated response based on document context
- **Sources**: Document references with page numbers
- **Confidence Score**: Similarity score for retrieved content
- **Full Context**: Complete retrieved text chunks

## 🛠️ Development

### Adding New Documents

1. Place PDF files in the `data/` directory
2. Run `python main.py` to reprocess all documents
3. The vector store will be updated automatically

### Customizing Chunking

Modify parameters in `src/chunking.py`:
```python
chunking_size=1000,    # Characters per chunk
chunk_overlap=200      # Overlap between chunks
```

### Changing LLM Models

Update model configuration in `src/search_retreiver.py`:
```python
model='llama-3.1-8b-instant'  # Groq models
# or
model='gpt-4-turbo'           # OpenAI models
```

## 📊 Technical Details

### Dependencies

- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Embedding generation
- **PyMuPDF**: PDF text extraction
- **Streamlit**: Web application framework
- **Groq**: Fast LLM inference API

### Performance

- **Embedding Dimension**: 384 (MiniLM model)
- **Chunk Size**: 1000 characters with 200 overlap
- **Similarity Metric**: Cosine similarity
- **Top-K Retrieval**: Configurable (default: 5)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **My Father**: For being my inspiration and guiding light


## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the troubleshooting section below

## 🔍 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure GROQ_API_KEY is set correctly
   - Check API key permissions

2. **"No documents found"**
   - Verify PDF files exist in `data/` directory
   - Run `python main.py` to reprocess documents

3. **Memory issues**
   - Reduce chunk size in `src/chunking.py`
   - Process fewer documents at once

4. **Slow responses**
   - Use smaller embedding model
   - Reduce top-k retrieval results

---

**Built with ❤️ **