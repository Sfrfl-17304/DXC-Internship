# DEMO VIDEO

https://drive.google.com/drive/folders/19uWQ29nO3WUz5xRy6sdLH9AB5l0uNJzD?usp=drive_link

# DXC HR Hybrid Search Engine

A sophisticated AI-powered recruitment tool that combines structured filtering with semantic search to help HR professionals find the perfect candidates from a database of CVs.

## Project Overview

This project implements a hybrid search system that:
- **Processes CVs** using AI to extract structured data (skills, experience, candidate names)
- **Provides natural language search** - describe your ideal candidate in plain English
- **Combines filtering and ranking** - uses MongoDB for structured queries and vector embeddings for semantic similarity
- **Offers manual overrides** - fine-tune search criteria with manual filters

## Architecture

- **Frontend**: Streamlit web application
- **Backend**: Python with LangChain framework
- **Database**: MongoDB for document storage and structured queries
- **AI Models**: 
  - Groq Llama3-70B for text extraction and analysis
  - Ollama mxbai-embed-large for vector embeddings
- **Search**: Hybrid approach combining structured filtering + semantic ranking

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Ollama (for embeddings)
- Groq API account

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DXC-Internship
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```bash
# .env file content
GROQ_API_KEY="your_groq_api_key_here"
MONGO_USERNAME="username"
MONGO_PASSWORD="password"
MONGO_DB="hr_dxc_database"
```

**How to get your Groq API Key:**
1. Go to [Groq Console](https://console.groq.com/)
2. Sign up/Login
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste it in your `.env` file

### 5. Install Ollama

**For Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**For Windows:**
Download and install from [ollama.ai](https://ollama.ai/)

**Pull the embedding model:**
```bash
ollama pull mxbai-embed-large
```

### 6. Start MongoDB

```bash
# Start MongoDB using Docker Compose
docker-compose up -d

# Verify MongoDB is running
docker ps
```

This will start:
- MongoDB on port 27017
- Mongo Express (web UI) on port 8081

## Project Structure

```
DXC-Internship/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── docker-compose.yml             # MongoDB setup
├── .env                           # Environment variables
├── README.md                      # This file
├── source_documents/              # Place CV files here
├── dxc_rag_pipeline/             # Core processing modules
│   ├── loaders.py
├── hybrid_search_mongodb/        # HR engine components
│   └── hr_engine/
│       └── cv_processor.py       # CV processing logic
└── Diagrams/                     # Documentation diagrams
    ├──HybridSearch.drawio
    └──HybridSearch.png
```

## Usage

### 1. Start the Application

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start the Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 2. Process CV Documents

1. **Upload CVs**: Place your CV files (PDF, DOCX) in the `source_documents/` folder
2. **Open the app**: Navigate to the sidebar "Admin Panel"
3. **Set folder path**: Enter `./source_documents` (or your custom path)
4. **Click "Process All CVs in Folder"**: The system will:
   - Extract text from each document
   - Use AI to identify candidate names, skills, and experience
   - Generate vector embeddings
   - Store everything in MongoDB

### 3. Search for Candidates

**Natural Language Search:**
```
"Senior Python developer with 5+ years experience in machine learning"
"React or Vue.js frontend developer"
"Data scientist with experience in TensorFlow and AWS"
```

**Manual Override Options:**
- Select specific skills from database dropdown
- Set experience range (min/max years)
- Choose AND/OR logic for skill matching

### 4. View Results

Results are displayed with:
- **Relevance score** (0.0 to 1.0) based on semantic similarity
- **Matching skills** highlighted
- **Full CV text** for detailed review
- **MongoDB query** details for transparency

## Advanced Configuration

### Custom CV Processing

Edit `hybrid_search_mongodb/hr_engine/cv_processor.py` to:
- Modify skill extraction prompts
- Add custom data fields
- Change embedding models

### Database Management

Access Mongo Express at `http://localhost:8081` to:
- View processed candidates
- Inspect collections
- Run custom queries

### Model Configuration

In `app.py`, you can modify:
- **LLM Model**: Change `model_name="llama3-70b-8192"` to other Groq models
- **Embedding Model**: Change `model="mxbai-embed-large"` to other Ollama models
- **Temperature**: Adjust AI creativity/consistency

## API Endpoints

The application uses:
- **Groq API**: For text analysis and skill extraction
- **Ollama API**: For generating embeddings (local)
- **MongoDB**: For data storage and retrieval

## Troubleshooting

### Common Issues

**MongoDB Connection Failed:**
```bash
# Check if MongoDB is running
docker ps

# Restart MongoDB
docker-compose down
docker-compose up -d
```

**Ollama Model Not Found:**
```bash
# Pull the required model
ollama pull mxbai-embed-large

# Check available models
ollama list
```

**Groq API Errors:**
- Verify your API key in `.env` file
- Check your Groq account quota
- Ensure internet connection for API calls

**Import Errors:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Check virtual environment is activated
which python  # Should point to your venv
```

### Performance Tips

- **Large CV Collections**: Process CVs in smaller batches
- **Slow Searches**: Reduce embedding dimensions or use smaller models
- **Memory Issues**: Restart the application periodically

## Features

### Current Features
- Natural language CV search
- AI-powered skill extraction
- Hybrid search (structured + semantic)
- Manual filter overrides
- Experience level detection
- Professional UI without emojis
- MongoDB query transparency
- Batch CV processing

### Future Enhancements
- [ ] REST API endpoints
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Custom skill taxonomies
- [ ] Interview scheduling integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is developed for DXC Technology internship purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console logs for error details
3. Ensure all prerequisites are properly installed
4. Verify environment variables are correctly set

---

**DXC Technology - Hybrid Search Engine for HR**  
*AI-Powered Recruitment Made Simple*
