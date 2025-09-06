# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start the application:**
```bash
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Environment setup:**
- Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Uses Python 3.13+ with uv package manager

**Development workflow:**
- Server runs on http://localhost:8000 with auto-reload
- Frontend served statically from `/frontend` directory  
- Documents auto-loaded from `/docs` on startup
- No test framework configured - verify changes manually via web interface
- No linting/formatting tools configured

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) chatbot system** built around a tool-based AI architecture:

### Core Architecture Pattern
- **Tool-Based RAG**: Claude decides whether to search course content or use general knowledge based on query analysis
- **Session Management**: Conversation history maintained across interactions for context continuity
- **Document Processing Pipeline**: Structured parsing → chunking → embedding → storage

### Key Components Flow
1. **Document Processor** (`document_processor.py`) - Parses course files with expected format:
   ```
   Course Title: [title]
   Course Link: [url] 
   Course Instructor: [instructor]
   
   Lesson 0: Introduction
   [content]
   ```

2. **RAG System** (`rag_system.py`) - Central orchestrator that coordinates:
   - Session management for conversation context
   - AI generation with tool availability
   - Document loading and course analytics

3. **AI Generator** (`ai_generator.py`) - Anthropic Claude integration with:
   - Pre-built system prompt optimizing for educational content
   - Tool-based decision making (search vs direct answer)
   - One search per query maximum rule

4. **Search Tools** (`search_tools.py`) - Tool interface for Claude:
   - `search_course_content` tool with course/lesson filtering
   - Semantic course name matching
   - Source tracking for citations

5. **Vector Store** (`vector_store.py`) - ChromaDB integration:
   - Separate collections for course metadata vs content chunks  
   - Sentence transformers for embeddings (`all-MiniLM-L6-v2`)
   - Smart filtering by course title and lesson number

### Data Models
- **Course** - Contains title (unique ID), instructor, lessons list
- **Lesson** - Number, title, optional link
- **CourseChunk** - Content with course/lesson metadata for vector storage

### Configuration
All settings centralized in `config.py`:
- Chunk size: 800 characters with 100 char overlap
- Max search results: 5
- Conversation history: 2 messages
- ChromaDB path: `./chroma_db`

### Frontend Integration
- Static HTML/JS served from FastAPI
- Real-time query processing with loading states
- Session-based conversation continuity
- Markdown response rendering with collapsible sources

### Document Processing Specifics
- **Chunking Strategy**: Sentence-boundary aware with configurable overlap
- **Context Enhancement**: Chunks prefixed with `"Course [title] Lesson [X] content:"`
- **Metadata Extraction**: Regex-based parsing of course headers and lesson markers
- **Fallback Handling**: Treats entire document as single content if no lesson structure found

The system automatically loads documents from `/docs` on startup and provides `/api/query` for Q&A and `/api/courses` for analytics.

## API Endpoints
- `POST /api/query` - Process queries with optional session_id for conversation continuity
- `GET /api/courses` - Get course analytics (total courses, titles)
- `GET /` - Web interface (served from `/frontend`)
- `GET /docs` - FastAPI auto-generated API documentation

## Critical Implementation Details

**Session Management**: 
- Sessions auto-created if not provided in query requests
- Conversation history limited to 2 messages (configurable in `config.py`)
- Session state managed in-memory (not persistent across server restarts)

**Document Processing Pipeline**:
- Expected format: `Course Title:`, `Course Link:`, `Course Instructor:`, then `Lesson X:` sections
- Processes `.txt` files from `/docs` directory on startup
- Creates embeddings using `all-MiniLM-L6-v2` model
- Chunks text at 800 chars with 100 char overlap
- Two ChromaDB collections: course metadata + content chunks

**AI Tool Architecture**:
- Claude decides autonomously whether to search course content or answer directly
- One search per query maximum rule enforced in `ai_generator.py`
- Search tool filters by course title and lesson number
- Sources automatically tracked and returned with responses

**Configuration System**:
- All settings centralized in `backend/config.py`
- Environment variables loaded via python-dotenv
- ChromaDB storage location: `./chroma_db`

- Always use uv to run the server, do not use pip directly
- always use uv to run the server and do not use pip directly