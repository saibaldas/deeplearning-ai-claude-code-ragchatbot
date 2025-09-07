"""Pytest configuration and fixtures for RAG chatbot tests"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add backend directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.fixtures.sample_course_data import (
    SAMPLE_COURSE,
    SAMPLE_COURSE_2, 
    SAMPLE_COURSE_CHUNKS,
    SAMPLE_COURSE_METADATA,
    SAMPLE_QUERIES
)
from tests.fixtures.mock_responses import (
    create_mock_anthropic_client,
    create_mock_chroma_collection,
    create_mock_vector_store,
    MOCK_CHROMA_SEARCH_RESULTS,
    MOCK_TOOL_USE_RESPONSE
)

# Import the actual classes
from models import Course, Lesson, CourseChunk
from config import Config
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Fixture providing a sample course object"""
    return SAMPLE_COURSE

@pytest.fixture
def sample_course_2():
    """Fixture providing a second sample course object"""
    return SAMPLE_COURSE_2

@pytest.fixture
def sample_course_chunks():
    """Fixture providing sample course chunks"""
    return SAMPLE_COURSE_CHUNKS

@pytest.fixture
def sample_course_metadata():
    """Fixture providing sample course metadata"""
    return SAMPLE_COURSE_METADATA

@pytest.fixture
def sample_queries():
    """Fixture providing sample test queries"""
    return SAMPLE_QUERIES

@pytest.fixture
def mock_anthropic_client():
    """Fixture providing a mock Anthropic client"""
    return create_mock_anthropic_client()

@pytest.fixture
def mock_chroma_collection():
    """Fixture providing a mock ChromaDB collection"""
    return create_mock_chroma_collection()

@pytest.fixture
def mock_vector_store():
    """Fixture providing a mock VectorStore"""
    return create_mock_vector_store()

@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    return Config(
        ANTHROPIC_API_KEY="test_key",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH="./test_chroma_db"
    )

@pytest.fixture
def sample_search_results():
    """Fixture providing sample SearchResults object"""
    return SearchResults(
        documents=MOCK_CHROMA_SEARCH_RESULTS['documents'][0],
        metadata=MOCK_CHROMA_SEARCH_RESULTS['metadatas'][0],
        distances=MOCK_CHROMA_SEARCH_RESULTS['distances'][0]
    )

@pytest.fixture
def empty_search_results():
    """Fixture providing empty SearchResults object"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

@pytest.fixture
def error_search_results():
    """Fixture providing SearchResults with error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Database connection failed"
    )

@pytest.fixture
def mock_session_manager():
    """Fixture providing a mock SessionManager"""
    from session_manager import SessionManager
    mock_manager = MagicMock(spec=SessionManager)
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_conversation_history.return_value = "User: Previous question\nAssistant: Previous answer"
    mock_manager.add_exchange.return_value = None
    return mock_manager

@pytest.fixture
def mock_tool_manager():
    """Fixture providing a mock ToolManager"""
    from search_tools import ToolManager
    mock_manager = MagicMock(spec=ToolManager)
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
    ]
    mock_manager.execute_tool.return_value = "Mock tool result"
    mock_manager.get_last_sources.return_value = ["Source 1", "Source 2"]
    mock_manager.reset_sources.return_value = None
    return mock_manager

# Environment setup fixtures
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for all tests"""
    # Set test environment variables
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    yield
    # Cleanup after test
    if "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"] == "test_key":
        del os.environ["ANTHROPIC_API_KEY"]

@pytest.fixture
def patch_anthropic():
    """Fixture to patch Anthropic client creation"""
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = create_mock_anthropic_client()
        mock_anthropic.return_value = mock_client
        yield mock_client

@pytest.fixture
def patch_chromadb():
    """Fixture to patch ChromaDB client creation"""
    with patch('chromadb.PersistentClient') as mock_chroma:
        mock_client = MagicMock()
        mock_collection = create_mock_chroma_collection()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        yield mock_client