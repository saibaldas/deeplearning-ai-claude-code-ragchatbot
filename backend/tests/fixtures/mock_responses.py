"""Mock responses for external APIs and services"""

from typing import Dict, List, Any
from unittest.mock import MagicMock

# Mock Anthropic API responses
class MockAnthropicResponse:
    """Mock Anthropic API response object"""
    
    def __init__(self, content_text: str = None, tool_use_data: Dict = None, stop_reason: str = "end_turn"):
        self.stop_reason = stop_reason
        if tool_use_data:
            self.content = [
                MockContentBlock(block_type="tool_use", **tool_use_data)
            ]
        else:
            self.content = [
                MockContentBlock(block_type="text", text=content_text or "Default response")
            ]
        # Add text property for compatibility
        self.text = content_text or "Default response"

class MockContentBlock:
    """Mock content block for Anthropic responses"""
    
    def __init__(self, block_type: str, text: str = None, name: str = None, id: str = None, input: Dict = None):
        self.type = block_type
        self.text = text
        self.name = name
        self.id = id or "tool_use_123"
        self.input = input or {}

# Sample tool use responses
MOCK_TOOL_USE_RESPONSE = MockAnthropicResponse(
    tool_use_data={
        "name": "search_course_content",
        "id": "tool_123",
        "input": {"query": "computer use capability", "course_name": "Computer Use"}
    },
    stop_reason="tool_use"
)

MOCK_OUTLINE_TOOL_USE_RESPONSE = MockAnthropicResponse(
    tool_use_data={
        "name": "get_course_outline", 
        "id": "tool_124",
        "input": {"course_name": "Computer Use"}
    },
    stop_reason="tool_use"
)

MOCK_TEXT_RESPONSE = MockAnthropicResponse(
    content_text="This is a general knowledge response about machine learning concepts.",
    stop_reason="end_turn"
)

# Mock ChromaDB query responses
MOCK_CHROMA_SEARCH_RESULTS = {
    'documents': [["Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic...", 
                  "That is, it can look at the screen, a computer usually running in a virtual machine..."]],
    'metadatas': [[
        {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 0, "chunk_index": 0},
        {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 0, "chunk_index": 1}
    ]],
    'distances': [[0.1, 0.2]]
}

MOCK_CHROMA_EMPTY_RESULTS = {
    'documents': [[]],
    'metadatas': [[]],
    'distances': [[]]
}

MOCK_CHROMA_COURSE_CATALOG_RESULTS = {
    'documents': [["Building Towards Computer Use with Anthropic"]],
    'metadatas': [[{
        "title": "Building Towards Computer Use with Anthropic",
        "instructor": "Colt Steele",
        "course_link": "https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        "lessons_json": '[{"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "https://learn.deeplearning.ai/lesson/intro"}, {"lesson_number": 1, "lesson_title": "API Basics", "lesson_link": "https://learn.deeplearning.ai/lesson/basics"}]',
        "lesson_count": 2
    }]],
    'distances': [[0.1]]
}

def create_mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MOCK_TEXT_RESPONSE
    return mock_client

def create_mock_chroma_collection():
    """Create a mock ChromaDB collection"""
    mock_collection = MagicMock()
    mock_collection.query.return_value = MOCK_CHROMA_SEARCH_RESULTS
    mock_collection.get.return_value = MOCK_CHROMA_COURSE_CATALOG_RESULTS
    mock_collection.add.return_value = None
    return mock_collection

def create_mock_vector_store():
    """Create a mock VectorStore with common methods"""
    mock_store = MagicMock()
    
    # Mock search method
    from vector_store import SearchResults
    mock_results = SearchResults(
        documents=MOCK_CHROMA_SEARCH_RESULTS['documents'][0],
        metadata=MOCK_CHROMA_SEARCH_RESULTS['metadatas'][0],
        distances=MOCK_CHROMA_SEARCH_RESULTS['distances'][0]
    )
    mock_store.search.return_value = mock_results
    
    # Mock course resolution
    mock_store._resolve_course_name.return_value = "Building Towards Computer Use with Anthropic"
    
    # Mock course by name
    mock_store.get_course_by_name.return_value = {
        "title": "Building Towards Computer Use with Anthropic",
        "instructor": "Colt Steele", 
        "course_link": "https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "https://learn.deeplearning.ai/lesson/intro"},
            {"lesson_number": 1, "lesson_title": "API Basics", "lesson_link": "https://learn.deeplearning.ai/lesson/basics"}
        ],
        "lesson_count": 2
    }
    
    # Mock lesson link
    mock_store.get_lesson_link.return_value = "https://learn.deeplearning.ai/lesson/intro"
    
    return mock_store