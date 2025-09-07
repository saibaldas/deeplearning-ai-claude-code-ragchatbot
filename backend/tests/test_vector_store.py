"""Tests for VectorStore functionality"""

import pytest
from unittest.mock import MagicMock, patch
import json

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test cases for SearchResults class"""
    
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'test1'}, {'course': 'test2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'course': 'test1'}, {'course': 'test2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_empty_constructor(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("No results found")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "No results found"
    
    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults([], [], [])
        assert results.is_empty() is True
    
    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(['doc'], [{'meta': 'data'}], [0.1])
        assert results.is_empty() is False


class TestVectorStore:
    """Test cases for VectorStore"""
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_init(self, mock_embedding_func, mock_client):
        """Test VectorStore initialization"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_collection = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Execute
        store = VectorStore("/test/path", "test-model", max_results=10)
        
        # Verify
        assert store.max_results == 10
        mock_client.assert_called_once()
        assert mock_client_instance.get_or_create_collection.call_count == 2  # catalog and content collections
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_successful(self, mock_embedding_func, mock_client, sample_search_results):
        """Test successful search operation"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock successful search
        mock_content_collection.query.return_value = {
            'documents': [['test doc']],
            'metadatas': [[{'course_title': 'test'}]],
            'distances': [[0.1]]
        }
        
        store = VectorStore("/test/path", "test-model", max_results=5)
        
        # Execute
        results = store.search("test query")
        
        # Verify
        assert not results.is_empty()
        assert results.error is None
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None
        )
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_course_filter(self, mock_embedding_func, mock_client):
        """Test search with course name filter"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock course resolution
        mock_catalog_collection.query.return_value = {
            'documents': [['Computer Use Course']],
            'metadatas': [[{'title': 'Building Towards Computer Use'}]],
            'distances': [[0.1]]
        }
        
        # Mock content search
        mock_content_collection.query.return_value = {
            'documents': [['course content']],
            'metadatas': [[{'course_title': 'Building Towards Computer Use'}]],
            'distances': [[0.2]]
        }
        
        store = VectorStore("/test/path", "test-model")
        
        # Execute
        results = store.search("test query", course_name="Computer Use")
        
        # Verify
        mock_catalog_collection.query.assert_called_once_with(
            query_texts=["Computer Use"],
            n_results=1
        )
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"course_title": "Building Towards Computer Use"}
        )
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_course_not_found(self, mock_embedding_func, mock_client):
        """Test search when course name doesn't match any course"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock empty course resolution
        mock_catalog_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        store = VectorStore("/test/path", "test-model")
        
        # Execute
        results = store.search("test query", course_name="Nonexistent Course")
        
        # Verify
        assert results.error == "No course found matching 'Nonexistent Course'"
        # Content search should not be called
        mock_content_collection.query.assert_not_called()
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_build_filter_combinations(self, mock_embedding_func, mock_client):
        """Test different filter combinations"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_or_create_collection.return_value = MagicMock()
        
        store = VectorStore("/test/path", "test-model")
        
        # Test no filter
        result = store._build_filter(None, None)
        assert result is None
        
        # Test course only
        result = store._build_filter("Test Course", None)
        assert result == {"course_title": "Test Course"}
        
        # Test lesson only
        result = store._build_filter(None, 1)
        assert result == {"lesson_number": 1}
        
        # Test both filters
        result = store._build_filter("Test Course", 1)
        assert result == {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 1}
        ]}
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_metadata(self, mock_embedding_func, mock_client, sample_course):
        """Test adding course metadata to vector store"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        store = VectorStore("/test/path", "test-model")
        
        # Execute
        store.add_course_metadata(sample_course)
        
        # Verify
        mock_catalog_collection.add.assert_called_once()
        call_args = mock_catalog_collection.add.call_args
        
        # Check documents
        assert call_args[1]["documents"] == [sample_course.title]
        
        # Check IDs
        assert call_args[1]["ids"] == [sample_course.title]
        
        # Check metadata
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == sample_course.title
        assert metadata["instructor"] == sample_course.instructor
        assert metadata["course_link"] == sample_course.course_link
        assert "lessons_json" in metadata
        assert metadata["lesson_count"] == len(sample_course.lessons)
        
        # Verify lessons JSON is valid
        lessons_data = json.loads(metadata["lessons_json"])
        assert len(lessons_data) == len(sample_course.lessons)
    
    @patch('chromadb.PersistentClient') 
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_content(self, mock_embedding_func, mock_client, sample_course_chunks):
        """Test adding course content chunks to vector store"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        store = VectorStore("/test/path", "test-model")
        
        # Execute
        store.add_course_content(sample_course_chunks[:2])  # Use first 2 chunks
        
        # Verify
        mock_content_collection.add.assert_called_once()
        call_args = mock_content_collection.add.call_args
        
        # Check documents
        expected_docs = [chunk.content for chunk in sample_course_chunks[:2]]
        assert call_args[1]["documents"] == expected_docs
        
        # Check metadata
        expected_metadata = [{
            "course_title": chunk.course_title,
            "lesson_number": chunk.lesson_number,
            "chunk_index": chunk.chunk_index
        } for chunk in sample_course_chunks[:2]]
        assert call_args[1]["metadatas"] == expected_metadata
        
        # Check IDs
        expected_ids = [f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}" 
                       for chunk in sample_course_chunks[:2]]
        assert call_args[1]["ids"] == expected_ids
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_by_name_success(self, mock_embedding_func, mock_client):
        """Test successful course retrieval by name"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock course name resolution
        mock_catalog_collection.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        
        # Mock course data retrieval
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Lesson 1", "lesson_link": "link1"},
            {"lesson_number": 2, "lesson_title": "Lesson 2", "lesson_link": "link2"}
        ])
        
        mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'title': 'Test Course',
                'instructor': 'Test Instructor',
                'course_link': 'http://test.com',
                'lessons_json': lessons_json,
                'lesson_count': 2
            }]
        }
        
        store = VectorStore("/test/path", "test-model")
        
        # Execute
        result = store.get_course_by_name("Test")
        
        # Verify
        assert result is not None
        assert result['title'] == 'Test Course'
        assert result['instructor'] == 'Test Instructor'
        assert result['course_link'] == 'http://test.com'
        assert 'lessons' in result
        assert len(result['lessons']) == 2
        assert 'lessons_json' not in result  # Should be removed
        
        # Verify lessons were parsed correctly
        lessons = result['lessons']
        assert lessons[0]['lesson_number'] == 1
        assert lessons[0]['lesson_title'] == 'Lesson 1'
        assert lessons[1]['lesson_number'] == 2
        assert lessons[1]['lesson_title'] == 'Lesson 2'
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_by_name_not_found(self, mock_embedding_func, mock_client):
        """Test course retrieval when course not found"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock empty course resolution
        mock_catalog_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        store = VectorStore("/test/path", "test-model")
        
        # Execute
        result = store.get_course_by_name("Nonexistent")
        
        # Verify
        assert result is None
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_lesson_link(self, mock_embedding_func, mock_client):
        """Test getting specific lesson link"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Lesson 1", "lesson_link": "http://lesson1.com"},
            {"lesson_number": 2, "lesson_title": "Lesson 2", "lesson_link": "http://lesson2.com"}
        ])
        
        mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': lessons_json
            }]
        }
        
        store = VectorStore("/test/path", "test-model")
        
        # Execute
        link = store.get_lesson_link("Test Course", 2)
        
        # Verify
        assert link == "http://lesson2.com"
        mock_catalog_collection.get.assert_called_once_with(ids=["Test Course"])
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_lesson_link_not_found(self, mock_embedding_func, mock_client):
        """Test getting lesson link when lesson doesn't exist"""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Lesson 1", "lesson_link": "http://lesson1.com"}
        ])
        
        mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': lessons_json
            }]
        }
        
        store = VectorStore("/test/path", "test-model")
        
        # Execute
        link = store.get_lesson_link("Test Course", 5)  # Non-existent lesson
        
        # Verify
        assert link is None