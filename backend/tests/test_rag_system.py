"""Integration tests for RAG System end-to-end functionality"""

import pytest
from unittest.mock import MagicMock, patch, call
import os
from tempfile import TemporaryDirectory

from rag_system import RAGSystem
from models import Course, Lesson
from tests.fixtures.mock_responses import (
    MockAnthropicResponse,
    MOCK_TEXT_RESPONSE,
    MOCK_TOOL_USE_RESPONSE
)


class TestRAGSystem:
    """Test cases for RAG System integration"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_init_components(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test RAG system initialization with all components"""
        # Execute
        rag_system = RAGSystem(test_config)
        
        # Verify all components are initialized
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        
        # Verify components initialized with correct parameters
        mock_doc_proc.assert_called_once_with(test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP)
        mock_vector_store.assert_called_once_with(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
        mock_ai_gen.assert_called_once_with(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        mock_session_mgr.assert_called_once_with(test_config.MAX_HISTORY)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')  
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_tools_registered(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test that both search and outline tools are registered"""
        # Execute
        rag_system = RAGSystem(test_config)
        
        # Verify tools are registered
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        assert len(tool_definitions) == 2
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_without_session(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test query processing without existing session"""
        # Setup
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Response about machine learning"
        
        mock_session_mgr_instance = mock_session_mgr.return_value
        mock_session_mgr_instance.get_conversation_history.return_value = None
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        response, sources = rag_system.query("What is machine learning?")
        
        # Verify
        assert response == "Response about machine learning"
        assert isinstance(sources, list)
        
        # Verify AI generator was called correctly
        mock_ai_gen_instance.generate_response.assert_called_once()
        call_args = mock_ai_gen_instance.generate_response.call_args
        
        assert "Answer this question about course materials: What is machine learning?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_with_session(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test query processing with existing session"""
        # Setup
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Follow-up response"
        
        mock_session_mgr_instance = mock_session_mgr.return_value
        mock_session_mgr_instance.get_conversation_history.return_value = "User: Previous question\nAssistant: Previous answer"
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        response, sources = rag_system.query("Follow-up question", session_id="test_session")
        
        # Verify
        assert response == "Follow-up response"
        
        # Verify session manager interactions
        mock_session_mgr_instance.get_conversation_history.assert_called_once_with("test_session")
        mock_session_mgr_instance.add_exchange.assert_called_once_with(
            "test_session", 
            "Follow-up question", 
            "Follow-up response"
        )
        
        # Verify conversation history passed to AI
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert call_args[1]["conversation_history"] == "User: Previous question\nAssistant: Previous answer"
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_with_tool_sources(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test that sources from tool searches are returned"""
        # Setup
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Course content response"
        
        rag_system = RAGSystem(test_config)
        
        # Mock tool manager to return sources
        rag_system.tool_manager.get_last_sources = MagicMock(return_value=["Source 1", "Source 2"])
        rag_system.tool_manager.reset_sources = MagicMock()
        
        # Execute
        response, sources = rag_system.query("Tell me about API basics")
        
        # Verify
        assert response == "Course content response"
        assert sources == ["Source 1", "Source 2"]
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_add_course_document_success(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test successful course document processing"""
        # Setup
        sample_course = Course(
            title="Test Course",
            instructor="Test Instructor", 
            lessons=[Lesson(lesson_number=1, title="Test Lesson")]
        )
        sample_chunks = [MagicMock(), MagicMock()]
        
        mock_doc_proc_instance = mock_doc_proc.return_value
        mock_doc_proc_instance.process_course_document.return_value = (sample_course, sample_chunks)
        
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.add_course_metadata = MagicMock()
        mock_vector_store_instance.add_course_content = MagicMock()
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        course, chunk_count = rag_system.add_course_document("test_file.txt")
        
        # Verify
        assert course == sample_course
        assert chunk_count == 2
        
        mock_doc_proc_instance.process_course_document.assert_called_once_with("test_file.txt")
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_store_instance.add_course_content.assert_called_once_with(sample_chunks)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_add_course_document_error(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test course document processing error handling"""
        # Setup
        mock_doc_proc_instance = mock_doc_proc.return_value
        mock_doc_proc_instance.process_course_document.side_effect = Exception("Processing failed")
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        course, chunk_count = rag_system.add_course_document("invalid_file.txt")
        
        # Verify
        assert course is None
        assert chunk_count == 0
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_clear_existing(self, mock_listdir, mock_exists, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test adding course folder with clear_existing=True"""
        # Setup
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.txt"]
        
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.clear_all_data = MagicMock()
        mock_vector_store_instance.get_existing_course_titles.return_value = []
        
        sample_course = Course(title="Test Course", lessons=[])
        mock_doc_proc_instance = mock_doc_proc.return_value
        mock_doc_proc_instance.process_course_document.return_value = (sample_course, [MagicMock()])
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        courses, chunks = rag_system.add_course_folder("/test/folder", clear_existing=True)
        
        # Verify
        assert courses == 2
        assert chunks == 2
        mock_vector_store_instance.clear_all_data.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_skip_existing(self, mock_listdir, mock_exists, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test adding course folder skips existing courses"""
        # Setup
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.txt"]
        
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.get_existing_course_titles.return_value = ["Existing Course"]
        
        # First course is new, second course already exists
        new_course = Course(title="New Course", lessons=[])
        existing_course = Course(title="Existing Course", lessons=[])
        
        mock_doc_proc_instance = mock_doc_proc.return_value
        mock_doc_proc_instance.process_course_document.side_effect = [
            (new_course, [MagicMock()]),
            (existing_course, [MagicMock()])
        ]
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        courses, chunks = rag_system.add_course_folder("/test/folder", clear_existing=False)
        
        # Verify - only new course should be added
        assert courses == 1
        assert chunks == 1
        
        # Verify only new course was added to vector store
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(new_course)
        mock_vector_store_instance.add_course_content.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    def test_add_course_folder_nonexistent_path(self, mock_exists, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test handling of nonexistent folder path"""
        # Setup
        mock_exists.return_value = False
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        courses, chunks = rag_system.add_course_folder("/nonexistent/folder")
        
        # Verify
        assert courses == 0
        assert chunks == 0
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test course analytics retrieval"""
        # Setup
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.get_course_count.return_value = 5
        mock_vector_store_instance.get_existing_course_titles.return_value = ["Course 1", "Course 2", "Course 3"]
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        analytics = rag_system.get_course_analytics()
        
        # Verify
        assert analytics["total_courses"] == 5
        assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]
        
        mock_vector_store_instance.get_course_count.assert_called_once()
        mock_vector_store_instance.get_existing_course_titles.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_end_to_end_content_search_query(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test end-to-end content search query processing"""
        # Setup - simulate tool use for content search
        tool_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "tool_123",
                "input": {"query": "computer use", "course_name": "Computer Use"}
            },
            stop_reason="tool_use"
        )
        final_response = MockAnthropicResponse(
            content_text="Computer use capability allows AI models to interact with computer interfaces...",
            stop_reason="end_turn"
        )
        
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Computer use capability allows AI models to interact with computer interfaces..."
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        response, sources = rag_system.query("Tell me about computer use capability")
        
        # Verify
        assert "Computer use capability allows AI models" in response
        mock_ai_gen_instance.generate_response.assert_called_once()
        
        # Verify correct tools were passed
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_end_to_end_outline_query(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test end-to-end course outline query processing"""
        # Setup
        outline_response = MockAnthropicResponse(
            content_text="**Computer Use Course**\nInstructor: Colt Steele\n**Course Lessons:**\n1. Introduction\n2. API Basics",
            stop_reason="end_turn"
        )
        
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "**Computer Use Course**\nInstructor: Colt Steele\n**Course Lessons:**\n1. Introduction\n2. API Basics"
        
        rag_system = RAGSystem(test_config)
        
        # Execute
        response, sources = rag_system.query("What is the structure of the Computer Use course?")
        
        # Verify
        assert "Computer Use Course" in response
        assert "Colt Steele" in response
        assert "Course Lessons" in response
        mock_ai_gen_instance.generate_response.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_error_propagation(self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test that errors are properly propagated through the system"""
        # Setup
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.side_effect = Exception("AI Generation failed")
        
        rag_system = RAGSystem(test_config)
        
        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            rag_system.query("Test query")
        
        assert "AI Generation failed" in str(exc_info.value)