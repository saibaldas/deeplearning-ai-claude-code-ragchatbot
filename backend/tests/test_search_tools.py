"""Tests for CourseSearchTool and CourseOutlineTool"""

import pytest
from unittest.mock import MagicMock, patch

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"] 
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_successful_search(self, mock_vector_store, sample_search_results):
        """Test successful search execution with results"""
        # Setup
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute("computer use capability")
        
        # Verify
        assert result is not None
        assert isinstance(result, str)
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Lesson 0" in result
        mock_vector_store.search.assert_called_once_with(
            query="computer use capability",
            course_name=None,
            lesson_number=None
        )
    
    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test search execution with course name filter"""
        # Setup
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute("API requests", course_name="Computer Use")
        
        # Verify
        mock_vector_store.search.assert_called_once_with(
            query="API requests",
            course_name="Computer Use",
            lesson_number=None
        )
        assert result is not None
    
    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test search execution with lesson number filter"""
        # Setup 
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute("introduction", lesson_number=1)
        
        # Verify
        mock_vector_store.search.assert_called_once_with(
            query="introduction",
            course_name=None,
            lesson_number=1
        )
        assert result is not None
    
    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test search execution with both course name and lesson filters"""
        # Setup
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute  
        result = tool.execute("API basics", course_name="Computer Use", lesson_number=1)
        
        # Verify
        mock_vector_store.search.assert_called_once_with(
            query="API basics",
            course_name="Computer Use", 
            lesson_number=1
        )
        assert result is not None
    
    def test_execute_with_search_error(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        # Setup
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute("test query")
        
        # Verify
        assert result == "Database connection failed"
    
    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        # Setup
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute("nonexistent content")
        
        # Verify
        assert "No relevant content found" in result
    
    def test_execute_empty_results_with_course_filter(self, mock_vector_store, empty_search_results):
        """Test empty results message includes filter information"""
        # Setup
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute("nonexistent", course_name="Test Course")
        
        # Verify
        assert "No relevant content found in course 'Test Course'" in result
    
    def test_execute_empty_results_with_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test empty results message includes lesson information"""
        # Setup
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute("nonexistent", lesson_number=2)
        
        # Verify
        assert "No relevant content found in lesson 2" in result
    
    def test_format_results(self, mock_vector_store, sample_search_results):
        """Test result formatting functionality"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        formatted = tool._format_results(sample_search_results)
        
        # Verify
        assert isinstance(formatted, str)
        assert "[Building Towards Computer Use with Anthropic - Lesson 0]" in formatted
        assert "Welcome to Building Toward Computer Use" in formatted
        assert "\n\n" in formatted  # Multiple results separated by double newlines
    
    def test_source_tracking(self, mock_vector_store, sample_search_results):
        """Test that sources are properly tracked"""
        # Setup
        mock_vector_store.get_lesson_link.return_value = "https://learn.deeplearning.ai/lesson/intro"
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        tool._format_results(sample_search_results)
        
        # Verify
        sources = tool.last_sources
        assert len(sources) == 2
        assert "Building Towards Computer Use with Anthropic - Lesson 0" in sources[0]
        assert "data-lesson-link" in sources[0]  # Check for embedded link


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that outline tool definition is correctly structured"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]
    
    def test_execute_successful_outline_retrieval(self, mock_vector_store):
        """Test successful course outline retrieval"""
        # Setup - mock_vector_store already has get_course_by_name configured
        tool = CourseOutlineTool(mock_vector_store)
        
        # Execute
        result = tool.execute("Computer Use")
        
        # Verify
        assert result is not None
        assert "**Building Towards Computer Use with Anthropic**" in result
        assert "Instructor: Colt Steele" in result
        assert "Course Link:" in result
        assert "**Course Lessons:**" in result
        assert "0. Introduction" in result
        assert "1. API Basics" in result
        mock_vector_store.get_course_by_name.assert_called_once_with("Computer Use")
    
    def test_execute_course_not_found(self, mock_vector_store):
        """Test handling when course is not found"""
        # Setup
        mock_vector_store.get_course_by_name.return_value = None
        tool = CourseOutlineTool(mock_vector_store)
        
        # Execute
        result = tool.execute("Nonexistent Course")
        
        # Verify
        assert "No course found matching 'Nonexistent Course'" in result
        mock_vector_store.get_course_by_name.assert_called_once_with("Nonexistent Course")
    
    def test_format_outline(self, mock_vector_store):
        """Test outline formatting with complete course data"""
        # Setup
        tool = CourseOutlineTool(mock_vector_store)
        course_data = {
            "title": "Test Course",
            "instructor": "Test Instructor",
            "course_link": "https://test.com/course",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "First Lesson"},
                {"lesson_number": 2, "lesson_title": "Second Lesson"}
            ]
        }
        
        # Execute
        formatted = tool._format_outline(course_data)
        
        # Verify
        assert "**Test Course**" in formatted
        assert "Instructor: Test Instructor" in formatted
        assert "Course Link: https://test.com/course" in formatted
        assert "**Course Lessons:**" in formatted
        assert "1. First Lesson" in formatted
        assert "2. Second Lesson" in formatted
    
    def test_format_outline_minimal_data(self, mock_vector_store):
        """Test outline formatting with minimal course data"""
        # Setup
        tool = CourseOutlineTool(mock_vector_store)
        course_data = {
            "title": "Minimal Course",
            "lessons": []
        }
        
        # Execute
        formatted = tool._format_outline(course_data)
        
        # Verify
        assert "**Minimal Course**" in formatted
        assert "Instructor: Unknown Instructor" in formatted
        assert "Course Link: No link available" in formatted
        assert "No lessons available" in formatted


class TestToolManager:
    """Test cases for ToolManager"""
    
    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        # Setup
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        manager.register_tool(search_tool)
        
        # Verify
        assert "search_course_content" in manager.tools
    
    def test_register_multiple_tools(self, mock_vector_store):
        """Test registering multiple tools"""
        # Setup
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        # Execute
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        # Verify
        assert len(manager.tools) == 2
        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools
    
    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting tool definitions"""
        # Setup
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute
        definitions = manager.get_tool_definitions()
        
        # Verify
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Test tool execution through manager"""
        # Setup
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute
        result = manager.execute_tool("search_course_content", query="test query")
        
        # Verify
        assert result is not None
        assert isinstance(result, str)
    
    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing non-existent tool"""
        # Setup
        manager = ToolManager()
        
        # Execute
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        # Verify
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last search"""
        # Setup
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute
        manager.execute_tool("search_course_content", query="test query")
        sources = manager.get_last_sources()
        
        # Verify
        assert isinstance(sources, list)
        assert len(sources) > 0
    
    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources"""
        # Setup
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute
        manager.execute_tool("search_course_content", query="test query")
        manager.reset_sources()
        sources = manager.get_last_sources()
        
        # Verify
        assert len(sources) == 0