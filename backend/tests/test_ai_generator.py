"""Tests for AI Generator tool calling and response processing"""

import pytest
from unittest.mock import MagicMock, patch, call

from ai_generator import AIGenerator
from tests.fixtures.mock_responses import (
    MockAnthropicResponse,
    MockContentBlock,
    MOCK_TOOL_USE_RESPONSE,
    MOCK_TEXT_RESPONSE
)


class TestAIGenerator:
    """Test cases for AIGenerator"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        api_key = "test_key"
        model = "claude-sonnet-4-20250514"
        
        with patch('anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key, model)
            
            mock_anthropic.assert_called_once_with(api_key=api_key)
            assert generator.model == model
            assert generator.base_params["model"] == model
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_generate_response_simple_text(self, patch_anthropic):
        """Test generating simple text response without tools"""
        # Setup
        patch_anthropic.messages.create.return_value = MOCK_TEXT_RESPONSE
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        
        # Execute
        result = generator.generate_response("What is machine learning?")
        
        # Verify
        assert result == "This is a general knowledge response about machine learning concepts."
        patch_anthropic.messages.create.assert_called_once()
        
        # Check the call parameters
        call_args = patch_anthropic.messages.create.call_args
        assert call_args[1]["model"] == "claude-sonnet-4-20250514"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "What is machine learning?"
    
    def test_generate_response_with_conversation_history(self, patch_anthropic):
        """Test generating response with conversation history"""
        # Setup
        patch_anthropic.messages.create.return_value = MOCK_TEXT_RESPONSE
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        history = "User: Previous question\nAssistant: Previous answer"
        
        # Execute
        result = generator.generate_response("Follow-up question", conversation_history=history)
        
        # Verify
        call_args = patch_anthropic.messages.create.call_args
        assert history in call_args[1]["system"]
        assert generator.SYSTEM_PROMPT in call_args[1]["system"]
    
    def test_generate_response_with_tools_no_tool_use(self, patch_anthropic, mock_tool_manager):
        """Test response generation with tools available but not used"""
        # Setup
        patch_anthropic.messages.create.return_value = MOCK_TEXT_RESPONSE
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        # Execute
        result = generator.generate_response(
            "What is AI?", 
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify
        assert result == "This is a general knowledge response about machine learning concepts."
        call_args = patch_anthropic.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self, patch_anthropic, mock_tool_manager):
        """Test response generation with tool use"""
        # Setup - first call returns tool use, second call returns final response
        tool_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "tool_123", 
                "input": {"query": "computer use"}
            },
            stop_reason="tool_use"
        )
        final_response = MockAnthropicResponse(
            content_text="Based on the course content, computer use involves...",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Course content about computer use"
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content"}]
        
        # Execute
        result = generator.generate_response(
            "Tell me about computer use",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify
        assert result == "Based on the course content, computer use involves..."
        assert patch_anthropic.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="computer use"
        )
    
    def test_handle_tool_execution_single_tool(self, patch_anthropic, mock_tool_manager):
        """Test handling execution of a single tool"""
        # Setup
        initial_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "tool_456",
                "input": {"query": "API basics", "course_name": "Computer Use"}
            },
            stop_reason="tool_use"
        )
        final_response = MockAnthropicResponse(
            content_text="Here's information about API basics...",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.return_value = final_response
        mock_tool_manager.execute_tool.return_value = "API basics content from course"
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        base_params = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Tell me about API basics"}],
            "system": generator.SYSTEM_PROMPT
        }
        
        # Execute
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        # Verify
        assert result == "Here's information about API basics..."
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="API basics",
            course_name="Computer Use"
        )
        
        # Check that the messages were properly structured
        call_args = patch_anthropic.messages.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 3  # Original user message + assistant tool use + user tool results
        
        # Check assistant message with tool use
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["content"]) == 1
        assert messages[1]["content"][0].type == "tool_use"
        
        # Check user message with tool results
        assert messages[2]["role"] == "user"
        assert len(messages[2]["content"]) == 1
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_456"
        assert messages[2]["content"][0]["content"] == "API basics content from course"
    
    def test_handle_tool_execution_multiple_tools(self, patch_anthropic, mock_tool_manager):
        """Test handling execution of multiple tools in one response"""
        # Setup
        content_blocks = [
            MockContentBlock("tool_use", name="search_course_content", id="tool_1", 
                           input={"query": "computer use"}),
            MockContentBlock("tool_use", name="get_course_outline", id="tool_2", 
                           input={"course_name": "Computer Use"})
        ]
        
        initial_response = MagicMock()
        initial_response.content = content_blocks
        
        final_response = MockAnthropicResponse(
            content_text="Here's comprehensive information...",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.return_value = final_response
        mock_tool_manager.execute_tool.side_effect = [
            "Search results about computer use",
            "Course outline for Computer Use"
        ]
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        base_params = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Tell me everything about computer use"}],
            "system": generator.SYSTEM_PROMPT
        }
        
        # Execute
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        # Verify
        assert result == "Here's comprehensive information..."
        assert mock_tool_manager.execute_tool.call_count == 2
        
        expected_calls = [
            call("search_course_content", query="computer use"),
            call("get_course_outline", course_name="Computer Use")
        ]
        mock_tool_manager.execute_tool.assert_has_calls(expected_calls)
        
        # Check that tool results were properly structured
        call_args = patch_anthropic.messages.create.call_args
        messages = call_args[1]["messages"]
        tool_results = messages[2]["content"]
        assert len(tool_results) == 2
        
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[0]["content"] == "Search results about computer use"
        
        assert tool_results[1]["type"] == "tool_result"
        assert tool_results[1]["tool_use_id"] == "tool_2"
        assert tool_results[1]["content"] == "Course outline for Computer Use"
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        prompt = AIGenerator.SYSTEM_PROMPT
        
        # Check for tool descriptions
        assert "search_course_content" in prompt
        assert "get_course_outline" in prompt
        
        # Check for usage guidelines
        assert "course outline/structure queries" in prompt
        assert "course content questions" in prompt
        assert "general knowledge questions" in prompt
        assert "Sequential tool calling" in prompt  # Updated for new functionality
        
        # Check for response protocol
        assert "No meta-commentary" in prompt
        assert "course title, course link, and complete numbered lesson list" in prompt
    
    def test_generate_response_api_error(self, patch_anthropic):
        """Test handling of Anthropic API errors"""
        # Setup
        patch_anthropic.messages.create.side_effect = Exception("API Error")
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        
        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")
        
        assert "API Error" in str(exc_info.value)
    
    def test_generate_response_tool_execution_error(self, patch_anthropic, mock_tool_manager):
        """Test handling of tool execution errors"""
        # Setup
        tool_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "tool_789",
                "input": {"query": "test"}
            },
            stop_reason="tool_use"
        )
        final_response = MockAnthropicResponse(
            content_text="I encountered an error...",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Tool execution failed"
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content"}]
        
        # Execute
        result = generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify - should still return a response even if tool execution has issues
        assert result == "I encountered an error..."
        mock_tool_manager.execute_tool.assert_called_once()
    
    def test_base_params_configuration(self):
        """Test that base parameters are properly configured"""
        with patch('anthropic.Anthropic'):
            generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
            
            assert generator.base_params["model"] == "claude-sonnet-4-20250514"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_no_conversation_history(self, patch_anthropic):
        """Test system prompt without conversation history"""
        # Setup
        patch_anthropic.messages.create.return_value = MOCK_TEXT_RESPONSE
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        
        # Execute
        generator.generate_response("Test query")
        
        # Verify
        call_args = patch_anthropic.messages.create.call_args
        system_content = call_args[1]["system"]
        
        # Should contain base prompt but not history section
        assert generator.SYSTEM_PROMPT in system_content
        assert "Previous conversation:" not in system_content


class TestSequentialToolCalling:
    """Test cases for sequential tool calling functionality"""
    
    def test_two_round_tool_execution(self, patch_anthropic, mock_tool_manager):
        """Test successful 2-round sequential tool calling"""
        # Setup - Round 1: get course outline, Round 2: search content
        round1_response = MockAnthropicResponse(
            tool_use_data={
                "name": "get_course_outline",
                "id": "tool_round1",
                "input": {"course_name": "Computer Use"}
            },
            stop_reason="tool_use"
        )
        
        round2_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content", 
                "id": "tool_round2",
                "input": {"query": "lesson 4 content", "course_name": "Computer Use"}
            },
            stop_reason="tool_use"
        )
        
        final_response = MockAnthropicResponse(
            content_text="Based on the course outline and content search, lesson 4 covers API basics...",
            stop_reason="end_turn"
        )
        
        # Mock API responses for 3 calls: Round 1 tool, Round 2 tool, Final synthesis
        patch_anthropic.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Mock tool execution results
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: 1. Introduction 2. API Basics 3. Advanced Usage 4. Computer Use Implementation",
            "Lesson 4 content: Computer use implementation involves screen capture and mouse control..."
        ]
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search content"}
        ]
        
        # Execute with sequential enabled
        result = generator.generate_response(
            "What is covered in lesson 4 of the Computer Use course?",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
            enable_sequential=True
        )
        
        # Verify
        assert result == "Based on the course outline and content search, lesson 4 covers API basics..."
        assert patch_anthropic.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify tool execution order
        expected_calls = [
            call("get_course_outline", course_name="Computer Use"),
            call("search_course_content", query="lesson 4 content", course_name="Computer Use")
        ]
        mock_tool_manager.execute_tool.assert_has_calls(expected_calls)
    
    def test_early_termination_no_tools(self, patch_anthropic, mock_tool_manager):
        """Test that query terminates early when no tools are needed"""
        # Setup - Direct response without tools
        direct_response = MockAnthropicResponse(
            content_text="This is a general knowledge question that doesn't require course data.",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.return_value = direct_response
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content"}]
        
        # Execute
        result = generator.generate_response(
            "What is machine learning?",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
            enable_sequential=True
        )
        
        # Verify
        assert result == "This is a general knowledge question that doesn't require course data."
        assert patch_anthropic.messages.create.call_count == 1  # Only one API call
        mock_tool_manager.execute_tool.assert_not_called()  # No tools used
    
    def test_single_round_sufficient(self, patch_anthropic, mock_tool_manager):
        """Test termination after single round when sufficient information gathered"""
        # Setup - Single tool use then direct response 
        tool_response = MockAnthropicResponse(
            tool_use_data={
                "name": "get_course_outline",
                "id": "tool_123",
                "input": {"course_name": "MCP"}
            },
            stop_reason="tool_use"
        )
        
        final_response = MockAnthropicResponse(
            content_text="The MCP course has 5 lessons covering protocol basics to advanced implementation.",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Course outline with 5 lessons"
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "get_course_outline"}]
        
        # Execute
        result = generator.generate_response(
            "How many lessons are in the MCP course?",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
            enable_sequential=True
        )
        
        # Verify
        assert "5 lessons" in result
        assert patch_anthropic.messages.create.call_count == 2  # Tool call + response
        assert mock_tool_manager.execute_tool.call_count == 1   # Only one tool used
    
    def test_max_rounds_reached(self, patch_anthropic, mock_tool_manager):
        """Test behavior when maximum rounds are reached"""
        # Setup - 2 tool rounds, then synthesis
        round1_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "tool_r1", 
                "input": {"query": "introduction"}
            },
            stop_reason="tool_use"
        )
        
        round2_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "tool_r2",
                "input": {"query": "advanced topics"}
            },
            stop_reason="tool_use"
        )
        
        synthesis_response = MockAnthropicResponse(
            content_text="Synthesis: The course covers both introduction and advanced topics comprehensively.",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.side_effect = [round1_response, round2_response, synthesis_response]
        mock_tool_manager.execute_tool.side_effect = [
            "Introduction content...",
            "Advanced topics content..."
        ]
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content"}]
        
        # Execute with max_rounds=2
        result = generator.generate_response(
            "Compare introduction and advanced topics",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
            enable_sequential=True
        )
        
        # Verify
        assert "Synthesis:" in result
        assert patch_anthropic.messages.create.call_count == 3  # 2 rounds + synthesis
        assert mock_tool_manager.execute_tool.call_count == 2   # 2 tool executions
    
    def test_tool_execution_error_handling(self, patch_anthropic, mock_tool_manager):
        """Test graceful handling of tool execution errors"""
        # Setup - First tool fails, second succeeds
        round1_response = MockAnthropicResponse(
            tool_use_data={
                "name": "get_course_outline",
                "id": "tool_fail",
                "input": {"course_name": "NonExistent"}
            },
            stop_reason="tool_use"
        )
        
        round2_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "tool_success", 
                "input": {"query": "alternative search"}
            },
            stop_reason="tool_use"
        )
        
        final_response = MockAnthropicResponse(
            content_text="Despite the error, I found alternative information...",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Mock first tool to fail, second to succeed
        mock_tool_manager.execute_tool.side_effect = [
            Exception("Course not found"),
            "Alternative search results"
        ]
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        
        # Execute
        result = generator.generate_response(
            "Find information about course",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
            enable_sequential=True
        )
        
        # Verify error was handled gracefully
        assert "alternative information" in result
        assert mock_tool_manager.execute_tool.call_count == 2
    
    def test_backward_compatibility_single_round(self, patch_anthropic, mock_tool_manager):
        """Test that sequential=False maintains single-round behavior"""
        # Setup - Single tool response
        tool_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "single_tool",
                "input": {"query": "test"}
            },
            stop_reason="tool_use"
        )
        
        final_response = MockAnthropicResponse(
            content_text="Single round result",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.side_effect = [tool_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Search result"
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content"}]
        
        # Execute with sequential disabled
        result = generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
            enable_sequential=False  # Disable sequential
        )
        
        # Verify single-round behavior
        assert result == "Single round result"
        assert patch_anthropic.messages.create.call_count == 2  # Tool + response
        assert mock_tool_manager.execute_tool.call_count == 1
    
    def test_context_preservation_across_rounds(self, patch_anthropic, mock_tool_manager):
        """Test that context is properly preserved between rounds"""
        # Setup responses
        round1_response = MockAnthropicResponse(
            tool_use_data={
                "name": "get_course_outline",
                "id": "context_r1",
                "input": {"course_name": "Test Course"}
            },
            stop_reason="tool_use"
        )
        
        round2_response = MockAnthropicResponse(
            tool_use_data={
                "name": "search_course_content",
                "id": "context_r2", 
                "input": {"query": "based on previous outline"}
            },
            stop_reason="tool_use"
        )
        
        final_response = MockAnthropicResponse(
            content_text="Context-aware final response",
            stop_reason="end_turn"
        )
        
        patch_anthropic.messages.create.side_effect = [round1_response, round2_response, final_response]
        mock_tool_manager.execute_tool.side_effect = ["Outline data", "Content data"]
        
        generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        
        # Execute with conversation history
        result = generator.generate_response(
            "Complex query requiring context",
            conversation_history="User: Previous question\nAssistant: Previous answer",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
            enable_sequential=True
        )
        
        # Verify
        assert result == "Context-aware final response"
        
        # Check that system content included conversation history in all calls
        for call_args in patch_anthropic.messages.create.call_args_list:
            system_content = call_args[1]["system"]
            assert "Previous conversation:" in system_content
            assert "Previous question" in system_content
    
    def test_system_prompt_sequential_tool_guidance(self):
        """Test that system prompt includes sequential tool calling guidance"""
        prompt = AIGenerator.SYSTEM_PROMPT
        
        # Check for sequential tool calling content
        assert "Sequential tool calling" in prompt
        assert "up to 2 rounds" in prompt
        assert "Multi-Step Query Examples" in prompt
        assert "When to Continue vs. Stop" in prompt
        
        # Verify removal of old single-tool constraint
        assert "One tool use per query maximum" not in prompt
        
        # Check for multi-round examples
        assert "Round 1:" in prompt and "Round 2:" in prompt
        assert "Compare the lesson structure between two courses" in prompt