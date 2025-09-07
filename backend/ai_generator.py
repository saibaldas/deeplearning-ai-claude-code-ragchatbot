import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search within course materials for specific content and detailed educational materials
2. **get_course_outline**: Get complete course information including title, instructor, course link, and all lesson numbers/titles

Tool Usage Guidelines:
- For **course outline/structure queries**: Use get_course_outline to retrieve course title, course link, instructor, and complete lesson list
- For **course content questions**: Use search_course_content to find specific materials within lessons
- For **general knowledge questions**: Answer using existing knowledge without searching
- **Sequential tool calling**: You can make multiple tool calls (up to 2 rounds) when complex queries require multi-step reasoning
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Multi-Step Query Examples:
- "Find a course covering the same topic as lesson 3 of Course X" → Round 1: get outline for Course X, Round 2: search for similar topics
- "Compare the lesson structure between two courses" → Round 1: get outline for first course, Round 2: get outline for second course
- "What content comes after the API basics lesson in the Computer Use course?" → Round 1: get course outline, Round 2: search content from subsequent lessons

When to Continue vs. Stop:
- **Continue to Round 2** if: You need additional information from a different source, comparison requires multiple outlines, or initial results suggest related content exists elsewhere
- **Stop after Round 1** if: Single tool call provides complete answer, no additional information would be helpful, or query is fully addressed
- **Stop immediately** if: Query can be answered with general knowledge, tools would not provide relevant information

Response Protocol:
- **No meta-commentary**: Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
- Do not mention "based on the search results," "using the tool," or "in my first/second search"
- For outline queries: Always include course title, course link, and complete numbered lesson list when available
- For multi-step queries: Synthesize information from all rounds into a cohesive response

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2,
                         enable_sequential: bool = True) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling for complex queries.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool rounds (default: 2)
            enable_sequential: Whether to enable sequential tool calling (default: True)
            
        Returns:
            Generated response as string
        """
        
        # Check if sequential tool calling is enabled and tools are available
        if enable_sequential and tools and tool_manager and max_tool_rounds > 1:
            return self._execute_tool_rounds(query, conversation_history, tools, tool_manager, max_tool_rounds)
        
        # Fall back to legacy single-round behavior for backward compatibility
        return self._generate_single_round_response(query, conversation_history, tools, tool_manager)
    
    def _generate_single_round_response(self, query: str, conversation_history: Optional[str],
                                      tools: Optional[List], tool_manager) -> str:
        """
        Generate response using legacy single-round tool execution.
        Maintains backward compatibility for existing functionality.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def _execute_tool_rounds(self, query: str, conversation_history: Optional[str], 
                           tools: List, tool_manager, max_rounds: int = 2) -> str:
        """
        Execute multiple rounds of tool calling for complex queries.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context  
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool rounds (default: 2)
            
        Returns:
            Final response after all rounds completed
        """
        round_results = []
        current_round = 1
        
        # Build initial system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initial API parameters
        current_messages = [{"role": "user", "content": query}]
        
        while current_round <= max_rounds:
            # Prepare API call for current round
            round_context = self._build_round_context(query, round_results, current_round)
            
            api_params = {
                **self.base_params,
                "messages": current_messages,
                "system": f"{system_content}\n\n{round_context}",
                "tools": tools,
                "tool_choice": {"type": "auto"}
            }
            
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # Check if tools were used
            if response.stop_reason == "tool_use":
                # Execute tools and collect results for this round
                tool_results = self._execute_round_tools(response, tool_manager, current_round)
                round_results.extend(tool_results)
                
                # Update message history for next round
                current_messages.append({"role": "assistant", "content": response.content})
                current_messages.append({"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result["tool_use_id"],
                        "content": result["content"]
                    } for result in tool_results
                ]})
                
                current_round += 1
            else:
                # No tools used, return response directly
                return response.content[0].text
        
        # Max rounds reached, get final synthesis
        return self._synthesize_final_response(query, current_messages, system_content, round_results)
    
    def _build_round_context(self, original_query: str, round_results: List[Dict], current_round: int) -> str:
        """
        Build context for the current round based on previous results.
        
        Args:
            original_query: The original user query
            round_results: Results from previous rounds
            current_round: Current round number
            
        Returns:
            Context string for this round
        """
        if not round_results:
            return f"Round {current_round}: Initial analysis of query."
        
        context_parts = [f"Round {current_round}: Previous tool results available:"]
        
        for i, result in enumerate(round_results[-3:], 1):  # Show last 3 results to avoid token overflow
            tool_name = result.get("tool_name", "unknown")
            content_preview = str(result.get("content", ""))[:200]
            context_parts.append(f"- Tool {i}: {tool_name} → {content_preview}...")
        
        context_parts.append("Use this information to decide if you need additional tool calls or can provide a complete answer.")
        
        return "\n".join(context_parts)
    
    def _execute_round_tools(self, response, tool_manager, round_number: int) -> List[Dict]:
        """
        Execute all tool calls for the current round.
        
        Args:
            response: Claude's response containing tool use requests
            tool_manager: Tool manager to execute tools
            round_number: Current round number
            
        Returns:
            List of tool results with metadata
        """
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                        "tool_name": content_block.name,
                        "round": round_number,
                        "parameters": content_block.input
                    })
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append({
                        "type": "tool_result", 
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}",
                        "tool_name": content_block.name,
                        "round": round_number,
                        "error": True
                    })
        
        return tool_results
    
    def _synthesize_final_response(self, original_query: str, messages: List[Dict], 
                                 system_content: str, round_results: List[Dict]) -> str:
        """
        Generate final synthesis response after all tool rounds completed.
        
        Args:
            original_query: The original user query
            messages: Complete message history from all rounds
            system_content: System prompt content
            round_results: All tool results from all rounds
            
        Returns:
            Final synthesized response
        """
        # Create synthesis prompt
        synthesis_context = f"""
        All tool rounds completed. Synthesize the information to answer the original query.
        
        Tool Results Summary:
        {self._format_round_results(round_results)}
        
        Provide a comprehensive answer to the original query using all gathered information.
        """
        
        # Final API call without tools for synthesis
        final_params = {
            **self.base_params,
            "messages": messages + [{"role": "user", "content": synthesis_context}],
            "system": system_content
        }
        
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def _format_round_results(self, round_results: List[Dict]) -> str:
        """
        Format round results for synthesis context.
        
        Args:
            round_results: List of tool results from all rounds
            
        Returns:
            Formatted string summarizing all results
        """
        if not round_results:
            return "No tool results available."
        
        formatted = []
        current_round = 1
        round_tools = []
        
        for result in round_results:
            if result.get("round", 1) != current_round:
                if round_tools:
                    formatted.append(f"Round {current_round}: {', '.join(round_tools)}")
                current_round = result.get("round", current_round)
                round_tools = []
            
            tool_name = result.get("tool_name", "unknown")
            content_preview = str(result.get("content", ""))[:150]
            round_tools.append(f"{tool_name} → {content_preview}...")
        
        if round_tools:
            formatted.append(f"Round {current_round}: {', '.join(round_tools)}")
        
        return "\n".join(formatted)