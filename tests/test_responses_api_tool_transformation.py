#!/usr/bin/env python3
"""
Test tool format transformation for Responses API

This tests the fix for handling both nested and flat tool formats
in the Responses API transformation layer.
"""

import pytest
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig
)


def test_nested_tool_format():
    """Test that nested tool format (correct Responses API format) is handled properly"""
    
    # Nested format as provided by Responses API
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    config = LiteLLMCompletionResponsesConfig()
    chat_tools, _ = config.transform_responses_api_tools_to_chat_completion_tools(tools)
    
    assert len(chat_tools) == 1
    assert chat_tools[0]["type"] == "function"
    assert chat_tools[0]["function"]["name"] == "get_weather"
    assert chat_tools[0]["function"]["description"] == "Get the current weather"
    assert chat_tools[0]["function"]["parameters"]["properties"]["location"]["type"] == "string"


def test_flat_tool_format_backwards_compatibility():
    """Test that flat tool format (backwards compatibility) still works"""
    
    # Flat format for backwards compatibility
    tools = [
        {
            "type": "function",
            "name": "calculate",
            "description": "Perform calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    ]
    
    config = LiteLLMCompletionResponsesConfig()
    chat_tools, _ = config.transform_responses_api_tools_to_chat_completion_tools(tools)
    
    assert len(chat_tools) == 1
    assert chat_tools[0]["type"] == "function"
    assert chat_tools[0]["function"]["name"] == "calculate"
    assert chat_tools[0]["function"]["description"] == "Perform calculations"
    assert chat_tools[0]["function"]["parameters"]["properties"]["expression"]["type"] == "string"


def test_multiple_tools_mixed_formats():
    """Test handling multiple tools with mixed formats"""
    
    tools = [
        # Nested format
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        # Flat format
        {
            "type": "function",
            "name": "calculate",
            "description": "Calculate",
            "parameters": {"type": "object", "properties": {}}
        }
    ]
    
    config = LiteLLMCompletionResponsesConfig()
    chat_tools, _ = config.transform_responses_api_tools_to_chat_completion_tools(tools)
    
    assert len(chat_tools) == 2
    assert chat_tools[0]["function"]["name"] == "get_weather"
    assert chat_tools[1]["function"]["name"] == "calculate"


def test_empty_tools_array_filtered():
    """Test that empty tools array is handled correctly"""
    
    config = LiteLLMCompletionResponsesConfig()
    
    # Test with empty tools array - should not crash
    chat_tools, web_search = config.transform_responses_api_tools_to_chat_completion_tools([])
    
    assert chat_tools == []
    assert web_search is None


def test_web_search_tool():
    """Test web search tool extraction"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "regular_function",
                "description": "Regular function",
                "parameters": {}
            }
        },
        {
            "type": "web_search"
        }
    ]
    
    config = LiteLLMCompletionResponsesConfig()
    chat_tools, web_search = config.transform_responses_api_tools_to_chat_completion_tools(tools)
    
    assert len(chat_tools) == 1
    assert chat_tools[0]["function"]["name"] == "regular_function"
    # web_search returns the web search tool object, not just the string
    assert web_search is not None


def test_tool_with_strict_parameter():
    """Test tool with strict parameter is preserved"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "strict_function",
                "description": "Function with strict mode",
                "parameters": {"type": "object", "properties": {}},
                "strict": True
            }
        }
    ]
    
    config = LiteLLMCompletionResponsesConfig()
    chat_tools, _ = config.transform_responses_api_tools_to_chat_completion_tools(tools)
    
    assert chat_tools[0]["function"]["strict"] == True


if __name__ == "__main__":
    # Run tests
    test_nested_tool_format()
    print("✅ Nested tool format test passed")
    
    test_flat_tool_format_backwards_compatibility()
    print("✅ Flat tool format backwards compatibility test passed")
    
    test_multiple_tools_mixed_formats()
    print("✅ Multiple tools mixed formats test passed")
    
    test_empty_tools_array_filtered()
    print("✅ Empty tools array filtering test passed")
    
    test_web_search_tool()
    print("✅ Web search tool extraction test passed")
    
    test_tool_with_strict_parameter()
    print("✅ Tool with strict parameter test passed")
    
    print("\n🎉 All tool transformation tests passed!")