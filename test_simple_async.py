#!/usr/bin/env python3
"""
Simple test to verify async functions work correctly without pytest dependency.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from benchmarks.backend_request_func import (
    async_request_openai_completions,
    RequestFuncInput,
)


class MockResponse:
    def __init__(self, chunks, status=200):
        self.status = status
        self.reason = "OK" if status == 200 else "Error"
        self.chunks = iter(chunks)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, *args):
        pass
        
    @property
    def content(self):
        return self
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        try:
            chunk = next(self.chunks)
            return chunk.encode('utf-8')
        except StopIteration:
            raise StopAsyncIteration


async def test_normal_streaming():
    """Test normal streaming with [DONE] marker"""
    print("Testing normal streaming...")
    
    chunks = [
        'data: {"choices": [{"text": "Hello"}]}',
        'data: {"choices": [{"text": " world"}]}', 
        'data: {"choices": [{"text": "!"}]}',
        'data: [DONE]'
    ]
    
    mock_response = MockResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Hi there",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=2,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        mock_session_instance.post.return_value = mock_response
        
        result = await async_request_openai_completions(request_input)
        
        print(f"  Success: {result.success}")
        print(f"  Generated text: '{result.generated_text}'")
        print(f"  TTFT: {result.ttft}")
        print(f"  Error: {result.error}")
        
        assert result.success, f"Expected success but got error: {result.error}"
        assert result.generated_text == "Hello world!", f"Expected 'Hello world!' but got '{result.generated_text}'"
        assert result.ttft > 0, "Expected positive TTFT"
        
        print("  ✓ Test passed!")


async def test_without_done_marker():
    """Test streaming without [DONE] marker"""
    print("\nTesting streaming without [DONE] marker...")
    
    chunks = [
        'data: {"choices": [{"text": "No"}]}',
        'data: {"choices": [{"text": " done"}]}',
        'data: {"choices": [{"text": " marker"}]}'
        # No [DONE] marker
    ]
    
    mock_response = MockResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Test",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        mock_session_instance.post.return_value = mock_response
        
        result = await async_request_openai_completions(request_input)
        
        print(f"  Success: {result.success}")
        print(f"  Generated text: '{result.generated_text}'")
        print(f"  Error: {result.error}")
        
        # Should still succeed even without [DONE] marker
        assert result.success, f"Expected success but got error: {result.error}"
        assert result.generated_text == "No done marker", f"Expected 'No done marker' but got '{result.generated_text}'"
        
        print("  ✓ Test passed!")


async def test_malformed_chunks():
    """Test handling of malformed chunks"""
    print("\nTesting malformed chunks...")
    
    chunks = [
        'data: {"choices": [{"text": "Good"}]}',
        'data: {bad json}',  # This should be skipped
        ':', # SSE comment, should be skipped
        'data: {"choices": [{"text": " data"}]}',
        'data: [DONE]'
    ]
    
    mock_response = MockResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Test",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        mock_session_instance.post.return_value = mock_response
        
        result = await async_request_openai_completions(request_input)
        
        print(f"  Success: {result.success}")
        print(f"  Generated text: '{result.generated_text}'")
        print(f"  Error: {result.error}")
        
        assert result.success, f"Expected success but got error: {result.error}"
        assert result.generated_text == "Good data", f"Expected 'Good data' but got '{result.generated_text}'"
        
        print("  ✓ Test passed!")


async def test_empty_response():
    """Test handling of empty response"""
    print("\nTesting empty response...")
    
    chunks = []  # No chunks at all
    
    mock_response = MockResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Test",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        mock_session_instance.post.return_value = mock_response
        
        result = await async_request_openai_completions(request_input)
        
        print(f"  Success: {result.success}")
        print(f"  Generated text: '{result.generated_text}'")
        print(f"  Error: {result.error}")
        
        assert not result.success, "Expected failure for empty response"
        assert "Never received a valid chunk" in result.error, f"Expected appropriate error message but got: {result.error}"
        
        print("  ✓ Test passed!")


async def main():
    """Run all tests"""
    print("Running async streaming tests...")
    print("=" * 50)
    
    try:
        await test_normal_streaming()
        await test_without_done_marker()
        await test_malformed_chunks()
        await test_empty_response()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)