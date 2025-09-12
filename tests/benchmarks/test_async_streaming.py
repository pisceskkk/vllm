#!/usr/bin/env python3
"""
Unit tests for the improved async request functions.

These tests verify that the stream handling improvements work correctly,
particularly for timeout handling and stream completion detection.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

from benchmarks.backend_request_func import (
    async_request_openai_completions,
    async_request_openai_chat_completions,
    RequestFuncInput,
    RequestFuncOutput
)


class MockStreamingResponse:
    """Mock aiohttp response that simulates streaming behavior"""
    
    def __init__(self, chunks, status=200, delay_between_chunks=0.1):
        self.status = status
        self.reason = "OK" if status == 200 else "Error"
        self.chunks = chunks
        self.delay = delay_between_chunks
    
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
        if not self.chunks:
            raise StopAsyncIteration
        
        chunk = self.chunks.pop(0)
        await asyncio.sleep(self.delay)
        return chunk.encode('utf-8')


@pytest.fixture
def sample_request_input():
    """Create a sample RequestFuncInput for testing"""
    return RequestFuncInput(
        prompt="Hello, how are you?",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=5,
        output_len=10,
        model="test-model"
    )


@pytest.mark.asyncio
async def test_completions_normal_stream():
    """Test normal streaming completion with [DONE] marker"""
    chunks = [
        'data: {"choices": [{"text": "I am"}]}',
        'data: {"choices": [{"text": " doing"}]}',
        'data: {"choices": [{"text": " well"}]}',
        'data: [DONE]'
    ]
    
    mock_response = MockStreamingResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Hello",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=10,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value = mock_response
        
        result = await async_request_openai_completions(request_input)
        
        assert result.success
        assert "I am doing well" == result.generated_text
        assert result.ttft > 0


@pytest.mark.asyncio
async def test_completions_without_done_marker():
    """Test streaming completion without [DONE] marker (should still succeed)"""
    chunks = [
        'data: {"choices": [{"text": "Hello"}]}',
        'data: {"choices": [{"text": " world"}]}',
        # Missing [DONE] marker
    ]
    
    mock_response = MockStreamingResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Hi",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value = mock_response
        
        result = await async_request_openai_completions(request_input)
        
        assert result.success  # Should still succeed even without [DONE]
        assert "Hello world" == result.generated_text


@pytest.mark.asyncio 
async def test_completions_malformed_chunks():
    """Test handling of malformed chunks"""
    chunks = [
        'data: {"choices": [{"text": "Good"}]}',
        'data: {invalid json}',  # This should be skipped
        'data: {"choices": [{"text": " response"}]}',
        'data: [DONE]'
    ]
    
    mock_response = MockStreamingResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Test",
        api_url="http://localhost:8000/v1/completions", 
        prompt_len=1,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value = mock_response
        
        result = await async_request_openai_completions(request_input)
        
        assert result.success
        assert "Good response" == result.generated_text


@pytest.mark.asyncio
async def test_completions_empty_response():
    """Test handling of empty response"""
    chunks = []
    
    mock_response = MockStreamingResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Test",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value = mock_response
        
        result = await async_request_openai_completions(request_input)
        
        assert not result.success
        assert "Never received a valid chunk" in result.error


@pytest.mark.asyncio
async def test_chat_completions_normal_stream():
    """Test normal chat completion streaming"""
    chunks = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " there"}}]}',
        'data: [DONE]'
    ]
    
    mock_response = MockStreamingResponse(chunks)
    
    request_input = RequestFuncInput(
        prompt="Hi",
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=1,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value = mock_response
        
        result = await async_request_openai_chat_completions(request_input)
        
        assert result.success
        assert "Hello there" == result.generated_text
        assert result.ttft > 0


@pytest.mark.asyncio
async def test_stream_timeout():
    """Test stream processing timeout"""
    # Create a response that takes too long between chunks
    chunks = [
        'data: {"choices": [{"text": "Slow"}]}',
        'data: {"choices": [{"text": " response"}]}',
    ]
    
    # Mock response with long delay that should trigger timeout
    mock_response = MockStreamingResponse(chunks, delay_between_chunks=70.0)  # Longer than 60s timeout
    
    request_input = RequestFuncInput(
        prompt="Test",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=5,
        model="test-model"
    )
    
    with patch('benchmarks.backend_request_func.aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value = mock_response
        
        # Patch time.perf_counter to simulate timeout condition
        original_time = time.perf_counter
        call_count = [0]
        
        def mock_time():
            call_count[0] += 1
            if call_count[0] <= 2:  # First couple calls return normal time
                return original_time() 
            else:  # Later calls simulate timeout
                return original_time() + 70.0  # Simulate 70 seconds passed
        
        with patch('benchmarks.backend_request_func.time.perf_counter', side_effect=mock_time):
            result = await async_request_openai_completions(request_input)
            
            assert not result.success
            assert "timed out" in result.error.lower()


if __name__ == "__main__":
    # Run a simple test
    asyncio.run(test_completions_normal_stream())
    print("Basic test passed!")