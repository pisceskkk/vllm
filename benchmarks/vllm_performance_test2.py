#!/usr/bin/env python3
"""
vLLM Performance Test Script

This script tests vLLM's async streaming performance, specifically designed to
reproduce and verify fixes for issues that occur with small max_tokens values
where the streaming response might hang indefinitely.

Usage:
    python vllm_performance_test2.py --model <model_name> --max-tokens <tokens>
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results from a single test request"""
    success: bool
    latency: float
    response_text: str
    error: str = ""
    timed_out: bool = False


async def send_request_async(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    max_tokens: int,
    model: str,
    timeout: float = 30.0
) -> TestResult:
    """
    Send an async request to vLLM API with streaming enabled.
    
    This function includes improvements to handle:
    1. Proper stream completion detection
    2. Timeout mechanism to prevent hanging
    3. Better error handling for async requests
    4. More logging for diagnostics
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer dummy_token",
    }
    
    start_time = time.time()
    generated_text = ""
    chunks_received = 0
    done_received = False
    
    logger.debug(f"Sending request with max_tokens={max_tokens}, timeout={timeout}s")
    
    try:
        # Create a timeout for the entire request
        request_timeout = aiohttp.ClientTimeout(total=timeout)
        
        async with session.post(
            url=api_url,
            json=payload,
            headers=headers,
            timeout=request_timeout
        ) as response:
            logger.debug(f"Response status: {response.status}")
            
            if response.status != 200:
                return TestResult(
                    success=False,
                    latency=time.time() - start_time,
                    response_text="",
                    error=f"HTTP {response.status}: {response.reason}"
                )
            
            # Process streaming response with additional safeguards
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                
                try:
                    chunk_str = chunk_bytes.decode("utf-8")
                    
                    # Skip SSE comments (ping messages)
                    if chunk_str.startswith(":"):
                        logger.debug("Skipping SSE comment")
                        continue
                    
                    # Remove SSE data prefix
                    chunk_data = chunk_str.removeprefix("data: ")
                    
                    # Check for completion marker
                    if chunk_data == "[DONE]":
                        logger.debug("Received [DONE] marker")
                        done_received = True
                        break
                    
                    # Parse JSON data
                    try:
                        data = json.loads(chunk_data)
                        chunks_received += 1
                        
                        # Extract content from choices
                        if choices := data.get("choices"):
                            if len(choices) > 0:
                                text = choices[0].get("text", "")
                                generated_text += text
                                logger.debug(f"Chunk {chunks_received}: received {len(text)} chars")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON chunk: {e}, chunk: {chunk_data[:100]}...")
                        continue
                        
                except UnicodeDecodeError as e:
                    logger.warning(f"Failed to decode chunk: {e}")
                    continue
            
            # Check if we properly received the completion
            if not done_received and chunks_received == 0:
                logger.warning("No chunks received and no [DONE] marker")
                return TestResult(
                    success=False,
                    latency=time.time() - start_time,
                    response_text="",
                    error="No data received from stream"
                )
            
            logger.info(f"Request completed: {chunks_received} chunks, [DONE]: {done_received}")
            return TestResult(
                success=True,
                latency=time.time() - start_time,
                response_text=generated_text
            )
            
    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {timeout}s")
        return TestResult(
            success=False,
            latency=time.time() - start_time,
            response_text="",
            error=f"Request timed out after {timeout}s",
            timed_out=True
        )
    except Exception as e:
        logger.error(f"Request failed with exception: {e}")
        return TestResult(
            success=False,
            latency=time.time() - start_time,
            response_text="",
            error=str(e)
        )


async def run_performance_test(
    api_url: str,
    model: str,
    max_tokens_list: List[int],
    num_requests: int = 5,
    timeout: float = 30.0
) -> None:
    """Run performance tests with various max_tokens values"""
    
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a cat.",
        "What is 2 + 2?",
    ]
    
    logger.info(f"Starting performance test with {len(max_tokens_list)} token limits")
    logger.info(f"API URL: {api_url}")
    logger.info(f"Model: {model}")
    logger.info(f"Requests per token limit: {num_requests}")
    logger.info(f"Request timeout: {timeout}s")
    
    async with aiohttp.ClientSession() as session:
        for max_tokens in max_tokens_list:
            logger.info(f"\n--- Testing max_tokens = {max_tokens} ---")
            
            successful_requests = 0
            total_latency = 0.0
            timeouts = 0
            errors = []
            
            for i in range(num_requests):
                prompt = test_prompts[i % len(test_prompts)]
                logger.info(f"Request {i+1}/{num_requests}: '{prompt[:30]}...'")
                
                result = await send_request_async(
                    session=session,
                    api_url=api_url,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    model=model,
                    timeout=timeout
                )
                
                if result.success:
                    successful_requests += 1
                    total_latency += result.latency
                    logger.info(f"  ✓ Success: {result.latency:.2f}s, {len(result.response_text)} chars")
                elif result.timed_out:
                    timeouts += 1
                    logger.error(f"  ✗ Timeout: {result.latency:.2f}s")
                else:
                    errors.append(result.error)
                    logger.error(f"  ✗ Error: {result.error}")
            
            # Report results for this max_tokens value
            logger.info(f"\nResults for max_tokens = {max_tokens}:")
            logger.info(f"  Successful requests: {successful_requests}/{num_requests}")
            logger.info(f"  Timeouts: {timeouts}")
            logger.info(f"  Errors: {len(errors)}")
            
            if successful_requests > 0:
                avg_latency = total_latency / successful_requests
                logger.info(f"  Average latency: {avg_latency:.2f}s")
            
            if errors:
                logger.info(f"  Error details: {set(errors)}")
            
            # Brief pause between test batches
            await asyncio.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="vLLM Performance Test")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/v1/completions",
        help="API endpoint URL"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Model name to test"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20],
        help="List of max_tokens values to test"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of requests per max_tokens value"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("vLLM Performance Test 2.0")
    logger.info("=" * 40)
    
    try:
        asyncio.run(run_performance_test(
            api_url=args.api_url,
            model=args.model,
            max_tokens_list=args.max_tokens,
            num_requests=args.num_requests,
            timeout=args.timeout
        ))
        logger.info("\nTest completed successfully!")
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()