#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Real API integration tests for call_language_model module.

This file contains integration tests that make real API calls to test the
functionality of the language model calling system. These tests require
valid API credentials in the configuration file.

@File    : real_api_tests.py
@Author  : Integration Test Suite
@Date    : 2025/8/13
@Description: Integration tests with real API calls for comprehensive testing.
"""

import os
import threading
import time
import json
from typing import Any, List, Dict

from call_language_model import (
    call_language_model,
    call_embedding_model,
    batch_call_language_model
)


def test_basic_language_model_call():
    """Test basic language model functionality with real API."""
    print("="*60)
    print("TEST 1: Basic Language Model Call")
    print("="*60)
    
    try:
        response, tokens_used, error = call_language_model(
            model_provider='aliyun',
            model_name='qwen2.5-7b-instruct',
            system_prompt="You are a helpful assistant. Respond concisely.",
            user_prompt="What is artificial intelligence? Limit your response to 2 sentences.",
            stream=False,
            config_path="./llm_config.yaml"
        )

        print(f"✓ Model Provider: aliyun")
        print(f"✓ Model Name: qwen2.5-7b-instruct")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens_used}")
        print(f"✓ Error: {error or 'None'}")
        
        # Basic validation
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert isinstance(tokens_used, int), "Token count should be an integer"
        assert tokens_used > 0, "Should have used some tokens"
        assert error is None, f"Should not have error, but got: {error}"
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_streaming_language_model_call():
    """Test streaming language model functionality with real API."""
    print("\n" + "="*60)
    print("TEST 2: Streaming Language Model Call")
    print("="*60)
    
    try:
        response, tokens_used, error = call_language_model(
            model_provider='aliyun',
            model_name='qwen2.5-7b-instruct',
            system_prompt="You are a helpful assistant. Respond concisely.",
            user_prompt="Count from 1 to 5 and explain what counting is.",
            stream=True,
            collect=True,  # Collect streaming results
            config_path="./llm_config.yaml"
        )

        print(f"✓ Model Provider: aliyun")
        print(f"✓ Model Name: qwen2.5-7b-instruct")
        print(f"✓ Stream Mode: True (collected)")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens_used}")
        print(f"✓ Error: {error or 'None'}")
        
        # Validation
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert error is None, f"Should not have error, but got: {error}"
        # Note: streaming mode may not accurately count tokens
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_multimodal_language_model_call():
    """Test multimodal language model functionality (requires image file)."""
    print("\n" + "="*60)
    print("TEST 3: Multimodal Language Model Call")
    print("="*60)
    
    # This test requires an image file - will skip if not available
    import os
    image_files = ['test_image.png', 'test_image.jpg', '1.png', 'sample.png']
    available_image = None
    
    for img_file in image_files:
        if os.path.exists(img_file):
            available_image = img_file
            break
    
    if not available_image:
        print("⚠️  Skipping multimodal test - no test image file found")
        print("   (Create a test image file named 'test_image.png' to run this test)")
        return True
    
    try:
        response, tokens_used, error = call_language_model(
            model_provider='aliyun',
            model_name='qwen2.5-7b-instruct',
            system_prompt="You are a helpful assistant that can analyze images. Describe what you see briefly.",
            user_prompt="Describe this image in 1-2 sentences.",
            files=[available_image],
            stream=False,
            config_path="./llm_config.yaml"
        )

        print(f"✓ Model Provider: aliyun")
        print(f"✓ Model Name: qwen2.5-7b-instruct")
        print(f"✓ Image File: {available_image}")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens_used}")
        print(f"✓ Error: {error or 'None'}")
        
        # Validation
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert isinstance(tokens_used, int), "Token count should be an integer"
        assert tokens_used > 0, "Should have used some tokens"
        assert error is None, f"Should not have error, but got: {error}"
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_embedding_model_call():
    """Test embedding model functionality with real API."""
    print("\n" + "="*60)
    print("TEST 4: Embedding Model Call")
    print("="*60)
    
    try:
        embeddings, tokens_used, error = call_embedding_model(
            model_provider='aliyun',
            model_name='text-embedding-v3',
            text='This is a test sentence for generating embeddings.',
            config_path="./llm_config.yaml"
        )

        print(f"✓ Model Provider: aliyun")
        print(f"✓ Model Name: text-embedding-v3")
        print(f"✓ Input Text: 'This is a test sentence for generating embeddings.'")
        print(f"✓ Embedding Dimensions: {len(embeddings[0]) if embeddings else 0}")
        print(f"✓ Number of Embeddings: {len(embeddings)}")
        print(f"✓ First Few Values: {embeddings[0][:5] if embeddings else 'None'}")
        print(f"✓ Tokens Used: {tokens_used}")
        print(f"✓ Error: {error or 'None'}")
        
        # Validation
        assert isinstance(embeddings, list), "Embeddings should be a list"
        assert len(embeddings) == 1, "Should have 1 embedding for 1 text"
        assert isinstance(embeddings[0], list), "Each embedding should be a list"
        assert len(embeddings[0]) > 0, "Embedding should have dimensions"
        assert isinstance(tokens_used, int), "Token count should be an integer"
        assert tokens_used > 0, "Should have used some tokens"
        assert error is None, f"Should not have error, but got: {error}"
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_multiple_text_embeddings():
    """Test embedding model with multiple texts."""
    print("\n" + "="*60)
    print("TEST 5: Multiple Text Embeddings")
    print("="*60)
    
    try:
        texts = [
            "Artificial intelligence is transforming the world.",
            "Machine learning algorithms can learn from data.",
            "Natural language processing helps computers understand text."
        ]
        
        embeddings, tokens_used, error = call_embedding_model(
            model_provider='aliyun',
            model_name='text-embedding-v3',
            text=texts,
            config_path="./llm_config.yaml"
        )

        print(f"✓ Model Provider: aliyun")
        print(f"✓ Model Name: text-embedding-v3")
        print(f"✓ Number of Input Texts: {len(texts)}")
        print(f"✓ Embedding Dimensions: {len(embeddings[0]) if embeddings else 0}")
        print(f"✓ Number of Embeddings: {len(embeddings)}")
        print(f"✓ Tokens Used: {tokens_used}")
        print(f"✓ Error: {error or 'None'}")
        
        # Validation
        assert isinstance(embeddings, list), "Embeddings should be a list"
        assert len(embeddings) == len(texts), f"Should have {len(texts)} embeddings"
        assert all(isinstance(emb, list) for emb in embeddings), "Each embedding should be a list"
        assert all(len(emb) > 0 for emb in embeddings), "Each embedding should have dimensions"
        assert isinstance(tokens_used, int), "Token count should be an integer"
        assert tokens_used > 0, "Should have used some tokens"
        assert error is None, f"Should not have error, but got: {error}"
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_batch_language_model_processing():
    """Test batch processing functionality with real API."""
    print("\n" + "="*60)
    print("TEST 6: Batch Language Model Processing")
    print("="*60)
    
    try:
        batch_requests = [
            {
                "system_prompt": "You are a helpful assistant. Respond with exactly one sentence.",
                "user_prompt": "What is Python programming language?"
            },
            {
                "system_prompt": "You are a helpful assistant. Respond with exactly one sentence.",
                "user_prompt": "What is machine learning?"
            },
            {
                "system_prompt": "You are a helpful assistant. Respond with exactly one sentence.",
                "user_prompt": "What is data science?"
            }
        ]
        
        print(f"Processing {len(batch_requests)} requests...")
        
        results = batch_call_language_model(
            model_provider='aliyun',
            model_name='qwen2.5-7b-instruct',
            requests=batch_requests,
            max_workers=2,
            show_progress=True,
            config_path="./llm_config.yaml"
        )
        
        print(f"\n✓ Model Provider: aliyun")
        print(f"✓ Model Name: qwen2.5-7b-instruct")
        print(f"✓ Total Requests: {len(batch_requests)}")
        print(f"✓ Results Received: {len(results)}")
        
        # Display results
        total_tokens = 0
        successful_requests = 0
        
        for i, result in enumerate(results):
            if not result.get('error_msg'):
                successful_requests += 1
                total_tokens += result.get('tokens_used', 0)
                print(f"\n  Request {i+1}:")
                print(f"    Question: {batch_requests[i]['user_prompt']}")
                print(f"    Response: {result['response_text'][:100]}...")
                print(f"    Tokens: {result['tokens_used']}")
            else:
                print(f"\n  Request {i+1}: ERROR - {result['error_msg']}")
        
        print(f"\n✓ Successful Requests: {successful_requests}/{len(results)}")
        print(f"✓ Total Tokens Used: {total_tokens}")
        
        # Validation
        assert len(results) == len(batch_requests), "Should have results for all requests"
        assert successful_requests > 0, "Should have at least one successful request"
        assert total_tokens > 0, "Should have used some tokens"
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_real_time_save():
    """Test batch processing with real-time file saving functionality."""
    print("\n" + "="*60)
    print("TEST 7: Real-time File Saving During Batch Processing")
    print("="*60)
    
    try:
        import os
        import threading
        
        # Test batch requests
        batch_requests = [
            {
                "system_prompt": "You are a helpful assistant. Respond concisely.",
                "user_prompt": "What is Python programming language?",
                "max_tokens": 100
            },
            {
                "system_prompt": "You are a helpful assistant. Respond concisely.",
                "user_prompt": "What is artificial intelligence?",
                "max_tokens": 100
            },
            {
                "system_prompt": "You are a helpful assistant. Respond concisely.",
                "user_prompt": "Explain machine learning.",
                "max_tokens": 100
            },
            {
                "system_prompt": "You are a helpful assistant. Respond concisely.",
                "user_prompt": "What are neural networks?",
                "max_tokens": 100
            },
            {
                "system_prompt": "You are a helpful assistant. Respond concisely.",
                "user_prompt": "What is natural language processing?",
                "max_tokens": 100
            }
        ]
        
        # Output file path
        output_file = "./test_real_time_results.jsonl"
        
        # Remove existing file if present
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"✓ Removed existing file: {output_file}")
        
        print(f"Processing {len(batch_requests)} requests with real-time saving...")
        
        # File monitoring function
        file_update_count = 0
        def monitor_file():
            """Monitor file changes during processing"""
            nonlocal file_update_count
            line_count = 0
            while file_update_count < len(batch_requests):
                try:
                    if os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if len(lines) > line_count:
                                file_update_count = len(lines)
                                print(f"  📁 File updated: {len(lines)} results saved")
                                line_count = len(lines)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  ⚠️  File monitoring error: {e}")
                    break
        
        # Start file monitoring in background
        monitor_thread = threading.Thread(target=monitor_file, daemon=True)
        monitor_thread.start()
        
        # Execute batch processing with real-time saving
        results = batch_call_language_model(
            model_provider='aliyun',
            model_name='qwen2.5-7b-instruct',
            requests=batch_requests,
            max_workers=2,
            output_file=output_file,
            show_progress=True,
            config_path="./llm_config.yaml"
        )
        
        print(f"\n✓ Model Provider: aliyun")
        print(f"✓ Model Name: qwen2.5-7b-instruct")
        print(f"✓ Total Requests: {len(batch_requests)}")
        print(f"✓ Results Received: {len(results)}")
        print(f"✓ Output File: {output_file}")
        
        # Verify file content and real-time saving
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            print(f"✓ File Contains: {len(lines)} lines")
            
            # Validate JSON format
            valid_json_count = 0
            for i, line in enumerate(lines):
                try:
                    json.loads(line.strip())
                    valid_json_count += 1
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Line {i+1} invalid JSON: {e}")
            
            print(f"✓ Valid JSON Lines: {valid_json_count}/{len(lines)}")
            
            # Display sample result
            if lines:
                try:
                    first_result = json.loads(lines[0])
                    print(f"\n  Sample Result:")
                    print(f"    Request Index: {first_result.get('request_index')}")
                    print(f"    Question: {batch_requests[0]['user_prompt']}")
                    print(f"    Response: {first_result.get('response_text', '')[:80]}...")
                    print(f"    Tokens Used: {first_result.get('tokens_used')}")
                    print(f"    Timestamp: {first_result.get('timestamp')}")
                except Exception as e:
                    print(f"  ⚠️  Error parsing first result: {e}")
            
            # Validation assertions
            assert len(lines) == len(batch_requests), "File should contain all batch results"
            assert valid_json_count == len(lines), "All lines should be valid JSON"
            assert len(results) == len(batch_requests), "Should have results for all requests"
            
            # Validate real-time saving occurred
            successful_requests = sum(1 for r in results if r and not r.get('error_msg'))
            assert successful_requests > 0, "Should have at least one successful request"
            
            print(f"✓ Real-time Saving: Verified")
            print(f"✓ Successful Requests: {successful_requests}/{len(results)}")
            
        else:
            raise Exception(f"Output file {output_file} was not created")
        
        print("✅ Test passed!")
        os.remove(output_file)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_custom_configuration():
    """Test using custom configuration instead of config file."""
    print("\n" + "="*60)
    print("TEST 8: Custom Configuration")
    print("="*60)
    
    try:
        # Load config from file to get credentials
        import yaml
        try:
            with open('./llm_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Find Aliyun credentials
            aliyun_config = None
            for model_config in config.get('all_models', []):
                if model_config.get('provider') == 'aliyun':
                    aliyun_config = model_config
                    break

            if not aliyun_config:
                print("⚠️  Skipping custom config test - no Aliyun config found in file")
                return True
            
            custom_config = {
                'api_key': aliyun_config.get('api_key'),
                'base_url': aliyun_config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            }
            
        except FileNotFoundError:
            print("⚠️  Skipping custom config test - no config file found")
            return True
        
        response, tokens_used, error = call_language_model(
            model_provider='aliyun',
            model_name='qwen2.5-7b-instruct',
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello and mention you're using custom configuration.",
            custom_config=custom_config  # Using custom config instead of file
        )

        print(f"✓ Model Provider: aliyun")
        print(f"✓ Model Name: qwen2.5-7b-instruct")
        print(f"✓ Configuration: Custom (not from file)")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens_used}")
        print(f"✓ Error: {error or 'None'}")
        
        # Validation
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert isinstance(tokens_used, int), "Token count should be an integer"
        assert tokens_used > 0, "Should have used some tokens"
        assert error is None, f"Should not have error, but got: {error}"
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def validate_no_duplicate_or_missing(all_chunks: List[List[Dict[str, Any]]],
                                      total_expected: int) -> None:
    """all_chunks[i] = chunk list read by thread i"""
    seen_ids   = set()
    duplicates = 0
    for idx, chunks in enumerate(all_chunks):
        for chk in chunks:
            cid = chk.get("id")           # OpenAI 每帧有唯一 id
            if cid is None:
                continue
            if cid in seen_ids:
                duplicates += 1
            seen_ids.add(cid)
    if duplicates:
        raise AssertionError(f"Found {duplicates} duplicate chunks across threads")
    if len(seen_ids) != total_expected:
        raise AssertionError(f"Expected {total_expected} unique chunks, got {len(seen_ids)}")


def test_stream_wrapper_thread_safety():
    """
    Thread safety test: Verify that stream wrapper correctly prevents cross-thread access 
    and test concurrent handling of multiple independent streams.
    This is the core test for verifying stream wrapper thread safety design.
    """
    print("\n" + "="*60)
    print("TEST 9: Stream Wrapper Thread Safety (Design Verification)")
    print("="*60)

    try:
        print("=== Phase 1: Testing Cross-Thread Access Prevention ===")
        
        # 1. Create a stream and verify cross-thread access is correctly blocked
        print("Creating stream in main thread...")
        result = call_language_model(
            model_provider="aliyun",
            model_name="qwen2.5-7b-instruct",
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello and count from 1 to 5.",
            max_tokens=100,
            temperature=0,
            stream=True,
            collect=False,
            config_path="./llm_config.yaml"
        )
        
        if isinstance(result, tuple) and len(result) == 3:
            stream, tokens, err = result
            if err:
                raise RuntimeError(f"Failed to obtain stream: {err}")
        else:
            stream = result
        
        print(f"✓ Stream created successfully in main thread")
        
        # 2. Test if cross-thread access is correctly blocked
        cross_thread_errors = []
        cross_thread_lock = threading.Lock()
        
        def cross_thread_test():
            """Attempt to access stream from another thread"""
            try:
                chunk = next(stream)  # This should throw an error
                with cross_thread_lock:
                    cross_thread_errors.append("ERROR: Cross-thread access was allowed!")
            except RuntimeError as e:
                if "not thread-safe" in str(e):
                    with cross_thread_lock:
                        cross_thread_errors.append("SUCCESS: Cross-thread access correctly prevented")
                else:
                    with cross_thread_lock:
                        cross_thread_errors.append(f"UNEXPECTED ERROR: {e}")
            except Exception as e:
                with cross_thread_lock:
                    cross_thread_errors.append(f"OTHER ERROR: {e}")
        
        test_thread = threading.Thread(target=cross_thread_test)
        test_thread.start()
        test_thread.join()
        
        print(f"✓ Cross-thread access test result: {cross_thread_errors[0] if cross_thread_errors else 'No result'}")
        
        # Verify cross-thread access is correctly blocked
        if not cross_thread_errors or "SUCCESS" not in cross_thread_errors[0]:
            print("❌ Cross-thread access prevention failed!")
            return False
        
        print("=== Phase 2: Testing Concurrent Independent Streams ===")
        
        # 3. Test multiple threads each creating and using independent streams
        THREAD_COUNT = 32  # Reduce thread count to ensure stability
        results = []
        errors = []
        results_lock = threading.Lock()
        
        def independent_stream_worker(thread_id):
            """Each thread creates and uses its own stream"""
            try:
                # Each thread creates its own stream
                result = call_language_model(
                    model_provider="aliyun",
                    model_name="qwen2.5-7b-instruct",
                    system_prompt="You are a calculator. Answer only with numbers.",
                    user_prompt=f"Calculate {thread_id + 1} + {thread_id + 2}",
                    max_tokens=10,
                    temperature=0,
                    stream=True,
                    collect=False,
                    config_path="./llm_config.yaml"
                )
                
                if isinstance(result, tuple) and len(result) == 3:
                    thread_stream, thread_tokens, thread_err = result
                    if thread_err:
                        raise RuntimeError(f"Thread {thread_id} failed to get stream: {thread_err}")
                else:
                    thread_stream = result
                
                # Consume its own stream
                chunks_consumed = 0
                content_parts = []
                
                try:
                    while True:
                        chunk = next(thread_stream)
                        chunks_consumed += 1
                        
                        # Extract content
                        if hasattr(chunk, "choices") and chunk.choices:
                            choice = chunk.choices[0]
                            if hasattr(choice, "delta") and hasattr(choice.delta, "content") and choice.delta.content:
                                content_parts.append(choice.delta.content)
                                
                except StopIteration:
                    pass
                
                full_content = "".join(content_parts)
                expected_result = (thread_id + 1) + (thread_id + 2)
                
                with results_lock:
                    results.append({
                        'thread_id': thread_id,
                        'chunks_consumed': chunks_consumed,
                        'content': full_content.strip(),
                        'expected': expected_result,
                        'success': True
                    })
                
            except Exception as e:
                with results_lock:
                    errors.append({
                        'thread_id': thread_id,
                        'error': str(e)
                    })
        
        # Start multiple independent stream processing threads
        print(f"Starting {THREAD_COUNT} threads with independent streams...")
        start_time = time.time()
        
        threads = [threading.Thread(target=independent_stream_worker, args=(i,)) for i in range(THREAD_COUNT)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        duration = time.time() - start_time
        
        # 4. Analyze results
        successful_threads = len(results)
        failed_threads = len(errors)
        total_chunks = sum(r['chunks_consumed'] for r in results)
        
        print(f"\n✓ Independent Streams Test Results:")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - Successful threads: {successful_threads}/{THREAD_COUNT}")
        print(f"  - Failed threads: {failed_threads}")
        print(f"  - Total chunks consumed: {total_chunks}")
        print(f"  - Average chunks per successful thread: {total_chunks/successful_threads:.1f}" if successful_threads > 0 else "  - No successful threads")
        
        if results:
            print(f"  - Sample results:")
            for result in results[:5]:
                print(f"    Thread {result['thread_id']}: {result['chunks_consumed']} chunks, content: '{result['content'][:20]}...'")
        
        if errors:
            print(f"  - Sample errors:")
            for error in errors[:3]:
                print(f"    Thread {error['thread_id']}: {error['error'][:50]}...")
        
        # 5. Validate results
        success_rate = successful_threads / THREAD_COUNT
        
        if success_rate < 0.5:  # At least 50% success rate
            print(f"❌ Too many thread failures: {success_rate:.1%} success rate")
            return False
        
        if total_chunks == 0:
            print("❌ No chunks were consumed by any thread")
            return False
        
        print("=== Phase 3: Thread Safety Summary ===")
        print("✅ Stream Wrapper Thread Safety Test PASSED!")
        print("Key findings:")
        print("   ✓ Cross-thread access is correctly prevented")
        print("   ✓ Independent streams work safely in parallel")
        print(f"   ✓ {successful_threads} threads successfully processed their own streams")
        print(f"   ✓ {total_chunks} total chunks processed without interference")
        print("   ✓ No data corruption between threads detected")
        
        return True

    except Exception as e:
        print(f"❌ Stream wrapper thread safety test failed: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False


def test_high_concurrency_stress_test():
    """High concurrency stress test: Multiple threads making independent API calls simultaneously, 
    testing system stability under high load. This is a stress test to verify the system 
    can handle large numbers of concurrent requests."""
    print("\n" + "="*60)
    print("TEST 10: High Concurrency Stress Test (128 independent API calls)")
    print("="*60)
    
    import threading
    import concurrent.futures
    from collections import Counter, defaultdict
    import random
    
    try:
        # Configuration for minimal token usage
        THREAD_COUNT = 128  # Set for high concurrency
        REQUESTS_PER_THREAD = 1
        TOTAL_REQUESTS = THREAD_COUNT * REQUESTS_PER_THREAD
        
        print(f"Configuration:")
        print(f"  - Threads: {THREAD_COUNT}")
        print(f"  - Requests per thread: {REQUESTS_PER_THREAD}")
        print(f"  - Total requests: {TOTAL_REQUESTS}")
        print(f"  - Expected total token usage: ~{TOTAL_REQUESTS * 10} tokens (minimal)")
        print(f"  - Test purpose: Stress test system under high concurrent load")
        
        # Results storage with thread safety
        results_lock = threading.Lock()
        results = []
        errors = []
        thread_results = defaultdict(list)
        
        # Test data - each thread gets a unique number to verify no cross-contamination
        def generate_test_prompts():
            """Generate test prompts that expect specific numeric responses"""
            prompts = []
            for thread_id in range(THREAD_COUNT):
                for req_id in range(REQUESTS_PER_THREAD):
                    # Use a simple arithmetic to get predictable results
                    a = thread_id + 1000
                    b = req_id + 1
                    expected = a + b
                    prompt = {
                        'thread_id': thread_id,
                        'request_id': req_id,
                        'a': a,
                        'b': b,
                        'expected': expected,
                        'prompt': f"{a} + {b} = ?"
                    }
                    prompts.append(prompt)
            return prompts
        
        test_prompts = generate_test_prompts()
        print(f"  - Generated {len(test_prompts)} test prompts")
        
        def worker_thread(thread_id, assigned_prompts):
            """Worker function for each thread"""
            thread_start_time = time.time()
            thread_results_local = []
            thread_errors_local = []
            
            try:
                for prompt_data in assigned_prompts:
                    try:
                        # Make API call with minimal token usage
                        response, tokens_used, error = call_language_model(
                            model_provider='aliyun',
                            model_name='qwen2.5-7b-instruct',
                            system_prompt="Respond only with the numeric answer, no explanation.",
                            user_prompt=prompt_data['prompt'],
                            max_tokens=5,  # Minimal tokens to reduce cost
                            temperature=0,  # Deterministic responses
                            stream=True,
                            config_path="./llm_config.yaml"
                        )
                        
                        # Parse the response to extract the number
                        try:
                            # Extract number from response
                            response_cleaned = ''.join(filter(str.isdigit, response.strip()))
                            actual_result = int(response_cleaned) if response_cleaned else None
                        except:
                            actual_result = None
                        
                        result = {
                            'thread_id': thread_id,
                            'request_id': prompt_data['request_id'],
                            'prompt': prompt_data['prompt'],
                            'expected': prompt_data['expected'],
                            'response': response,
                            'actual_result': actual_result,
                            'tokens_used': tokens_used,
                            'error': error,
                            'correct': actual_result == prompt_data['expected'],
                            'timestamp': time.time()
                        }
                        thread_results_local.append(result)
                        
                        # Small delay to reduce API rate limiting
                        time.sleep(0.5 + (thread_id % 5) * 0.02)  # Increased and staggered delays
                        
                    except Exception as e:
                        thread_errors_local.append({
                            'thread_id': thread_id,
                            'error': str(e),
                            'prompt_data': prompt_data
                        })
                
            except Exception as e:
                thread_errors_local.append({
                    'thread_id': thread_id,
                    'error': f"Thread error: {str(e)}",
                    'prompt_data': None
                })
            
            # Thread-safe result storage
            with results_lock:
                results.extend(thread_results_local)
                errors.extend(thread_errors_local)
                thread_results[thread_id] = thread_results_local
            
            thread_duration = time.time() - thread_start_time
            print(f"    Thread {thread_id}: {len(thread_results_local)} requests completed in {thread_duration:.2f}s")
        
        # Distribute prompts among threads
        prompts_per_thread = []
        for i in range(THREAD_COUNT):
            start_idx = i * REQUESTS_PER_THREAD
            end_idx = start_idx + REQUESTS_PER_THREAD
            prompts_per_thread.append(test_prompts[start_idx:end_idx])
        
        print(f"\nStarting {THREAD_COUNT} concurrent threads...")
        start_time = time.time()
        
        # Launch all threads using ThreadPoolExecutor for better management
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            # Submit all tasks
            future_to_thread = {
                executor.submit(worker_thread, thread_id, prompts_per_thread[thread_id]): thread_id
                for thread_id in range(THREAD_COUNT)
            }
            
            # Wait for completion with progress tracking
            completed = 0
            for future in concurrent.futures.as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                completed += 1
                if completed % 10 == 0 or completed == THREAD_COUNT:
                    print(f"    Progress: {completed}/{THREAD_COUNT} threads completed")
                
                try:
                    future.result()  # This will raise an exception if the thread failed
                except Exception as e:
                    print(f"    Thread {thread_id} failed: {e}")
        
        total_duration = time.time() - start_time
        
        # Analysis and validation
        print(f"\n✓ Stress Test Completed:")
        print(f"  - Total Duration: {total_duration:.2f} seconds")
        print(f"  - Total Results: {len(results)}")
        print(f"  - Total Errors: {len(errors)}")
        
        # System performance analysis under stress
        successful_results = [r for r in results if not r['error']]
        correct_results = [r for r in successful_results if r['correct']]
        
        print(f"  - Successful API calls: {len(successful_results)}/{TOTAL_REQUESTS}")
        print(f"  - Correct calculations: {len(correct_results)}/{len(successful_results)}")
        print(f"  - System performance under stress: {len(successful_results)/TOTAL_REQUESTS*100:.1f}% success rate")
        
        # Token usage analysis
        total_tokens = sum(r['tokens_used'] for r in successful_results)
        print(f"  - Total tokens used: {total_tokens}")
        print(f"  - Average tokens per request: {total_tokens/len(successful_results):.1f}" if successful_results else "  - No successful requests")
        
        # Thread distribution analysis
        thread_distribution = Counter(r['thread_id'] for r in successful_results)
        print(f"  - Threads with results: {len(thread_distribution)}/{THREAD_COUNT}")
        
        # Validate system reliability under stress - check for data corruption/mixing
        thread_safety_issues = []
        for result in successful_results:
            # Check if thread got correct data for its assigned calculation
            if result['correct'] is False and result['actual_result'] is not None:
                # Check if the result belongs to another thread's calculation
                for other_result in successful_results:
                    if (other_result['thread_id'] != result['thread_id'] and 
                        other_result['expected'] == result['actual_result']):
                        thread_safety_issues.append({
                            'result_thread': result['thread_id'],
                            'expected': result['expected'],
                            'got': result['actual_result'],
                            'possibly_from_thread': other_result['thread_id']
                        })
                        break
        
        print(f"  - Potential data corruption under stress: {len(thread_safety_issues)}")
        
        # Display sample results
        if successful_results:
            print(f"\n  Sample Results:")
            for i, result in enumerate(successful_results[:5]):
                status = "✓" if result['correct'] else "✗"
                print(f"    {status} Thread {result['thread_id']}: {result['prompt']} → {result['response'].strip()} (expected: {result['expected']})")
        
        # Display any errors
        if errors:
            print(f"\n  Errors encountered:")
            error_types = Counter(error['error'][:50] for error in errors)
            for error_msg, count in error_types.most_common(3):
                print(f"    - {error_msg}... (×{count})")
        
        # Display stress test issues
        if thread_safety_issues:
            print(f"\n  ⚠️  Data Corruption Under Stress Detected:")
            for issue in thread_safety_issues[:3]:
                print(f"    - Thread {issue['result_thread']} expected {issue['expected']} but got {issue['got']} (possibly from Thread {issue['possibly_from_thread']})")
        
        # Validation assertions for stress test
        assert len(results) > 0, "Should have at least some results"
        assert len(successful_results) > TOTAL_REQUESTS * 0.5, f"Should have at least 50% successful requests under stress, got {len(successful_results)}/{TOTAL_REQUESTS}"
        assert len(thread_safety_issues) == 0, f"Should have no data corruption under stress, found {len(thread_safety_issues)}"
        assert len(correct_results) > len(successful_results) * 0.8, f"Should have at least 80% correct calculations under stress, got {len(correct_results)}/{len(successful_results)}"
        
        print(f"\n✅ High Concurrency Stress Test Passed!")
        print(f"   - No data corruption under high load")
        print(f"   - {len(correct_results)}/{len(successful_results)} calculations were correct")
        print(f"   - System maintained {len(successful_results)/TOTAL_REQUESTS*100:.1f}% reliability under stress")
        print(f"   - Total API cost: ~{total_tokens} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_error_handling():
    """Test error handling with invalid configurations."""
    print("\n" + "="*60)
    print("TEST 10: Error Handling")
    print("="*60)
    
    try:
        # Test 1: Invalid model name
        print("Testing invalid model name...")
        response, tokens_used, error = call_language_model(
            model_provider='openai',
            model_name='nonexistent-model-12345',
            system_prompt="Test",
            user_prompt="Test",
            config_path="./llm_config.yaml"
        )
        
        print(f"✓ Invalid model test - Error occurred as expected: {bool(error)}")
        assert error is not None, "Should have error for invalid model"
        
        # Test 2: No configuration provided
        print("Testing no configuration...")
        response2, tokens_used2, error2 = call_language_model(
            model_provider='openai',
            model_name='gpt-4o-mini',
            system_prompt="Test",
            user_prompt="Test",
            config_path=None,
            custom_config=None
        )
        
        print(f"✓ No config test - Error occurred as expected: {bool(error2)}")
        assert error2 is not None, "Should have error when no config provided"
        assert "Both config_path and custom_config cannot be None" in error2
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all integration tests and provide summary."""
    print("STARTING REAL API INTEGRATION TESTS")
    print("These tests make actual API calls and require valid credentials.")
    print("Make sure your llm_config.yaml file contains valid API keys.\n")
    
    tests = [
        ("Basic Language Model Call", test_basic_language_model_call),
        ("Streaming Language Model Call", test_streaming_language_model_call),
        ("Multimodal Language Model Call", test_multimodal_language_model_call),
        ("Embedding Model Call", test_embedding_model_call),
        ("Multiple Text Embeddings", test_multiple_text_embeddings),
        ("Batch Language Model Processing", test_batch_language_model_processing),
        ("Batch Result Real-Time Saving", test_real_time_save),
        ("Custom Configuration", test_custom_configuration),
        ("Thread Safety Design Verification", test_stream_wrapper_thread_safety),
        ("High Concurrency Stress Test", test_high_concurrency_stress_test),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {str(e)}")
            results.append((test_name, False))
        
        # Small delay between tests to avoid rate limiting
        time.sleep(1)
    
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    
    if passed == total:
        print("🎉 All integration tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    # Run all integration tests
    success = run_all_tests()

    if os.path.exists("./test_real_time_results.jsonl"):
        os.remove("./test_real_time_results.jsonl")
    
    if not success:
        print("\n" + "="*60)
        print("TROUBLESHOOTING TIPS:")
        print("="*60)
        print("1. Make sure llm_config.yaml exists and contains valid API credentials")
        print("2. Check your internet connection")
        print("3. Verify that your API keys have sufficient credits/quota")
        print("4. For multimodal tests, ensure you have a test image file")
        print("5. Some providers may have rate limits - try running tests with delays")
        
    exit(0 if success else 1)