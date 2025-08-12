#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Real API integration tests for call_language_model module.

This file contains integration tests that make real API calls to test the
functionality of the language model calling system. These tests require
valid API credentials in the configuration file.

@File    : real_api_tests.py
@Author  : Integration Test Suite
@Date    : 2025/7/31
@Description: Integration tests with real API calls for comprehensive testing.
"""

import time
import json
from typing import List, Dict

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
    """测试批量调用时的实时文件保存功能"""
    
    # 测试请求数据
    test_requests = [
        {"system_prompt":"You are a helpful assistant.", "user_prompt": "简单介绍一下Python编程语言", "max_tokens": 100},
        {"system_prompt":"You are a helpful assistant.", "user_prompt": "什么是人工智能？", "max_tokens": 100},
        {"system_prompt":"You are a helpful assistant.", "user_prompt": "解释什么是机器学习", "max_tokens": 100},
        {"system_prompt":"You are a helpful assistant.", "user_prompt": "深度学习的基本概念", "max_tokens": 100},
        {"system_prompt":"You are a helpful assistant.", "user_prompt": "什么是自然语言处理？", "max_tokens": 100}
    ]
    
    # 输出文件路径
    output_file = "./test_real_time_results.jsonl"
    
    # 删除已存在的文件
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除现有文件: {output_file}")
    
    print("开始批量调用测试...")
    print("观察文件是否实时更新...")
    
    # 监控文件的函数
    def monitor_file():
        """监控文件变化"""
        line_count = 0
        while True:
            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) > line_count:
                            print(f"文件已更新: 现在有 {len(lines)} 行结果")
                            line_count = len(lines)
                time.sleep(1)
            except Exception as e:
                print(f"监控文件时出错: {e}")
                break
    
    # 启动文件监控（在后台）
    import threading
    monitor_thread = threading.Thread(target=monitor_file, daemon=True)
    monitor_thread.start()
    
    # 执行批量调用
    results = batch_call_language_model(
        requests=test_requests,
        model_provider="aliyun",
        model_name="qwen-plus",  # 使用较便宜的模型进行测试
        max_workers=2,  # 限制并发数
        output_file=output_file,
        show_progress=True
    )
    
    print(f"\n批量调用完成！")
    print(f"总共处理了 {len(results)} 个请求")
    
    # 验证文件内容
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"文件中保存了 {len(lines)} 行结果")
            
            # 检查每行是否都是有效的JSON
            valid_json_count = 0
            for i, line in enumerate(lines):
                try:
                    json.loads(line.strip())
                    valid_json_count += 1
                except json.JSONDecodeError as e:
                    print(f"第 {i+1} 行不是有效的JSON: {e}")
            
            print(f"其中 {valid_json_count} 行是有效的JSON格式")
            
            # 显示第一个结果的示例
            if lines:
                try:
                    first_result = json.loads(lines[0])
                    print(f"\n第一个结果示例:")
                    print(f"- 请求索引: {first_result.get('request_index')}")
                    print(f"- 响应文本: {first_result.get('response_text', '')[:100]}...")
                    print(f"- 使用的token数: {first_result.get('tokens_used')}")
                    print(f"- 时间戳: {first_result.get('timestamp')}")
                except Exception as e:
                    print(f"解析第一个结果时出错: {e}")
    else:
        print(f"输出文件 {output_file} 不存在")


def test_custom_configuration():
    """Test using custom configuration instead of config file."""
    print("\n" + "="*60)
    print("TEST 7: Custom Configuration")
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


def test_error_handling():
    """Test error handling with invalid configurations."""
    print("\n" + "="*60)
    print("TEST 8: Error Handling")
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