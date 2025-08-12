#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Demo script for testing call_language_model functions.

This script demonstrates how to use the test suite and run various
mock tests for the language model calling functions. Updated to showcase
the new OpenAI /responses endpoint and reasoning capabilities.

@File    : demo_test.py
@Author  : Demo Script
@Date    : 2025/8/12
@Description: Demonstrate testing of language model functions with mock data.
"""

import os
import sys
from unittest.mock import patch, Mock

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from call_language_model import (
    call_language_model,
    call_embedding_model,
    batch_call_language_model
)


def demo_language_model_testing():
    """Demonstrate language model testing with mock data."""
    print("="*60)
    print("DEMO: Testing Language Model Calls with Mock Data")
    print("="*60)
    
    # Test 1: Basic language model call with OpenAI /responses endpoint
    print("\n1. Testing Basic OpenAI Language Model Call (/responses endpoint):")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "Hello! I'm a mock OpenAI model using the /responses endpoint. This demonstrates the new architecture supporting reasoning models like gpt-5.",
            150,  # mock token count
            None  # no error
        )
        mock_model.return_value = mock_instance
        
        # Call the function
        response, tokens, error = call_language_model(
            model_provider='openai',  # Uses /responses endpoint
            model_name='gpt-5',
            system_prompt='You are a helpful AI assistant.',
            user_prompt='Explain the difference between OpenAI /responses and /chat/completions endpoints.',
            reasoning={
                'effort': 'high',  # OpenAI reasoning parameter
                'summary': 'auto'  # auto summarize reasoning content
            },
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai (uses /responses endpoint)")
        print(f"✓ Model Name: gpt-5")
        print(f"✓ Additional Parameters: reasoning_effort='high'")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
    
    # Test 2: OpenAI-compatible model call with /chat/completions endpoint
    print("\n2. Testing OpenAI-Compatible Language Model Call (/chat/completions endpoint):")
    print("-" * 40)
    
    with patch('call_language_model.OpenAICompatibleModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "Hello! I'm a mock OpenAI-compatible model using the /chat/completions endpoint. This is used for providers like Aliyun, Volcengine, etc.",
            120,  # mock token count
            None  # no error
        )
        mock_model.return_value = mock_instance
        
        # Call the function
        response, tokens, error = call_language_model(
            model_provider='aliyun',  # Uses /chat/completions endpoint
            model_name='qwen-max',
            system_prompt='You are a helpful AI assistant.',
            user_prompt='Explain how third-party providers work with OpenAI-compatible APIs.',
            temperature=0.8,
            max_tokens=1000,
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: aliyun (uses /chat/completions endpoint)")
        print(f"✓ Model Name: qwen-max")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
    
    # Test 3: Reasoning content demo
    print("\n3. Testing Reasoning Content Handling:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance with reasoning content
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "<think>\nLet me analyze this step by step:\n1. First, I need to understand the problem\n2. Then I can break it down\n3. Finally, provide a solution\n</think>\n\nBased on my analysis, here's the solution to your problem...",
            250,  # mock token count
            None  # no error
        )
        mock_model.return_value = mock_instance
        
        # Call the function
        response, tokens, error = call_language_model(
            model_provider='openai',
            model_name='gpt-5-mini',
            system_prompt='You are a helpful AI assistant.',
            user_prompt='Solve this complex logical puzzle step by step.',
            reasoning={
                'effort': 'high',
                'summary': 'auto'
            },
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: gpt-5-mini (reasoning model)")
        print(f"✓ Reasoning Content: {'✓ Found' if '<think>' in response else '✗ Not found'}")
        print(f"✓ Response Preview: {response[:150]}...")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
    
    # Test 2: Multimodal language model call
    print("\n4. Testing Multimodal Language Model Call:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "I can see the image you've uploaded. This is a mock response simulating multimodal capability with the enhanced OpenAI /responses endpoint.",
            200,  # mock token count
            None  # no error
        )
        # Add the _encode_image method to the mock instance
        mock_instance._encode_image.return_value = 'data:image/png;base64,mock_base64_data'
        mock_model.return_value = mock_instance
        
        # Call the function with files
        response, tokens, error = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            system_prompt='You are a helpful AI assistant that can analyze images.',
            user_prompt='Describe what you see in this image.',
            files=['mock_image.png'],  # mock image file
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: gpt-4o")
        print(f"✓ Files: ['mock_image.png']")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
    
    # Test 3: Streaming language model call with reasoning
    print("\n5. Testing Streaming Language Model Call with Reasoning:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate_stream.return_value = (
            "<think>\nLet me think about this story...\nI need to create a narrative about a robot learning art...\nThis involves creativity and learning themes...\n</think>\n\nOnce upon a time, there was a robot named ArtBot who discovered the joy of painting...",
            0,    # streaming doesn't count tokens accurately
            None  # no error
        )
        mock_model.return_value = mock_instance
        
        # Call the function in streaming mode
        response, tokens, error = call_language_model(
            model_provider='openai',
            model_name='gpt-5',
            system_prompt='You are a creative AI assistant.',
            user_prompt='Write a short story about a robot learning to paint.',
            stream=True,
            collect=True,  # collect streaming results
            reasoning={
                'effort': 'high',  # OpenAI reasoning parameter
                'summary': 'auto'  # auto summarize reasoning content
            },
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: gpt-5 (reasoning model)")
        print(f"✓ Stream Mode: True (collected)")
        print(f"✓ Reasoning Content: {'✓ Found' if '<think>' in response else '✗ Not found'}")
        print(f"✓ Response Preview: {response[:150]}...")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")


def demo_embedding_model_testing():
    """Demonstrate embedding model testing with mock data."""
    print("\n\n" + "="*60)
    print("DEMO: Testing Embedding Model Calls with Mock Data")
    print("="*60)
    
    # Test 1: Single text embedding
    print("\n1. Testing Single Text Embedding:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIEmbeddingModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate_embeddings.return_value = (
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],  # mock embedding vector
            25,   # mock token count
            None  # no error
        )
        mock_model.return_value = mock_instance
        
        # Call the function
        embeddings, tokens, error = call_embedding_model(
            model_provider='openai',
            model_name='text-embedding-3-small',
            text='This is a test sentence for generating embeddings.',
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: text-embedding-3-small")
        print(f"✓ Input Text: 'This is a test sentence for generating embeddings.'")
        print(f"✓ Embedding Dimensions: {len(embeddings[0]) if embeddings else 0}")
        print(f"✓ Number of Embeddings: {len(embeddings)}")
        print(f"✓ First Few Values: {embeddings[0][:4] if embeddings else 'None'}")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
    
    # Test 2: Multiple text embeddings
    print("\n2. Testing Multiple Text Embeddings:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIEmbeddingModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate_embeddings.return_value = (
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # embedding for text 1
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],  # embedding for text 2
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]   # embedding for text 3
            ],
            75,   # mock token count for 3 texts
            None  # no error
        )
        mock_model.return_value = mock_instance
        
        # Call the function with multiple texts
        texts = [
            "Artificial intelligence is transforming the world.",
            "Machine learning algorithms can learn from data.",
            "Natural language processing helps computers understand text."
        ]
        
        embeddings, tokens, error = call_embedding_model(
            model_provider='openai',
            model_name='text-embedding-3-small',
            text=texts,
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: text-embedding-3-small")
        print(f"✓ Number of Input Texts: {len(texts)}")
        print(f"✓ Embedding Dimensions: {len(embeddings[0]) if embeddings else 0}")
        print(f"✓ Number of Embeddings: {len(embeddings)}")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")


def demo_batch_processing_testing():
    """Demonstrate batch processing testing with mock data."""
    print("\n\n" + "="*60)
    print("DEMO: Testing Batch Language Model Processing with Mock Data")
    print("="*60)
    
    print("\n1. Testing Batch Processing:")
    print("-" * 40)
    
    # Mock the individual call_language_model function
    with patch('call_language_model.call_language_model') as mock_call:
        # Setup mock responses for different requests
        mock_responses = [
            ("Python is a high-level programming language known for its simplicity and readability.", 45, None),
            ("Machine learning is a subset of AI that enables computers to learn without explicit programming.", 52, None),
            ("Data science combines statistics, programming, and domain knowledge to extract insights from data.", 48, None),
            ("Cloud computing provides on-demand access to computing resources over the internet.", 41, None),
            ("Blockchain is a distributed ledger technology that ensures transparency and security.", 46, None)
        ]
        
        mock_call.side_effect = mock_responses
        
        # Prepare batch requests
        batch_requests = [
            {
                "system_prompt": "You are a tech expert. Provide clear, concise explanations.",
                "user_prompt": "Explain Python programming language in one sentence."
            },
            {
                "system_prompt": "You are a tech expert. Provide clear, concise explanations.",
                "user_prompt": "What is machine learning?"
            },
            {
                "system_prompt": "You are a tech expert. Provide clear, concise explanations.",
                "user_prompt": "Define data science."
            },
            {
                "system_prompt": "You are a tech expert. Provide clear, concise explanations.",
                "user_prompt": "What is cloud computing?"
            },
            {
                "system_prompt": "You are a tech expert. Provide clear, concise explanations.",
                "user_prompt": "Explain blockchain technology."
            }
        ]
        
        # Call batch processing function
        results = batch_call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            requests=batch_requests,
            max_workers=3,
            show_progress=False,  # Disable progress bar for demo
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: gpt-4o")
        print(f"✓ Total Requests: {len(batch_requests)}")
        print(f"✓ Max Workers: 3")
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


def demo_error_handling_testing():
    """Demonstrate error handling testing."""
    print("\n\n" + "="*60)
    print("DEMO: Testing Error Handling with Mock Data")
    print("="*60)
    
    # Test 1: Invalid configuration
    print("\n1. Testing Invalid Configuration:")
    print("-" * 40)
    
    response, tokens, error = call_language_model(
        model_provider='openai',
        model_name='gpt-4o',
        system_prompt='Test prompt',
        user_prompt='Test prompt',
        config_path=None,  # No config provided
        custom_config=None
    )
    
    print(f"✓ Expected Error Occurred: {bool(error)}")
    print(f"✓ Error Message: {error}")
    print(f"✓ Response: '{response}'")
    print(f"✓ Tokens: {tokens}")
    
    # Test 2: Model not found in config
    print("\n2. Testing Model Not Found in Config:")
    print("-" * 40)
    
    response, tokens, error = call_language_model(
        model_provider='nonexistent_provider',
        model_name='nonexistent_model',
        system_prompt='Test prompt',
        user_prompt='Test prompt',
        config_path='./test_config.yaml'
    )
    
    print(f"✓ Expected Error Occurred: {bool(error)}")
    print(f"✓ Error Message: {error}")
    print(f"✓ Response: '{response}'")
    print(f"✓ Tokens: {tokens}")
    
    # Test 3: Embedding model error
    print("\n3. Testing Embedding Model Error:")
    print("-" * 40)
    
    embeddings, tokens, error = call_embedding_model(
        model_provider='openai',
        model_name='nonexistent_embedding_model',
        text='Test text',
        config_path='./test_config.yaml'
    )
    
    print(f"✓ Expected Error Occurred: {bool(error)}")
    print(f"✓ Error Message: {error}")
    print(f"✓ Embeddings: {embeddings}")
    print(f"✓ Tokens: {tokens}")


def demo_custom_config_testing():
    """Demonstrate custom configuration testing."""
    print("\n\n" + "="*60)
    print("DEMO: Testing Custom Configuration and Advanced Parameters")
    print("="*60)
    
    print("\n1. Testing Custom Config Instead of File:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "This response was generated using custom configuration instead of a config file. The new architecture supports flexible parameter passing.",
            120,
            None
        )
        mock_model.return_value = mock_instance
        
        # Custom configuration
        custom_config = {
            'api_key': 'custom-api-key-xyz',
            'base_url': 'https://custom-api-endpoint.com/v1'
        }
        
        # Call with custom config
        response, tokens, error = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            system_prompt='You are a helpful assistant.',
            user_prompt='Respond using the custom configuration.',
            custom_config=custom_config  # Using custom config instead of file
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: gpt-4o")
        print(f"✓ Custom API Key: {custom_config['api_key']}")
        print(f"✓ Custom Base URL: {custom_config['base_url']}")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
    
    print("\n2. Testing Advanced Parameters (kwargs):")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "<think>\nThis is a complex reasoning problem that requires high-effort thinking...\nLet me analyze the mathematical relationships...\nApplying logical deduction...\n</think>\n\nBased on my detailed analysis, the solution is...",
            300,
            None
        )
        mock_model.return_value = mock_instance
        
        # Call with advanced parameters
        response, tokens, error = call_language_model(
            model_provider='openai',
            model_name='gpt-5',
            system_prompt='You are a mathematical reasoning expert.',
            user_prompt='Solve this complex optimization problem with detailed reasoning.',
            reasoning={
                'effort': 'high',  # OpenAI reasoning parameter
                'summary': 'auto'  # auto summarize reasoning content
            },
            max_output_tokens=3000,  # Control response length
            temperature=0.1,  # Low temperature for precise reasoning
            custom_config={
                'api_key': 'reasoning-model-key',
                'base_url': 'https://api.openai.com/v1'
            }
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: gpt-5")
        print(f"✓ Advanced Parameters:")
        print(f"    - reasoning_effort: high")
        print(f"    - reasoning_summary: auto")
        print(f"    - max_output_tokens: 3000")
        print(f"    - temperature: 0.1")
        print(f"✓ Reasoning Content: {'✓ Found' if '<think>' in response else '✗ Not found'}")
        print(f"✓ Response Preview: {response[:150]}...")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
        
        # Verify the parameters were passed correctly
        call_args = mock_instance.generate.call_args
        if call_args:
            print(f"✓ Parameters passed to model: {list(call_args.kwargs.keys())}")
    
    print("\n3. Testing OpenAI-Compatible Provider with Custom Parameters:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAICompatibleModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "This is a response from an OpenAI-compatible provider with custom parameters. The system correctly routes different providers to their appropriate endpoints.",
            180,
            None
        )
        mock_model.return_value = mock_instance
        
        # Call with custom parameters for compatible provider
        response, tokens, error = call_language_model(
            model_provider='volcengine',
            model_name='doubao-1-5-pro-256k-250115',
            system_prompt='You are a helpful assistant.',
            user_prompt='Explain the difference between API endpoints.',
            temperature=0.7,
            max_tokens=1500,
            top_p=0.9,  # Provider-specific parameter
            frequency_penalty=0.1,  # Another provider-specific parameter
            custom_config={
                'api_key': 'volcengine-api-key',
                'base_url': 'https://ark.cn-beijing.volces.com/api/v3/'
            }
        )
        
        print(f"✓ Model Provider: volcengine (uses OpenAI-compatible endpoint)")
        print(f"✓ Model Name: doubao-1-5-pro-256k-250115")
        print(f"✓ Custom Parameters:")
        print(f"    - temperature: 0.7")
        print(f"    - max_tokens: 1500")
        print(f"    - top_p: 0.9")
        print(f"    - frequency_penalty: 0.1")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
        
        # Verify the parameters were passed correctly
        call_args = mock_instance.generate.call_args
        if call_args:
            print(f"✓ Parameters passed to model: {list(call_args.kwargs.keys())}")


def main():
    """Run all demo tests."""
    print("STARTING COMPREHENSIVE DEMO OF LANGUAGE MODEL TESTING")
    print("This demo shows how to test language model functions using mock data")
    print("without making real API calls.")
    
    try:
        # Run all demo functions
        demo_language_model_testing()
        demo_embedding_model_testing()
        demo_batch_processing_testing()
        demo_error_handling_testing()
        demo_custom_config_testing()
        
        print("\n\n" + "="*60)
        print("ALL DEMO TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ OpenAI native /responses endpoint support (for reasoning models)")
        print("✓ OpenAI-compatible /chat/completions endpoint support")
        print("✓ Reasoning content detection and formatting")
        print("✓ Advanced parameter passing (reasoning_effort, max_output_tokens, etc.)")
        print("✓ Basic language model calls with mock responses")
        print("✓ Multimodal language model calls (text + images)")
        print("✓ Streaming language model calls with reasoning content")
        print("✓ Single and multiple text embeddings")
        print("✓ Batch processing with parallel execution")
        print("✓ Error handling for various scenarios")
        print("✓ Custom configuration usage")
        print("✓ Flexible parameter passing for different providers")
        print("✓ Enhanced model routing based on provider type")
        print("\nAll tests used mock data - no real API calls were made!")
        print("The new architecture supports both OpenAI's advanced reasoning models")
        print("and traditional OpenAI-compatible providers seamlessly!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
