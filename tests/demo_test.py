#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Demo script for testing call_language_model functions.

This script demonstrates how to use the test suite and run various
mock tests for the language model calling functions.

@File    : demo_test.py
@Author  : Demo Script
@Date    : 2025/7/31
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
    
    # Test 1: Basic language model call
    print("\n1. Testing Basic Language Model Call:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "Hello! I'm a mock language model. This is a simulated response to demonstrate testing without real API calls.",
            150,  # mock token count
            None  # no error
        )
        mock_model.return_value = mock_instance
        
        # Call the function
        response, tokens, error = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            system_prompt='You are a helpful AI assistant.',
            user_prompt='Explain what artificial intelligence is in simple terms.',
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: gpt-4o")
        print(f"✓ Response: {response}")
        print(f"✓ Tokens Used: {tokens}")
        print(f"✓ Error: {error or 'None'}")
    
    # Test 2: Multimodal language model call
    print("\n2. Testing Multimodal Language Model Call:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "I can see the image you've uploaded. This is a mock response simulating multimodal capability. In a real scenario, I would describe what I see in the image.",
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
    
    # Test 3: Streaming language model call
    print("\n3. Testing Streaming Language Model Call:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate_stream.return_value = (
            "This is a mock streaming response. In real streaming, this would come piece by piece, but for testing we return the complete collected result.",
            0,    # streaming doesn't count tokens accurately
            None  # no error
        )
        mock_model.return_value = mock_instance
        
        # Call the function in streaming mode
        response, tokens, error = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            system_prompt='You are a helpful AI assistant.',
            user_prompt='Write a short story about a robot learning to paint.',
            stream=True,
            collect=True,  # collect streaming results
            config_path='./test_config.yaml'
        )
        
        print(f"✓ Model Provider: openai")
        print(f"✓ Model Name: gpt-4o")
        print(f"✓ Stream Mode: True (collected)")
        print(f"✓ Response: {response}")
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
    print("DEMO: Testing Custom Configuration with Mock Data")
    print("="*60)
    
    print("\n1. Testing Custom Config Instead of File:")
    print("-" * 40)
    
    with patch('call_language_model.OpenAIModel') as mock_model:
        # Create mock instance
        mock_instance = Mock()
        mock_instance.generate.return_value = (
            "This response was generated using custom configuration instead of a config file.",
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
        print("✓ Basic language model calls with mock responses")
        print("✓ Multimodal language model calls (text + images)")
        print("✓ Streaming language model calls")
        print("✓ Single and multiple text embeddings")
        print("✓ Batch processing with parallel execution")
        print("✓ Error handling for various scenarios")
        print("✓ Custom configuration usage")
        print("\nAll tests used mock data - no real API calls were made!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
