#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Test suite for call_language_model module using mock data.

This test file provides comprehensive testing for all language model calling methods
without making real API calls. Uses mock objects to simulate API responses.

@File    : test_call_language_model.py
@Author  : Test Suite
@Date    : 2025/7/31
@Description: Test language model and embedding model functions with mock data.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional

# Import the module to be tested
from call_language_model import (
    ModelConfig,
    OpenAIModel,
    OllamaModel,
    OpenAIEmbeddingModel,
    OllamaEmbeddingModel,
    call_language_model,
    call_embedding_model,
    batch_call_language_model
)


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig class."""
    
    def setUp(self):
        """Set up test fixtures with mock configuration."""
        self.test_config = {
            'all_models': [
                {
                    'provider': 'openai',
                    'model_name': ['gpt-4o', 'gpt-4o-mini'],
                    'api_key': 'test-openai-key',
                    'base_url': 'https://api.openai.com/v1'
                },
                {
                    'provider': 'ollama',
                    'model_name': ['llama3.1:8b', 'qwen3:7b'],
                    'base_url': 'http://localhost:11434'
                }
            ],
            'embedding_models': [
                {
                    'provider': 'openai',
                    'model_name': ['text-embedding-3-small', 'text-embedding-3-large'],
                    'api_key': 'test-openai-key',
                    'base_url': 'https://api.openai.com/v1'
                },
                {
                    'provider': 'ollama',
                    'model_name': ['nomic-embed-text', 'mxbai-embed-large'],
                    'base_url': 'http://localhost:11434'
                }
            ]
        }
        
        # Create temporary config file
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        import yaml
        yaml.dump(self.test_config, self.temp_config_file)
        self.temp_config_file.close()
        
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config_file.name)
    
    def test_load_config_success(self):
        """Test successful configuration loading."""
        config = ModelConfig(self.temp_config_file.name)
        self.assertIsNotNone(config.config)
        self.assertIn('all_models', config.config)
        self.assertIn('embedding_models', config.config)
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            ModelConfig('non_existent_file.yaml')
    
    def test_get_credentials_valid_model(self):
        """Test getting credentials for valid model."""
        config = ModelConfig(self.temp_config_file.name)
        credentials = config.get_credentials('openai', 'gpt-4o')
        
        self.assertEqual(credentials['provider'], 'openai')
        self.assertEqual(credentials['model_name'], 'gpt-4o')
        self.assertEqual(credentials['api_key'], 'test-openai-key')
    
    def test_get_credentials_invalid_model(self):
        """Test getting credentials for invalid model."""
        config = ModelConfig(self.temp_config_file.name)
        credentials = config.get_credentials('openai', 'invalid-model')
        
        self.assertEqual(credentials, {})
    
    def test_get_credentials_skip_checking(self):
        """Test getting credentials with skip checking enabled."""
        config = ModelConfig(self.temp_config_file.name)
        credentials = config.get_credentials('openai', 'any-model', skip_checking=True)
        
        self.assertEqual(credentials['provider'], 'openai')
        self.assertEqual(credentials['model_name'], 'any-model')
    
    def test_get_embedding_credentials_valid_model(self):
        """Test getting embedding credentials for valid model."""
        config = ModelConfig(self.temp_config_file.name)
        credentials = config.get_embedding_credentials('openai', 'text-embedding-3-small')
        
        self.assertEqual(credentials['provider'], 'openai')
        self.assertEqual(credentials['model_name'], 'text-embedding-3-small')


class TestOpenAIModel(unittest.TestCase):
    """Test cases for OpenAIModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_credentials = {
            'provider': 'openai',
            'model_name': 'gpt-4o',
            'api_key': 'test-key',
            'base_url': 'https://api.openai.com/v1'
        }
        
        # Mock ChatCompletion response
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        
        # Create a simple object for message that only has content attribute
        class MockMessage:
            def __init__(self):
                self.content = "This is a test response."
        
        self.mock_response.choices[0].message = MockMessage()
        self.mock_response.usage = Mock()
        self.mock_response.usage.total_tokens = 100
    
    @patch('call_language_model.OpenAI')
    def test_openai_model_init(self, mock_openai):
        """Test OpenAI model initialization."""
        model = OpenAIModel(self.mock_credentials)
        
        mock_openai.assert_called_once_with(
            api_key='test-key',
            base_url='https://api.openai.com/v1'
        )
        self.assertEqual(model.credentials, self.mock_credentials)
    
    @patch('call_language_model.OpenAI')
    def test_generate_success(self, mock_openai):
        """Test successful text generation."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.mock_response
        mock_openai.return_value = mock_client
        
        model = OpenAIModel(self.mock_credentials)
        result = model.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, world!"
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "This is a test response.")
        self.assertEqual(tokens_used, 100)
        self.assertIsNone(error)
    
    @patch('call_language_model.OpenAI')
    def test_generate_with_files(self, mock_openai):
        """Test generation with multimodal files."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.mock_response
        mock_openai.return_value = mock_client
        
        # Mock image encoding
        with patch.object(OpenAIModel, '_encode_image', return_value='data:image/png;base64,test'):
            model = OpenAIModel(self.mock_credentials)
            result = model.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Describe this image.",
                files=['test.png']
            )
            
            response_text, tokens_used, error = result
            self.assertEqual(response_text, "This is a test response.")
            self.assertEqual(tokens_used, 100)
            self.assertIsNone(error)
    
    @patch('call_language_model.OpenAI')
    def test_generate_stream_collected(self, mock_openai):
        """Test streaming generation with collected results."""
        # Create custom delta classes that only have content attribute
        class MockDelta1:
            def __init__(self):
                self.content = "Hello"
        
        class MockDelta2:
            def __init__(self):
                self.content = " world!"
        
        # Mock streaming response with proper delta structure
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta = MockDelta1()
        
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta = MockDelta2()
        
        mock_stream = [mock_chunk1, mock_chunk2]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai.return_value = mock_client
        
        model = OpenAIModel(self.mock_credentials)
        result = model.generate_stream(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello!",
            collect=True
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "Hello world!")
        self.assertEqual(tokens_used, 0)  # Streaming doesn't count tokens
        self.assertIsNone(error)


class TestOllamaModel(unittest.TestCase):
    """Test cases for OllamaModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_credentials = {
            'provider': 'ollama',
            'model_name': 'llama3.1:8b',
            'base_url': 'http://localhost:11434'
        }
        
        # Mock Ollama response
        self.mock_response = Mock()
        self.mock_response.message = Mock()
        self.mock_response.message.content = "This is a test response from Ollama."
        self.mock_response.eval_count = 50
        self.mock_response.prompt_eval_count = 30
    
    @patch('call_language_model.ollama')
    def test_generate_success(self, mock_ollama):
        """Test successful text generation with Ollama."""
        mock_ollama.chat.return_value = self.mock_response
        
        model = OllamaModel(self.mock_credentials)
        result = model.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, world!"
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "This is a test response from Ollama.")
        self.assertEqual(tokens_used, 80)  # eval_count + prompt_eval_count
        self.assertIsNone(error)
    
    @patch('call_language_model.ollama')
    def test_generate_stream_collected(self, mock_ollama):
        """Test streaming generation with collected results."""
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.message = Mock()
        mock_chunk1.message.content = "Hello"
        
        mock_chunk2 = Mock()
        mock_chunk2.message = Mock()
        mock_chunk2.message.content = " from Ollama!"
        
        mock_stream = [mock_chunk1, mock_chunk2]
        mock_ollama.chat.return_value = mock_stream
        
        model = OllamaModel(self.mock_credentials)
        result = model.generate_stream(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello!",
            collect=True
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "Hello from Ollama!")
        self.assertEqual(tokens_used, 0)  # Streaming doesn't count tokens precisely
        self.assertIsNone(error)


class TestOpenAIEmbeddingModel(unittest.TestCase):
    """Test cases for OpenAIEmbeddingModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_credentials = {
            'provider': 'openai',
            'model_name': 'text-embedding-3-small',
            'api_key': 'test-key',
            'base_url': 'https://api.openai.com/v1'
        }
        
        # Mock embedding response
        self.mock_response = Mock()
        self.mock_response.data = [Mock(), Mock()]
        self.mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        self.mock_response.data[1].embedding = [0.4, 0.5, 0.6]
        self.mock_response.usage = Mock()
        self.mock_response.usage.total_tokens = 50
    
    @patch('call_language_model.OpenAI')
    def test_generate_embeddings_single_text(self, mock_openai):
        """Test embedding generation for single text."""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = self.mock_response
        mock_openai.return_value = mock_client
        
        model = OpenAIEmbeddingModel(self.mock_credentials)
        result = model.generate_embeddings("Test text")
        
        embeddings, tokens_used, error = result
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(tokens_used, 50)
        self.assertIsNone(error)
    
    @patch('call_language_model.OpenAI')
    def test_generate_embeddings_multiple_texts(self, mock_openai):
        """Test embedding generation for multiple texts."""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = self.mock_response
        mock_openai.return_value = mock_client
        
        model = OpenAIEmbeddingModel(self.mock_credentials)
        result = model.generate_embeddings(["Text 1", "Text 2"])
        
        embeddings, tokens_used, error = result
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(tokens_used, 50)
        self.assertIsNone(error)


class TestOllamaEmbeddingModel(unittest.TestCase):
    """Test cases for OllamaEmbeddingModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_credentials = {
            'provider': 'ollama',
            'model_name': 'nomic-embed-text',
            'base_url': 'http://localhost:11434'
        }
        
        # Mock Ollama embedding response
        self.mock_response = Mock()
        self.mock_response.embedding = [0.1, 0.2, 0.3, 0.4]
        self.mock_response.eval_count = 25
    
    @patch('call_language_model.ollama')
    def test_generate_embeddings_single_text(self, mock_ollama):
        """Test embedding generation for single text with Ollama."""
        mock_ollama.embeddings.return_value = self.mock_response
        
        model = OllamaEmbeddingModel(self.mock_credentials)
        result = model.generate_embeddings("Test text")
        
        embeddings, tokens_used, error = result
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(tokens_used, 25)
        self.assertIsNone(error)
    
    @patch('call_language_model.ollama')
    def test_generate_embeddings_multiple_texts(self, mock_ollama):
        """Test embedding generation for multiple texts with Ollama."""
        mock_ollama.embeddings.return_value = self.mock_response
        
        model = OllamaEmbeddingModel(self.mock_credentials)
        result = model.generate_embeddings(["Text 1", "Text 2"])
        
        embeddings, tokens_used, error = result
        self.assertEqual(len(embeddings), 2)  # Two texts processed
        self.assertEqual(tokens_used, 50)  # 25 * 2
        self.assertIsNone(error)


class TestMainFunctions(unittest.TestCase):
    """Test cases for main calling functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'all_models': [
                {
                    'provider': 'openai',
                    'model_name': ['gpt-4o'],
                    'api_key': 'test-openai-key',
                    'base_url': 'https://api.openai.com/v1'
                }
            ],
            'embedding_models': [
                {
                    'provider': 'openai',
                    'model_name': ['text-embedding-3-small'],
                    'api_key': 'test-openai-key',
                    'base_url': 'https://api.openai.com/v1'
                }
            ]
        }
        
        # Create temporary config file
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        import yaml
        yaml.dump(self.test_config, self.temp_config_file)
        self.temp_config_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config_file.name)
    
    @patch('call_language_model.OpenAIModel')
    def test_call_language_model_with_config_file(self, mock_openai_model):
        """Test call_language_model function with config file."""
        # Mock model instance
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = ("Test response", 100, None)
        mock_openai_model.return_value = mock_model_instance
        
        result = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            system_prompt='You are a helpful assistant.',
            user_prompt='Hello!',
            config_path=self.temp_config_file.name
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "Test response")
        self.assertEqual(tokens_used, 100)
        self.assertIsNone(error)
    
    @patch('call_language_model.OpenAIModel')
    def test_call_language_model_with_custom_config(self, mock_openai_model):
        """Test call_language_model function with custom config."""
        # Mock model instance
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = ("Custom response", 150, None)
        mock_openai_model.return_value = mock_model_instance
        
        custom_config = {
            'api_key': 'custom-key',
            'base_url': 'https://custom.api.com/v1'
        }
        
        result = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            system_prompt='You are a helpful assistant.',
            user_prompt='Hello!',
            custom_config=custom_config
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "Custom response")
        self.assertEqual(tokens_used, 150)
        self.assertIsNone(error)
    
    @patch('call_language_model.OpenAIEmbeddingModel')
    def test_call_embedding_model(self, mock_embedding_model):
        """Test call_embedding_model function."""
        # Mock embedding model instance
        mock_model_instance = Mock()
        mock_model_instance.generate_embeddings.return_value = ([[0.1, 0.2, 0.3]], 50, None)
        mock_embedding_model.return_value = mock_model_instance
        
        result = call_embedding_model(
            model_provider='openai',
            model_name='text-embedding-3-small',
            text='Test text',
            config_path=self.temp_config_file.name
        )
        
        embeddings, tokens_used, error = result
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3]])
        self.assertEqual(tokens_used, 50)
        self.assertIsNone(error)
    
    @patch('call_language_model.call_language_model')
    def test_batch_call_language_model(self, mock_call_function):
        """Test batch_call_language_model function."""
        # Mock individual call results
        mock_call_function.side_effect = [
            ("Response 1", 100, None),
            ("Response 2", 120, None),
            ("Response 3", 80, None)
        ]
        
        batch_requests = [
            {
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Question 1"
            },
            {
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Question 2"
            },
            {
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Question 3"
            }
        ]
        
        results = batch_call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            requests=batch_requests,
            max_workers=2,
            show_progress=False,
            config_path=self.temp_config_file.name
        )
        
        self.assertEqual(len(results), 3)
        
        # Check first result
        self.assertEqual(results[0]['request_index'], 0)
        self.assertEqual(results[0]['response_text'], "Response 1")
        self.assertEqual(results[0]['tokens_used'], 100)
        self.assertIsNone(results[0]['error_msg'])
    
    def test_batch_call_language_model_empty_requests(self):
        """Test batch_call_language_model with empty requests."""
        results = batch_call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            requests=[],
            config_path=self.temp_config_file.name
        )
        
        self.assertEqual(results, [])
    
    def test_batch_call_language_model_invalid_requests(self):
        """Test batch_call_language_model with invalid request format."""
        invalid_requests = [
            {"invalid_field": "value"}  # Missing required fields
        ]
        
        results = batch_call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            requests=invalid_requests,
            config_path=self.temp_config_file.name
        )
        
        self.assertEqual(len(results), 1)
        self.assertIn("missing required fields", results[0]['error_msg'])


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling scenarios."""
    
    def test_call_language_model_no_config(self):
        """Test call_language_model with no configuration provided."""
        result = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            system_prompt='Test',
            user_prompt='Test',
            config_path=None,
            custom_config=None
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "")
        self.assertEqual(tokens_used, 0)
        self.assertIn("Both config_path and custom_config cannot be None", error)
    
    def test_call_embedding_model_no_config(self):
        """Test call_embedding_model with no configuration provided."""
        result = call_embedding_model(
            model_provider='openai',
            model_name='text-embedding-3-small',
            text='Test',
            config_path=None,
            custom_config=None
        )
        
        embeddings, tokens_used, error = result
        self.assertEqual(embeddings, [])
        self.assertEqual(tokens_used, 0)
        self.assertIn("Both config_path and custom_config cannot be None", error)


def run_demo_tests():
    """Run demonstration tests showing all functionality."""
    print("="*60)
    print("RUNNING DEMONSTRATION TESTS")
    print("="*60)
    
    # Test configuration loading
    print("\n1. Testing Configuration Loading:")
    test_config = {
        'all_models': [
            {
                'provider': 'openai',
                'model_name': ['gpt-4o', 'gpt-4o-mini'],
                'api_key': 'demo-key',
                'base_url': 'https://api.openai.com/v1'
            }
        ],
        'embedding_models': [
            {
                'provider': 'openai',
                'model_name': ['text-embedding-3-small'],
                'api_key': 'demo-key',
                'base_url': 'https://api.openai.com/v1'
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        config = ModelConfig(config_path)
        credentials = config.get_credentials('openai', 'gpt-4o')
        print(f"✓ Config loaded successfully")
        print(f"✓ Found credentials for openai/gpt-4o: {bool(credentials)}")
        
        # Test language model with mock
        print("\n2. Testing Language Model Call (Mocked):")
        with patch('call_language_model.OpenAIModel') as mock_model:
            mock_instance = Mock()
            mock_instance.generate.return_value = ("Hello! This is a test response.", 45, None)
            mock_model.return_value = mock_instance
            
            response, tokens, error = call_language_model(
                model_provider='openai',
                model_name='gpt-4o',
                system_prompt='You are a helpful assistant.',
                user_prompt='Say hello!',
                config_path=config_path
            )
            
            print(f"✓ Response: {response}")
            print(f"✓ Tokens used: {tokens}")
            print(f"✓ Error: {error}")
        
        # Test embedding model with mock
        print("\n3. Testing Embedding Model Call (Mocked):")
        with patch('call_language_model.OpenAIEmbeddingModel') as mock_embedding:
            mock_instance = Mock()
            mock_instance.generate_embeddings.return_value = ([[0.1, 0.2, 0.3, 0.4, 0.5]], 10, None)
            mock_embedding.return_value = mock_instance
            
            embeddings, tokens, error = call_embedding_model(
                model_provider='openai',
                model_name='text-embedding-3-small',
                text='This is a test text for embedding.',
                config_path=config_path
            )
            
            print(f"✓ Embeddings shape: {len(embeddings)}x{len(embeddings[0]) if embeddings else 0}")
            print(f"✓ Tokens used: {tokens}")
            print(f"✓ Error: {error}")
        
        # Test batch calling with mock
        print("\n4. Testing Batch Language Model Call (Mocked):")
        with patch('call_language_model.call_language_model') as mock_call:
            mock_call.side_effect = [
                ("This is response 1", 30, None),
                ("This is response 2", 35, None),
                ("This is response 3", 40, None)
            ]
            
            batch_requests = [
                {"system_prompt": "You are helpful.", "user_prompt": "Question 1"},
                {"system_prompt": "You are helpful.", "user_prompt": "Question 2"},
                {"system_prompt": "You are helpful.", "user_prompt": "Question 3"}
            ]
            
            results = batch_call_language_model(
                model_provider='openai',
                model_name='gpt-4o',
                requests=batch_requests,
                max_workers=2,
                show_progress=False,
                config_path=config_path
            )
            
            print(f"✓ Batch processing completed: {len(results)} results")
            total_tokens = sum(r['tokens_used'] for r in results)
            successful = sum(1 for r in results if not r['error_msg'])
            print(f"✓ Total tokens used: {total_tokens}")
            print(f"✓ Successful requests: {successful}/{len(results)}")
        
        # Test error handling
        print("\n5. Testing Error Handling:")
        response, tokens, error = call_language_model(
            model_provider='invalid_provider',
            model_name='invalid_model',
            system_prompt='Test',
            user_prompt='Test',
            config_path=config_path
        )
        print(f"✓ Error handling works: {bool(error)}")
        print(f"✓ Error message: {error}")
        
    finally:
        os.unlink(config_path)
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATION TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    # Run demonstration first
    run_demo_tests()
    
    # Then run unit tests
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    unittest.main(verbosity=2)
