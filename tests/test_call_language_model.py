#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Test suite for call_language_model module using mock data.

This test file provides comprehensive testing for all language model calling methods
without making real API calls. Uses mock objects to simulate API responses.
Tests cover the updated architecture with OpenAI /responses endpoint and OpenAI-compatible
/chat/completions endpoint, along with enhanced reasoning support.

@File    : test_call_language_model.py
@Author  : Test Suite
@Date    : 2025/9/10
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
    OpenAIResponsesModel,
    OpenAICompatibleModel,
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
        #!/usr/bin/python3
        # -*- coding: utf-8 -*-
        """Updated test suite matching refactored implementation (requests-based).

        Previous tests mocked SDK clients (OpenAI / ollama). The library now directly
        uses requests.Session.post + streaming iter_lines. These tests mock HTTP
        responses at the requests layer instead.
        """

        import os
        import json
        import time
        import tempfile
        import unittest
        from unittest.mock import patch, Mock
        from types import SimpleNamespace
        import yaml

        from call_language_model import (
            ModelConfig,
            OpenAIResponsesModel,
            OpenAICompatibleModel,
            OllamaModel,
            OpenAIEmbeddingModel,
            OllamaEmbeddingModel,
            call_language_model,
            call_embedding_model,
            batch_call_language_model,
        )


        class MockHTTPResponse:
            """Lightweight mock for requests.Response supporting json(), iter_lines()."""
            def __init__(self, json_data=None, status_code=200, lines=None):
                self._json_data = json_data or {}
                self.status_code = status_code
                self._lines = lines or []  # list of bytes or str
                self.text = json.dumps(self._json_data)

            def json(self):
                return self._json_data

            def raise_for_status(self):
                if 400 <= self.status_code:
                    raise Exception(f"HTTP {self.status_code}")

            def iter_lines(self):
                for ln in self._lines:
                    if isinstance(ln, str):
                        yield ln.encode('utf-8')
                    else:
                        yield ln


        class BaseConfigTest(unittest.TestCase):
            def setUp(self):
                self.test_config = {
                    'all_models': [
                        {'provider': 'openai', 'model_name': ['gpt-4o'], 'api_key': 'k', 'base_url': 'https://api.openai.com/v1'},
                        {'provider': 'aliyun', 'model_name': ['qwen-max'], 'api_key': 'k2', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'},
                        {'provider': 'ollama', 'model_name': ['llama3.1:8b'], 'base_url': 'http://localhost:11434'}
                    ],
                    'embedding_models': [
                        {'provider': 'openai', 'model_name': ['text-embedding-3-small'], 'api_key': 'k', 'base_url': 'https://api.openai.com/v1'},
                        {'provider': 'ollama', 'model_name': ['nomic-embed-text'], 'base_url': 'http://localhost:11434'}
                    ]
                }
                self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                yaml.safe_dump(self.test_config, self.temp_config_file)
                self.temp_config_file.close()

            def tearDown(self):
                os.unlink(self.temp_config_file.name)


        class TestModelConfig(BaseConfigTest):
            def test_get_credentials(self):
                cfg = ModelConfig(self.temp_config_file.name)
                cred = cfg.get_credentials('openai', 'gpt-4o')
                self.assertEqual(cred['model_name'], 'gpt-4o')


        class TestOpenAIResponsesModel(unittest.TestCase):
            def setUp(self):
                self.credentials = {'provider': 'openai', 'model_name': 'gpt-4o', 'api_key': 'k', 'base_url': 'https://api.openai.com/v1'}

            @patch('call_language_model.requests.Session.post')
            def test_generate_success(self, mock_post):
                json_body = {
                    'output': [
                        {'content': [{'text': 'Reasoning?'}], 'summary': []},
                        {'content': [{'text': 'Final answer.'}]}
                    ],
                    'usage': {'total_tokens': 42}
                }
                mock_post.return_value = MockHTTPResponse(json_body)
                model = OpenAIResponsesModel(self.credentials)
                text, tokens, err = model.generate(system_prompt='s', user_prompt='u')
                self.assertIn('Final answer', text)
                self.assertEqual(tokens, 42)
                self.assertIsNone(err)

            @patch('call_language_model.requests.Session.post')
            def test_generate_stream_collected(self, mock_post):
                # SSE chunks for streaming
                chunks = [
                    'data: {"type": "response.output_text.delta", "delta": "Hello"}',
                    'data: {"type": "response.output_text.delta", "delta": " world"}',
                    'data: {"type": "response.output_text.delta", "delta": "!"}',
                    'data: {"type": "other", "x": 1}',
                    'data: {"response": {"usage": {"total_tokens": 10}}}',
                    'data: [DONE]'
                ]
                mock_post.return_value = MockHTTPResponse({}, lines=chunks)
                model = OpenAIResponsesModel(self.credentials)
                text, tokens, err = model.generate_stream(system_prompt='s', user_prompt='u', collect=True)
                self.assertEqual(text, 'Hello world!')
                self.assertEqual(tokens, 10)
                self.assertIsNone(err)


        class TestOpenAICompatibleModel(unittest.TestCase):
            def setUp(self):
                self.credentials = {'provider': 'aliyun', 'model_name': 'qwen-max', 'api_key': 'k', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'}

            @patch('call_language_model.requests.Session.post')
            def test_generate_success(self, mock_post):
                json_body = {
                    'choices': [{'message': {'content': 'Compat answer'}}],
                    'usage': {'total_tokens': 11}
                }
                mock_post.return_value = MockHTTPResponse(json_body)
                model = OpenAICompatibleModel(self.credentials)
                text, tokens, err = model.generate(system_prompt='s', user_prompt='u')
                self.assertEqual(text, 'Compat answer')
                self.assertEqual(tokens, 11)
                self.assertIsNone(err)

            @patch('call_language_model.requests.Session.post')
            def test_generate_stream_collected(self, mock_post):
                # SSE style streaming JSON lines
                chunks = [
                    'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                    'data: {"choices": [{"delta": {"content": " there"}}]}',
                    'data: {"choices": [{"delta": {"content": "!"}}], "usage": {"total_tokens": 7}}',
                    'data: [DONE]'
                ]
                mock_post.return_value = MockHTTPResponse({}, lines=chunks)
                model = OpenAICompatibleModel(self.credentials)
                text, tokens, err = model.generate_stream(system_prompt='s', user_prompt='u', collect=True)
                self.assertEqual(text, 'Hello there!')
                self.assertEqual(tokens, 7)
                self.assertIsNone(err)


        class TestOllamaModel(unittest.TestCase):
            def setUp(self):
                self.credentials = {'provider': 'ollama', 'model_name': 'llama3.1:8b', 'base_url': 'http://localhost:11434'}

            @patch('call_language_model.requests.Session.post')
            def test_generate_success(self, mock_post):
                json_body = {
                    'message': {'content': 'Hi from Ollama'},
                    'prompt_eval_count': 5,
                    'eval_count': 7
                }
                mock_post.return_value = MockHTTPResponse(json_body)
                model = OllamaModel(self.credentials)
                text, tokens, err = model.generate(system_prompt='s', user_prompt='u')
                self.assertEqual(text, 'Hi from Ollama')
                self.assertEqual(tokens, 12)
                self.assertIsNone(err)

            @patch('call_language_model.requests.Session.post')
            def test_generate_stream_collected(self, mock_post):
                lines = [
                    json.dumps({'message': {'content': 'Part1 '}}),
                    json.dumps({'message': {'content': 'Part2'}}),
                    json.dumps({'done': True, 'prompt_eval_count': 2, 'eval_count': 3})
                ]
                mock_post.return_value = MockHTTPResponse({}, lines=lines)
                model = OllamaModel(self.credentials)
                text, tokens, err = model.generate_stream(system_prompt='s', user_prompt='u', collect=True)
                self.assertEqual(text, 'Part1 Part2')
                self.assertEqual(tokens, 5)
                self.assertIsNone(err)


        class TestEmbeddings(unittest.TestCase):
            def setUp(self):
                self.openai_embed_credentials = {'provider': 'openai', 'model_name': 'text-embedding-3-small', 'api_key': 'k', 'base_url': 'https://api.openai.com/v1'}
                self.ollama_embed_credentials = {'provider': 'ollama', 'model_name': 'nomic-embed-text', 'base_url': 'http://localhost:11434'}

            @patch('call_language_model.requests.Session.post')
            def test_openai_embeddings(self, mock_post):
                body = {
                    'data': [
                        {'embedding': [0.1, 0.2]},
                        {'embedding': [0.3, 0.4]}
                    ],
                    'usage': {'total_tokens': 9}
                }
                mock_post.return_value = MockHTTPResponse(body)
                model = OpenAIEmbeddingModel(self.openai_embed_credentials)
                embs, tokens, err = model.generate_embeddings(['a', 'b'])
                self.assertEqual(len(embs), 2)
                self.assertEqual(tokens, 9)
                self.assertIsNone(err)

            @patch('call_language_model.requests.Session.post')
            def test_ollama_embeddings(self, mock_post):
                # Each call returns one embedding; simulate two inputs
                responses = [
                    MockHTTPResponse({'embedding': [0.1, 0.2], 'eval_count': 3}),
                    MockHTTPResponse({'embedding': [0.3, 0.4], 'eval_count': 4}),
                ]
                mock_post.side_effect = responses
                model = OllamaEmbeddingModel(self.ollama_embed_credentials)
                embs, tokens, err = model.generate_embeddings(['x', 'y'])
                self.assertEqual(len(embs), 2)
                self.assertEqual(tokens, 7)
                self.assertIsNone(err)


        class TestFacadeFunctions(BaseConfigTest):
            @patch('call_language_model.OpenAIResponsesModel.generate')
            def test_call_language_model_openai(self, mock_gen):
                mock_gen.return_value = ("R", 5, None)
                txt, tokens, err = call_language_model('openai', 'gpt-4o', system_prompt='s', user_prompt='u', config_path=self.temp_config_file.name)
                self.assertEqual(txt, 'R')
                self.assertEqual(tokens, 5)
                self.assertIsNone(err)

            @patch('call_language_model.OpenAICompatibleModel.generate')
            def test_call_language_model_compatible(self, mock_gen):
                mock_gen.return_value = ("RC", 6, None)
                # add provider to config (already in base config setUp)
                txt, tokens, err = call_language_model('aliyun', 'qwen-max', system_prompt='s', user_prompt='u', config_path=self.temp_config_file.name)
                self.assertEqual(tokens, 6)
                self.assertIsNone(err)

            @patch('call_language_model.OllamaModel.generate')
            def test_call_language_model_ollama(self, mock_gen):
                mock_gen.return_value = ("RO", 3, None)
                txt, tokens, err = call_language_model('ollama', 'llama3.1:8b', system_prompt='s', user_prompt='u', config_path=self.temp_config_file.name)
                self.assertEqual(tokens, 3)
                self.assertIsNone(err)

            @patch('call_language_model.call_language_model')
            def test_batch_call_language_model(self, mock_single):
                mock_single.side_effect = [("A", 1, None), ("B", 2, None)]
                reqs = [
                    {'system_prompt': 's', 'user_prompt': 'u1'},
                    {'system_prompt': 's', 'user_prompt': 'u2'}
                ]
                results = batch_call_language_model('openai', 'gpt-4o', reqs, show_progress=False, config_path=self.temp_config_file.name)
                self.assertEqual(len(results), 2)
                self.assertIsNone(results[0]['error_msg'])


        class TestErrorHandling(unittest.TestCase):
            def test_missing_config(self):
                txt, tokens, err = call_language_model('openai', 'gpt-4o', system_prompt='s', user_prompt='u', config_path=None, custom_config=None)
                self.assertNotEqual(err, None)


        if __name__ == '__main__':
            unittest.main(verbosity=2)
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.test_config, self.temp_config_file)
        self.temp_config_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config_file.name)
    
    @patch('call_language_model.OpenAIResponsesModel')
    def test_call_language_model_openai_provider(self, mock_openai_model):
        """Test call_language_model function with OpenAI provider (uses /responses endpoint)."""
        # Mock model instance
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = ("OpenAI response", 100, None)
        mock_openai_model.return_value = mock_model_instance
        
        result = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',
            system_prompt='You are a helpful assistant.',
            user_prompt='Hello!',
            config_path=self.temp_config_file.name
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "OpenAI response")
        self.assertEqual(tokens_used, 100)
        self.assertIsNone(error)
        mock_openai_model.assert_called_once()
    
    @patch('call_language_model.OpenAICompatibleModel')
    def test_call_language_model_compatible_provider(self, mock_compatible_model):
        """Test call_language_model function with OpenAI-compatible provider (uses /chat/completions endpoint)."""
        # Add aliyun provider to test config
        self.test_config['all_models'].append({
            'provider': 'aliyun',
            'model_name': ['qwen-max'],
            'api_key': 'test-aliyun-key',
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        })
        
        # Update config file
        import yaml
        with open(self.temp_config_file.name, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Mock model instance
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = ("Aliyun response", 120, None)
        mock_compatible_model.return_value = mock_model_instance
        
        result = call_language_model(
            model_provider='aliyun',
            model_name='qwen-max',
            system_prompt='You are a helpful assistant.',
            user_prompt='Hello!',
            config_path=self.temp_config_file.name
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "Aliyun response")
        self.assertEqual(tokens_used, 120)
        self.assertIsNone(error)
        mock_compatible_model.assert_called_once()
    
    @patch('call_language_model.OpenAIResponsesModel')
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
    
    @patch('call_language_model.OpenAIResponsesModel')
    def test_call_language_model_with_additional_params(self, mock_openai_model):
        """Test call_language_model function with additional parameters (kwargs)."""
        # Mock model instance
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = ("Response with thinking", 200, None)
        mock_openai_model.return_value = mock_model_instance
        
        result = call_language_model(
            model_provider='openai',
            model_name='gpt-4o',  # Use a model that exists in test config
            system_prompt='You are a helpful assistant.',
            user_prompt='Solve this problem step by step.',
            thinking_effort='high',  # Correct parameter name for OpenAI
            max_completion_tokens=2000,  # Correct parameter name
            config_path=self.temp_config_file.name
        )
        
        response_text, tokens_used, error = result
        self.assertEqual(response_text, "Response with thinking")
        self.assertEqual(tokens_used, 200)
        self.assertIsNone(error)
        
        # Verify that the additional parameters were passed to the model
        call_args = mock_model_instance.generate.call_args
        self.assertIn('thinking_effort', call_args.kwargs)
        self.assertIn('max_completion_tokens', call_args.kwargs)
        self.assertEqual(call_args.kwargs['thinking_effort'], 'high')
        self.assertEqual(call_args.kwargs['max_completion_tokens'], 2000)

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
        with patch('call_language_model.OpenAIResponsesModel') as mock_model:
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
