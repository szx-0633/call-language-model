# Testing Guide for call_language_model

This directory contains comprehensive testing files for the `call_language_model` module that use mock data instead of making real API calls.

## Test Files Overview

### 1. `test_call_language_model.py`

**Comprehensive unit test suite using unittest framework**

- **Purpose**: Complete unit testing of all classes and functions
- **Features**:
  - Tests for `ModelConfig` class (configuration loading and validation)
  - Tests for `OpenAIModel` and `OllamaModel` classes
  - Tests for `OpenAIEmbeddingModel` and `OllamaEmbeddingModel` classes
  - Tests for main calling functions (`call_language_model`, `call_embedding_model`, `batch_call_language_model`)
  - Error handling and edge case testing
  - Mock data for all API responses

**How to run**:

```bash
python test_call_language_model.py
```

### 2. `demo_test.py`

**Interactive demonstration script**

- **Purpose**: Shows practical usage examples with mock data
- **Features**:
  - Basic language model calls
  - Multimodal calls (text + images)
  - Streaming responses
  - Embedding generation (single and multiple texts)
  - Batch processing demonstrations
  - Error handling examples
  - Custom configuration usage

**How to run**:

```bash
python demo_test.py
```

### 3. `real_api_tests.py`

**Real API integration tests**

- **Purpose**: Comprehensive integration testing with actual API calls
- **Features**:
  - Basic language model functionality testing
  - Streaming response testing
  - Multimodal (text + images) testing
  - Embedding model testing (single and multiple texts)
  - Batch processing with real API calls
  - Custom configuration testing
  - Error handling validation
  - Performance and token usage monitoring

**How to run**:

```bash
python real_api_tests.py
```

**⚠️ Warning**: These tests make real API calls and may incur costs!

### 4. `test_config.yaml`

**Mock configuration file for testing**

- Contains sample configurations for different providers
- Used by test files to simulate real configuration scenarios
- Includes both language models and embedding models

### 5. `run_tests.py`

**Unified test runner script**

- **Purpose**: Single entry point for running all types of tests
- **Features**:
  - Run mock tests only
  - Run unit tests with unittest framework
  - Run integration tests with real API calls
  - Run demo tests
  - Run all tests in sequence
  - Command-line options for test selection

**How to run**:

```bash
# Run mock tests only (default)
python run_tests.py --type mock

# Run unit tests
python run_tests.py --type unit

# Run integration tests (real API calls)
python run_tests.py --type integration

# Run demo
python run_tests.py --type demo

# Run all tests
python run_tests.py --type all
```

## Key Testing Features

### 🔄 **Mock API Responses**
All tests use mock data instead of real API calls:
- No actual network requests made
- No API keys required for testing
- Predictable, controlled test environment
- Fast execution without network delays

### 📊 **Comprehensive Coverage**
Tests cover all major functionality:
- ✅ Configuration loading and validation
- ✅ Model initialization and credential handling
- ✅ Text generation (streaming and non-streaming)
- ✅ Multimodal support (text + images)
- ✅ Embedding generation
- ✅ Batch processing with parallel execution
- ✅ Error handling and edge cases
- ✅ Custom configuration support

### 🛠 **Test Categories**

#### Unit Tests (`test_call_language_model.py`)
1. **Configuration Tests**
   - Valid/invalid config file loading
   - Model credential retrieval
   - Skip checking functionality

2. **Model Class Tests**
   - OpenAI model initialization and generation
   - Ollama model initialization and generation
   - Embedding model functionality
   - Response parsing

3. **Main Function Tests**
   - Language model calling with different parameters
   - Embedding model calling
   - Batch processing functionality
   - Error scenarios

#### Demo Tests (`demo_test.py`)
1. **Basic Usage Examples**
   - Simple text generation
   - Multimodal requests
   - Streaming responses

2. **Advanced Features**
   - Batch processing demonstration
   - Custom configuration usage
   - Error handling examples

## Running Tests

### Option 1: Using the Unified Test Runner (Recommended)

```bash
# Run mock tests only (safe, no API calls)
python run_tests.py --type mock

# Run unit tests with detailed output
python run_tests.py --type unit

# Run demo with mock data
python run_tests.py --type demo

# Run integration tests (REAL API CALLS - costs money!)
python run_tests.py --type integration

# Run all tests (mock, unit, demo, then integration)
python run_tests.py --type all
```

### Option 2: Run Individual Test Files

```bash
# Run mock unit tests
python test_call_language_model.py

# Run demo tests
python demo_test.py

# Run real API integration tests (WARNING: costs money!)
python real_api_tests.py
```

### Option 3: Run with Python's unittest module

```bash
# Run unit tests with verbose output
python -m unittest test_call_language_model -v

# Run specific test class
python -m unittest test_call_language_model.TestModelConfig -v
```

## Integration Tests with Real APIs

The `real_api_tests.py` file contains comprehensive integration tests that make actual API calls. These tests are designed to validate the entire system works correctly with real language model providers.

### ⚠️ Important Warnings

1. **Costs Money**: Integration tests make real API calls which may incur charges
2. **Requires Credentials**: You need valid API keys in your `llm_config.yaml` file
3. **Rate Limits**: Some providers have rate limits - tests include delays
4. **Internet Required**: Tests require stable internet connection

### Integration Test Coverage

1. **Basic Language Model Call** - Tests simple text generation
2. **Streaming Language Model Call** - Tests streaming response collection
3. **Multimodal Language Model Call** - Tests image + text input (requires test image)
4. **Embedding Model Call** - Tests single text embedding generation
5. **Multiple Text Embeddings** - Tests batch embedding generation
6. **Batch Language Model Processing** - Tests parallel processing of multiple requests
7. **Custom Configuration** - Tests using custom config instead of file
8. **Error Handling** - Tests various error scenarios

### Prerequisites for Integration Tests

1. **Configuration File**: Create `llm_config.yaml` with valid credentials:

```yaml
all_models:
  - provider: "openai"
    model_name: ["gpt-4o", "gpt-4o-mini"]
    api_key: "your-openai-api-key"
    base_url: "https://api.openai.com/v1"

embedding_models:
  - provider: "openai"
    model_name: ["text-embedding-3-small"]
    api_key: "your-openai-api-key"
    base_url: "https://api.openai.com/v1"
```

2. **Test Image (Optional)**: For multimodal tests, place a test image file named `test_image.png` in the project directory

3. **API Credits**: Ensure your API keys have sufficient credits/quota

### Running Integration Tests Safely

```bash
# Preview what tests will run without executing
python real_api_tests.py --help

# Run with confirmation prompt (default)
python run_tests.py --type integration

# Run specific integration test function
python -c "from real_api_tests import test_basic_language_model_call; test_basic_language_model_call()"
```

## Mock Data Examples

### Language Model Mock Response
```python
mock_response = {
    "response_text": "This is a mock response from the language model.",
    "tokens_used": 150,
    "error": None
}
```

### Embedding Model Mock Response
```python
mock_embeddings = {
    "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]],
    "tokens_used": 25,
    "error": None
}
```

### Batch Processing Mock Results
```python
mock_batch_results = [
    {
        "request_index": 0,
        "response_text": "Response 1",
        "tokens_used": 100,
        "error_msg": None
    },
    # ... more results
]
```

## Test Configuration

The `test_config.yaml` file contains mock configurations for:

- **OpenAI Provider**: GPT models and embedding models
- **Volcengine Provider**: Doubao models
- **Ollama Provider**: Local models like Llama and Qwen

## Benefits of Mock Testing

1. **🚀 Fast Execution**: No network calls, tests run instantly
2. **🔒 No API Keys Needed**: Can test without real credentials
3. **🎯 Predictable Results**: Controlled test environment
4. **💰 Cost-Free**: No API usage charges during testing
5. **🔄 Repeatable**: Same results every time
6. **🧪 Edge Case Testing**: Easy to simulate error conditions

## Adding New Tests

To add new tests, follow these patterns:

### For Unit Tests
```python
def test_new_functionality(self):
    """Test description."""
    with patch('call_language_model.SomeClass') as mock_class:
        mock_instance = Mock()
        mock_instance.some_method.return_value = expected_result
        mock_class.return_value = mock_instance
        
        # Test your functionality
        result = call_your_function()
        
        # Assert results
        self.assertEqual(result, expected_result)
```

### For Demo Tests
```python
def demo_new_feature():
    """Demonstrate new feature with mock data."""
    print("Testing New Feature:")
    
    with patch('call_language_model.SomeClass') as mock_class:
        # Setup mock
        # Call function
        # Display results
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure `call_language_model.py` is in the same directory
2. **YAML Error**: Check that `test_config.yaml` is properly formatted
3. **Mock Failures**: Ensure you're patching the correct module paths

### Debug Tips

1. Use `python -v` for verbose import information
2. Add print statements in test methods for debugging
3. Check mock setup if tests fail unexpectedly

## Next Steps

After running these tests successfully:

1. **Modify Real Code**: Make changes to `call_language_model.py`
2. **Run Tests Again**: Ensure changes don't break existing functionality
3. **Add New Tests**: Create tests for new features you add
4. **Integration Testing**: Test with real APIs in a separate environment

---

**Note**: These tests use mock data and do not make real API calls. For integration testing with real APIs, create a separate test environment with actual credentials and configuration files.
