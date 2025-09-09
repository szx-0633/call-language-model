# Quick Start Guide for Testing

## 🚀 Quick Start

### 0. Move call_language_model.py to the same directory as this file

```bash
cd tests
cp ../call_language_model.py ./call_language_model.py
```

### 1. Safe Testing (No API Calls, No Costs)

```bash
# Run all mock tests - completely safe, no real API calls
python run_tests.py --type mock

# Or run demo with interactive examples
python demo_test.py
```

### 2. Unit Testing (No API Calls)

```bash
# Run comprehensive unit tests
python run_tests.py --type unit

# Or run directly
python test_call_language_model.py
```

### 3. Integration Testing (⚠️ Real API Calls - Costs Money!)

**Prerequisites:**
- Create `llm_config.yaml` with valid API credentials
- Ensure you have API credits/quota
- Stable internet connection

```bash
# Run with confirmation prompt
python run_tests.py --type integration

# Or run directly (will ask for confirmation)
python real_api_tests.py
```

## 📁 Test File Summary

| File | Purpose | API Calls | Cost |
|------|---------|-----------|------|
| `test_call_language_model.py` | Unit tests with mocks | ❌ No | Free |
| `demo_test.py` | Interactive demos | ❌ No | Free |
| `real_api_tests.py` | Integration tests | ✅ Yes | 💰 Costs money |
| `run_tests.py` | Unified test runner | Configurable | Depends on selection |

## 🔧 Setup for Integration Tests

1. **Create configuration file** (`llm_config.yaml`):

```yaml
all_models:
  - provider: "openai"
    model_name: ["gpt-4o-mini", "gpt-4o"]
    api_key: "sk-your-actual-openai-key"
    base_url: "https://api.openai.com/v1"
  - provider: "aliyun"
    model_name: ["qwen2.5-7b-instruct"]
    api_key: "sk-your-actual-aliyun-key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

embedding_models:
  - provider: "openai"
    model_name: ["text-embedding-3-small"]
    api_key: "sk-your-actual-openai-key"
    base_url: "https://api.openai.com/v1"
  - provider: "aliyun"
    model_name: ["text-embedding-v3"]
    api_key: "sk-your-actual-aliyun-key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

2. **Optional: Add test image** for multimodal tests:
   - Place any image file as `test_image.png` in project directory

## 🎯 Test Recommendations

### For Development
```bash
# Start with mock tests during development
python run_tests.py --type mock
python run_tests.py --type unit
```

### For Validation
```bash
# Test one integration function at a time
python -c "from real_api_tests import test_basic_language_model_call; test_basic_language_model_call()"
```

### For CI/CD
```bash
# Only run safe tests in automated environments
python run_tests.py --type unit --no-confirm
```

### For Full Testing
```bash
# Run everything (will prompt before costly tests)
python run_tests.py --type all
```

## 💡 Cost Estimation

Integration tests typically use:
- **Basic test**: ~50-100 tokens (~$0.001-0.002)
- **Embedding test**: ~20-30 tokens (~$0.0001)
- **Batch test**: ~200-300 tokens (~$0.002-0.005)
- **Full suite**: ~500-1000 tokens (~$0.01-0.02)

*Costs are approximate and depend on model pricing*

## 🐛 Troubleshooting

### Test Failures
1. **Mock tests fail**: Check Python environment and imports
2. **Unit tests fail**: Verify mock setup and module structure
3. **Integration tests fail**: Check API keys, credits, and network

### Common Issues
- `ModuleNotFoundError`: Install missing dependencies (`pip install openai ollama pyyaml tqdm`)
- `FileNotFoundError`: Create `llm_config.yaml` with valid credentials
- `API errors`: Check API key validity and account credits

### Getting Help
- Review test output for specific error messages
- Verify configuration file format
- Test with single API call first
