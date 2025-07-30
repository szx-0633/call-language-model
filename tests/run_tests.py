#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Quick test runner for real API integration tests.

This script provides options to run individual tests or all tests
for the call_language_model integration testing.

@File    : run_tests.py
@Author  : Test Runner
@Date    : 2025/7/31
@Description: Test runner for both mock and real API tests.
"""

import sys
import argparse


def run_mock_tests():
    """Run mock tests (no real API calls)."""
    print("Running Mock Tests (No Real API Calls)...")
    print("="*50)
    
    try:
        import test_call_language_model
        # Run the demonstration tests
        test_call_language_model.run_demo_tests()
        return True
    except Exception as e:
        print(f"❌ Mock tests failed: {str(e)}")
        return False


def run_unit_tests():
    """Run unit tests with unittest framework."""
    print("Running Unit Tests...")
    print("="*50)
    
    import unittest
    import test_call_language_model
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_call_language_model)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests (real API calls)."""
    print("Running Integration Tests (Real API Calls)...")
    print("="*50)
    print("⚠️  WARNING: These tests will make real API calls and may incur costs!")
    print("Make sure you have valid API credentials in llm_config.yaml")
    
    response = input("\nDo you want to continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Integration tests cancelled.")
        return True
    
    try:
        import real_api_tests
        return real_api_tests.run_all_tests()
    except Exception as e:
        print(f"❌ Integration tests failed: {str(e)}")
        return False


def run_demo():
    """Run a simple demo with mock data."""
    print("Running Demo with Mock Data...")
    print("="*50)
    
    try:
        import demo_test
        demo_test.main()
        return True
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Test runner for call_language_model')
    parser.add_argument('--type', choices=['mock', 'unit', 'integration', 'demo', 'all'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--no-confirm', action='store_true', 
                       help='Skip confirmation for integration tests')
    
    args = parser.parse_args()
    
    print("CALL_LANGUAGE_MODEL TEST RUNNER")
    print("="*60)
    print(f"Test type: {args.type}")
    print()
    
    success = True
    
    if args.type == 'mock' or args.type == 'all':
        success &= run_mock_tests()
        print()
    
    if args.type == 'unit' or args.type == 'all':
        success &= run_unit_tests()
        print()
    
    if args.type == 'demo' or args.type == 'all':
        success &= run_demo()
        print()
    
    if args.type == 'integration' or args.type == 'all':
        if args.no_confirm:
            # Skip confirmation for automated runs
            try:
                import real_api_tests
                success &= real_api_tests.run_all_tests()
            except Exception as e:
                print(f"❌ Integration tests failed: {str(e)}")
                success = False
        else:
            success &= run_integration_tests()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("❌ SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
