#!/usr/bin/env python3
"""
Test script to verify the sarcasm detection project setup.
"""

import sys
import os
import pandas as pd
import json

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import sklearn
        import nltk
        print("✓ Basic libraries imported successfully")
        
        # Test our custom modules with better path handling
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        sys.path.insert(0, src_path)
        
        from preprocessing.text_preprocessor import TextPreprocessor
        from models.sarcasm_detector import SarcasmDetector
        print("✓ Custom modules imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_loading():
    """Test if the dataset can be loaded."""
    print("\nTesting data loading...")
    
    try:
        # Check if dataset exists
        dataset_path = 'data/Sarcasm_Headlines_Dataset.json'
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset not found at {dataset_path}")
            return False
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        df = pd.DataFrame(data)
        print(f"✓ Dataset loaded successfully: {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Class distribution: {df['is_sarcastic'].value_counts().to_dict()}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_preprocessing():
    """Test the text preprocessing functionality."""
    print("\nTesting text preprocessing...")
    
    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        sys.path.insert(0, src_path)
        
        from preprocessing.text_preprocessor import TextPreprocessor
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Test preprocessing
        test_texts = [
            "Scientists discover that coffee is actually good for you!",
            "Breaking: Water is wet, study confirms.",
            "New study shows benefits of Mediterranean diet"
        ]
        
        processed_texts = preprocessor.preprocess_batch(test_texts)
        features = preprocessor.extract_features_batch(test_texts)
        
        print(f"✓ Preprocessing completed successfully")
        print(f"  Processed texts: {len(processed_texts)}")
        print(f"  Features extracted: {features.shape[1]} features")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False

def test_model_initialization():
    """Test model initialization."""
    print("\nTesting model initialization...")
    
    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        sys.path.insert(0, src_path)
        
        from models.sarcasm_detector import SarcasmDetector
        
        # Test different model types
        model_types = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
        
        for model_type in model_types:
            detector = SarcasmDetector(model_type=model_type)
            print(f"✓ {model_type} model initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

def test_project_structure():
    """Test if the project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        'data',
        'models', 
        'notebooks',
        'src',
        'src/api',
        'src/preprocessing',
        'src/models',
        'src/utils',
        'tests'
    ]
    
    required_files = [
        'requirements.txt',
        'README.md',
        'data/Sarcasm_Headlines_Dataset.json',
        'src/preprocessing/text_preprocessor.py',
        'src/models/sarcasm_detector.py',
        'src/api/main.py',
        'notebooks/01_data_exploration.ipynb',
        'notebooks/02_text_preprocessing.ipynb',
        'notebooks/03_model_training.ipynb',
        'notebooks/04_model_evaluation.ipynb'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            all_good = False
    
    return all_good

def test_nltk_data():
    """Test if NLTK data is available."""
    print("\nTesting NLTK data...")
    
    try:
        import nltk
        
        # Test required NLTK data
        required_data = ['punkt', 'stopwords', 'wordnet']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' else f'corpora/{data_name}')
                print(f"✓ NLTK {data_name} data available")
            except LookupError:
                print(f"✗ NLTK {data_name} data missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ NLTK test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("SARCASM DETECTION PROJECT - SETUP TEST")
    print("=" * 50)
    
    tests = [
        test_project_structure,
        test_nltk_data,
        test_imports,
        test_data_loading,
        test_preprocessing,
        test_model_initialization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Run: jupyter notebook notebooks/01_data_exploration.ipynb")
        print("2. Follow the notebooks in order")
        print("3. Train models and deploy the API")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Download NLTK data: python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\"")
        print("- Check if dataset is in the correct location")
        print("- Verify Python environment and imports")

if __name__ == "__main__":
    main() 