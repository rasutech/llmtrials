"""
Test script to verify LLM integration is working properly
Run this to ensure your LLM is being called correctly
"""

import pandas as pd
import logging
from vsql_analyzer import force_llm_analysis, AnalysisConfig, VSQLAnalyzer

# Configure logging to see LLM calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_llm_with_sample_data():
    """Test LLM integration with sample SQL data"""
    
    # Create sample SQL data for testing
    sample_data = [
        {
            'SQL_ID': 'TEST001',
            'SQL_FULLTEXT': 'SELECT * FROM users u LEFT JOIN orders o ON u.user_id = o.user_id WHERE o.user_id IS NULL',
            'PARSING_SCHEMA_NAME': 'TESTSCHEMA',
            'LAST_LOAD_TIME': '2024-01-01 10:00:00'
        },
        {
            'SQL_ID': 'TEST002', 
            'SQL_FULLTEXT': 'DELETE FROM orders WHERE order_id NOT IN (SELECT order_id FROM order_items)',
            'PARSING_SCHEMA_NAME': 'TESTSCHEMA',
            'LAST_LOAD_TIME': '2024-01-01 10:05:00'
        },
        {
            'SQL_ID': 'TEST003',
            'SQL_FULLTEXT': 'SELECT COUNT(*) FROM products p JOIN categories c ON p.category_id = c.id GROUP BY c.name HAVING COUNT(*) > 100',
            'PARSING_SCHEMA_NAME': 'TESTSCHEMA', 
            'LAST_LOAD_TIME': '2024-01-01 10:10:00'
        }
    ]
    
    # Create test CSV file
    df = pd.DataFrame(sample_data)
    test_csv = './test_vsql_data.csv'
    df.to_csv(test_csv, index=False)
    
    print("Created test data with 3 SQL statements")
    print("Testing LLM integration...")
    
    # Test 1: Force LLM analysis
    print("\n=== Test 1: Force LLM Analysis ===")
    try:
        results = force_llm_analysis(test_csv, batch_size=10, max_batches=1)
        print(f"✓ LLM calls made: {results.get('total_llm_calls', 0)}")
        print(f"✓ Records processed: {results.get('total_records_processed', 0)}")
    except Exception as e:
        print(f"✗ Force LLM test failed: {e}")
    
    # Test 2: Direct LLM call test
    print("\n=== Test 2: Direct LLM Call Test ===")
    try:
        from ollama_wrapper import answer_from_ollama
        
        test_prompt = """Classify this SQL as integrity_check, integrity_fix, or normal_usage:
        
        SQL: SELECT * FROM users WHERE user_id NOT IN (SELECT user_id FROM orders)
        
        Respond in JSON: {"purpose": "integrity_check", "confidence": 0.9}"""
        
        response = answer_from_ollama(test_prompt, "mistral-nemo:latest")
        print(f"✓ LLM Response: {response}")
    except Exception as e:
        print(f"✗ Direct LLM call failed: {e}")
        print("Please check your ollama_wrapper.py and answer_from_ollama function")
    
    # Test 3: Custom config with forced LLM
    print("\n=== Test 3: Custom Config with LLM ===")
    try:
        config = AnalysisConfig(
            csv_path=test_csv,
            batch_size=5,
            max_batches=1,
            force_llm_analysis=True,  # This forces LLM calls
            skip_duplicates=False,   # Don't skip for testing
            analysis_types=["pattern"],
            output_dir="./llm_test_output"
        )
        
        analyzer = VSQLAnalyzer(config)
        results = analyzer.run_full_analysis()
        
        print(f"✓ Custom config LLM calls: {results.get('total_llm_calls', 0)}")
    except Exception as e:
        print(f"✗ Custom config test failed: {e}")
    
    # Cleanup
    import os
    if os.path.exists(test_csv):
        os.remove(test_csv)

def debug_pattern_matcher():
    """Debug why pattern matcher might be catching everything"""
    from ultra_optimized_analyzer import OptimizedPatternMatcher
    
    print("\n=== Pattern Matcher Debug ===")
    matcher = OptimizedPatternMatcher()
    
    test_sqls = [
        "SELECT * FROM users u LEFT JOIN orders o ON u.user_id = o.user_id WHERE o.user_id IS NULL",
        "DELETE FROM orders WHERE order_id NOT IN (SELECT order_id FROM order_items)",
        "SELECT name, email FROM customers WHERE status = 'active'",
        "UPDATE products SET price = price * 1.1 WHERE category = 'electronics'"
    ]
    
    for i, sql in enumerate(test_sqls, 1):
        purpose, sub_type, confidence = matcher.classify_sql(sql)
        print(f"SQL {i}: purpose={purpose}, sub_type={sub_type}, confidence={confidence}")
        
        if purpose == 'unknown':
            print(f"  → This SQL would trigger LLM analysis")
        else:
            print(f"  → This SQL matched pattern, no LLM needed")

if __name__ == "__main__":
    print("Testing LLM Integration for V$SQL Analysis")
    print("=" * 50)
    
    # Debug pattern matcher first
    debug_pattern_matcher()
    
    # Test LLM integration
    test_llm_with_sample_data()
    
    print("\nTo force LLM usage in your analysis:")
    print("1. Use force_llm_analysis() function")
    print("2. Use --mode force_llm in command line")
    print("3. Set force_llm_analysis=True in config")
    print("4. Lower llm_threshold to 0.3 or 0.4 for more LLM usage")
