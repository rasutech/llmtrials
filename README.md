# V$SQL Data Integrity Analysis Framework

A comprehensive Python framework for analyzing Oracle v$sql data to discover table relationships and identify data integrity challenges in your application.

## Overview

This framework processes v$sql export data to:
- **Discover table relationships** through JOIN analysis
- **Identify data integrity issues** from SQL patterns
- **Understand real data challenges** in your application
- **Avoid duplicate processing** with intelligent caching
- **Process data in configurable batches** for efficiency testing

## Quick Start

### 1. Test LLM Integration First

**IMPORTANT**: Test LLM integration before processing your data:
å
```python
# Run the LLM integration test
python test_llm_integration.py
```

Or test with your actual data:

```python
from vsql_analyzer import force_llm_analysis

# Force LLM analysis on first 100 records
results = force_llm_analysis('your_vsql.csv', batch_size=100, max_batches=1)
print(f"LLM calls made: {results['total_llm_calls']}")
```

### 2. Test Framework Efficacy

After confirming LLM works, test batch processing:

```python
from vsql_analyzer import test_microbatch_efficacy

# Test different batch sizes
results = test_microbatch_efficacy('your_vsql.csv')
print("Optimal batch size determined from test results")
```

### 3. Analyze Integrity Patterns (Fast)

Focus on identifying data integrity issues:

```python
from vsql_analyzer import analyze_integrity_patterns

# Process 10 batches of 1000 records each
results = analyze_integrity_patterns(
    csv_path='your_vsql.csv',
    batch_size=1000,
    max_batches=10
)
```

### 4. Discover Table Relationships (Detailed)

Understand how tables are connected:

```python
from vsql_analyzer import discover_table_relationships

# Process 5 batches of 5000 records each
results = discover_table_relationships(
    csv_path='your_vsql.csv',
    batch_size=5000,
    max_batches=5
)
```

### 5. Comprehensive Analysis

Run all analysis types:

```python
from vsql_analyzer import comprehensive_analysis

# Full analysis with 2000 record batches
results = comprehensive_analysis(
    csv_path='your_vsql.csv',
    batch_size=2000
)
```

## Command Line Usage

The framework includes a command-line interface:

```bash
# Test LLM integration first
python test_llm_integration.py

# Force LLM analysis on 100 records
python vsql_analyzer.py your_vsql.csv --mode force_llm --batch-size 100 --max-batches 1

# Test efficacy with different batch sizes
python vsql_analyzer.py your_vsql.csv --mode test --output-dir ./test_results

# Analyze integrity patterns
python vsql_analyzer.py your_vsql.csv --mode integrity --batch-size 1000 --max-batches 10

# LLM-heavy analysis (low confidence threshold)
python vsql_analyzer.py your_vsql.csv --mode llm_heavy --batch-size 1000 --max-batches 5

# Comprehensive analysis
python vsql_analyzer.py your_vsql.csv --mode comprehensive --batch-size 2000
```

## Key Features

### SQL Deduplication
- **Automatic duplicate detection** using normalized SQL fingerprints
- **Persistent caching** to avoid reprocessing across sessions
- **Configurable** - can be disabled if needed

### Microbatch Processing
- **Configurable batch sizes**: 100, 1000, 10000, or any size
- **Progress tracking** with detailed logging
- **Resumable processing** with intermediate results saved
- **Memory efficient** for large datasets

### Multi-Analysis Approach
1. **Pattern Analysis** (Fast): Uses regex patterns to classify SQL
2. **Relationship Analysis** (Detailed): Discovers table relationships
3. **Join Analysis** (Comprehensive): Maps all join fields and patterns

### Intelligent Processing
- **Skips processed tables** when sufficient data is available
- **Prioritizes LLM usage** for complex/unknown patterns only
- **Thread-safe processing** for parallel execution

## Output Structure

### Generated Files
```
analysis_output/
├── analysis_summary.md           # High-level findings
├── batch_processing.log          # Detailed processing log
├── batch_N_results.json          # Individual batch results
├── microbatch_test_results.json  # Efficacy test results
├── sql_hashes.pkl                # Deduplication cache
└── processed_tables.pkl          # Table processing cache
```

### Analysis Types Output

#### Pattern Analysis
- Classification of SQL by purpose (integrity_check, integrity_fix, normal_usage)
- Identification of problematic tables
- Statistical distribution of SQL types

#### Relationship Analysis
- Table-to-table relationships discovered from JOINs
- Potential foreign key relationships
- Circular dependency detection
- Missing relationship identification

#### Join Analysis
- Detailed join field mapping
- Multi-path relationship discovery
- Join pattern consistency analysis
- Visual relationship graphs

## Configuration Options

### Basic Configuration
```python
from vsql_analyzer import AnalysisConfig, VSQLAnalyzer

config = AnalysisConfig(
    csv_path='your_vsql.csv',
    batch_size=1000,               # Records per batch
    max_batches=10,                # Limit for testing
    num_threads=8,                 # Parallel processing
    llm_model='mistral-nemo:latest',
    output_dir='./my_analysis',
    skip_duplicates=True,          # Enable deduplication
    analysis_types=['pattern', 'relationship']  # Choose analysis types
)

analyzer = VSQLAnalyzer(config)
results = analyzer.run_full_analysis()
```

### Analysis Type Selection
- **`pattern`**: Fast regex-based classification
- **`relationship`**: Table relationship discovery
- **`join`**: Detailed join field analysis

Choose fewer types for faster processing, more types for comprehensive analysis.

## Processing Strategy

### For Initial Testing (Recommended)
1. **Start with pattern analysis** on 1000-10000 records
2. **Evaluate results** to understand your data characteristics
3. **Scale up gradually** based on findings

### For Production Analysis
1. **Enable all analysis types** for comprehensive insights
2. **Use appropriate batch sizes** (2000-5000 records)
3. **Monitor processing speed** and adjust accordingly

### Deduplication Benefits
- **Reduces LLM costs** by avoiding reprocessing identical SQL
- **Improves performance** by skipping known patterns
- **Maintains consistency** across analysis runs

## Common Usage Patterns

### 1. Quick Assessment (100 records)
```python
results = analyze_integrity_patterns('vsql.csv', batch_size=100, max_batches=1)
```

### 2. Medium Testing (10,000 records)
```python
results = discover_table_relationships('vsql.csv', batch_size=1000, max_batches=10)
```

### 3. Large-Scale Analysis (All data)
```python
config = AnalysisConfig(
    csv_path='vsql.csv',
    batch_size=5000,
    analysis_types=['pattern', 'relationship', 'join']
)
analyzer = VSQLAnalyzer(config)
results = analyzer.run_full_analysis()
```

## Understanding Results

### Integrity Issue Types
- **integrity_check**: SQL finding data problems
- **integrity_fix**: SQL fixing data problems  
- **normal_usage**: Regular application queries

### Relationship Types
- **INNER JOIN**: Standard relationships
- **LEFT JOIN**: May indicate optional relationships or data quality issues
- **Implicit joins**: Old-style WHERE clause joins

### Key Metrics
- **Processing speed**: Records per second
- **LLM efficiency**: Percentage requiring LLM analysis
- **Duplicate rate**: Percentage of duplicate SQL statements

## Troubleshooting

### Performance Issues
- Reduce `batch_size` for memory constraints
- Reduce `num_threads` for CPU constraints
- Use `analysis_types=['pattern']` for fastest processing

### LLM Connection Issues
- Verify `ollama_wrapper.py` and `answer_from_ollama` function
- Check LLM model availability
- Review LLM response format in logs

### Memory Issues
- Process smaller batches
- Enable `skip_duplicates=True`
- Limit `max_batches` for initial runs

## Advanced Usage

### Custom Analysis Pipeline
```python
from vsql_analyzer import VSQLAnalyzer, AnalysisConfig

# Custom configuration
config = AnalysisConfig(
    csv_path='large_vsql.csv',
    batch_size=3000,
    num_threads=12,
    analysis_types=['pattern', 'relationship'],
    skip_duplicates=True,
    output_dir='./custom_analysis'
)

analyzer = VSQLAnalyzer(config)

# Run analysis
results = analyzer.run_full_analysis()

# Access detailed results
integrity_issues = results['most_problematic_tables']
processing_stats = results['total_processing_time']
```

### Focusing on Specific Tables
After initial analysis, you can focus on specific problematic tables using the individual analyzer modules.

## Requirements

- pandas
- numpy
- networkx
- matplotlib
- ollama_wrapper (your LLM interface)

## File Dependencies

Ensure these files are in your working directory:
- `ultra_optimized_analyzer.py`
- `relationship_discovery_analyzer.py` 
- `join_field_discovery_analyzer.py`
- `ollama_wrapper.py`

The framework automatically imports and uses these modules for different analysis types.
