"""
V$SQL Data Integrity Analysis Framework

A comprehensive framework for analyzing v$sql data to discover table relationships
and identify data integrity issues using pattern matching and LLM analysis.

Features:
- SQL deduplication to avoid reprocessing
- Microbatch processing (100, 1000, 10000 records)
- Table relationship discovery
- Data integrity issue detection
- Progress tracking and caching
"""

import pandas as pd
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Import the existing analyzers
from ultra_optimized_analyzer import MultiThreadedAnalyzer, ProcessingStats
from relationship_discovery_analyzer import RelationshipBasedIntegrityAnalyzer
from join_field_discovery_analyzer import JoinFieldDiscoveryEngine, JoinBasedIntegrityAnalyzer

# Import LLM wrapper
from ollama_wrapper import answer_from_ollama

@dataclass
class AnalysisConfig:
    """Configuration for analysis runs"""
    csv_path: str
    batch_size: int = 1000
    max_batches: Optional[int] = None
    num_threads: int = 8
    llm_model: str = "mistral-nemo:latest"
    output_dir: str = "./analysis_output"
    skip_duplicates: bool = True
    enable_caching: bool = True
    analysis_types: List[str] = field(default_factory=lambda: ["pattern", "relationship", "join"])

class SQLDeduplicator:
    """Handles SQL deduplication to avoid reprocessing identical queries"""
    
    def __init__(self, cache_file: str = "sql_hashes.pkl"):
        self.cache_file = cache_file
        self.processed_hashes: Set[str] = set()
        self.sql_fingerprints: Dict[str, Dict] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load processed SQL hashes from cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.processed_hashes = data.get('hashes', set())
                    self.sql_fingerprints = data.get('fingerprints', {})
                logging.info(f"Loaded {len(self.processed_hashes)} processed SQL hashes")
            except Exception as e:
                logging.warning(f"Could not load SQL cache: {e}")
    
    def save_cache(self):
        """Save processed hashes to cache"""
        try:
            data = {
                'hashes': self.processed_hashes,
                'fingerprints': self.sql_fingerprints
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            logging.debug(f"Saved {len(self.processed_hashes)} SQL hashes to cache")
        except Exception as e:
            logging.error(f"Could not save SQL cache: {e}")
    
    def get_sql_hash(self, sql: str) -> str:
        """Generate normalized hash for SQL"""
        # Normalize SQL for comparison
        normalized = self._normalize_sql(sql)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for deduplication"""
        # Remove extra whitespace and convert to uppercase
        normalized = ' '.join(sql.upper().split())
        
        # Replace bind variables with placeholders
        normalized = re.sub(r':\w+', ':VAR', normalized)
        normalized = re.sub(r"'[^']*'", "'STRING'", normalized)
        normalized = re.sub(r'\d+', 'NUM', normalized)
        
        return normalized
    
    def is_duplicate(self, sql: str) -> bool:
        """Check if SQL has been processed before"""
        sql_hash = self.get_sql_hash(sql)
        return sql_hash in self.processed_hashes
    
    def mark_processed(self, sql: str, sql_id: str, metadata: Dict = None):
        """Mark SQL as processed"""
        sql_hash = self.get_sql_hash(sql)
        self.processed_hashes.add(sql_hash)
        self.sql_fingerprints[sql_hash] = {
            'sql_id': sql_id,
            'processed_at': time.time(),
            'metadata': metadata or {}
        }

class BatchProcessor:
    """Handles microbatch processing of v$sql data"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.deduplicator = SQLDeduplicator() if config.skip_duplicates else None
        self.batch_results: List[Dict] = []
        
        # Setup logging
        os.makedirs(config.output_dir, exist_ok=True)
        log_file = os.path.join(config.output_dir, 'batch_processing.log')
        self._setup_logging(log_file)
    
    def _setup_logging(self, log_file: str):
        """Setup logging for batch processing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_total_rows(self) -> int:
        """Get total number of rows in the CSV file"""
        return sum(1 for _ in open(self.config.csv_path)) - 1
    
    def run_microbatch_test(self, batch_sizes: List[int] = [100, 1000, 10000]) -> Dict[str, Any]:
        """Run test batches of different sizes to validate approach"""
        results = {}
        
        for batch_size in batch_sizes:
            logging.info(f"Testing batch size: {batch_size}")
            
            start_time = time.time()
            batch_result = self.process_batch(0, batch_size, test_mode=True)
            processing_time = time.time() - start_time
            
            results[f"batch_{batch_size}"] = {
                'processing_time': processing_time,
                'records_processed': batch_result.get('records_processed', 0),
                'duplicates_skipped': batch_result.get('duplicates_skipped', 0),
                'llm_calls': batch_result.get('llm_calls', 0),
                'speed_per_second': batch_result.get('records_processed', 0) / max(processing_time, 1)
            }
            
            logging.info(f"Batch {batch_size}: {processing_time:.2f}s, "
                        f"{results[f'batch_{batch_size}']['speed_per_second']:.1f} records/sec")
        
        # Save test results
        test_results_file = os.path.join(self.config.output_dir, 'microbatch_test_results.json')
        with open(test_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def process_batch(self, start_index: int, batch_size: int, test_mode: bool = False) -> Dict[str, Any]:
        """Process a single batch of records"""
        end_index = start_index + batch_size
        
        logging.info(f"Processing batch: rows {start_index}-{end_index}")
        
        # Load batch data
        df = pd.read_csv(
            self.config.csv_path,
            skiprows=range(1, start_index + 1) if start_index > 0 else None,
            nrows=batch_size,
            usecols=['SQL_ID', 'SQL_FULLTEXT', 'PARSING_SCHEMA_NAME', 'LAST_LOAD_TIME']
        )
        
        batch_stats = {
            'records_loaded': len(df),
            'records_processed': 0,
            'duplicates_skipped': 0,
            'llm_calls': 0,
            'analysis_results': []
        }
        
        # Process each record
        for idx, row in df.iterrows():
            sql = row['SQL_FULLTEXT']
            sql_id = row['SQL_ID']
            
            # Check for duplicates
            if self.deduplicator and self.deduplicator.is_duplicate(sql):
                batch_stats['duplicates_skipped'] += 1
                continue
            
            # Process based on analysis types
            record_result = {
                'sql_id': sql_id,
                'analysis': {}
            }
            
            if 'pattern' in self.config.analysis_types:
                # Use pattern-based analysis (fast)
                from ultra_optimized_analyzer import OptimizedPatternMatcher
                matcher = OptimizedPatternMatcher()
                purpose, sub_type, confidence = matcher.classify_sql(sql)
                record_result['analysis']['pattern'] = {
                    'purpose': purpose,
                    'sub_type': sub_type,
                    'confidence': confidence
                }
            
            if 'relationship' in self.config.analysis_types and not test_mode:
                # Use relationship analysis
                from relationship_discovery_analyzer import RelationshipDiscoveryEngine
                rel_engine = RelationshipDiscoveryEngine()
                rel_analysis = rel_engine.analyze_sql(sql, sql_id)
                record_result['analysis']['relationships'] = rel_analysis
            
            if 'join' in self.config.analysis_types and not test_mode:
                # Use join field analysis
                join_engine = JoinFieldDiscoveryEngine()
                join_analysis = join_engine.analyze_sql_for_joins(sql, sql_id)
                record_result['analysis']['joins'] = join_analysis
            
            # Mark as processed
            if self.deduplicator:
                self.deduplicator.mark_processed(sql, sql_id, record_result['analysis'])
            
            batch_stats['records_processed'] += 1
            batch_stats['analysis_results'].append(record_result)
            
            # Count LLM calls (estimated)
            if len(sql) > 200 and 'pattern' in record_result['analysis']:
                if record_result['analysis']['pattern']['purpose'] == 'unknown':
                    batch_stats['llm_calls'] += 1
        
        # Save deduplicator cache
        if self.deduplicator:
            self.deduplicator.save_cache()
        
        return batch_stats
    
    def process_all_batches(self) -> List[Dict[str, Any]]:
        """Process all batches in the file"""
        total_rows = self.get_total_rows()
        num_batches = (total_rows + self.config.batch_size - 1) // self.config.batch_size
        
        if self.config.max_batches:
            num_batches = min(num_batches, self.config.max_batches)
        
        logging.info(f"Processing {total_rows:,} rows in {num_batches} batches of {self.config.batch_size:,}")
        
        all_results = []
        
        for batch_num in range(num_batches):
            start_index = batch_num * self.config.batch_size
            
            batch_result = self.process_batch(start_index, self.config.batch_size)
            batch_result['batch_number'] = batch_num
            batch_result['start_index'] = start_index
            
            all_results.append(batch_result)
            
            # Save intermediate results
            batch_file = os.path.join(self.config.output_dir, f'batch_{batch_num}_results.json')
            with open(batch_file, 'w') as f:
                json.dump(batch_result, f, indent=2, default=str)
        
        return all_results

class VSQLAnalyzer:
    """Main analyzer that orchestrates the entire analysis pipeline"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.batch_processor = BatchProcessor(config)
        
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        start_time = time.time()
        
        logging.info("="*60)
        logging.info("STARTING V$SQL ANALYSIS")
        logging.info("="*60)
        logging.info(f"Configuration: {self.config}")
        
        # Process all batches
        batch_results = self.batch_processor.process_all_batches()
        
        # Aggregate results
        aggregated_results = self._aggregate_batch_results(batch_results)
        
        # Generate final reports
        self._generate_final_reports(aggregated_results)
        
        total_time = time.time() - start_time
        
        final_summary = {
            'total_processing_time': total_time,
            'batches_processed': len(batch_results),
            'total_records_processed': sum(b['records_processed'] for b in batch_results),
            'total_duplicates_skipped': sum(b['duplicates_skipped'] for b in batch_results),
            'total_llm_calls': sum(b['llm_calls'] for b in batch_results),
            'average_processing_speed': sum(b['records_processed'] for b in batch_results) / total_time,
            'config_used': self.config.__dict__
        }
        
        logging.info("="*60)
        logging.info("ANALYSIS COMPLETE")
        logging.info("="*60)
        for key, value in final_summary.items():
            if key != 'config_used':
                logging.info(f"{key}: {value}")
        
        return final_summary
    
    def _aggregate_batch_results(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all batches"""
        aggregated = {
            'integrity_patterns': Counter(),
            'table_relationships': {},
            'join_patterns': {},
            'most_problematic_tables': Counter(),
            'temporal_patterns': []
        }
        
        # Aggregate pattern analysis
        for batch in batch_results:
            for result in batch.get('analysis_results', []):
                if 'pattern' in result['analysis']:
                    pattern_data = result['analysis']['pattern']
                    aggregated['integrity_patterns'][pattern_data['purpose']] += 1
                    
                    if pattern_data['purpose'] in ['integrity_check', 'integrity_fix']:
                        # Extract table names to identify problematic tables
                        sql = result.get('sql_fulltext', '')
                        tables = self._extract_table_names(sql)
                        for table in tables:
                            aggregated['most_problematic_tables'][table] += 1
        
        return aggregated
    
    def _extract_table_names(self, sql: str) -> Set[str]:
        """Quick extraction of table names from SQL"""
        import re
        tables = set()
        patterns = [
            r'FROM\s+([A-Za-z_]\w*)',
            r'JOIN\s+([A-Za-z_]\w*)',
            r'INTO\s+([A-Za-z_]\w*)',
            r'UPDATE\s+([A-Za-z_]\w*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.update(match.upper() for match in matches)
        
        return tables
    
    def _generate_final_reports(self, aggregated_results: Dict):
        """Generate comprehensive final reports"""
        # Create summary report
        report_content = f"""# V$SQL Analysis Summary Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Configuration Used
- Batch Size: {self.config.batch_size:,}
- Analysis Types: {', '.join(self.config.analysis_types)}
- LLM Model: {self.config.llm_model}
- Deduplication: {self.config.skip_duplicates}

## Key Findings

### SQL Purpose Distribution
"""
        
        for purpose, count in aggregated_results['integrity_patterns'].most_common():
            percentage = (count / sum(aggregated_results['integrity_patterns'].values())) * 100
            report_content += f"- **{purpose}**: {count:,} queries ({percentage:.1f}%)\n"
        
        report_content += "\n### Most Problematic Tables\n"
        report_content += "Tables appearing most frequently in integrity-related SQL:\n\n"
        
        for table, count in aggregated_results['most_problematic_tables'].most_common(10):
            report_content += f"- **{table}**: {count:,} integrity-related queries\n"
        
        # Save report
        report_file = os.path.join(self.config.output_dir, 'analysis_summary.md')
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logging.info(f"Final report saved to: {report_file}")

def create_analysis_config(csv_path: str, **kwargs) -> AnalysisConfig:
    """Create analysis configuration with defaults"""
    return AnalysisConfig(csv_path=csv_path, **kwargs)

# Convenience functions for different analysis scenarios

def test_microbatch_efficacy(csv_path: str, output_dir: str = "./test_output") -> Dict[str, Any]:
    """Test different batch sizes to determine optimal configuration"""
    config = create_analysis_config(
        csv_path=csv_path,
        output_dir=output_dir,
        analysis_types=["pattern"]  # Fast analysis for testing
    )
    
    processor = BatchProcessor(config)
    return processor.run_microbatch_test()

def analyze_integrity_patterns(csv_path: str, batch_size: int = 1000, max_batches: int = 10) -> Dict[str, Any]:
    """Focus on identifying data integrity patterns"""
    config = create_analysis_config(
        csv_path=csv_path,
        batch_size=batch_size,
        max_batches=max_batches,
        analysis_types=["pattern"],
        output_dir="./integrity_analysis"
    )
    
    analyzer = VSQLAnalyzer(config)
    return analyzer.run_full_analysis()

def discover_table_relationships(csv_path: str, batch_size: int = 5000, max_batches: int = 5) -> Dict[str, Any]:
    """Focus on discovering table relationships"""
    config = create_analysis_config(
        csv_path=csv_path,
        batch_size=batch_size,
        max_batches=max_batches,
        analysis_types=["relationship", "join"],
        output_dir="./relationship_analysis"
    )
    
    analyzer = VSQLAnalyzer(config)
    return analyzer.run_full_analysis()

def comprehensive_analysis(csv_path: str, batch_size: int = 2000) -> Dict[str, Any]:
    """Run comprehensive analysis with all features"""
    config = create_analysis_config(
        csv_path=csv_path,
        batch_size=batch_size,
        analysis_types=["pattern", "relationship", "join"],
        output_dir="./comprehensive_analysis"
    )
    
    analyzer = VSQLAnalyzer(config)
    return analyzer.run_full_analysis()

# Example usage patterns
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='V$SQL Analysis Framework')
    parser.add_argument('csv_path', help='Path to v$sql CSV file')
    parser.add_argument('--mode', choices=['test', 'integrity', 'relationships', 'comprehensive'], 
                       default='test', help='Analysis mode')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size')
    parser.add_argument('--max-batches', type=int, help='Maximum number of batches')
    parser.add_argument('--output-dir', default='./analysis_output', help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        print("Running microbatch efficacy test...")
        results = test_microbatch_efficacy(args.csv_path, args.output_dir)
        print(f"Test results saved to {args.output_dir}")
        
    elif args.mode == 'integrity':
        print("Analyzing integrity patterns...")
        results = analyze_integrity_patterns(
            args.csv_path, 
            batch_size=args.batch_size,
            max_batches=args.max_batches
        )
        
    elif args.mode == 'relationships':
        print("Discovering table relationships...")
        results = discover_table_relationships(
            args.csv_path,
            batch_size=args.batch_size,
            max_batches=args.max_batches
        )
        
    elif args.mode == 'comprehensive':
        print("Running comprehensive analysis...")
        results = comprehensive_analysis(
            args.csv_path,
            batch_size=args.batch_size
        )
    
    print("\nAnalysis complete!")