import pandas as pd
import re
import json
import time
import threading
import queue
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
import hashlib
import pickle
import os
from pathlib import Path

# Import your LLM wrapper
from ollama_wrapper import answer_from_ollama

# Enhanced logging configuration
class ColoredFormatter(logging.Formatter):
    """Colored logging formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

# Configure enhanced logging
def setup_logging(log_file: str = 'sql_analysis.log'):
    """Setup enhanced logging with file and console output"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Console handler with color
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(threadName)s] - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

@dataclass
class ProcessingStats:
    """Track processing statistics"""
    total_sqls: int = 0
    processed_sqls: int = 0
    skipped_sqls: int = 0
    integrity_checks: int = 0
    integrity_fixes: int = 0
    normal_usage: int = 0
    llm_calls: int = 0
    pattern_matches: int = 0
    processing_time: float = 0.0
    errors: int = 0
    
    def get_progress_percentage(self) -> float:
        return (self.processed_sqls / max(self.total_sqls, 1)) * 100
    
    def get_processing_speed(self) -> float:
        return self.processed_sqls / max(self.processing_time, 1)

class TableTracker:
    """Track processed tables to enable smart skipping"""
    
    def __init__(self, cache_file: str = 'processed_tables.pkl'):
        self.processed_tables = set()
        self.table_patterns = defaultdict(Counter)
        self.table_integrity_status = {}
        self.cache_file = cache_file
        self.lock = threading.Lock()
        self._load_cache()
    
    def _load_cache(self):
        """Load processed tables from cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.processed_tables = data.get('processed_tables', set())
                    self.table_patterns = data.get('table_patterns', defaultdict(Counter))
                    self.table_integrity_status = data.get('table_integrity_status', {})
                logger.info(f"Loaded {len(self.processed_tables)} processed tables from cache")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
    
    def save_cache(self):
        """Save processed tables to cache"""
        with self.lock:
            try:
                data = {
                    'processed_tables': self.processed_tables,
                    'table_patterns': dict(self.table_patterns),
                    'table_integrity_status': self.table_integrity_status
                }
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.debug(f"Saved {len(self.processed_tables)} processed tables to cache")
            except Exception as e:
                logger.error(f"Could not save cache: {e}")
    
    def should_process_sql(self, sql: str, tables: Set[str]) -> bool:
        """Determine if SQL should be processed based on tables"""
        with self.lock:
            # If all tables are already well-processed, skip
            if tables and all(self._is_table_well_processed(table) for table in tables):
                return False
            return True
    
    def _is_table_well_processed(self, table: str) -> bool:
        """Check if a table has been sufficiently processed"""
        if table not in self.processed_tables:
            return False
        
        # Check if we have enough patterns for this table
        patterns = self.table_patterns.get(table, Counter())
        total_patterns = sum(patterns.values())
        
        # Consider well-processed if we have > 50 patterns
        return total_patterns > 50
    
    def mark_table_processed(self, table: str, pattern_type: str):
        """Mark a table as processed with pattern type"""
        with self.lock:
            self.processed_tables.add(table)
            self.table_patterns[table][pattern_type] += 1
    
    def get_table_summary(self, table: str) -> Dict[str, Any]:
        """Get processing summary for a table"""
        with self.lock:
            return {
                'processed': table in self.processed_tables,
                'patterns': dict(self.table_patterns.get(table, Counter())),
                'integrity_status': self.table_integrity_status.get(table, 'unknown')
            }

class OptimizedPatternMatcher:
    """Highly optimized pattern matcher with caching"""
    
    def __init__(self):
        self.compiled_patterns = {}
        self._compile_all_patterns()
        self.match_cache = {}
        
    def _compile_all_patterns(self):
        """Pre-compile all regex patterns"""
        patterns = {
            # Integrity check patterns
            'orphan_check': r'SELECT.*FROM\s+(\w+).*NOT\s+IN.*SELECT.*FROM\s+(\w+)',
            'duplicate_check': r'SELECT.*COUNT.*GROUP\s+BY.*HAVING.*COUNT.*>\s*1',
            'missing_ref_check': r'SELECT.*LEFT\s+JOIN.*WHERE.*IS\s+NULL',
            'existence_check': r'SELECT.*NOT\s+EXISTS',
            
            # Integrity fix patterns
            'orphan_fix': r'DELETE.*FROM\s+(\w+).*NOT\s+IN',
            'duplicate_fix': r'DELETE.*(?:ROWID|ROW_NUMBER)',
            'update_fix': r'UPDATE.*SET.*(?:FROM|WHERE.*IN.*SELECT)',
            'merge_fix': r'MERGE\s+INTO.*WHEN\s+MATCHED',
            
            # Normal usage patterns
            'simple_select': r'^SELECT.*FROM\s+\w+\s+WHERE\s+\w+\s*=\s*:',
            'simple_insert': r'^INSERT\s+INTO\s+\w+.*VALUES\s*\(',
            'simple_update': r'^UPDATE\s+\w+\s+SET.*WHERE\s+\w+\s*=\s*:',
            'simple_delete': r'^DELETE\s+FROM\s+\w+\s+WHERE\s+\w+\s*=\s*:'
        }
        
        for name, pattern in patterns.items():
            self.compiled_patterns[name] = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    
    def classify_sql(self, sql: str) -> Tuple[str, str, float]:
        """
        Fast SQL classification
        Returns: (purpose, sub_type, confidence)
        """
        # Check cache
        sql_hash = hashlib.md5(sql.encode()).hexdigest()[:16]
        if sql_hash in self.match_cache:
            return self.match_cache[sql_hash]
        
        result = ('unknown', '', 0.0)
        
        # Check patterns in priority order
        for pattern_name, pattern in self.compiled_patterns.items():
            if pattern.search(sql):
                if 'check' in pattern_name:
                    result = ('integrity_check', pattern_name, 0.85)
                elif 'fix' in pattern_name:
                    result = ('integrity_fix', pattern_name, 0.85)
                elif 'simple' in pattern_name:
                    result = ('normal_usage', pattern_name, 0.9)
                break
        
        self.match_cache[sql_hash] = result
        return result

class MultiThreadedAnalyzer:
    """Main analyzer with multi-threading and range processing"""
    
    def __init__(self, 
                 csv_path: str,
                 start_index: int = 0,
                 end_index: Optional[int] = None,
                 num_threads: int = 8,
                 llm_model: str = "mistral-nemo:latest",
                 skip_processed_tables: bool = True):
        """
        Initialize analyzer with range processing
        
        Args:
            csv_path: Path to v$sql CSV file
            start_index: Starting row index (0-based)
            end_index: Ending row index (exclusive), None for all
            num_threads: Number of processing threads
            llm_model: LLM model to use
            skip_processed_tables: Whether to skip already processed tables
        """
        self.csv_path = csv_path
        self.start_index = start_index
        self.end_index = end_index
        self.num_threads = num_threads
        self.llm_model = llm_model
        self.skip_processed_tables = skip_processed_tables
        
        # Initialize components
        self.table_tracker = TableTracker()
        self.pattern_matcher = OptimizedPatternMatcher()
        self.stats = ProcessingStats()
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Thread synchronization
        self.stop_event = threading.Event()
        self.progress_lock = threading.Lock()
        
        logger.info(f"Initialized analyzer: threads={num_threads}, model={llm_model}, "
                   f"range=[{start_index}:{end_index or 'end'}]")
    
    def load_data(self) -> pd.DataFrame:
        """Load data with range selection"""
        logger.info(f"Loading v$sql file: {self.csv_path}")
        
        # First, get total rows
        total_rows = sum(1 for _ in open(self.csv_path)) - 1  # Subtract header
        logger.info(f"Total rows in file: {total_rows:,}")
        
        # Determine actual range
        actual_start = max(0, self.start_index)
        actual_end = min(self.end_index or total_rows, total_rows) if self.end_index else total_rows
        rows_to_read = actual_end - actual_start
        
        logger.info(f"Loading rows {actual_start:,} to {actual_end:,} ({rows_to_read:,} rows)")
        
        # Load only the specified range
        df = pd.read_csv(
            self.csv_path,
            usecols=['SQL_ID', 'SQL_FULLTEXT', 'PARSING_SCHEMA_NAME', 'LAST_LOAD_TIME'],
            skiprows=range(1, actual_start + 1) if actual_start > 0 else None,
            nrows=rows_to_read
        )
        
        df['LAST_LOAD_TIME'] = pd.to_datetime(df['LAST_LOAD_TIME'], errors='coerce')
        df = df.sort_values('LAST_LOAD_TIME')
        
        self.stats.total_sqls = len(df)
        logger.info(f"Loaded {len(df):,} SQL statements")
        
        return df
    
    def extract_tables(self, sql: str) -> Set[str]:
        """Extract table names from SQL"""
        tables = set()
        
        # Quick extraction patterns
        patterns = [
            r'FROM\s+([A-Za-z_]\w*)',
            r'JOIN\s+([A-Za-z_]\w*)',
            r'INTO\s+([A-Za-z_]\w*)',
            r'UPDATE\s+([A-Za-z_]\w*)',
            r'DELETE\s+FROM\s+([A-Za-z_]\w*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.update(match.upper() for match in matches if match)
        
        return tables
    
    def process_sql_worker(self, thread_id: int):
        """Worker thread for processing SQLs"""
        logger.info(f"Worker {thread_id} started")
        
        while not self.stop_event.is_set():
            try:
                # Get item from queue with timeout
                item = self.processing_queue.get(timeout=1)
                if item is None:  # Poison pill
                    break
                
                idx, row = item
                
                # Process the SQL
                result = self._process_single_sql(idx, row, thread_id)
                
                # Put result in results queue
                self.results_queue.put(result)
                
                # Update progress
                with self.progress_lock:
                    self.stats.processed_sqls += 1
                    if self.stats.processed_sqls % 100 == 0:
                        self._log_progress()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {thread_id} error: {e}")
                self.stats.errors += 1
        
        logger.info(f"Worker {thread_id} finished")
    
    def _process_single_sql(self, idx: int, row: pd.Series, thread_id: int) -> Dict[str, Any]:
        """Process a single SQL statement"""
        sql = row['SQL_FULLTEXT']
        sql_id = row['SQL_ID']
        
        # Extract tables
        tables = self.extract_tables(sql)
        
        # Check if should skip based on processed tables
        if self.skip_processed_tables and tables:
            if not self.table_tracker.should_process_sql(sql, tables):
                logger.debug(f"[Thread-{thread_id}] Skipping SQL {sql_id} - tables already processed")
                self.stats.skipped_sqls += 1
                return {
                    'index': idx,
                    'sql_id': sql_id,
                    'purpose': 'skipped',
                    'tables': list(tables),
                    'reason': 'tables_already_processed'
                }
        
        # Pattern-based classification first
        purpose, sub_type, confidence = self.pattern_matcher.classify_sql(sql)
        
        if purpose != 'unknown':
            self.stats.pattern_matches += 1
            classification_method = 'pattern'
        else:
            # Use LLM only for unknown patterns > 200 chars
            if len(sql) > 200:
                purpose, sub_type, confidence = self._classify_with_llm(sql, thread_id)
                classification_method = 'llm'
                self.stats.llm_calls += 1
            else:
                classification_method = 'length_filter'
        
        # Update statistics
        if purpose == 'integrity_check':
            self.stats.integrity_checks += 1
        elif purpose == 'integrity_fix':
            self.stats.integrity_fixes += 1
        elif purpose == 'normal_usage':
            self.stats.normal_usage += 1
        
        # Mark tables as processed
        for table in tables:
            self.table_tracker.mark_table_processed(table, purpose)
        
        return {
            'index': idx,
            'sql_id': sql_id,
            'purpose': purpose,
            'sub_type': sub_type,
            'confidence': confidence,
            'tables': list(tables),
            'method': classification_method,
            'thread_id': thread_id
        }
    
    def _classify_with_llm(self, sql: str, thread_id: int) -> Tuple[str, str, float]:
        """LLM classification with error handling"""
        try:
            sql_truncated = sql[:500]
            prompt = f"""Classify SQL as:
1. integrity_check - Finding data issues
2. integrity_fix - Fixing data issues
3. normal_usage - Regular query

SQL: {sql_truncated}

Return JSON: {{"purpose": "category", "sub_type": "specific_type", "confidence": 0.0-1.0}}"""
            
            response = answer_from_ollama(prompt, self.llm_model)
            result = json.loads(response)
            
            return (
                result.get('purpose', 'unknown'),
                result.get('sub_type', ''),
                result.get('confidence', 0.5)
            )
        except Exception as e:
            logger.debug(f"[Thread-{thread_id}] LLM error: {e}")
            return ('unknown', '', 0.0)
    
    def _log_progress(self):
        """Log detailed progress information"""
        progress = self.stats.get_progress_percentage()
        speed = self.stats.get_processing_speed()
        eta_seconds = (self.stats.total_sqls - self.stats.processed_sqls) / max(speed, 1)
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        logger.info(
            f"Progress: {progress:.1f}% | "
            f"Processed: {self.stats.processed_sqls:,}/{self.stats.total_sqls:,} | "
            f"Speed: {speed:.0f} SQL/s | "
            f"ETA: {eta_str} | "
            f"Checks: {self.stats.integrity_checks:,} | "
            f"Fixes: {self.stats.integrity_fixes:,} | "
            f"Skipped: {self.stats.skipped_sqls:,}"
        )
    
    def analyze(self) -> Dict[str, Any]:
        """Main analysis function with multi-threading"""
        start_time = time.time()
        
        # Load data
        df = self.load_data()
        
        # Start worker threads
        workers = []
        for i in range(self.num_threads):
            worker = threading.Thread(
                target=self.process_sql_worker,
                args=(i,),
                name=f"Worker-{i}"
            )
            worker.start()
            workers.append(worker)
        
        # Queue all items for processing
        logger.info(f"Queuing {len(df):,} items for processing...")
        for idx, row in df.iterrows():
            self.processing_queue.put((idx, row))
        
        # Add poison pills to stop workers
        for _ in range(self.num_threads):
            self.processing_queue.put(None)
        
        # Collect results
        results = []
        processed_count = 0
        
        while processed_count < len(df):
            try:
                result = self.results_queue.get(timeout=1)
                results.append(result)
                processed_count += 1
            except queue.Empty:
                continue
        
        # Wait for all workers to finish
        for worker in workers:
            worker.join()
        
        # Save table tracker cache
        self.table_tracker.save_cache()
        
        # Calculate final statistics
        self.stats.processing_time = time.time() - start_time
        
        # Add results to dataframe
        results_df = pd.DataFrame(results)
        df = df.merge(results_df[['index', 'purpose', 'sub_type', 'confidence']], 
                     left_index=True, right_on='index', how='left')
        
        # Find sequences
        sequences = self._find_sequences(df)
        
        # Generate final report
        final_stats = {
            'total_sqls': self.stats.total_sqls,
            'processed_sqls': self.stats.processed_sqls,
            'skipped_sqls': self.stats.skipped_sqls,
            'integrity_checks': self.stats.integrity_checks,
            'integrity_fixes': self.stats.integrity_fixes,
            'normal_usage': self.stats.normal_usage,
            'pattern_matches': self.stats.pattern_matches,
            'llm_calls': self.stats.llm_calls,
            'errors': self.stats.errors,
            'processing_time_seconds': round(self.stats.processing_time, 2),
            'processing_speed_per_second': round(self.stats.get_processing_speed(), 2),
            'sequences_found': len(sequences)
        }
        
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        for key, value in final_stats.items():
            logger.info(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
        
        return {
            'statistics': final_stats,
            'dataframe': df,
            'sequences': sequences,
            'table_summary': self._generate_table_summary()
        }
    
    def _find_sequences(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find check->fix sequences"""
        sequences = []
        
        checks = df[df['purpose'] == 'integrity_check']
        fixes = df[df['purpose'] == 'integrity_fix']
        
        for _, check in checks.iterrows():
            # Find fixes within 5 minutes
            time_window = pd.Timedelta(minutes=5)
            potential_fixes = fixes[
                (fixes['LAST_LOAD_TIME'] > check['LAST_LOAD_TIME']) &
                (fixes['LAST_LOAD_TIME'] <= check['LAST_LOAD_TIME'] + time_window)
            ]
            
            for _, fix in potential_fixes.iterrows():
                sequences.append({
                    'check_id': check['SQL_ID'],
                    'fix_id': fix['SQL_ID'],
                    'check_type': check.get('sub_type', ''),
                    'fix_type': fix.get('sub_type', ''),
                    'time_diff_seconds': (fix['LAST_LOAD_TIME'] - check['LAST_LOAD_TIME']).total_seconds()
                })
        
        return sequences
    
    def _generate_table_summary(self) -> Dict[str, Any]:
        """Generate summary of processed tables"""
        summary = {}
        
        for table in sorted(self.table_tracker.processed_tables):
            summary[table] = self.table_tracker.get_table_summary(table)
        
        return summary

def run_analysis_with_range(csv_path: str,
                          start_index: int = 0,
                          end_index: Optional[int] = None,
                          output_dir: str = './analysis_output',
                          num_threads: int = 8,
                          model: str = 'mistral-nemo:latest',
                          skip_processed: bool = True) -> Dict[str, Any]:
    """
    Run analysis on a specific range of SQLs
    
    Args:
        csv_path: Path to v$sql CSV
        start_index: Starting row (0-based)
        end_index: Ending row (exclusive), None for all remaining
        output_dir: Output directory
        num_threads: Number of processing threads
        model: LLM model to use
        skip_processed: Skip already processed tables
    
    Example:
        # Process first 1000 rows
        run_analysis_with_range('data.csv', 0, 1000)
        
        # Process rows 5000-6000
        run_analysis_with_range('data.csv', 5000, 6000)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging for this run
    log_file = os.path.join(output_dir, f'analysis_{start_index}_{end_index}.log')
    setup_logging(log_file)
    
    logger.info("="*60)
    logger.info(f"Starting analysis: rows [{start_index}:{end_index or 'end'}]")
    logger.info("="*60)
    
    # Run analysis
    analyzer = MultiThreadedAnalyzer(
        csv_path=csv_path,
        start_index=start_index,
        end_index=end_index,
        num_threads=num_threads,
        llm_model=model,
        skip_processed_tables=skip_processed
    )
    
    results = analyzer.analyze()
    
    # Save results
    output_file = os.path.join(output_dir, f'results_{start_index}_{end_index}.json')
    with open(output_file, 'w') as f:
        json.dump({
            'statistics': results['statistics'],
            'sequences': results['sequences'][:100],  # Top 100
            'table_summary': results['table_summary']
        }, f, indent=2)
    
    # Save classified SQLs
    df_output = os.path.join(output_dir, f'classified_sqls_{start_index}_{end_index}.csv')
    results['dataframe'][['SQL_ID', 'purpose', 'sub_type', 'confidence']].to_csv(
        df_output, index=False
    )
    
    logger.info(f"Results saved to: {output_dir}")
    
    return results

# Convenience functions for common ranges
def analyze_first_n(csv_path: str, n: int = 1000, **kwargs):
    """Analyze first N rows"""
    return run_analysis_with_range(csv_path, 0, n, **kwargs)

def analyze_chunk(csv_path: str, chunk_number: int, chunk_size: int = 1000, **kwargs):
    """Analyze a specific chunk"""
    start = chunk_number * chunk_size
    end = start + chunk_size
    return run_analysis_with_range(csv_path, start, end, **kwargs)

def analyze_in_batches(csv_path: str, batch_size: int = 5000, max_batches: int = None, **kwargs):
    """Analyze file in batches"""
    # Get total rows
    total_rows = sum(1 for _ in open(csv_path)) - 1
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    if max_batches:
        num_batches = min(num_batches, max_batches)
    
    logger.info(f"Processing {total_rows:,} rows in {num_batches} batches of {batch_size:,}")
    
    all_results = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total_rows)
        
        logger.info(f"\nProcessing batch {i+1}/{num_batches}: rows [{start:,}-{end:,}]")
        
        results = run_analysis_with_range(
            csv_path, start, end, 
            output_dir=f'./batch_output/batch_{i}',
            **kwargs
        )
        all_results.append(results)
    
    return all_results

# Example usage
if __name__ == "__main__":
    # Example 1: Analyze first 100 rows for testing
    results = analyze_first_n(
        'your_vsql.csv',
        n=100,
        num_threads=4,
        model='mistral-nemo:latest'
    )
    
    # Example 2: Analyze specific range
    results = run_analysis_with_range(
        'your_vsql.csv',
        start_index=1000,
        end_index=2000,
        num_threads=8
    )
    
    # Example 3: Process in chunks of 5000
    all_results = analyze_in_batches(
        'your_vsql.csv',
        batch_size=5000,
        max_batches=8,  # Process first 40,000 rows
        num_threads=8
    )