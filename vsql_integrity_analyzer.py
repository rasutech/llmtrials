import pandas as pd
import re
import json
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, DML
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# Import your LLM wrapper
from ollama_wrapper import answer_from_ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SQLPattern:
    """Represents a pattern of SQL statements"""
    pattern_id: str
    pattern_type: str  # 'check', 'fix', 'normal_usage', 'report', 'maintenance'
    characteristic_keywords: List[str]
    example_sqls: List[str]
    frequency: int
    confidence: float
    related_tables: Set[str]

@dataclass
class IntegrityFixSequence:
    """Represents a discovered integrity check->fix sequence"""
    sequence_id: str
    check_sql: str
    fix_sql: str
    check_pattern: str
    fix_pattern: str
    tables_involved: Set[str]
    issue_type: str
    confidence: float
    occurrence_count: int

class VSQLAnalyzer:
    """Analyzes v$sql CSV file to discover data integrity patterns"""
    
    def __init__(self, csv_path: str, llm_model: str = "qwen3:30b"):
        self.csv_path = csv_path
        self.llm_model = llm_model
        self.df = None
        self.sql_patterns = defaultdict(list)
        self.integrity_sequences = []
        self.schema_discovery = SchemaDiscoveryFromSQL()
        
    def load_vsql_file(self) -> pd.DataFrame:
        """Load the v$sql CSV file with specified columns"""
        logger.info(f"Loading v$sql file from {self.csv_path}")
        
        # Read CSV with specific columns
        self.df = pd.read_csv(self.csv_path, 
                             usecols=['SQL_ID', 'SQL_FULLTEXT', 'PARSING_SCHEMA_NAME', 'LAST_LOAD_TIME'])
        
        # Convert LAST_LOAD_TIME to datetime
        self.df['LAST_LOAD_TIME'] = pd.to_datetime(self.df['LAST_LOAD_TIME'], errors='coerce')
        
        # Sort by time to understand sequence
        self.df = self.df.sort_values('LAST_LOAD_TIME')
        
        logger.info(f"Loaded {len(self.df)} SQL statements")
        return self.df
    
    def classify_sql_purpose(self, sql: str) -> Dict[str, Any]:
        """Classify SQL by its likely purpose using pattern matching and LLM"""
        sql_upper = sql.upper().strip()
        
        classification = {
            'sql': sql,
            'purpose': 'unknown',
            'confidence': 0.0,
            'indicators': []
        }
        
        # Pattern-based classification
        integrity_check_patterns = [
            (r'SELECT.*COUNT.*HAVING.*COUNT.*>', 'duplicate_check'),
            (r'SELECT.*NOT\s+IN.*SELECT', 'orphan_check'),
            (r'SELECT.*LEFT\s+JOIN.*WHERE.*IS\s+NULL', 'missing_reference_check'),
            (r'SELECT.*NOT\s+EXISTS', 'existence_check'),
            (r'WITH.*RECURSIVE.*SELECT', 'circular_reference_check'),
            (r'SELECT.*GROUP\s+BY.*HAVING', 'aggregation_anomaly_check'),
            (r'SELECT.*MINUS\s+SELECT|SELECT.*EXCEPT\s+SELECT', 'set_difference_check')
        ]
        
        integrity_fix_patterns = [
            (r'DELETE.*WHERE.*NOT\s+IN.*SELECT', 'orphan_removal'),
            (r'UPDATE.*SET.*=.*\(.*SELECT', 'reference_correction'),
            (r'MERGE\s+INTO.*WHEN\s+MATCHED.*UPDATE', 'data_synchronization'),
            (r'INSERT.*ON\s+CONFLICT.*UPDATE', 'upsert_fix'),
            (r'DELETE.*USING.*JOIN', 'cascading_cleanup'),
            (r'UPDATE.*FROM.*JOIN.*SET', 'bulk_correction'),
            (r'DELETE.*WHERE.*IN.*\(.*SELECT.*GROUP\s+BY.*HAVING', 'duplicate_removal')
        ]
        
        normal_usage_patterns = [
            (r'^SELECT.*FROM.*WHERE.*=\s*:\w+$', 'parameterized_query'),
            (r'^SELECT.*FROM.*WHERE.*=\s*\d+$', 'single_record_fetch'),
            (r'^INSERT\s+INTO.*VALUES\s*\(', 'single_insert'),
            (r'^UPDATE.*SET.*WHERE.*=\s*:\w+$', 'single_update'),
            (r'SELECT.*ROWNUM\s*<|LIMIT\s+\d+|FETCH\s+FIRST', 'pagination_query')
        ]
        
        # Check patterns
        for pattern, purpose in integrity_check_patterns:
            if re.search(pattern, sql_upper):
                classification['purpose'] = 'integrity_check'
                classification['sub_type'] = purpose
                classification['confidence'] = 0.8
                classification['indicators'].append(f'Pattern: {purpose}')
                return classification
        
        for pattern, purpose in integrity_fix_patterns:
            if re.search(pattern, sql_upper):
                classification['purpose'] = 'integrity_fix'
                classification['sub_type'] = purpose
                classification['confidence'] = 0.8
                classification['indicators'].append(f'Pattern: {purpose}')
                return classification
        
        for pattern, purpose in normal_usage_patterns:
            if re.search(pattern, sql_upper):
                classification['purpose'] = 'normal_usage'
                classification['sub_type'] = purpose
                classification['confidence'] = 0.7
                classification['indicators'].append(f'Pattern: {purpose}')
                return classification
        
        # If no pattern matches, use LLM for complex cases
        if len(sql) > 200 and classification['purpose'] == 'unknown':
            classification = self._classify_with_llm(sql)
        
        return classification
    
    def _classify_with_llm(self, sql: str) -> Dict[str, Any]:
        """Use LLM to classify complex SQL statements"""
        prompt = f"""Analyze this SQL statement and determine its purpose in a database system.

SQL:
{sql[:1000]}  # Truncate very long SQLs

Classify it as one of:
1. integrity_check - Checking for data quality issues (orphans, duplicates, invalid references)
2. integrity_fix - Fixing data quality issues (deleting orphans, updating references)
3. normal_usage - Regular application queries (fetching data, simple CRUD)
4. reporting - Analytical or reporting queries
5. maintenance - Database maintenance operations

Respond with JSON format:
{{
    "purpose": "category",
    "sub_type": "specific_issue_type",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""
        
        try:
            response = answer_from_ollama(prompt, self.llm_model)
            result = json.loads(response)
            
            return {
                'purpose': result.get('purpose', 'unknown'),
                'sub_type': result.get('sub_type', ''),
                'confidence': result.get('confidence', 0.5),
                'indicators': [result.get('reasoning', '')]
            }
        except Exception as e:
            logger.debug(f"LLM classification failed: {e}")
            return {
                'purpose': 'unknown',
                'confidence': 0.0,
                'indicators': []
            }
    
    def discover_integrity_sequences(self) -> List[IntegrityFixSequence]:
        """Discover check->fix sequences in the SQL history"""
        logger.info("Discovering integrity check->fix sequences...")
        
        # First, classify all SQLs
        logger.info("Classifying SQL statements...")
        self.df['classification'] = self.df['SQL_FULLTEXT'].apply(self.classify_sql_purpose)
        
        # Extract classification details
        self.df['purpose'] = self.df['classification'].apply(lambda x: x['purpose'])
        self.df['sub_type'] = self.df['classification'].apply(lambda x: x.get('sub_type', ''))
        self.df['confidence'] = self.df['classification'].apply(lambda x: x['confidence'])
        
        # Group by schema and time windows
        sequences = []
        
        for schema in self.df['PARSING_SCHEMA_NAME'].unique():
            schema_df = self.df[self.df['PARSING_SCHEMA_NAME'] == schema].copy()
            
            # Look for check->fix patterns within time windows (e.g., 5 minutes)
            checks = schema_df[schema_df['purpose'] == 'integrity_check']
            fixes = schema_df[schema_df['purpose'] == 'integrity_fix']
            
            for _, check_row in checks.iterrows():
                check_time = check_row['LAST_LOAD_TIME']
                check_tables = self._extract_tables(check_row['SQL_FULLTEXT'])
                
                # Find fixes within 5 minutes after the check
                time_window = pd.Timedelta(minutes=5)
                potential_fixes = fixes[
                    (fixes['LAST_LOAD_TIME'] > check_time) & 
                    (fixes['LAST_LOAD_TIME'] <= check_time + time_window)
                ]
                
                for _, fix_row in potential_fixes.iterrows():
                    fix_tables = self._extract_tables(fix_row['SQL_FULLTEXT'])
                    
                    # Check if they operate on same tables
                    common_tables = check_tables & fix_tables
                    if common_tables:
                        sequence = IntegrityFixSequence(
                            sequence_id=f"{check_row['SQL_ID']}_{fix_row['SQL_ID']}",
                            check_sql=check_row['SQL_FULLTEXT'],
                            fix_sql=fix_row['SQL_FULLTEXT'],
                            check_pattern=check_row['sub_type'],
                            fix_pattern=fix_row['sub_type'],
                            tables_involved=common_tables,
                            issue_type=self._infer_issue_type(check_row['sub_type'], fix_row['sub_type']),
                            confidence=min(check_row['confidence'], fix_row['confidence']),
                            occurrence_count=1
                        )
                        sequences.append(sequence)
        
        # Group similar sequences
        self.integrity_sequences = self._group_similar_sequences(sequences)
        
        logger.info(f"Discovered {len(self.integrity_sequences)} unique integrity sequences")
        return self.integrity_sequences
    
    def _extract_tables(self, sql: str) -> Set[str]:
        """Extract table names from SQL"""
        tables = set()
        
        # Parse SQL
        try:
            parsed = sqlparse.parse(sql)[0]
            tokens = parsed.flatten()
            
            keywords = ['FROM', 'JOIN', 'INTO', 'UPDATE', 'TABLE']
            
            for i, token in enumerate(tokens):
                if token.value.upper() in keywords:
                    # Look for next non-whitespace token
                    j = i + 1
                    while j < len(tokens) and tokens[j].is_whitespace:
                        j += 1
                    if j < len(tokens) and not tokens[j].is_keyword:
                        table_name = tokens[j].value.strip('`"[]').upper()
                        # Filter out common non-table tokens
                        if table_name and not table_name in ['SELECT', 'SET', 'VALUES', '(']:
                            tables.add(table_name)
        except:
            # Fallback to regex
            patterns = [
                r'FROM\s+(\w+)',
                r'JOIN\s+(\w+)',
                r'INTO\s+(\w+)',
                r'UPDATE\s+(\w+)',
                r'DELETE\s+FROM\s+(\w+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, sql, re.IGNORECASE)
                tables.update(match.upper() for match in matches)
        
        return tables
    
    def _infer_issue_type(self, check_type: str, fix_type: str) -> str:
        """Infer the type of integrity issue from check and fix patterns"""
        issue_mapping = {
            ('duplicate_check', 'duplicate_removal'): 'duplicate_records',
            ('orphan_check', 'orphan_removal'): 'orphaned_records',
            ('missing_reference_check', 'reference_correction'): 'invalid_references',
            ('circular_reference_check', 'cascading_cleanup'): 'circular_dependencies',
            ('existence_check', 'bulk_correction'): 'missing_data'
        }
        
        return issue_mapping.get((check_type, fix_type), 'data_inconsistency')
    
    def _group_similar_sequences(self, sequences: List[IntegrityFixSequence]) -> List[IntegrityFixSequence]:
        """Group similar sequences to find recurring patterns"""
        if not sequences:
            return []
        
        # Create feature vectors for sequences
        def sequence_to_features(seq):
            return f"{seq.check_pattern}_{seq.fix_pattern}_{'_'.join(sorted(seq.tables_involved))}"
        
        sequence_groups = defaultdict(list)
        for seq in sequences:
            key = sequence_to_features(seq)
            sequence_groups[key].append(seq)
        
        # Merge similar sequences
        merged_sequences = []
        for key, group in sequence_groups.items():
            if group:
                # Take the first as representative and update count
                representative = group[0]
                representative.occurrence_count = len(group)
                merged_sequences.append(representative)
        
        return merged_sequences
    
    def analyze_with_llm_batch(self, sample_size: int = 100) -> Dict[str, Any]:
        """Analyze patterns using LLM in batches"""
        logger.info("Analyzing SQL patterns with LLM...")
        
        # Sample different types of SQLs
        samples = {
            'integrity_checks': self.df[self.df['purpose'] == 'integrity_check'].sample(
                min(sample_size, len(self.df[self.df['purpose'] == 'integrity_check']))
            ),
            'integrity_fixes': self.df[self.df['purpose'] == 'integrity_fix'].sample(
                min(sample_size, len(self.df[self.df['purpose'] == 'integrity_fix']))
            ),
            'sequences': self.integrity_sequences[:min(20, len(self.integrity_sequences))]
        }
        
        insights = {}
        
        # Analyze integrity patterns
        if len(samples['sequences']) > 0:
            sequence_descriptions = []
            for seq in samples['sequences']:
                desc = f"""
Issue Type: {seq.issue_type}
Tables: {', '.join(seq.tables_involved)}
Check Pattern: {seq.check_pattern}
Fix Pattern: {seq.fix_pattern}
Occurrences: {seq.occurrence_count}
"""
                sequence_descriptions.append(desc)
            
            prompt = f"""As a database expert, analyze these discovered data integrity patterns from a Fiber Inventory Management system:

{chr(10).join(sequence_descriptions[:10])}

Please provide:
1. Common integrity issues in this system
2. Root causes of these issues
3. Recommendations for prevention
4. Priority order for fixing these issues

Respond in JSON format with keys: common_issues, root_causes, prevention_recommendations, priority_order
"""
            
            try:
                response = answer_from_ollama(prompt, self.llm_model)
                insights['integrity_analysis'] = json.loads(response)
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                insights['integrity_analysis'] = None
        
        return insights

class SchemaDiscoveryFromSQL:
    """Discovers schema from SQL statements in v$sql"""
    
    def __init__(self):
        self.tables = defaultdict(lambda: {
            'columns': set(),
            'relationships': defaultdict(set),
            'usage_count': 0,
            'operation_types': Counter()
        })
        
    def analyze_sql_for_schema(self, sql: str, operation_type: str = None) -> Dict[str, Set[str]]:
        """Extract schema information from a single SQL"""
        tables_columns = defaultdict(set)
        
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql)[0]
            
            # Extract operation type if not provided
            if not operation_type:
                for token in parsed.tokens:
                    if token.ttype in sqlparse.tokens.DML:
                        operation_type = token.value.upper()
                        break
            
            # Extract tables and track operations
            tables = self._extract_tables_detailed(parsed)
            for table in tables:
                self.tables[table]['usage_count'] += 1
                if operation_type:
                    self.tables[table]['operation_types'][operation_type] += 1
            
            # Extract columns with table context
            self._extract_columns_with_context(sql, tables_columns)
            
            # Update global schema
            for table, columns in tables_columns.items():
                self.tables[table]['columns'].update(columns)
            
            # Extract relationships from JOINs
            self._extract_relationships(sql)
            
        except Exception as e:
            logger.debug(f"Schema extraction error: {e}")
        
        return dict(tables_columns)
    
    def _extract_tables_detailed(self, parsed) -> Set[str]:
        """Extract table names from parsed SQL"""
        tables = set()
        
        # Walk through tokens
        from_seen = False
        join_seen = False
        
        for token in parsed.tokens:
            if isinstance(token, sqlparse.sql.Token):
                value_upper = token.value.upper()
                if value_upper in ['FROM', 'JOIN', 'INTO', 'UPDATE', 'TABLE']:
                    from_seen = True
                    continue
            
            if from_seen and hasattr(token, 'get_real_name'):
                table_name = token.get_real_name()
                if table_name:
                    tables.add(table_name.upper())
                    from_seen = False
        
        return tables
    
    def _extract_columns_with_context(self, sql: str, tables_columns: Dict[str, Set[str]]) -> None:
        """Extract columns with their table context"""
        # Pattern for table.column
        table_column_pattern = r'(\w+)\.(\w+)'
        matches = re.findall(table_column_pattern, sql, re.IGNORECASE)
        
        for table, column in matches:
            tables_columns[table.upper()].add(column.upper())
        
        # Pattern for column lists in INSERT
        insert_pattern = r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)'
        insert_matches = re.findall(insert_pattern, sql, re.IGNORECASE)
        
        for table, columns_str in insert_matches:
            columns = [col.strip().upper() for col in columns_str.split(',')]
            tables_columns[table.upper()].update(columns)
    
    def _extract_relationships(self, sql: str) -> None:
        """Extract relationships from JOIN conditions"""
        join_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        matches = re.findall(join_pattern, sql, re.IGNORECASE)
        
        for t1, c1, t2, c2 in matches:
            t1, t2 = t1.upper(), t2.upper()
            if t1 != t2:
                self.tables[t1]['relationships'][t2].add((c1.upper(), c2.upper()))
                self.tables[t2]['relationships'][t1].add((c2.upper(), c1.upper()))

class IntegrityReportGenerator:
    """Generates comprehensive reports from analysis"""
    
    def __init__(self, analyzer: VSQLAnalyzer):
        self.analyzer = analyzer
        self.timestamp = datetime.now()
    
    def generate_full_report(self, output_dir: str) -> Dict[str, Any]:
        """Generate complete analysis report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze patterns
        llm_insights = self.analyzer.analyze_with_llm_batch()
        
        # Generate statistics
        stats = self._generate_statistics()
        
        # Create report structure
        report = {
            'metadata': {
                'timestamp': str(self.timestamp),
                'total_sqls': len(self.analyzer.df),
                'date_range': {
                    'start': str(self.analyzer.df['LAST_LOAD_TIME'].min()),
                    'end': str(self.analyzer.df['LAST_LOAD_TIME'].max())
                }
            },
            'statistics': stats,
            'discovered_sequences': [
                {
                    'sequence_id': seq.sequence_id,
                    'issue_type': seq.issue_type,
                    'tables': list(seq.tables_involved),
                    'occurrences': seq.occurrence_count,
                    'confidence': seq.confidence,
                    'check_pattern': seq.check_pattern,
                    'fix_pattern': seq.fix_pattern
                }
                for seq in self.analyzer.integrity_sequences[:50]  # Top 50
            ],
            'llm_insights': llm_insights,
            'schema_summary': self._generate_schema_summary()
        }
        
        # Save JSON report
        with open(os.path.join(output_dir, 'integrity_analysis.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(report, output_dir)
        
        # Generate visualizations
        self._generate_visualizations(output_dir)
        
        return report
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate statistical summary"""
        df = self.analyzer.df
        
        return {
            'sql_purposes': dict(df['purpose'].value_counts()),
            'schemas': dict(df['PARSING_SCHEMA_NAME'].value_counts()),
            'integrity_issues': {
                'total_checks': len(df[df['purpose'] == 'integrity_check']),
                'total_fixes': len(df[df['purpose'] == 'integrity_fix']),
                'discovered_sequences': len(self.analyzer.integrity_sequences),
                'unique_issue_types': len(set(seq.issue_type for seq in self.analyzer.integrity_sequences))
            },
            'temporal_distribution': {
                'sqls_per_hour': df.groupby(df['LAST_LOAD_TIME'].dt.hour).size().to_dict(),
                'sqls_per_day': df.groupby(df['LAST_LOAD_TIME'].dt.date).size().to_dict()
            }
        }
    
    def _generate_schema_summary(self) -> Dict[str, Any]:
        """Generate schema discovery summary"""
        schema = self.analyzer.schema_discovery
        
        # Top tables by usage
        top_tables = sorted(
            [(name, info['usage_count']) for name, info in schema.tables.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        return {
            'total_tables': len(schema.tables),
            'top_tables_by_usage': top_tables,
            'tables_with_relationships': sum(1 for t in schema.tables.values() if t['relationships']),
            'total_relationships': sum(len(t['relationships']) for t in schema.tables.values())
        }
    
    def _generate_markdown_report(self, report: Dict, output_dir: str) -> None:
        """Generate human-readable markdown report"""
        md = f"""# V$SQL Data Integrity Analysis Report
Generated: {report['metadata']['timestamp']}

## Overview
- **Total SQL Statements**: {report['metadata']['total_sqls']:,}
- **Analysis Period**: {report['metadata']['date_range']['start']} to {report['metadata']['date_range']['end']}
- **Discovered Integrity Sequences**: {len(report['discovered_sequences'])}

## SQL Purpose Distribution
| Purpose | Count | Percentage |
|---------|-------|------------|
"""
        
        total = report['metadata']['total_sqls']
        for purpose, count in report['statistics']['sql_purposes'].items():
            percentage = (count / total) * 100
            md += f"| {purpose} | {count:,} | {percentage:.1f}% |\n"
        
        md += f"""
## Data Integrity Findings

### Summary
- **Integrity Checks Found**: {report['statistics']['integrity_issues']['total_checks']:,}
- **Integrity Fixes Found**: {report['statistics']['integrity_issues']['total_fixes']:,}
- **Check-Fix Sequences**: {report['statistics']['integrity_issues']['discovered_sequences']}
- **Unique Issue Types**: {report['statistics']['integrity_issues']['unique_issue_types']}

### Top Integrity Issues
"""
        
        # Group sequences by issue type
        issue_groups = defaultdict(list)
        for seq in report['discovered_sequences']:
            issue_groups[seq['issue_type']].append(seq)
        
        for issue_type, sequences in sorted(issue_groups.items(), 
                                          key=lambda x: sum(s['occurrences'] for s in x[1]), 
                                          reverse=True)[:5]:
            total_occurrences = sum(s['occurrences'] for s in sequences)
            md += f"\n#### {issue_type.replace('_', ' ').title()}\n"
            md += f"- **Total Occurrences**: {total_occurrences}\n"
            md += f"- **Affected Tables**: {', '.join(set(t for s in sequences for t in s['tables']))}\n"
            md += f"- **Common Patterns**: {', '.join(set(s['check_pattern'] for s in sequences if s['check_pattern']))}\n"
        
        # Add LLM insights if available
        if report.get('llm_insights', {}).get('integrity_analysis'):
            insights = report['llm_insights']['integrity_analysis']
            md += "\n## AI-Generated Insights\n"
            
            if 'common_issues' in insights:
                md += "\n### Common Issues Identified\n"
                for issue in insights['common_issues']:
                    md += f"- {issue}\n"
            
            if 'prevention_recommendations' in insights:
                md += "\n### Prevention Recommendations\n"
                for rec in insights['prevention_recommendations']:
                    md += f"- {rec}\n"
        
        # Save markdown
        with open(os.path.join(output_dir, 'integrity_report.md'), 'w') as f:
            f.write(md)
    
    def _generate_visualizations(self, output_dir: str) -> None:
        """Generate visualization charts"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. SQL Purpose Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        purposes = self.analyzer.df['purpose'].value_counts()
        purposes.plot(kind='bar', ax=ax)
        ax.set_title('Distribution of SQL Statement Purposes')
        ax.set_xlabel('Purpose')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sql_purpose_distribution.png'))
        plt.close()
        
        # 2. Temporal Distribution of Integrity Operations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Hourly distribution
        integrity_df = self.analyzer.df[self.analyzer.df['purpose'].isin(['integrity_check', 'integrity_fix'])]
        if not integrity_df.empty:
            hourly = integrity_df.groupby([integrity_df['LAST_LOAD_TIME'].dt.hour, 'purpose']).size().unstack(fill_value=0)
            hourly.plot(kind='bar', ax=ax1, stacked=True)
            ax1.set_title('Integrity Operations by Hour of Day')
            ax1.set_xlabel('Hour')
            ax1.set_ylabel('Count')
            ax1.legend(['Checks', 'Fixes'])
        
        # Daily trend
        daily = integrity_df.groupby([integrity_df['LAST_LOAD_TIME'].dt.date, 'purpose']).size().unstack(fill_value=0)
        if len(daily) > 1:
            daily.plot(ax=ax2)
            ax2.set_title('Integrity Operations Over Time')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Count')
            ax2.legend(['Checks', 'Fixes'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'))
        plt.close()
        
        # 3. Issue Type Distribution
        if self.analyzer.integrity_sequences:
            fig, ax = plt.subplots(figsize=(10, 6))
            issue_counts = defaultdict(int)
            for seq in self.analyzer.integrity_sequences:
                issue_counts[seq.issue_type] += seq.occurrence_count
            
            issues_df = pd.DataFrame(list(issue_counts.items()), columns=['Issue Type', 'Count'])
            issues_df = issues_df.sort_values('Count', ascending=True)
            
            issues_df.plot(x='Issue Type', y='Count', kind='barh', ax=ax, legend=False)
            ax.set_title('Data Integrity Issues by Type')
            ax.set_xlabel('Occurrence Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'issue_type_distribution.png'))
            plt.close()

# Main execution function
def analyze_vsql_integrity(csv_path: str, 
                          llm_model: str = "qwen3:30b",
                          output_dir: str = './vsql_analysis') -> Dict[str, Any]:
    """Main function to analyze v$sql for data integrity patterns"""
    
    # Initialize analyzer
    analyzer = VSQLAnalyzer(csv_path, llm_model)
    
    # Load data
    analyzer.load_vsql_file()
    
    # Discover integrity sequences
    sequences = analyzer.discover_integrity_sequences()
    
    # Analyze schema from SQLs
    logger.info("Discovering schema from SQL statements...")
    for _, row in analyzer.df.iterrows():
        analyzer.schema_discovery.analyze_sql_for_schema(
            row['SQL_FULLTEXT'], 
            row.get('purpose')
        )
    
    # Generate report
    report_generator = IntegrityReportGenerator(analyzer)
    report = report_generator.generate_full_report(output_dir)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    
    # Print summary
    print(f"\n=== Analysis Summary ===")
    print(f"Total SQLs Analyzed: {len(analyzer.df):,}")
    print(f"Integrity Checks Found: {len(analyzer.df[analyzer.df['purpose'] == 'integrity_check']):,}")
    print(f"Integrity Fixes Found: {len(analyzer.df[analyzer.df['purpose'] == 'integrity_fix']):,}")
    print(f"Check-Fix Sequences Discovered: {len(sequences)}")
    print(f"\nTop 5 Issue Types:")
    
    issue_counts = defaultdict(int)
    for seq in sequences:
        issue_counts[seq.issue_type] += seq.occurrence_count
    
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {issue_type}: {count} occurrences")
    
    return report

# Example usage
if __name__ == "__main__":
    # Analyze your v$sql CSV file
    results = analyze_vsql_integrity(
        csv_path='path/to/your/vsql_export.csv',
        llm_model='qwen3:30b',  # or 'gpt-oss' depending on your model
        output_dir='./fiber_integrity_analysis'
    )
