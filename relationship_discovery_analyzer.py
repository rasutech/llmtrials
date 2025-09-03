import pandas as pd
import re
import json
import time
import threading
import queue
from typing import Dict, List, Set, Tuple, Optional, Any, DefaultDict
from dataclasses import dataclass, fieldå
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import loggingå
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

# Import your LLM wrapper
from ollama_wrapper import answer_from_ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class TableRelationship:
    """Represents a discovered relationship between tables"""
    source_table: str
    target_table: str
    join_columns: List[Tuple[str, str]]  # [(source_col, target_col), ...]
    join_type: str  # INNER, LEFT, RIGHT, FULL
    frequency: int = 1
    example_sqls: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def __hash__(self):
        return hash((self.source_table, self.target_table, tuple(self.join_columns)))

@dataclass
class TableProfile:
    """Comprehensive profile of a database table"""
    name: str
    columns: Set[str] = field(default_factory=set)
    primary_key_candidates: Set[str] = field(default_factory=set)
    foreign_key_candidates: Dict[str, str] = field(default_factory=dict)  # column -> referenced_table
    relationships: List[TableRelationship] = field(default_factory=list)
    join_frequency: int = 0
    select_frequency: int = 0
    update_frequency: int = 0
    delete_frequency: int = 0
    insert_frequency: int = 0
    common_filter_columns: Counter = field(default_factory=Counter)
    common_join_columns: Counter = field(default_factory=Counter)

@dataclass
class PotentialIntegrityIssue:
    """Represents a potential data integrity issue discovered from relationships"""
    issue_id: str
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    affected_tables: List[str]
    evidence: Dict[str, Any]
    recommended_action: str
    confidence: float

class RelationshipDiscoveryEngine:
    """Discovers table relationships from SELECT queries"""
    
    def __init__(self, llm_model: str = "mistral-nemo:latest"):
        self.table_profiles: Dict[str, TableProfile] = {}
        self.relationships: Dict[Tuple[str, str], TableRelationship] = {}
        self.relationship_graph = nx.MultiDiGraph()
        self.sql_patterns = self._compile_patterns()
        self.llm_model = llm_model
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for SQL analysis"""
        return {
            'join': re.compile(
                r'(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+'
                r'(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+'
                r'(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+'
                r'ON\s+(.*?)(?:WHERE|GROUP|ORDER|LIMIT|$)',
                re.IGNORECASE | re.DOTALL
            ),
            'simple_join': re.compile(
                r'FROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s*,\s*(\w+)(?:\s+(?:AS\s+)?(\w+))?',
                re.IGNORECASE
            ),
            'where_join': re.compile(
                r'WHERE\s+.*?(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)',
                re.IGNORECASE
            ),
            'table_column': re.compile(
                r'(\w+)\.(\w+)',
                re.IGNORECASE
            ),
            'filter_condition': re.compile(
                r'WHERE\s+.*?(\w+)\.(\w+)\s*(?:=|<|>|<=|>=|<>|!=|LIKE|IN)',
                re.IGNORECASE
            )
        }
    
    def analyze_sql(self, sql: str, sql_id: str) -> Dict[str, Any]:
        """Analyze a single SQL statement for relationships"""
        analysis = {
            'sql_id': sql_id,
            'tables': set(),
            'relationships': [],
            'columns_used': defaultdict(set),
            'filter_columns': defaultdict(set),
            'operation_type': self._get_operation_type(sql)
        }
        
        # Try LLM-based analysis first for better accuracy
        llm_analysis = self._analyze_with_llm(sql, sql_id)
        if llm_analysis and llm_analysis.get('success'):
            # Use LLM results
            analysis['tables'] = set(llm_analysis.get('tables', []))
            analysis['relationships'] = llm_analysis.get('relationships', [])
            analysis['columns_used'] = defaultdict(set, llm_analysis.get('columns_used', {}))
            analysis['filter_columns'] = defaultdict(set, llm_analysis.get('filter_columns', {}))
            logger.debug(f"Used LLM analysis for SQL {sql_id}")
        else:
            # Fallback to regex-based analysis
            logger.debug(f"Using regex fallback for SQL {sql_id}")
            
            # Only analyze SELECT statements for relationships
            if analysis['operation_type'] != 'SELECT':
                return self._analyze_non_select(sql, analysis)
            
            # Extract JOIN relationships
            self._extract_join_relationships(sql, analysis)
            
            # Extract table.column references
            self._extract_column_references(sql, analysis)
            
            # Extract filter conditions
            self._extract_filter_conditions(sql, analysis)
        
        # Update table profiles
        self._update_table_profiles(analysis, sql)
        
        return analysis
    
    def _get_operation_type(self, sql: str) -> str:
        """Determine the SQL operation type"""
        sql_upper = sql.strip().upper()
        for op in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']:
            if sql_upper.startswith(op):
                return op
        return 'OTHER'
    
    def _analyze_non_select(self, sql: str, analysis: Dict) -> Dict:
        """Analyze non-SELECT statements for table usage"""
        op_type = analysis['operation_type']
        
        if op_type == 'INSERT':
            match = re.search(r'INSERT\s+INTO\s+(\w+)', sql, re.IGNORECASE)
            if match:
                table = match.group(1).upper()
                analysis['tables'].add(table)
                self._ensure_table_profile(table).insert_frequency += 1
                
        elif op_type == 'UPDATE':
            match = re.search(r'UPDATE\s+(\w+)', sql, re.IGNORECASE)
            if match:
                table = match.group(1).upper()
                analysis['tables'].add(table)
                self._ensure_table_profile(table).update_frequency += 1
                
        elif op_type == 'DELETE':
            match = re.search(r'DELETE\s+FROM\s+(\w+)', sql, re.IGNORECASE)
            if match:
                table = match.group(1).upper()
                analysis['tables'].add(table)
                self._ensure_table_profile(table).delete_frequency += 1
        
        return analysis
    
    def _extract_join_relationships(self, sql: str, analysis: Dict) -> None:
        """Extract JOIN relationships from SQL"""
        # Standard JOIN syntax
        for match in self.sql_patterns['join'].finditer(sql):
            table1 = match.group(1).upper()
            alias1 = match.group(2).upper() if match.group(2) else table1
            join_type = match.group(3).upper() if match.group(3) else 'INNER'
            table2 = match.group(4).upper()
            alias2 = match.group(5).upper() if match.group(5) else table2
            join_condition = match.group(6)
            
            analysis['tables'].update([table1, table2])
            
            # Parse join condition
            join_columns = self._parse_join_condition(join_condition, alias1, alias2, table1, table2)
            
            if join_columns:
                rel = {
                    'source_table': table1,
                    'target_table': table2,
                    'join_type': join_type,
                    'join_columns': join_columns
                }
                analysis['relationships'].append(rel)
        
        # Old-style comma joins
        comma_matches = list(self.sql_patterns['simple_join'].finditer(sql))
        if comma_matches:
            # Look for WHERE conditions that might be joins
            where_joins = self.sql_patterns['where_join'].findall(sql)
            for t1, c1, t2, c2 in where_joins:
                t1, t2 = t1.upper(), t2.upper()
                if t1 != t2:
                    analysis['relationships'].append({
                        'source_table': t1,
                        'target_table': t2,
                        'join_type': 'INNER',
                        'join_columns': [(c1.upper(), c2.upper())]
                    })
    
    def _parse_join_condition(self, condition: str, alias1: str, alias2: str, 
                            table1: str, table2: str) -> List[Tuple[str, str]]:
        """Parse JOIN condition to extract column pairs"""
        join_columns = []
        
        # Look for equality conditions
        eq_pattern = re.compile(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)')
        
        for match in eq_pattern.finditer(condition):
            t1, c1, t2, c2 = match.groups()
            t1, c1, t2, c2 = t1.upper(), c1.upper(), t2.upper(), c2.upper()
            
            # Map aliases back to tables
            if t1 == alias1:
                t1 = table1
            if t2 == alias2:
                t2 = table2
            
            # Ensure correct order
            if t1 == table1 and t2 == table2:
                join_columns.append((c1, c2))
            elif t1 == table2 and t2 == table1:
                join_columns.append((c2, c1))
        
        return join_columns
    
    def _extract_column_references(self, sql: str, analysis: Dict) -> None:
        """Extract all table.column references"""
        for match in self.sql_patterns['table_column'].finditer(sql):
            table, column = match.groups()
            table, column = table.upper(), column.upper()
            analysis['columns_used'][table].add(column)
            
            # Add to table profile
            profile = self._ensure_table_profile(table)
            profile.columns.add(column)
    
    def _extract_filter_conditions(self, sql: str, analysis: Dict) -> None:
        """Extract columns used in WHERE conditions"""
        for match in self.sql_patterns['filter_condition'].finditer(sql):
            table, column = match.groups()
            table, column = table.upper(), column.upper()
            analysis['filter_columns'][table].add(column)
            
            # Track common filter columns
            profile = self._ensure_table_profile(table)
            profile.common_filter_columns[column] += 1
    
    def _analyze_with_llm(self, sql: str, sql_id: str) -> Dict[str, Any]:
        """Analyze SQL using LLM for better accuracy"""
        try:
            # Truncate SQL for LLM processing
            sql_truncated = sql[:1200] if len(sql) > 1200 else sql
            
            prompt = f"""You are a SQL parser. Analyze this SQL statement and extract table relationships, columns, and joins.

SQL ID: {sql_id}
SQL: {sql_truncated}

IMPORTANT: Respond ONLY with valid JSON. No explanations, no markdown, no extra text.

Identify:
1. All tables mentioned in the query
2. All relationships (joins) between tables  
3. Columns used from each table
4. Filter/WHERE conditions with columns

Expected JSON format:
{{
  "success": true,
  "tables": ["TABLE1", "TABLE2"],
  "relationships": [
    {{
      "source_table": "TABLE1",
      "target_table": "TABLE2", 
      "join_type": "INNER",
      "join_columns": [["COL1", "COL2"]]
    }}
  ],
  "columns_used": {{
    "TABLE1": ["COL1", "COL2"],
    "TABLE2": ["COL3", "COL4"]
  }},
  "filter_columns": {{
    "TABLE1": ["COL1"],
    "TABLE2": ["COL3"]
  }}
}}

Use uppercase for table/column names. Return ONLY the JSON object."""

            response = answer_from_ollama(prompt, self.llm_model)
            
            # Better debugging - log the actual response
            logger.info(f"LLM response for {sql_id} (first 200 chars): {response[:200]}")
            
            # Clean the response to extract JSON
            response_cleaned = response.strip()
            
            # Try to find JSON in the response
            if not response_cleaned:
                logger.warning(f"Empty response from LLM for {sql_id}")
                raise ValueError("Empty response from LLM")
            
            # Look for JSON block markers
            if '```json' in response_cleaned:
                start = response_cleaned.find('```json') + 7
                end = response_cleaned.find('```', start)
                if end > start:
                    response_cleaned = response_cleaned[start:end].strip()
                    logger.debug(f"Extracted JSON from markdown block for {sql_id}")
            elif '```' in response_cleaned:
                start = response_cleaned.find('```') + 3
                end = response_cleaned.find('```', start)
                if end > start:
                    response_cleaned = response_cleaned[start:end].strip()
                    logger.debug(f"Extracted content from code block for {sql_id}")
            
            # Try to find JSON object
            if '{' in response_cleaned and '}' in response_cleaned:
                start = response_cleaned.find('{')
                end = response_cleaned.rfind('}') + 1
                response_cleaned = response_cleaned[start:end]
                logger.debug(f"Extracted JSON object for {sql_id}")
            else:
                logger.warning(f"No JSON object found in response for {sql_id}: {response_cleaned}")
                raise ValueError("No JSON object found in response")
            
            logger.debug(f"Cleaned JSON for {sql_id}: {response_cleaned}")
            result = json.loads(response_cleaned)
            
            # Validate and clean the result
            if result.get('success'):
                # Ensure all table names are uppercase
                if 'tables' in result:
                    result['tables'] = [t.upper() for t in result['tables']]
                
                # Clean relationships
                if 'relationships' in result:
                    for rel in result['relationships']:
                        if 'source_table' in rel:
                            rel['source_table'] = rel['source_table'].upper()
                        if 'target_table' in rel:
                            rel['target_table'] = rel['target_table'].upper()
                        if 'join_columns' in rel:
                            rel['join_columns'] = [
                                (c1.upper(), c2.upper()) if isinstance(c, list) and len(c) == 2 
                                else c for c in rel['join_columns']
                                for c1, c2 in [c] if isinstance(c, list) and len(c) == 2
                            ]
                
                # Clean columns_used
                if 'columns_used' in result:
                    result['columns_used'] = {
                        table.upper(): [col.upper() for col in cols]
                        for table, cols in result['columns_used'].items()
                    }
                
                # Clean filter_columns  
                if 'filter_columns' in result:
                    result['filter_columns'] = {
                        table.upper(): [col.upper() for col in cols]
                        for table, cols in result['filter_columns'].items()
                    }
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM analysis failed for SQL {sql_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _ensure_table_profile(self, table: str) -> TableProfile:
        """Ensure table profile exists"""
        if table not in self.table_profiles:
            self.table_profiles[table] = TableProfile(name=table)
        return self.table_profiles[table]
    
    def _update_table_profiles(self, analysis: Dict, sql: str) -> None:
        """Update table profiles with analysis results"""
        # Update operation frequencies
        for table in analysis['tables']:
            profile = self._ensure_table_profile(table)
            if analysis['operation_type'] == 'SELECT':
                profile.select_frequency += 1
        
        # Update columns from LLM or regex analysis
        for table, columns in analysis['columns_used'].items():
            profile = self._ensure_table_profile(table)
            if isinstance(columns, set):
                profile.columns.update(columns)
            elif isinstance(columns, list):
                profile.columns.update(columns)
        
        # Update filter columns
        for table, filter_cols in analysis['filter_columns'].items():
            profile = self._ensure_table_profile(table)
            if isinstance(filter_cols, set):
                for col in filter_cols:
                    profile.common_filter_columns[col] += 1
            elif isinstance(filter_cols, list):
                for col in filter_cols:
                    profile.common_filter_columns[col] += 1
        
        # Update relationships
        for rel_data in analysis['relationships']:
            source = rel_data['source_table']
            target = rel_data['target_table']
            join_columns = rel_data.get('join_columns', [])
            join_type = rel_data.get('join_type', 'INNER')
            
            # Create or update relationship
            key = (source, target)
            if key not in self.relationships:
                self.relationships[key] = TableRelationship(
                    source_table=source,
                    target_table=target,
                    join_columns=join_columns,
                    join_type=join_type
                )
            else:
                # Update existing relationship
                self.relationships[key].frequency += 1
                # Merge join columns
                existing_joins = set(self.relationships[key].join_columns)
                new_joins = set(join_columns)
                self.relationships[key].join_columns = list(existing_joins | new_joins)
            
            # Add example SQL (limit to 3)
            if len(self.relationships[key].example_sqls) < 3:
                self.relationships[key].example_sqls.append(sql[:200] + '...')
            
            # Update graph
            self.relationship_graph.add_edge(
                source, target,
                join_columns=join_columns,
                join_type=join_type
            )
            
            # Update profiles
            self._ensure_table_profile(source).join_frequency += 1
            self._ensure_table_profile(target).join_frequency += 1
            
            # Track join columns
            for src_col, tgt_col in join_columns:
                self._ensure_table_profile(source).common_join_columns[src_col] += 1
                self._ensure_table_profile(target).common_join_columns[tgt_col] += 1

class IntegrityIssueDetector:
    """Detects potential integrity issues from discovered relationships"""
    
    def __init__(self, discovery_engine: RelationshipDiscoveryEngine, llm_model: str = "mistral-nemo:latest"):
        self.discovery = discovery_engine
        self.llm_model = llm_model
        self.detected_issues: List[PotentialIntegrityIssue] = []
    
    def analyze_integrity(self) -> List[PotentialIntegrityIssue]:
        """Analyze discovered relationships for potential integrity issues"""
        logger.info("Analyzing relationships for integrity issues...")
        
        # Run various integrity checks
        self._check_orphaned_records_potential()
        self._check_missing_relationships()
        self._check_circular_dependencies()
        self._check_inconsistent_join_patterns()
        self._check_missing_foreign_keys()
        self._check_unusual_cardinality()
        self._analyze_complex_patterns_with_llm()
        
        # Sort by severity and confidence
        self.detected_issues.sort(
            key=lambda x: (
                {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x.severity],
                -x.confidence
            )
        )
        
        logger.info(f"Detected {len(self.detected_issues)} potential integrity issues")
        return self.detected_issues
    
    def _check_orphaned_records_potential(self) -> None:
        """Check for potential orphaned records based on join patterns"""
        for (source, target), rel in self.discovery.relationships.items():
            source_profile = self.discovery.table_profiles[source]
            target_profile = self.discovery.table_profiles[target]
            
            # If source is frequently joined but also frequently accessed alone
            if source_profile.select_frequency > rel.frequency * 2:
                # Check if LEFT JOINs are common
                left_join_ratio = sum(1 for edge in self.discovery.relationship_graph[source][target].values()
                                    if edge.get('join_type') == 'LEFT') / max(rel.frequency, 1)
                
                if left_join_ratio > 0.3:  # 30% or more LEFT JOINs
                    issue = PotentialIntegrityIssue(
                        issue_id=f"ORPHAN_{source}_{target}",
                        issue_type="potential_orphaned_records",
                        severity="high" if source_profile.select_frequency > 100 else "medium",
                        description=f"Table {source} is often LEFT JOINed with {target}, suggesting potential orphaned records",
                        affected_tables=[source, target],
                        evidence={
                            'left_join_ratio': left_join_ratio,
                            'total_joins': rel.frequency,
                            'join_columns': rel.join_columns
                        },
                        recommended_action=f"Add foreign key constraint on {source}.{rel.join_columns[0][0]} "
                                         f"referencing {target}.{rel.join_columns[0][1]}",
                        confidence=min(0.7 + left_join_ratio * 0.3, 0.95)
                    )
                    self.detected_issues.append(issue)
    
    def _check_missing_relationships(self) -> None:
        """Check for tables that should be related but aren't"""
        # Look for tables with similar column names that aren't joined
        column_to_tables = defaultdict(set)
        
        for table, profile in self.discovery.table_profiles.items():
            for column in profile.columns:
                # Look for potential foreign key columns
                if column.endswith('_ID') or column.endswith('_KEY'):
                    column_to_tables[column].add(table)
        
        # Check for missing relationships
        for column, tables in column_to_tables.items():
            if len(tables) > 1:
                tables_list = list(tables)
                for i in range(len(tables_list)):
                    for j in range(i + 1, len(tables_list)):
                        t1, t2 = tables_list[i], tables_list[j]
                        
                        # Check if relationship exists
                        if (t1, t2) not in self.discovery.relationships and \
                           (t2, t1) not in self.discovery.relationships:
                            
                            issue = PotentialIntegrityIssue(
                                issue_id=f"MISSING_REL_{t1}_{t2}_{column}",
                                issue_type="missing_relationship",
                                severity="medium",
                                description=f"Tables {t1} and {t2} both have column {column} but are never joined",
                                affected_tables=[t1, t2],
                                evidence={
                                    'common_column': column,
                                    't1_usage': self.discovery.table_profiles[t1].select_frequency,
                                    't2_usage': self.discovery.table_profiles[t2].select_frequency
                                },
                                recommended_action=f"Verify if {t1}.{column} should reference {t2}.{column} or vice versa",
                                confidence=0.6
                            )
                            self.detected_issues.append(issue)
    
    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies in relationships"""
        try:
            cycles = list(nx.simple_cycles(self.discovery.relationship_graph))
            
            for cycle in cycles:
                if len(cycle) > 2:  # Ignore self-references and simple bidirectional
                    issue = PotentialIntegrityIssue(
                        issue_id=f"CIRCULAR_{'_'.join(cycle)}",
                        issue_type="circular_dependency",
                        severity="high",
                        description=f"Circular dependency detected: {' -> '.join(cycle + [cycle[0]])}",
                        affected_tables=cycle,
                        evidence={
                            'cycle_length': len(cycle),
                            'cycle_path': cycle
                        },
                        recommended_action="Review relationship design to eliminate circular dependencies",
                        confidence=0.9
                    )
                    self.detected_issues.append(issue)
        except:
            pass  # Graph might not have cycles
    
    def _check_inconsistent_join_patterns(self) -> None:
        """Check for inconsistent join patterns between same tables"""
        for (source, target), rel in self.discovery.relationships.items():
            if len(rel.join_columns) > 1:
                # Multiple different join conditions for same table pair
                issue = PotentialIntegrityIssue(
                    issue_id=f"INCONSISTENT_JOIN_{source}_{target}",
                    issue_type="inconsistent_join_pattern",
                    severity="medium",
                    description=f"Multiple join patterns between {source} and {target}",
                    affected_tables=[source, target],
                    evidence={
                        'join_patterns': rel.join_columns,
                        'frequency': rel.frequency
                    },
                    recommended_action="Standardize join conditions or verify if multiple relationships are intended",
                    confidence=0.7
                )
                self.detected_issues.append(issue)
    
    def _check_missing_foreign_keys(self) -> None:
        """Identify likely foreign key relationships without constraints"""
        for (source, target), rel in self.discovery.relationships.items():
            # High frequency joins suggest foreign key relationship
            if rel.frequency > 10:
                for src_col, tgt_col in rel.join_columns:
                    # Check column naming patterns
                    if (src_col.endswith('_ID') or src_col.endswith('_KEY') or 
                        src_col == f"{target}_ID" or src_col == f"{target.rstrip('S')}_ID"):
                        
                        issue = PotentialIntegrityIssue(
                            issue_id=f"MISSING_FK_{source}_{src_col}_{target}_{tgt_col}",
                            issue_type="missing_foreign_key",
                            severity="high",
                            description=f"Likely missing foreign key: {source}.{src_col} -> {target}.{tgt_col}",
                            affected_tables=[source, target],
                            evidence={
                                'join_frequency': rel.frequency,
                                'column_pattern': f"{src_col} matches FK pattern"
                            },
                            recommended_action=f"CREATE FOREIGN KEY on {source}.{src_col} REFERENCES {target}.{tgt_col}",
                            confidence=0.85
                        )
                        self.detected_issues.append(issue)
    
    def _check_unusual_cardinality(self) -> None:
        """Check for unusual cardinality patterns"""
        for table, profile in self.discovery.table_profiles.items():
            # Tables with high update/delete frequency relative to selects
            if profile.select_frequency > 0:
                update_ratio = profile.update_frequency / profile.select_frequency
                delete_ratio = profile.delete_frequency / profile.select_frequency
                
                if update_ratio > 0.5 or delete_ratio > 0.3:
                    issue = PotentialIntegrityIssue(
                        issue_id=f"HIGH_MUTATION_{table}",
                        issue_type="high_mutation_rate",
                        severity="medium",
                        description=f"Table {table} has high update/delete rate relative to reads",
                        affected_tables=[table],
                        evidence={
                            'update_ratio': update_ratio,
                            'delete_ratio': delete_ratio,
                            'select_count': profile.select_frequency
                        },
                        recommended_action="Review data lifecycle and consider archival strategy or integrity constraints",
                        confidence=0.7
                    )
                    self.detected_issues.append(issue)
    
    def _analyze_complex_patterns_with_llm(self) -> None:
        """Use LLM to analyze complex relationship patterns"""
        # Select most complex relationship patterns
        complex_tables = sorted(
            self.discovery.table_profiles.items(),
            key=lambda x: len(x[1].relationships),
            reverse=True
        )[:10]  # Top 10 most connected tables
        
        for table, profile in complex_tables:
            if len(profile.relationships) > 3:
                relationships_desc = []
                for rel in profile.relationships[:5]:
                    relationships_desc.append(
                        f"- {rel.source_table} -> {rel.target_table} on {rel.join_columns}"
                    )
                
                prompt = f"""Analyze these table relationships for potential data integrity issues:

Table: {table}
Relationships:
{chr(10).join(relationships_desc)}

Common join columns: {list(profile.common_join_columns.most_common(5))}
Operation frequencies: SELECT={profile.select_frequency}, UPDATE={profile.update_frequency}, DELETE={profile.delete_frequency}

Identify potential integrity issues and respond in JSON:
{{"issues": [{{"type": "issue_type", "severity": "high/medium/low", "description": "...", "recommendation": "..."}}]}}"""

                try:
                    response = answer_from_ollama(prompt, self.llm_model)
                    llm_issues = json.loads(response).get('issues', [])
                    
                    for llm_issue in llm_issues:
                        issue = PotentialIntegrityIssue(
                            issue_id=f"LLM_{table}_{llm_issue['type']}",
                            issue_type=llm_issue['type'],
                            severity=llm_issue['severity'],
                            description=llm_issue['description'],
                            affected_tables=[table],
                            evidence={'llm_analysis': True},
                            recommended_action=llm_issue['recommendation'],
                            confidence=0.6
                        )
                        self.detected_issues.append(issue)
                        
                except Exception as e:
                    logger.debug(f"LLM analysis failed for {table}: {e}")

class RelationshipBasedIntegrityAnalyzer:
    """Main analyzer that orchestrates the relationship discovery and integrity analysis"""
    
    def __init__(self, csv_path: str, 
                 start_index: int = 0,
                 end_index: Optional[int] = None,
                 num_threads: int = 8,
                 llm_model: str = "mistral-nemo:latest"):
        self.csv_path = csv_path
        self.start_index = start_index
        self.end_index = end_index
        self.num_threads = num_threads
        self.llm_model = llm_model
        self.discovery_engine = RelationshipDiscoveryEngine()
        self.stats = {
            'total_sqls': 0,
            'select_queries': 0,
            'relationships_found': 0,
            'tables_discovered': 0,
            'processing_time': 0.0
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Main analysis function"""
        start_time = time.time()
        
        # Load data
        logger.info(f"Loading v$sql file: {self.csv_path}")
        df = self._load_data()
        
        # Process SQLs
        logger.info("Analyzing SQL statements for relationships...")
        self._process_sqls_parallel(df)
        
        # Detect integrity issues
        integrity_detector = IntegrityIssueDetector(self.discovery_engine, self.llm_model)
        integrity_issues = integrity_detector.analyze_integrity()
        
        # Generate statistics
        self.stats['processing_time'] = time.time() - start_time
        self.stats['relationships_found'] = len(self.discovery_engine.relationships)
        self.stats['tables_discovered'] = len(self.discovery_engine.table_profiles)
        
        # Log summary
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Total SQLs processed: {self.stats['total_sqls']:,}")
        logger.info(f"SELECT queries analyzed: {self.stats['select_queries']:,}")
        logger.info(f"Tables discovered: {self.stats['tables_discovered']}")
        logger.info(f"Relationships found: {self.stats['relationships_found']}")
        logger.info(f"Integrity issues detected: {len(integrity_issues)}")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        
        return {
            'statistics': self.stats,
            'table_profiles': self.discovery_engine.table_profiles,
            'relationships': list(self.discovery_engine.relationships.values()),
            'integrity_issues': integrity_issues,
            'relationship_graph': self.discovery_engine.relationship_graph
        }
    
    def _load_data(self) -> pd.DataFrame:
        """Load data with range selection"""
        # First get total rows
        total_rows = sum(1 for _ in open(self.csv_path)) - 1
        
        # Determine range
        actual_start = max(0, self.start_index)
        actual_end = min(self.end_index or total_rows, total_rows)
        rows_to_read = actual_end - actual_start
        
        logger.info(f"Loading rows {actual_start:,} to {actual_end:,} ({rows_to_read:,} rows)")
        
        df = pd.read_csv(
            self.csv_path,
            usecols=['SQL_ID', 'SQL_FULLTEXT', 'PARSING_SCHEMA_NAME', 'LAST_LOAD_TIME'],
            skiprows=range(1, actual_start + 1) if actual_start > 0 else None,
            nrows=rows_to_read
        )
        
        self.stats['total_sqls'] = len(df)
        return df
    
    def _process_sqls_parallel(self, df: pd.DataFrame) -> None:
        """Process SQLs in parallel"""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            
            for idx, row in df.iterrows():
                future = executor.submit(
                    self._process_single_sql,
                    row['SQL_FULLTEXT'],
                    row['SQL_ID']
                )
                futures.append(future)
            
            # Process results
            for i, future in enumerate(as_completed(futures)):
                if i % 1000 == 0:
                    logger.info(f"Progress: {i}/{len(futures)} SQLs processed")
                
                try:
                    analysis = future.result()
                    if analysis['operation_type'] == 'SELECT':
                        self.stats['select_queries'] += 1
                except Exception as e:
                    logger.debug(f"Error processing SQL: {e}")
    
    def _process_single_sql(self, sql: str, sql_id: str) -> Dict[str, Any]:
        """Process a single SQL"""
        return self.discovery_engine.analyze_sql(sql, sql_id)
    
    def generate_report(self, output_dir: str, results: Dict[str, Any]) -> None:
        """Generate comprehensive report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save raw results
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
            json.dump({
                'statistics': results['statistics'],
                'integrity_issues': [
                    {
                        'issue_id': issue.issue_id,
                        'type': issue.issue_type,
                        'severity': issue.severity,
                        'description': issue.description,
                        'tables': issue.affected_tables,
                        'recommendation': issue.recommended_action,
                        'confidence': issue.confidence
                    }
                    for issue in results['integrity_issues']
                ]
            }, f, indent=2)
        
        # 2. Generate relationship visualization
        self._visualize_relationships(
            results['relationship_graph'],
            os.path.join(output_dir, 'relationship_graph.png')
        )
        
        # 3. Generate markdown report
        report = self._generate_markdown_report(results)
        with open(os.path.join(output_dir, 'integrity_report.md'), 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to: {output_dir}")
    
    def _visualize_relationships(self, graph: nx.MultiDiGraph, output_path: str) -> None:
        """Create relationship graph visualization"""
        plt.figure(figsize=(20, 16))
        
        # Layout
        pos = nx.spring_layout(graph, k=3, iterations=50)
        
        # Draw nodes
        node_sizes = []
        node_colors = []
        for node in graph.nodes():
            profile = self.discovery_engine.table_profiles.get(node)
            if profile:
                # Size based on usage frequency
                size = min(3000, 500 + profile.select_frequency * 10)
                node_sizes.append(size)
                # Color based on mutation rate
                if profile.update_frequency + profile.delete_frequency > profile.select_frequency * 0.5:
                    node_colors.append('lightcoral')  # High mutation
                else:
                    node_colors.append('lightblue')  # Normal
            else:
                node_sizes.append(500)
                node_colors.append('lightgray')
        
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.5)
        
        # Labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
        
        plt.title('Database Table Relationships', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report"""
        report = f"""# Database Relationship and Integrity Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total SQLs Analyzed**: {results['statistics']['total_sqls']:,}
- **SELECT Queries**: {results['statistics']['select_queries']:,}
- **Tables Discovered**: {results['statistics']['tables_discovered']}
- **Relationships Found**: {results['statistics']['relationships_found']}
- **Potential Integrity Issues**: {len(results['integrity_issues'])}
- **Processing Time**: {results['statistics']['processing_time']:.2f} seconds

## Top Integrity Issues

"""
        # Group issues by severity
        issues_by_severity = defaultdict(list)
        for issue in results['integrity_issues']:
            issues_by_severity[issue.severity].append(issue)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in issues_by_severity:
                report += f"### {severity.upper()} Severity Issues\n\n"
                
                for issue in issues_by_severity[severity][:10]:  # Top 10 per severity
                    report += f"#### {issue.issue_type.replace('_', ' ').title()}\n"
                    report += f"- **Description**: {issue.description}\n"
                    report += f"- **Affected Tables**: {', '.join(issue.affected_tables)}\n"
                    report += f"- **Recommendation**: {issue.recommended_action}\n"
                    report += f"- **Confidence**: {issue.confidence:.0%}\n\n"
        
        # Add top relationships
        report += "## Most Frequent Table Relationships\n\n"
        top_relationships = sorted(
            results['relationships'],
            key=lambda x: x.frequency,
            reverse=True
        )[:20]
        
        report += "| Source Table | Target Table | Join Columns | Frequency | Join Type |\n"
        report += "|--------------|--------------|--------------|-----------|------------|\n"
        
        for rel in top_relationships:
            join_cols = ', '.join([f"{s}->{t}" for s, t in rel.join_columns[:2]])
            report += f"| {rel.source_table} | {rel.target_table} | {join_cols} | {rel.frequency} | {rel.join_type} |\n"
        
        return report

# Main execution function
def analyze_database_relationships(csv_path: str,
                                 start_index: int = 0,
                                 end_index: Optional[int] = None,
                                 output_dir: str = './relationship_analysis',
                                 num_threads: int = 8,
                                 llm_model: str = "mistral-nemo:latest") -> Dict[str, Any]:
    """
    Analyze database relationships and identify integrity issues
    
    Args:
        csv_path: Path to v$sql CSV file
        start_index: Starting row index
        end_index: Ending row index (None for all)
        output_dir: Output directory for results
        num_threads: Number of processing threads
        llm_model: LLM model to use
    """
    # Create analyzer
    analyzer = RelationshipBasedIntegrityAnalyzer(
        csv_path=csv_path,
        start_index=start_index,
        end_index=end_index,
        num_threads=num_threads,
        llm_model=llm_model
    )
    
    # Run analysis
    results = analyzer.analyze()
    
    # Generate report
    analyzer.generate_report(output_dir, results)
    
    return results

# Example usage
if __name__ == "__main__":
    # Analyze first 10000 rows
    results = analyze_database_relationships(
        csv_path='your_vsql.csv',
        start_index=0,
        end_index=10000,
        num_threads=8,
        llm_model='mistral-nemo:latest'
    )
    
    print(f"\nTop 5 Integrity Issues:")
    for issue in results['integrity_issues'][:5]:
        print(f"- [{issue.severity.upper()}] {issue.description}")
        print(f"  Recommendation: {issue.recommended_action}\n")
