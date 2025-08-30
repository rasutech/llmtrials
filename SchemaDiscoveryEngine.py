import re
import json
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, DML
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Column:
    """Represents a database column with its properties"""
    name: str
    table: str
    data_type: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    is_nullable: bool = True
    has_index: bool = False

@dataclass
class Table:
    """Represents a database table with discovered properties"""
    name: str
    columns: Dict[str, Column] = field(default_factory=dict)
    primary_keys: Set[str] = field(default_factory=set)
    foreign_keys: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # column -> (ref_table, ref_column)
    relationships: List['TableRelationship'] = field(default_factory=list)
    row_count_estimate: Optional[int] = None
    usage_frequency: int = 0
    operation_types: Counter = field(default_factory=Counter)

@dataclass
class TableRelationship:
    """Enhanced relationship with discovered properties"""
    source_table: str
    target_table: str
    join_columns: List[Tuple[str, str]]  # [(source_col, target_col)]
    relationship_type: str  # 'one-to-one', 'one-to-many', 'many-to-many'
    confidence: float
    discovery_method: str  # 'join', 'foreign_key', 'naming_convention', 'llm_inference'
    sample_queries: List[str] = field(default_factory=list)

class SchemaDiscoveryEngine:
    """Dynamically discovers database schema from SQL statements"""
    
    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self.relationships: List[TableRelationship] = []
        self.column_patterns = self._compile_patterns()
        self.naming_conventions = self._detect_naming_conventions()
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for schema discovery"""
        return {
            'foreign_key': re.compile(r'(\w+)_id$', re.IGNORECASE),
            'primary_key': re.compile(r'^id$|_id$|(\w+)_pk$', re.IGNORECASE),
            'create_table': re.compile(r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)', re.IGNORECASE | re.DOTALL),
            'alter_table': re.compile(r'ALTER\s+TABLE\s+(\w+)\s+ADD\s+(?:CONSTRAINT\s+)?(\w+)?\s*FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)', re.IGNORECASE),
            'index_creation': re.compile(r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+\w+\s+ON\s+(\w+)\s*\(([^)]+)\)', re.IGNORECASE),
            'join_pattern': re.compile(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', re.IGNORECASE),
            'table_alias': re.compile(r'(\w+)\s+(?:AS\s+)?(\w+)\s+(?:ON|WHERE|JOIN)', re.IGNORECASE)
        }
    
    def _detect_naming_conventions(self) -> Dict[str, Any]:
        """Detect naming conventions used in the database"""
        return {
            'foreign_key_suffixes': ['_id', '_key', '_ref', '_fk'],
            'primary_key_names': ['id', 'pk', 'oid', 'guid'],
            'junction_table_indicators': ['_to_', '_x_', '_map', '_link', '_assoc'],
            'timestamp_columns': ['created_at', 'updated_at', 'modified_date', 'timestamp'],
            'status_columns': ['status', 'state', 'is_active', 'deleted', 'enabled']
        }
    
    def discover_schema(self, sql_statements: List[str]) -> Dict[str, Table]:
        """Main method to discover complete schema from SQL statements"""
        logger.info("Starting dynamic schema discovery...")
        
        # Phase 1: Extract tables and columns from all SQL types
        for i, sql in enumerate(sql_statements):
            if i % 1000 == 0:
                logger.info(f"Processing SQL {i}/{len(sql_statements)}")
            
            try:
                self._process_sql_statement(sql)
            except Exception as e:
                logger.debug(f"Error processing SQL: {e}")
                continue
        
        # Phase 2: Infer relationships from patterns
        self._infer_relationships()
        
        # Phase 3: Detect junction tables and many-to-many relationships
        self._detect_junction_tables()
        
        # Phase 4: Analyze naming patterns for additional relationships
        self._analyze_naming_patterns()
        
        logger.info(f"Discovered {len(self.tables)} tables with {len(self.relationships)} relationships")
        
        return self.tables
    
    def _process_sql_statement(self, sql: str) -> None:
        """Process a single SQL statement to extract schema information"""
        sql_upper = sql.upper()
        
        # Handle CREATE TABLE statements
        create_match = self.column_patterns['create_table'].search(sql)
        if create_match:
            self._process_create_table(create_match.group(1), create_match.group(2))
            return
        
        # Handle ALTER TABLE for foreign keys
        alter_match = self.column_patterns['alter_table'].search(sql)
        if alter_match:
            self._process_alter_table(alter_match)
            return
        
        # Parse with sqlparse for other statement types
        try:
            parsed = sqlparse.parse(sql)[0]
            self._extract_from_parsed_sql(parsed)
        except:
            # Fallback to regex-based extraction
            self._extract_tables_from_sql(sql)
    
    def _process_create_table(self, table_name: str, columns_def: str) -> None:
        """Process CREATE TABLE statement"""
        table_name = table_name.upper()
        if table_name not in self.tables:
            self.tables[table_name] = Table(name=table_name)
        
        table = self.tables[table_name]
        
        # Parse column definitions
        column_lines = columns_def.split(',')
        for line in column_lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract column name and properties
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0].strip('`"[]').upper()
                
                # Check for PRIMARY KEY
                if 'PRIMARY KEY' in line.upper():
                    table.primary_keys.add(col_name)
                    
                # Check for FOREIGN KEY
                fk_match = re.search(r'REFERENCES\s+(\w+)\s*\((\w+)\)', line, re.IGNORECASE)
                if fk_match:
                    ref_table = fk_match.group(1).upper()
                    ref_column = fk_match.group(2).upper()
                    table.foreign_keys[col_name] = (ref_table, ref_column)
                
                # Add column
                column = Column(
                    name=col_name,
                    table=table_name,
                    is_primary_key=col_name in table.primary_keys,
                    is_foreign_key=col_name in table.foreign_keys,
                    is_nullable='NOT NULL' not in line.upper()
                )
                table.columns[col_name] = column
    
    def _process_alter_table(self, match: re.Match) -> None:
        """Process ALTER TABLE statement for foreign keys"""
        table_name = match.group(1).upper()
        column_name = match.group(3).upper()
        ref_table = match.group(4).upper()
        ref_column = match.group(5).upper()
        
        if table_name not in self.tables:
            self.tables[table_name] = Table(name=table_name)
        
        table = self.tables[table_name]
        table.foreign_keys[column_name] = (ref_table, ref_column)
        
        if column_name in table.columns:
            table.columns[column_name].is_foreign_key = True
            table.columns[column_name].references_table = ref_table
            table.columns[column_name].references_column = ref_column
    
    def _extract_from_parsed_sql(self, parsed) -> None:
        """Extract schema information from parsed SQL"""
        # Get the SQL type
        sql_type = self._get_statement_type(parsed)
        
        # Extract tables based on SQL type
        if sql_type in ['SELECT', 'UPDATE', 'DELETE', 'INSERT']:
            tables = self._extract_tables_from_tokens(parsed.tokens)
            
            for table_name in tables:
                table_name = table_name.upper()
                if table_name not in self.tables:
                    self.tables[table_name] = Table(name=table_name)
                
                self.tables[table_name].usage_frequency += 1
                self.tables[table_name].operation_types[sql_type] += 1
        
        # Extract JOINs for relationships
        sql_text = str(parsed)
        join_matches = self.column_patterns['join_pattern'].findall(sql_text)
        for match in join_matches:
            self._process_join_condition(match)
    
    def _get_statement_type(self, parsed) -> str:
        """Get the type of SQL statement"""
        for token in parsed.tokens:
            if token.ttype is DML:
                return token.value.upper()
        return 'OTHER'
    
    def _extract_tables_from_tokens(self, tokens) -> Set[str]:
        """Extract table names from SQL tokens"""
        tables = set()
        from_seen = False
        
        for token in tokens:
            if token.is_keyword and token.value.upper() in ['FROM', 'JOIN', 'INTO', 'UPDATE']:
                from_seen = True
            elif from_seen and isinstance(token, Identifier):
                table_name = token.get_real_name()
                if table_name:
                    tables.add(table_name)
            elif from_seen and isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    table_name = identifier.get_real_name()
                    if table_name:
                        tables.add(table_name)
        
        return tables
    
    def _extract_tables_from_sql(self, sql: str) -> None:
        """Fallback regex-based table extraction"""
        # Common patterns for table names
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)',
            r'UPDATE\s+(\w+)',
            r'DELETE\s+FROM\s+(\w+)',
            r'INSERT\s+INTO\s+(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            for table_name in matches:
                table_name = table_name.upper()
                if table_name not in self.tables:
                    self.tables[table_name] = Table(name=table_name)
                self.tables[table_name].usage_frequency += 1
    
    def _process_join_condition(self, match: Tuple[str, str, str, str]) -> None:
        """Process a JOIN condition to track relationships"""
        t1, c1, t2, c2 = [x.upper() for x in match]
        
        # Ensure tables exist
        for table in [t1, t2]:
            if table not in self.tables:
                self.tables[table] = Table(name=table)
        
        # Add columns if not exists
        if c1 not in self.tables[t1].columns:
            self.tables[t1].columns[c1] = Column(name=c1, table=t1)
        if c2 not in self.tables[t2].columns:
            self.tables[t2].columns[c2] = Column(name=c2, table=t2)
        
        # Track join relationship
        self._add_relationship(t1, t2, [(c1, c2)], 'join')
    
    def _infer_relationships(self) -> None:
        """Infer relationships based on foreign key patterns"""
        for table_name, table in self.tables.items():
            # Check foreign keys from ALTER TABLE statements
            for col_name, (ref_table, ref_col) in table.foreign_keys.items():
                self._add_relationship(table_name, ref_table, [(col_name, ref_col)], 'foreign_key')
            
            # Check column naming patterns
            for col_name, column in table.columns.items():
                # Look for foreign key naming patterns
                if self.column_patterns['foreign_key'].match(col_name):
                    # Try to find referenced table
                    potential_table = col_name.replace('_ID', '').replace('_KEY', '').upper()
                    if potential_table in self.tables:
                        self._add_relationship(table_name, potential_table, [(col_name, 'ID')], 'naming_convention')
    
    def _detect_junction_tables(self) -> None:
        """Detect junction tables for many-to-many relationships"""
        for table_name, table in self.tables.items():
            # Check if table name suggests junction table
            is_junction = any(indicator in table_name for indicator in self.naming_conventions['junction_table_indicators'])
            
            # Check if table has mainly foreign keys
            if len(table.columns) > 0:
                fk_ratio = len(table.foreign_keys) / len(table.columns)
                if fk_ratio > 0.6 or is_junction:
                    # Likely a junction table
                    self._process_junction_table(table_name, table)
    
    def _process_junction_table(self, table_name: str, table: Table) -> None:
        """Process a junction table to create many-to-many relationships"""
        # Find the two main foreign keys
        fk_tables = list(set(ref_table for ref_table, _ in table.foreign_keys.values()))
        
        if len(fk_tables) >= 2:
            # Create many-to-many relationship between the first two tables
            self._add_relationship(
                fk_tables[0], 
                fk_tables[1], 
                [(table_name, table_name)],  # Junction table as evidence
                'many_to_many_junction',
                relationship_type='many-to-many'
            )
    
    def _analyze_naming_patterns(self) -> None:
        """Analyze column naming patterns for additional relationships"""
        # Group columns by name pattern
        column_groups = defaultdict(list)
        
        for table_name, table in self.tables.items():
            for col_name in table.columns:
                # Extract base name (remove common suffixes)
                base_name = re.sub(r'(_ID|_KEY|_REF|_FK)$', '', col_name, flags=re.IGNORECASE)
                column_groups[base_name].append((table_name, col_name))
        
        # Find relationships from common column names
        for base_name, occurrences in column_groups.items():
            if len(occurrences) > 1 and base_name.upper() in self.tables:
                # Likely foreign key references to the base_name table
                target_table = base_name.upper()
                for source_table, col_name in occurrences:
                    if source_table != target_table:
                        self._add_relationship(
                            source_table, 
                            target_table, 
                            [(col_name, 'ID')], 
                            'naming_pattern'
                        )
    
    def _add_relationship(self, source: str, target: str, 
                         join_columns: List[Tuple[str, str]], 
                         method: str,
                         relationship_type: Optional[str] = None) -> None:
        """Add a discovered relationship"""
        # Check if relationship already exists
        for rel in self.relationships:
            if (rel.source_table == source and rel.target_table == target and 
                rel.join_columns == join_columns):
                # Update confidence if found again
                rel.confidence = min(1.0, rel.confidence + 0.1)
                return
        
        # Determine relationship type if not specified
        if not relationship_type:
            relationship_type = self._infer_relationship_type(source, target, join_columns)
        
        # Create new relationship
        relationship = TableRelationship(
            source_table=source,
            target_table=target,
            join_columns=join_columns,
            relationship_type=relationship_type,
            confidence=0.5 if method == 'naming_convention' else 0.8,
            discovery_method=method
        )
        
        self.relationships.append(relationship)
        
        # Add to table's relationships
        if source in self.tables:
            self.tables[source].relationships.append(relationship)
    
    def _infer_relationship_type(self, source: str, target: str, 
                                join_columns: List[Tuple[str, str]]) -> str:
        """Infer the type of relationship"""
        source_table = self.tables.get(source)
        target_table = self.tables.get(target)
        
        if not source_table or not target_table:
            return 'unknown'
        
        # Check if join column is primary key in either table
        source_col = join_columns[0][0]
        target_col = join_columns[0][1]
        
        source_is_pk = source_col in source_table.primary_keys
        target_is_pk = target_col in target_table.primary_keys
        
        if source_is_pk and target_is_pk:
            return 'one-to-one'
        elif target_is_pk:
            return 'many-to-one'
        elif source_is_pk:
            return 'one-to-many'
        else:
            return 'many-to-many'
    
    def generate_schema_report(self) -> Dict[str, Any]:
        """Generate comprehensive schema discovery report"""
        report = {
            'summary': {
                'total_tables': len(self.tables),
                'total_relationships': len(self.relationships),
                'tables_by_usage': [],
                'relationship_types': defaultdict(int),
                'discovery_methods': defaultdict(int)
            },
            'tables': {},
            'relationships': [],
            'potential_issues': []
        }
        
        # Tables by usage frequency
        tables_by_usage = sorted(
            [(name, table.usage_frequency) for name, table in self.tables.items()],
            key=lambda x: x[1],
            reverse=True
        )
        report['summary']['tables_by_usage'] = tables_by_usage[:20]
        
        # Relationship statistics
        for rel in self.relationships:
            report['summary']['relationship_types'][rel.relationship_type] += 1
            report['summary']['discovery_methods'][rel.discovery_method] += 1
        
        # Detailed table information
        for name, table in self.tables.items():
            report['tables'][name] = {
                'columns': len(table.columns),
                'primary_keys': list(table.primary_keys),
                'foreign_keys': dict(table.foreign_keys),
                'usage_frequency': table.usage_frequency,
                'operation_types': dict(table.operation_types),
                'relationships_count': len(table.relationships)
            }
        
        # Relationships
        report['relationships'] = [
            {
                'source': rel.source_table,
                'target': rel.target_table,
                'type': rel.relationship_type,
                'confidence': rel.confidence,
                'method': rel.discovery_method,
                'join_columns': rel.join_columns
            }
            for rel in sorted(self.relationships, key=lambda x: x.confidence, reverse=True)
        ]
        
        # Identify potential issues
        report['potential_issues'] = self._identify_schema_issues()
        
        return report
    
    def _identify_schema_issues(self) -> List[Dict[str, Any]]:
        """Identify potential schema issues"""
        issues = []
        
        # Tables without primary keys
        for name, table in self.tables.items():
            if not table.primary_keys:
                issues.append({
                    'type': 'missing_primary_key',
                    'table': name,
                    'severity': 'high',
                    'description': f'Table {name} has no identified primary key'
                })
        
        # Isolated tables (no relationships)
        for name, table in self.tables.items():
            if not table.relationships and table.usage_frequency > 10:
                issues.append({
                    'type': 'isolated_table',
                    'table': name,
                    'severity': 'medium',
                    'description': f'Table {name} has no relationships but is frequently used'
                })
        
        # Potential missing foreign keys
        for name, table in self.tables.items():
            for col_name in table.columns:
                if (self.column_patterns['foreign_key'].match(col_name) and 
                    col_name not in table.foreign_keys):
                    issues.append({
                        'type': 'potential_missing_fk',
                        'table': name,
                        'column': col_name,
                        'severity': 'low',
                        'description': f'Column {col_name} looks like a foreign key but has no constraint'
                    })
        
        return issues
    
    def visualize_schema(self, output_path: str, max_tables: int = 50) -> None:
        """Create visual representation of discovered schema"""
        G = nx.DiGraph()
        
        # Select most important tables
        important_tables = sorted(
            self.tables.items(),
            key=lambda x: x[1].usage_frequency,
            reverse=True
        )[:max_tables]
        
        important_table_names = {name for name, _ in important_tables}
        
        # Add nodes
        for name, table in important_tables:
            node_size = min(5000, 1000 + table.usage_frequency * 10)
            G.add_node(name, size=node_size, 
                      color='lightblue' if table.primary_keys else 'lightcoral')
        
        # Add edges for relationships
        for rel in self.relationships:
            if (rel.source_table in important_table_names and 
                rel.target_table in important_table_names):
                G.add_edge(
                    rel.source_table, 
                    rel.target_table,
                    weight=rel.confidence,
                    type=rel.relationship_type,
                    method=rel.discovery_method
                )
        
        # Create visualization
        plt.figure(figsize=(20, 16))
        
        # Use hierarchical layout for better visibility
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes
        node_colors = [G.nodes[node].get('color', 'lightblue') for node in G.nodes()]
        node_sizes = [G.nodes[node].get('size', 1000) for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7)
        
        # Draw edges with different styles for different relationship types
        edge_styles = {
            'one-to-one': 'solid',
            'one-to-many': 'dashed',
            'many-to-many': 'dotted',
            'unknown': 'dashdot'
        }
        
        for rel_type, style in edge_styles.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) 
                    if d.get('type') == rel_type]
            if edges:
                nx.draw_networkx_edges(G, pos, edges, style=style, 
                                     alpha=0.5, arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Add legend
        plt.legend(
            handles=[
                plt.Line2D([0], [0], color='black', linestyle='solid', label='one-to-one'),
                plt.Line2D([0], [0], color='black', linestyle='dashed', label='one-to-many'),
                plt.Line2D([0], [0], color='black', linestyle='dotted', label='many-to-many'),
                plt.scatter([0], [0], color='lightblue', s=100, label='Has Primary Key'),
                plt.scatter([0], [0], color='lightcoral', s=100, label='No Primary Key')
            ],
            loc='upper right'
        )
        
        plt.title('Dynamically Discovered Database Schema', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

class AdaptiveIntegrityAnalyzer:
    """Analyzes integrity based on discovered schema"""
    
    def __init__(self, schema_engine: SchemaDiscoveryEngine):
        self.schema = schema_engine
        self.integrity_rules = []
        self.llm_insights = {}
    
    def generate_integrity_rules(self) -> List[Dict[str, Any]]:
        """Generate integrity rules based on discovered schema"""
        rules = []
        
        # Rule 1: Foreign Key Integrity
        for table_name, table in self.schema.tables.items():
            for col_name, (ref_table, ref_col) in table.foreign_keys.items():
                rules.append({
                    'rule_id': f'FK_{table_name}_{col_name}',
                    'type': 'foreign_key_integrity',
                    'description': f'{table_name}.{col_name} must reference valid {ref_table}.{ref_col}',
                    'check_sql': f"""
                        SELECT t1.* FROM {table_name} t1 
                        LEFT JOIN {ref_table} t2 ON t1.{col_name} = t2.{ref_col}
                        WHERE t2.{ref_col} IS NULL AND t1.{col_name} IS NOT NULL
                    """,
                    'fix_sql': f"""
                        DELETE FROM {table_name} 
                        WHERE {col_name} NOT IN (SELECT {ref_col} FROM {ref_table})
                        AND {col_name} IS NOT NULL
                    """,
                    'severity': 'high'
                })
        
        # Rule 2: Orphaned Records in Related Tables
        for rel in self.schema.relationships:
            if rel.relationship_type in ['one-to-many', 'many-to-one']:
                rules.append({
                    'rule_id': f'ORPHAN_{rel.source_table}_{rel.target_table}',
                    'type': 'orphaned_records',
                    'description': f'Check for orphaned records between {rel.source_table} and {rel.target_table}',
                    'check_sql': f"""
                        SELECT * FROM {rel.source_table} s
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {rel.target_table} t
                            WHERE s.{rel.join_columns[0][0]} = t.{rel.join_columns[0][1]}
                        )
                    """,
                    'severity': 'medium'
                })
        
        # Rule 3: Duplicate Detection Based on Natural Keys
        for table_name, table in self.schema.tables.items():
            # Find potential natural key combinations
            non_pk_columns = [col for col in table.columns if col not in table.primary_keys]
            if non_pk_columns and len(non_pk_columns) <= 3:
                cols_str = ', '.join(non_pk_columns[:3])
                rules.append({
                    'rule_id': f'DUP_{table_name}',
                    'type': 'duplicate_detection',
                    'description': f'Check for duplicates in {table_name} based on {cols_str}',
                    'check_sql': f"""
                        SELECT {cols_str}, COUNT(*) as cnt
                        FROM {table_name}
                        GROUP BY {cols_str}
                        HAVING COUNT(*) > 1
                    """,
                    'severity': 'medium'
                })
        
        # Rule 4: Cyclic Dependencies
        for table_name, table in self.schema.tables.items():
            if len(table.foreign_keys) > 1:
                rules.append({
                    'rule_id': f'CYCLE_{table_name}',
                    'type': 'cyclic_dependency',
                    'description': f'Check for cyclic dependencies in {table_name}',
                    'check_sql': f"""
                        -- Complex recursive CTE to detect cycles
                        -- Specific implementation depends on the relationships
                    """,
                    'severity': 'high'
                })
        
        # Rule 5: Data Type Consistency
        column_types = defaultdict(lambda: defaultdict(set))
        for table_name, table in self.schema.tables.items():
            for col_name, column in table.columns.items():
                base_name = re.sub(r'(_ID|_KEY|_REF|_FK)$', '', col_name, flags=re.IGNORECASE)
                column_types[base_name][table_name].add(col_name)
        
        for base_name, table_cols in column_types.items():
            if len(table_cols) > 1:
                rules.append({
                    'rule_id': f'CONSISTENCY_{base_name}',
                    'type': 'data_type_consistency',
                    'description': f'Ensure consistent data types for {base_name} across tables',
                    'tables': list(table_cols.keys()),
                    'severity': 'low'
                })
        
        return rules
    
    def analyze_with_llm(self, sql_patterns: List[str], llm_endpoint: str) -> Dict[str, Any]:
        """Use LLM to understand complex integrity patterns"""
        # Group similar SQL patterns
        pattern_groups = self._group_similar_patterns(sql_patterns)
        
        insights = {}
        for group_name, patterns in pattern_groups.items():
            prompt = f"""
            Analyze these SQL patterns from a database with the following key tables:
            {', '.join(list(self.schema.tables.keys())[:20])}
            
            SQL Pattern Group: {group_name}
            Sample SQLs:
            {chr(10).join(patterns[:5])}
            
            Please identify:
            1. What integrity issue these queries are addressing
            2. The business rule being enforced
            3. Potential risks or improvements
            4. Related tables and their relationships
            
            Return as JSON with keys: issue_type, business_rule, risks, improvements, affected_tables
            """
            
            try:
                response = requests.post(
                    llm_endpoint,
                    json={'prompt': prompt, 'max_tokens': 500},
                    timeout=30
                )
                insights[group_name] = response.json()
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                insights[group_name] = None
        
        return insights
    
    def _group_similar_patterns(self, sql_patterns: List[str]) -> Dict[str, List[str]]:
        """Group similar SQL patterns using pattern matching"""
        groups = defaultdict(list)
        
        for sql in sql_patterns:
            # Normalize SQL for grouping
            normalized = re.sub(r'\b\d+\b', 'N', sql)  # Replace numbers with N
            normalized = re.sub(r"'[^']*'", "'STR'", normalized)  # Replace strings
            
            # Simple grouping by operation and main table
            operation = sql.split()[0].upper()
            tables = self._extract_main_table(sql)
            
            group_key = f"{operation}_{tables}"
            groups[group_key].append(sql)
        
        return dict(groups)
    
    def _extract_main_table(self, sql: str) -> str:
        """Extract the main table from SQL"""
        patterns = [
            (r'FROM\s+(\w+)', 'FROM'),
            (r'UPDATE\s+(\w+)', 'UPDATE'),
            (r'INSERT\s+INTO\s+(\w+)', 'INSERT'),
            (r'DELETE\s+FROM\s+(\w+)', 'DELETE')
        ]
        
        for pattern, _ in patterns:
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return 'UNKNOWN'

# Main execution function
def analyze_database_integrity(sql_file_path: str,
                             llm_endpoints: Dict[str, str],
                             output_dir: str = './integrity_analysis') -> Dict[str, Any]:
    """Complete database integrity analysis with dynamic schema discovery"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting comprehensive database analysis...")
    
    # Step 1: Read SQL file
    with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split into individual statements
    sql_statements = [s.strip() for s in sqlparse.split(content) if s.strip()]
    logger.info(f"Found {len(sql_statements)} SQL statements")
    
    # Step 2: Discover schema dynamically
    schema_engine = SchemaDiscoveryEngine()
    discovered_tables = schema_engine.discover_schema(sql_statements)
    
    # Generate schema report
    schema_report = schema_engine.generate_schema_report()
    
    # Save schema report
    with open(os.path.join(output_dir, 'discovered_schema.json'), 'w') as f:
        json.dump(schema_report, f, indent=2)
    
    # Visualize schema
    schema_engine.visualize_schema(
        os.path.join(output_dir, 'discovered_schema_visualization.png')
    )
    
    # Step 3: Generate integrity rules based on discovered schema
    integrity_analyzer = AdaptiveIntegrityAnalyzer(schema_engine)
    integrity_rules = integrity_analyzer.generate_integrity_rules()
    
    # Save integrity rules
    with open(os.path.join(output_dir, 'generated_integrity_rules.json'), 'w') as f:
        json.dump(integrity_rules, f, indent=2)
    
    # Step 4: Analyze patterns with LLM
    logger.info("Analyzing SQL patterns with LLM...")
    sample_sqls = sql_statements[:1000]  # Sample for LLM analysis
    llm_insights = integrity_analyzer.analyze_with_llm(
        sample_sqls, 
        llm_endpoints.get('qwen', 'http://localhost:8080/v1/completions')
    )
    
    # Step 5: Generate comprehensive report
    final_report = {
        'summary': {
            'total_tables_discovered': len(discovered_tables),
            'total_relationships': len(schema_engine.relationships),
            'total_integrity_rules': len(integrity_rules),
            'timestamp': str(datetime.now())
        },
        'top_tables': schema_report['summary']['tables_by_usage'][:10],
        'relationship_summary': dict(schema_report['summary']['relationship_types']),
        'potential_issues': schema_report['potential_issues'][:20],
        'integrity_rules_summary': {
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int)
        },
        'llm_insights': llm_insights
    }
    
    # Summarize integrity rules
    for rule in integrity_rules:
        final_report['integrity_rules_summary']['by_type'][rule['type']] += 1
        final_report['integrity_rules_summary']['by_severity'][rule.get('severity', 'unknown')] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    final_report['integrity_rules_summary']['by_type'] = dict(final_report['integrity_rules_summary']['by_type'])
    final_report['integrity_rules_summary']['by_severity'] = dict(final_report['integrity_rules_summary']['by_severity'])
    
    # Save final report
    with open(os.path.join(output_dir, 'integrity_analysis_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(final_report, schema_report, integrity_rules)
    with open(os.path.join(output_dir, 'integrity_analysis_report.md'), 'w') as f:
        f.write(markdown_report)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    
    return final_report

def generate_markdown_report(final_report: Dict, schema_report: Dict, integrity_rules: List[Dict]) -> str:
    """Generate a human-readable markdown report"""
    report = f"""# Database Integrity Analysis Report
Generated: {final_report['summary']['timestamp']}

## Executive Summary
- **Tables Discovered**: {final_report['summary']['total_tables_discovered']}
- **Relationships Found**: {final_report['summary']['total_relationships']}
- **Integrity Rules Generated**: {final_report['summary']['total_integrity_rules']}

## Top 10 Most Used Tables
| Table Name | Usage Count |
|------------|-------------|
"""
    
    for table, count in final_report['top_tables']:
        report += f"| {table} | {count} |\n"
    
    report += "\n## Discovered Relationships\n"
    for rel_type, count in final_report['relationship_summary'].items():
        report += f"- **{rel_type}**: {count} relationships\n"
    
    report += "\n## Potential Schema Issues\n"
    for issue in final_report['potential_issues'][:10]:
        report += f"- **{issue['type']}** ({issue['severity']}): {issue['description']}\n"
    
    report += "\n## Generated Integrity Rules\n"
    report += "### By Type\n"
    for rule_type, count in final_report['integrity_rules_summary']['by_type'].items():
        report += f"- **{rule_type}**: {count} rules\n"
    
    report += "\n### Sample Rules\n"
    for rule in integrity_rules[:5]:
        report += f"""
#### {rule['rule_id']}
- **Type**: {rule['type']}
- **Severity**: {rule.get('severity', 'unknown')}
- **Description**: {rule['description']}
"""
    
    return report

# Example usage
if __name__ == "__main__":
    llm_endpoints = {
        'qwen': 'http://your-server:port/qwen3-30b/v1/completions',
        'gpt': 'http://your-server:port/gpt-oss/v1/completions'
    }
    
    results = analyze_database_integrity(
        sql_file_path='path/to/your/vsql_file.sql',
        llm_endpoints=llm_endpoints,
        output_dir='./dynamic_integrity_analysis'
    )
    
    print("Analysis complete!")
    print(f"Discovered {results['summary']['total_tables_discovered']} tables")
    print(f"Found {results['summary']['total_relationships']} relationships")
    print(f"Generated {results['summary']['total_integrity_rules']} integrity rules")
