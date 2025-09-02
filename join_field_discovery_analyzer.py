import pandas as pd
import re
import json
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import logging

# Import your LLM wrapper
from ollama_wrapper import answer_from_ollama

logger = logging.getLogger(__name__)

@dataclass
class JoinPattern:
    """Represents a specific way two tables are joined"""
    source_table: str
    target_table: str
    source_column: str
    target_column: str
    join_type: str  # INNER, LEFT, RIGHT, FULL
    frequency: int = 1
    example_sqls: List[str] = field(default_factory=list)
    contexts: Set[str] = field(default_factory=set)  # e.g., "with EQUIPMENT", "with SIGNAL"
    
    def __hash__(self):
        return hash((self.source_table, self.target_table, self.source_column, self.target_column))
    
    @property
    def pattern_key(self) -> str:
        return f"{self.source_table}.{self.source_column} -> {self.target_table}.{self.target_column}"

@dataclass
class TableJoinProfile:
    """Complete join profile for a table"""
    table_name: str
    join_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # column -> stats
    join_patterns: List[JoinPattern] = field(default_factory=list)
    column_roles: Dict[str, str] = field(default_factory=dict)  # column -> role (PK, FK, join_key)
    
class JoinFieldDiscoveryEngine:
    """Discovers all join fields and patterns between tables"""
    
    def __init__(self):
        self.join_patterns: Dict[Tuple[str, str, str, str], JoinPattern] = {}
        self.table_profiles: Dict[str, TableJoinProfile] = {}
        self.multi_path_relationships: Dict[Tuple[str, str], List[JoinPattern]] = defaultdict(list)
        self.column_join_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'used_as_join': 0,
            'tables_joined': set(),
            'join_types': Counter(),
            'likely_role': 'unknown'
        })
        
    def analyze_sql_for_joins(self, sql: str, sql_id: str) -> Dict[str, Any]:
        """Extract all join patterns from a SQL statement"""
        analysis = {
            'sql_id': sql_id,
            'join_patterns': [],
            'tables_involved': set(),
            'join_context': self._extract_join_context(sql)
        }
        
        # Only analyze SELECT statements
        if not sql.strip().upper().startswith('SELECT'):
            return analysis
        
        # Extract all types of joins
        self._extract_explicit_joins(sql, analysis)
        self._extract_implicit_joins(sql, analysis)
        self._extract_subquery_joins(sql, analysis)
        
        # Update statistics
        self._update_join_statistics(analysis)
        
        return analysis
    
    def _extract_explicit_joins(self, sql: str, analysis: Dict) -> None:
        """Extract explicit JOIN statements"""
        # Pattern for JOIN...ON statements
        join_pattern = re.compile(
            r'(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+'
            r'(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+'
            r'(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+'
            r'ON\s+((?:[^W]|W(?!HERE))+?)(?:WHERE|GROUP|ORDER|HAVING|LIMIT|UNION|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        for match in join_pattern.finditer(sql):
            t1 = match.group(1).upper()
            alias1 = (match.group(2) or t1).upper()
            join_type = (match.group(3) or 'INNER').upper()
            t2 = match.group(4).upper()
            alias2 = (match.group(5) or t2).upper()
            join_condition = match.group(6)
            
            # Parse join conditions
            join_pairs = self._parse_join_conditions(join_condition, alias1, alias2, t1, t2)
            
            for src_col, tgt_col in join_pairs:
                pattern = {
                    'source_table': t1,
                    'target_table': t2,
                    'source_column': src_col,
                    'target_column': tgt_col,
                    'join_type': join_type,
                    'context': analysis['join_context']
                }
                analysis['join_patterns'].append(pattern)
                analysis['tables_involved'].update([t1, t2])
    
    def _extract_implicit_joins(self, sql: str, analysis: Dict) -> None:
        """Extract implicit joins from WHERE clause"""
        # Look for table1.col = table2.col patterns in WHERE
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|HAVING|LIMIT|$)', 
                               sql, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            # Find equality conditions between different tables
            eq_pattern = re.compile(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)')
            
            for match in eq_pattern.finditer(where_clause):
                t1, c1, t2, c2 = match.groups()
                t1, c1, t2, c2 = t1.upper(), c1.upper(), t2.upper(), c2.upper()
                
                if t1 != t2:  # Different tables
                    pattern = {
                        'source_table': t1,
                        'target_table': t2,
                        'source_column': c1,
                        'target_column': c2,
                        'join_type': 'IMPLICIT',
                        'context': analysis['join_context']
                    }
                    analysis['join_patterns'].append(pattern)
                    analysis['tables_involved'].update([t1, t2])
    
    def _extract_subquery_joins(self, sql: str, analysis: Dict) -> None:
        """Extract joins through subqueries (IN, EXISTS)"""
        # Pattern for IN subqueries
        in_pattern = re.compile(
            r'(\w+)\.(\w+)\s+IN\s*\(\s*SELECT\s+(?:\w+\.)?(\w+)\s+FROM\s+(\w+)',
            re.IGNORECASE
        )
        
        for match in in_pattern.finditer(sql):
            t1, c1, c2, t2 = match.groups()
            t1, c1, c2, t2 = t1.upper(), c1.upper(), c2.upper(), t2.upper()
            
            pattern = {
                'source_table': t1,
                'target_table': t2,
                'source_column': c1,
                'target_column': c2,
                'join_type': 'SUBQUERY_IN',
                'context': analysis['join_context']
            }
            analysis['join_patterns'].append(pattern)
            analysis['tables_involved'].update([t1, t2])
    
    def _parse_join_conditions(self, condition: str, alias1: str, alias2: str, 
                              table1: str, table2: str) -> List[Tuple[str, str]]:
        """Parse complex join conditions including AND/OR"""
        join_pairs = []
        
        # Split by AND (most common)
        and_parts = re.split(r'\s+AND\s+', condition, flags=re.IGNORECASE)
        
        for part in and_parts:
            # Look for equality conditions
            eq_match = re.search(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', part)
            if eq_match:
                t1, c1, t2, c2 = eq_match.groups()
                t1, c1, t2, c2 = t1.upper(), c1.upper(), t2.upper(), c2.upper()
                
                # Map aliases to actual tables
                actual_t1 = table1 if t1 == alias1 else (table2 if t1 == alias2 else t1)
                actual_t2 = table1 if t2 == alias1 else (table2 if t2 == alias2 else t2)
                
                # Ensure we have the correct source->target order
                if actual_t1 == table1 and actual_t2 == table2:
                    join_pairs.append((c1, c2))
                elif actual_t1 == table2 and actual_t2 == table1:
                    join_pairs.append((c2, c1))
        
        return join_pairs
    
    def _extract_join_context(self, sql: str) -> str:
        """Extract context about what other tables are involved in the query"""
        # Get all table names mentioned
        table_pattern = re.compile(r'(?:FROM|JOIN)\s+(\w+)', re.IGNORECASE)
        tables = set(match.group(1).upper() for match in table_pattern.finditer(sql))
        
        # Create context string
        if len(tables) > 2:
            return f"multi_table_join_{len(tables)}"
        else:
            return "simple_join"
    
    def _update_join_statistics(self, analysis: Dict) -> None:
        """Update join pattern statistics"""
        for pattern_data in analysis['join_patterns']:
            # Create or update join pattern
            key = (
                pattern_data['source_table'],
                pattern_data['target_table'],
                pattern_data['source_column'],
                pattern_data['target_column']
            )
            
            if key not in self.join_patterns:
                self.join_patterns[key] = JoinPattern(
                    source_table=pattern_data['source_table'],
                    target_table=pattern_data['target_table'],
                    source_column=pattern_data['source_column'],
                    target_column=pattern_data['target_column'],
                    join_type=pattern_data['join_type']
                )
            else:
                self.join_patterns[key].frequency += 1
            
            # Add context
            self.join_patterns[key].contexts.add(pattern_data['context'])
            
            # Update multi-path relationships
            table_pair = (pattern_data['source_table'], pattern_data['target_table'])
            if self.join_patterns[key] not in self.multi_path_relationships[table_pair]:
                self.multi_path_relationships[table_pair].append(self.join_patterns[key])
            
            # Update column statistics
            src_col_key = f"{pattern_data['source_table']}.{pattern_data['source_column']}"
            tgt_col_key = f"{pattern_data['target_table']}.{pattern_data['target_column']}"
            
            self.column_join_stats[src_col_key]['used_as_join'] += 1
            self.column_join_stats[src_col_key]['tables_joined'].add(pattern_data['target_table'])
            self.column_join_stats[src_col_key]['join_types'][pattern_data['join_type']] += 1
            
            self.column_join_stats[tgt_col_key]['used_as_join'] += 1
            self.column_join_stats[tgt_col_key]['tables_joined'].add(pattern_data['source_table'])
            self.column_join_stats[tgt_col_key]['join_types'][pattern_data['join_type']] += 1
            
            # Update table profiles
            self._update_table_profile(pattern_data['source_table'], pattern_data['source_column'], 'source')
            self._update_table_profile(pattern_data['target_table'], pattern_data['target_column'], 'target')
    
    def _update_table_profile(self, table: str, column: str, role: str) -> None:
        """Update table profile with join information"""
        if table not in self.table_profiles:
            self.table_profiles[table] = TableJoinProfile(table_name=table)
        
        profile = self.table_profiles[table]
        
        if column not in profile.join_columns:
            profile.join_columns[column] = {
                'join_count': 0,
                'as_source': 0,
                'as_target': 0,
                'joined_tables': set(),
                'join_types': Counter()
            }
        
        profile.join_columns[column]['join_count'] += 1
        if role == 'source':
            profile.join_columns[column]['as_source'] += 1
        else:
            profile.join_columns[column]['as_target'] += 1
    
    def analyze_join_patterns(self) -> Dict[str, Any]:
        """Analyze all discovered join patterns"""
        analysis = {
            'total_patterns': len(self.join_patterns),
            'multi_path_relationships': {},
            'key_columns': {},
            'join_complexity': {}
        }
        
        # Analyze multi-path relationships
        for (t1, t2), patterns in self.multi_path_relationships.items():
            if len(patterns) > 1:
                analysis['multi_path_relationships'][f"{t1}-{t2}"] = {
                    'path_count': len(patterns),
                    'paths': [
                        {
                            'join_key': p.pattern_key,
                            'frequency': p.frequency,
                            'join_type': p.join_type
                        }
                        for p in sorted(patterns, key=lambda x: x.frequency, reverse=True)
                    ]
                }
        
        # Identify key columns
        for col_key, stats in self.column_join_stats.items():
            if stats['used_as_join'] > 5:  # Threshold for key column
                # Determine likely role
                if col_key.endswith('_ID') or col_key.endswith('ID'):
                    likely_role = 'primary_key' if len(stats['tables_joined']) > 2 else 'foreign_key'
                else:
                    likely_role = 'join_key'
                
                analysis['key_columns'][col_key] = {
                    'join_frequency': stats['used_as_join'],
                    'tables_joined': list(stats['tables_joined']),
                    'likely_role': likely_role,
                    'join_types': dict(stats['join_types'])
                }
        
        return analysis
    
    def visualize_join_patterns(self, output_path: str, focus_tables: Optional[List[str]] = None):
        """Create comprehensive visualization of join patterns"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # Left plot: Table relationship graph with join fields
        self._plot_relationship_graph(ax1, focus_tables)
        
        # Right plot: Join pattern details
        self._plot_join_details(ax2, focus_tables)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_relationship_graph(self, ax, focus_tables: Optional[List[str]] = None):
        """Plot graph showing tables and their join relationships"""
        G = nx.MultiDiGraph()
        
        # Add nodes and edges
        for pattern in self.join_patterns.values():
            if focus_tables and not (pattern.source_table in focus_tables or 
                                   pattern.target_table in focus_tables):
                continue
            
            G.add_edge(
                pattern.source_table,
                pattern.target_table,
                label=f"{pattern.source_column}->{pattern.target_column}",
                frequency=pattern.frequency,
                join_type=pattern.join_type
            )
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No relationships found', ha='center', va='center')
            return
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_sizes = []
        for node in G.nodes():
            profile = self.table_profiles.get(node)
            if profile:
                size = min(3000, 500 + len(profile.join_columns) * 200)
            else:
                size = 500
            node_sizes.append(size)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, ax=ax)
        
        # Draw edges with labels
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            if (u, v) not in edge_labels:
                edge_labels[(u, v)] = []
            edge_labels[(u, v)].append(data['label'])
        
        # Combine multiple edges between same nodes
        for key, labels in edge_labels.items():
            edge_labels[key] = '\n'.join(labels[:3])  # Show max 3
            if len(labels) > 3:
                edge_labels[key] += f'\n... +{len(labels)-3} more'
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        ax.set_title('Table Relationships with Join Fields', fontsize=14)
        ax.axis('off')
    
    def _plot_join_details(self, ax, focus_tables: Optional[List[str]] = None):
        """Plot detailed join pattern information"""
        # Get top join patterns
        top_patterns = sorted(self.join_patterns.values(), 
                            key=lambda x: x.frequency, reverse=True)[:20]
        
        if focus_tables:
            top_patterns = [p for p in top_patterns 
                          if p.source_table in focus_tables or p.target_table in focus_tables]
        
        if not top_patterns:
            ax.text(0.5, 0.5, 'No join patterns found', ha='center', va='center')
            return
        
        # Create bar chart
        patterns_str = [f"{p.source_table}.{p.source_column}\n→\n{p.target_table}.{p.target_column}" 
                       for p in top_patterns[:15]]
        frequencies = [p.frequency for p in top_patterns[:15]]
        
        y_pos = range(len(patterns_str))
        bars = ax.barh(y_pos, frequencies)
        
        # Color by join type
        colors = {
            'INNER': 'green',
            'LEFT': 'orange',
            'RIGHT': 'red',
            'FULL': 'purple',
            'IMPLICIT': 'blue',
            'SUBQUERY_IN': 'brown'
        }
        
        for i, (bar, pattern) in enumerate(zip(bars, top_patterns[:15])):
            bar.set_color(colors.get(pattern.join_type, 'gray'))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(patterns_str, fontsize=8)
        ax.set_xlabel('Frequency')
        ax.set_title('Top Join Patterns by Frequency', fontsize=14)
        
        # Add legend
        legend_elements = [mpatches.Patch(color=color, label=jtype) 
                         for jtype, color in colors.items()]
        ax.legend(handles=legend_elements, loc='lower right')
        
        ax.grid(axis='x', alpha=0.3)

class JoinBasedIntegrityAnalyzer:
    """Analyzes integrity issues based on discovered join patterns"""
    
    def __init__(self, discovery_engine: JoinFieldDiscoveryEngine, llm_model: str = "mistral-nemo:latest"):
        self.discovery = discovery_engine
        self.llm_model = llm_model
        
    def analyze_integrity_from_joins(self) -> List[Dict[str, Any]]:
        """Analyze integrity issues from join patterns"""
        issues = []
        
        # Check for inconsistent join patterns
        issues.extend(self._check_inconsistent_joins())
        
        # Check for missing foreign keys
        issues.extend(self._check_missing_foreign_keys())
        
        # Check for unusual join patterns
        issues.extend(self._check_unusual_patterns())
        
        # Analyze complex patterns with LLM
        issues.extend(self._analyze_complex_joins_with_llm())
        
        return issues
    
    def _check_inconsistent_joins(self) -> List[Dict[str, Any]]:
        """Check for tables joined in multiple different ways"""
        issues = []
        
        for (t1, t2), patterns in self.discovery.multi_path_relationships.items():
            if len(patterns) > 1:
                # Calculate consistency score
                total_freq = sum(p.frequency for p in patterns)
                dominant_pattern = max(patterns, key=lambda x: x.frequency)
                consistency = dominant_pattern.frequency / total_freq
                
                if consistency < 0.8:  # Less than 80% use same pattern
                    issue = {
                        'type': 'inconsistent_join_pattern',
                        'severity': 'medium' if consistency > 0.5 else 'high',
                        'tables': [t1, t2],
                        'description': f"Tables {t1} and {t2} are joined using {len(patterns)} different patterns",
                        'details': {
                            'patterns': [
                                {
                                    'join': p.pattern_key,
                                    'frequency': p.frequency,
                                    'percentage': f"{(p.frequency/total_freq)*100:.1f}%"
                                }
                                for p in patterns
                            ]
                        },
                        'recommendation': f"Standardize joins between {t1} and {t2}. "
                                        f"Consider using the dominant pattern: {dominant_pattern.pattern_key}"
                    }
                    issues.append(issue)
        
        return issues
    
    def _check_missing_foreign_keys(self) -> List[Dict[str, Any]]:
        """Identify likely foreign keys based on join patterns"""
        issues = []
        
        for pattern in self.discovery.join_patterns.values():
            if pattern.frequency > 10:  # Frequently joined
                # Check if columns follow FK naming pattern
                if (pattern.source_column.endswith('_ID') or 
                    pattern.source_column.endswith('_KEY') or
                    pattern.source_column == f"{pattern.target_table}_ID"):
                    
                    issue = {
                        'type': 'missing_foreign_key',
                        'severity': 'high',
                        'tables': [pattern.source_table, pattern.target_table],
                        'description': f"Likely missing FK: {pattern.pattern_key}",
                        'details': {
                            'join_frequency': pattern.frequency,
                            'join_types': list(pattern.contexts),
                            'column_pattern': 'Follows FK naming convention'
                        },
                        'recommendation': f"ALTER TABLE {pattern.source_table} "
                                        f"ADD CONSTRAINT FK_{pattern.source_table}_{pattern.source_column} "
                                        f"FOREIGN KEY ({pattern.source_column}) "
                                        f"REFERENCES {pattern.target_table}({pattern.target_column})"
                    }
                    issues.append(issue)
        
        return issues
    
    def _check_unusual_patterns(self) -> List[Dict[str, Any]]:
        """Check for unusual join patterns that might indicate issues"""
        issues = []
        
        # Check for circular joins
        for pattern in self.discovery.join_patterns.values():
            # Self-referencing tables
            if pattern.source_table == pattern.target_table:
                issue = {
                    'type': 'self_referencing_join',
                    'severity': 'low',
                    'tables': [pattern.source_table],
                    'description': f"Table {pattern.source_table} has self-referencing join",
                    'details': {
                        'join_column': f"{pattern.source_column} -> {pattern.target_column}",
                        'frequency': pattern.frequency
                    },
                    'recommendation': "Verify if hierarchical relationship is intended"
                }
                issues.append(issue)
        
        return issues
    
    def _analyze_complex_joins_with_llm(self) -> List[Dict[str, Any]]:
        """Use LLM to analyze complex join patterns"""
        issues = []
        
        # Get tables with most complex join patterns
        complex_tables = sorted(
            self.discovery.table_profiles.items(),
            key=lambda x: len(x[1].join_columns),
            reverse=True
        )[:5]
        
        for table, profile in complex_tables:
            if len(profile.join_columns) > 3:
                join_desc = []
                for col, stats in profile.join_columns.items():
                    join_desc.append(
                        f"- {col}: joins with {len(stats['joined_tables'])} tables, "
                        f"used {stats['join_count']} times"
                    )
                
                prompt = f"""Analyze these join patterns for table {table}:

Join Columns:
{chr(10).join(join_desc[:10])}

This table has {len(profile.join_columns)} different columns used for joining.

Identify potential issues with this join pattern.
Respond in JSON: {{"issues": [{{"type": "...", "severity": "high/medium/low", "description": "...", "recommendation": "..."}}]}}"""

                try:
                    response = answer_from_ollama(prompt, self.llm_model)
                    llm_issues = json.loads(response).get('issues', [])
                    
                    for llm_issue in llm_issues:
                        issue = {
                            'type': f"llm_{llm_issue['type']}",
                            'severity': llm_issue['severity'],
                            'tables': [table],
                            'description': llm_issue['description'],
                            'details': {
                                'llm_analysis': True,
                                'join_column_count': len(profile.join_columns)
                            },
                            'recommendation': llm_issue['recommendation']
                        }
                        issues.append(issue)
                
                except Exception as e:
                    logger.debug(f"LLM analysis failed: {e}")
        
        return issues

def generate_join_analysis_report(discovery_engine: JoinFieldDiscoveryEngine, 
                                integrity_analyzer: JoinBasedIntegrityAnalyzer,
                                output_path: str) -> str:
    """Generate comprehensive report on join patterns and integrity"""
    
    patterns_analysis = discovery_engine.analyze_join_patterns()
    integrity_issues = integrity_analyzer.analyze_integrity_from_joins()
    
    report = f"""# Database Join Pattern Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Join Patterns Discovered**: {patterns_analysis['total_patterns']}
- **Multi-Path Relationships**: {len(patterns_analysis['multi_path_relationships'])}
- **Key Join Columns Identified**: {len(patterns_analysis['key_columns'])}
- **Integrity Issues Found**: {len(integrity_issues)}

## Multi-Path Relationships (Tables Joined in Multiple Ways)

These table pairs have multiple join patterns, which might indicate:
- Different types of relationships between tables
- Inconsistent database design
- Evolution of schema over time

"""
    
    # Multi-path relationships
    for rel_key, rel_data in patterns_analysis['multi_path_relationships'].items():
        report += f"\n### {rel_key}\n"
        report += f"**Number of Different Join Patterns**: {rel_data['path_count']}\n\n"
        
        for i, path in enumerate(rel_data['paths'], 1):
            report += f"{i}. `{path['join_key']}` - Used {path['frequency']} times ({path['join_type']} JOIN)\n"
    
    # Key columns
    report += "\n## Key Join Columns\n\n"
    report += "| Column | Join Frequency | Role | Tables Joined |\n"
    report += "|--------|----------------|------|---------------|\n"
    
    key_cols = sorted(patterns_analysis['key_columns'].items(), 
                     key=lambda x: x[1]['join_frequency'], reverse=True)[:20]
    
    for col, data in key_cols:
        tables = ', '.join(data['tables_joined'][:3])
        if len(data['tables_joined']) > 3:
            tables += f" +{len(data['tables_joined'])-3} more"
        
        report += f"| {col} | {data['join_frequency']} | {data['likely_role']} | {tables} |\n"
    
    # Integrity issues
    report += "\n## Integrity Issues\n\n"
    
    issues_by_type = defaultdict(list)
    for issue in integrity_issues:
        issues_by_type[issue['type']].append(issue)
    
    for issue_type, issues in issues_by_type.items():
        report += f"\n### {issue_type.replace('_', ' ').title()}\n\n"
        
        for issue in issues[:5]:  # Top 5 per type
            report += f"**{issue['description']}**\n"
            report += f"- Severity: {issue['severity']}\n"
            report += f"- Tables: {', '.join(issue['tables'])}\n"
            
            if 'details' in issue:
                report += "- Details:\n"
                for key, value in issue['details'].items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        report += f"  - {key}:\n"
                        for item in value[:3]:
                            report += f"    - {item}\n"
                    else:
                        report += f"  - {key}: {value}\n"
            
            report += f"- **Recommendation**: {issue['recommendation']}\n\n"
    
    # Example complex join
    report += "\n## Example Complex Join Patterns\n\n"
    
    complex_patterns = [p for p in discovery_engine.join_patterns.values() 
                       if len(p.contexts) > 1][:3]
    
    for pattern in complex_patterns:
        report += f"### {pattern.pattern_key}\n"
        report += f"- Frequency: {pattern.frequency}\n"
        report += f"- Join Type: {pattern.join_type}\n"
        report += f"- Used in contexts: {', '.join(list(pattern.contexts))}\n\n"
    
    return report

# Main analysis function
def analyze_join_fields(csv_path: str,
                       output_dir: str = './join_analysis',
                       sample_size: int = 10000,
                       llm_model: str = "mistral-nemo:latest") -> Dict[str, Any]:
    """
    Analyze all join fields and patterns in the database
    
    Returns detailed information about:
    - All different ways tables are joined
    - Which fields are used for joining
    - Multi-path relationships
    - Potential integrity issues
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize engines
    discovery_engine = JoinFieldDiscoveryEngine()
    
    # Load and process SQL statements
    logger.info(f"Loading SQL statements from {csv_path}")
    df = pd.read_csv(csv_path, nrows=sample_size)
    
    logger.info(f"Analyzing {len(df)} SQL statements for join patterns...")
    
    # Process each SQL
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"Progress: {idx}/{len(df)}")
        
        try:
            discovery_engine.analyze_sql_for_joins(row['SQL_FULLTEXT'], row['SQL_ID'])
        except Exception as e:
            logger.debug(f"Error processing SQL {row['SQL_ID']}: {e}")
    
    # Analyze integrity
    integrity_analyzer = JoinBasedIntegrityAnalyzer(discovery_engine, llm_model)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    discovery_engine.visualize_join_patterns(
        os.path.join(output_dir, 'join_patterns.png')
    )
    
    # Generate report
    logger.info("Generating report...")
    report = generate_join_analysis_report(discovery_engine, integrity_analyzer, output_dir)
    
    with open(os.path.join(output_dir, 'join_analysis_report.md'), 'w') as f:
        f.write(report)
    
    # Save detailed results
    results = {
        'total_patterns': len(discovery_engine.join_patterns),
        'tables_analyzed': len(discovery_engine.table_profiles),
        'multi_path_count': len([p for p in discovery_engine.multi_path_relationships.values() 
                                if len(p) > 1]),
        'sample_patterns': [
            {
                'pattern': p.pattern_key,
                'frequency': p.frequency,
                'join_type': p.join_type
            }
            for p in list(discovery_engine.join_patterns.values())[:50]
        ]
    }
    
    with open(os.path.join(output_dir, 'join_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    
    # Print summary
    print(f"\n=== Join Pattern Analysis Summary ===")
    print(f"Total Join Patterns: {len(discovery_engine.join_patterns)}")
    print(f"Tables with Joins: {len(discovery_engine.table_profiles)}")
    print(f"Multi-Path Relationships: {results['multi_path_count']}")
    
    print(f"\nTop 5 Most Frequent Join Patterns:")
    top_patterns = sorted(discovery_engine.join_patterns.values(), 
                         key=lambda x: x.frequency, reverse=True)[:5]
    for i, pattern in enumerate(top_patterns, 1):
        print(f"{i}. {pattern.pattern_key} - {pattern.frequency} times")
    
    return discovery_engine

# Example usage focusing on specific tables
if __name__ == "__main__":
    # Basic analysis
    engine = analyze_join_fields(
        csv_path='your_vsql.csv',
        sample_size=10000,
        llm_model='mistral-nemo:latest'
    )
    
    # To focus on specific tables (e.g., FIBER relationships)
    engine.visualize_join_patterns(
        output_path='./fiber_joins.png',
        focus_tables=['FIBER', 'FIBERSPLICE', 'FIBERCABLE', 'SIGNAL', 'EQUIPMENT']
    )