import pandas as pd
import re
import json
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx

# Import your LLM wrapper
from ollama_wrapper import answer_from_ollama

logger = logging.getLogger(__name__)

@dataclass
class DynamicIntegrityRule:
    """Represents a dynamically discovered integrity rule"""
    rule_id: str
    rule_name: str
    description: str
    affected_tables: List[str]
    check_pattern: str
    fix_pattern: Optional[str]
    example_check_sql: str
    example_fix_sql: Optional[str]
    severity: str  # 'critical', 'high', 'medium', 'low'
    frequency: int
    confidence: float
    business_impact: str

@dataclass
class TableProfile:
    """Profile of a discovered table based on SQL usage"""
    name: str
    columns: Set[str]
    primary_operations: Counter  # SELECT, INSERT, UPDATE, DELETE counts
    relationship_tables: Set[str]
    integrity_check_frequency: int
    integrity_fix_frequency: int
    common_join_columns: Dict[str, int]
    estimated_importance: float

class AdaptiveRuleGenerator:
    """Generates integrity rules based on discovered patterns"""
    
    def __init__(self, vsql_analyzer, llm_model: str = "qwen3:30b"):
        self.analyzer = vsql_analyzer
        self.llm_model = llm_model
        self.table_profiles = {}
        self.discovered_rules = []
        self.pattern_library = self._initialize_pattern_library()
        
    def _initialize_pattern_library(self) -> Dict[str, Any]:
        """Initialize library of known integrity patterns"""
        return {
            'orphan_patterns': {
                'check': [
                    r'SELECT.*FROM\s+(\w+).*NOT\s+IN.*SELECT.*FROM\s+(\w+)',
                    r'SELECT.*LEFT\s+JOIN.*WHERE.*IS\s+NULL',
                    r'SELECT.*NOT\s+EXISTS.*SELECT.*FROM'
                ],
                'fix': [
                    r'DELETE.*WHERE.*NOT\s+IN',
                    r'DELETE.*LEFT\s+JOIN.*WHERE.*IS\s+NULL'
                ],
                'description': 'Records referencing non-existent parent records'
            },
            'duplicate_patterns': {
                'check': [
                    r'SELECT.*COUNT\(\*\).*GROUP\s+BY.*HAVING\s+COUNT.*>\s*1',
                    r'SELECT.*ROW_NUMBER\(\).*OVER.*PARTITION\s+BY'
                ],
                'fix': [
                    r'DELETE.*WHERE.*ROWID.*NOT\s+IN.*MIN\(ROWID\)',
                    r'DELETE.*ROW_NUMBER.*>\s*1'
                ],
                'description': 'Duplicate records based on business keys'
            },
            'circular_patterns': {
                'check': [
                    r'WITH\s+RECURSIVE.*SELECT.*CONNECT\s+BY',
                    r'WITH.*AS.*SELECT.*UNION\s+ALL.*SELECT'
                ],
                'fix': [
                    r'UPDATE.*SET.*=\s*NULL.*WHERE',
                    r'DELETE.*WHERE.*IN.*WITH\s+RECURSIVE'
                ],
                'description': 'Circular references in hierarchical data'
            },
            'inconsistent_patterns': {
                'check': [
                    r'SELECT.*CASE\s+WHEN.*THEN.*ELSE.*END',
                    r'SELECT.*WHERE.*<>.*SELECT'
                ],
                'fix': [
                    r'UPDATE.*SET.*=.*CASE\s+WHEN',
                    r'MERGE.*WHEN\s+MATCHED.*UPDATE'
                ],
                'description': 'Data inconsistencies between related records'
            }
        }
    
    def profile_tables(self) -> Dict[str, TableProfile]:
        """Create profiles for all discovered tables"""
        logger.info("Profiling discovered tables...")
        
        # Analyze each SQL to build table profiles
        for _, row in self.analyzer.df.iterrows():
            sql = row['SQL_FULLTEXT']
            purpose = row.get('purpose', 'unknown')
            
            # Extract tables and their usage
            tables = self._extract_tables_from_sql(sql)
            operation = self._get_operation_type(sql)
            
            for table in tables:
                if table not in self.table_profiles:
                    self.table_profiles[table] = TableProfile(
                        name=table,
                        columns=set(),
                        primary_operations=Counter(),
                        relationship_tables=set(),
                        integrity_check_frequency=0,
                        integrity_fix_frequency=0,
                        common_join_columns=defaultdict(int),
                        estimated_importance=0.0
                    )
                
                profile = self.table_profiles[table]
                profile.primary_operations[operation] += 1
                
                if purpose == 'integrity_check':
                    profile.integrity_check_frequency += 1
                elif purpose == 'integrity_fix':
                    profile.integrity_fix_frequency += 1
                
                # Extract columns
                columns = self._extract_columns_for_table(sql, table)
                profile.columns.update(columns)
                
                # Extract relationships
                related_tables = self._extract_related_tables(sql, table)
                profile.relationship_tables.update(related_tables)
                
                # Extract join columns
                join_columns = self._extract_join_columns(sql, table)
                for col in join_columns:
                    profile.common_join_columns[col] += 1
        
        # Calculate importance scores
        self._calculate_table_importance()
        
        logger.info(f"Profiled {len(self.table_profiles)} tables")
        return self.table_profiles
    
    def _extract_tables_from_sql(self, sql: str) -> Set[str]:
        """Extract table names from SQL"""
        tables = set()
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
    
    def _get_operation_type(self, sql: str) -> str:
        """Get the primary operation type of SQL"""
        sql_upper = sql.upper().strip()
        for op in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE']:
            if sql_upper.startswith(op):
                return op
        return 'OTHER'
    
    def _extract_columns_for_table(self, sql: str, table: str) -> Set[str]:
        """Extract columns for a specific table"""
        columns = set()
        
        # Pattern for table.column
        pattern = rf'{table}\.(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        columns.update(match.upper() for match in matches)
        
        return columns
    
    def _extract_related_tables(self, sql: str, target_table: str) -> Set[str]:
        """Extract tables related to target table through JOINs"""
        related = set()
        
        # Look for JOIN patterns involving target table
        join_pattern = rf'{target_table}\s+\w*\s*JOIN\s+(\w+)|(\w+)\s+\w*\s*JOIN\s+{target_table}'
        matches = re.findall(join_pattern, sql, re.IGNORECASE)
        
        for match_tuple in matches:
            for match in match_tuple:
                if match:
                    related.add(match.upper())
        
        return related
    
    def _extract_join_columns(self, sql: str, table: str) -> List[str]:
        """Extract columns used in JOIN conditions for a table"""
        columns = []
        
        # Pattern for join conditions
        pattern = rf'{table}\.(\w+)\s*=\s*\w+\.\w+|\w+\.\w+\s*=\s*{table}\.(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        
        for match_tuple in matches:
            for match in match_tuple:
                if match:
                    columns.append(match.upper())
        
        return columns
    
    def _calculate_table_importance(self) -> None:
        """Calculate importance score for each table"""
        if not self.table_profiles:
            return
        
        # Factors for importance
        max_usage = max(sum(p.primary_operations.values()) for p in self.table_profiles.values())
        max_integrity = max(p.integrity_check_frequency + p.integrity_fix_frequency 
                          for p in self.table_profiles.values())
        max_relationships = max(len(p.relationship_tables) for p in self.table_profiles.values())
        
        for profile in self.table_profiles.values():
            usage_score = sum(profile.primary_operations.values()) / max(max_usage, 1)
            integrity_score = (profile.integrity_check_frequency + profile.integrity_fix_frequency) / max(max_integrity, 1)
            relationship_score = len(profile.relationship_tables) / max(max_relationships, 1)
            
            # Weighted importance
            profile.estimated_importance = (
                0.3 * usage_score + 
                0.4 * integrity_score + 
                0.3 * relationship_score
            )
    
    def discover_integrity_rules(self) -> List[DynamicIntegrityRule]:
        """Discover integrity rules from SQL patterns"""
        logger.info("Discovering integrity rules from patterns...")
        
        # First, profile tables
        self.profile_tables()
        
        # Analyze integrity sequences for rule patterns
        for sequence in self.analyzer.integrity_sequences:
            rule = self._create_rule_from_sequence(sequence)
            if rule:
                self.discovered_rules.append(rule)
        
        # Use LLM to understand complex patterns
        self._enhance_rules_with_llm()
        
        # Prioritize rules
        self._prioritize_rules()
        
        logger.info(f"Discovered {len(self.discovered_rules)} integrity rules")
        return self.discovered_rules
    
    def _create_rule_from_sequence(self, sequence) -> Optional[DynamicIntegrityRule]:
        """Create an integrity rule from a check-fix sequence"""
        # Determine rule category
        rule_category = self._categorize_sequence(sequence)
        
        if not rule_category:
            return None
        
        # Generate rule ID
        tables_str = '_'.join(sorted(sequence.tables_involved)[:2])
        rule_id = f"RULE_{rule_category}_{tables_str}_{sequence.sequence_id[:8]}"
        
        # Calculate severity based on table importance
        severity = self._calculate_severity(sequence.tables_involved)
        
        return DynamicIntegrityRule(
            rule_id=rule_id,
            rule_name=f"{sequence.issue_type.replace('_', ' ').title()} Rule",
            description=f"Detects and fixes {sequence.issue_type} in {', '.join(sequence.tables_involved)}",
            affected_tables=list(sequence.tables_involved),
            check_pattern=sequence.check_pattern,
            fix_pattern=sequence.fix_pattern,
            example_check_sql=sequence.check_sql[:500],  # Truncate for readability
            example_fix_sql=sequence.fix_sql[:500] if sequence.fix_sql else None,
            severity=severity,
            frequency=sequence.occurrence_count,
            confidence=sequence.confidence,
            business_impact=self._assess_business_impact(sequence)
        )
    
    def _categorize_sequence(self, sequence) -> Optional[str]:
        """Categorize a sequence into known pattern types"""
        check_sql_upper = sequence.check_sql.upper()
        
        for category, patterns in self.pattern_library.items():
            for check_pattern in patterns['check']:
                if re.search(check_pattern, check_sql_upper):
                    return category.replace('_patterns', '')
        
        return 'custom'
    
    def _calculate_severity(self, tables: Set[str]) -> str:
        """Calculate severity based on table importance"""
        if not tables:
            return 'low'
        
        # Get average importance of involved tables
        importances = [
            self.table_profiles[table].estimated_importance 
            for table in tables 
            if table in self.table_profiles
        ]
        
        if not importances:
            return 'medium'
        
        avg_importance = np.mean(importances)
        
        if avg_importance > 0.7:
            return 'critical'
        elif avg_importance > 0.5:
            return 'high'
        elif avg_importance > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _assess_business_impact(self, sequence) -> str:
        """Assess the business impact of an integrity issue"""
        # Simple heuristic based on issue type and tables
        impact_keywords = {
            'critical': ['CUSTOMER', 'ORDER', 'PAYMENT', 'ACCOUNT', 'SIGNAL'],
            'high': ['FIBER', 'SPLICE', 'EQUIPMENT', 'PORT', 'CABLE'],
            'medium': ['SEGMENT', 'SPAN', 'ROUTE', 'PATH'],
            'low': ['LOG', 'TEMP', 'BACKUP', 'ARCHIVE']
        }
        
        tables_upper = [t.upper() for t in sequence.tables_involved]
        
        for impact_level, keywords in impact_keywords.items():
            if any(keyword in table for keyword in keywords for table in tables_upper):
                return f"{impact_level.title()} - Affects {sequence.issue_type.replace('_', ' ')}"
        
        return "Medium - General data quality impact"
    
    def _enhance_rules_with_llm(self) -> None:
        """Use LLM to enhance understanding of complex rules"""
        logger.info("Enhancing rules with LLM analysis...")
        
        # Group rules by pattern for batch analysis
        pattern_groups = defaultdict(list)
        for rule in self.discovered_rules:
            pattern_groups[rule.check_pattern].append(rule)
        
        # Analyze top patterns with LLM
        for pattern, rules in sorted(pattern_groups.items(), 
                                   key=lambda x: sum(r.frequency for r in x[1]), 
                                   reverse=True)[:10]:
            
            sample_rules = rules[:3]
            prompt = f"""Analyze these data integrity rules from a Fiber Management System:

Pattern Type: {pattern}
Affected Tables: {', '.join(set(t for r in sample_rules for t in r.affected_tables))}

Example Check SQL:
{sample_rules[0].example_check_sql}

Example Fix SQL:
{sample_rules[0].example_fix_sql if sample_rules[0].example_fix_sql else 'No fix SQL available'}

Based on this pattern:
1. What specific integrity issue is being addressed?
2. What could cause this issue in a fiber management context?
3. What are the business implications if not fixed?
4. Suggest preventive measures

Respond in JSON format with keys: issue_description, root_causes, business_impact, prevention_measures
"""
            
            try:
                response = answer_from_ollama(prompt, self.llm_model)
                insights = json.loads(response)
                
                # Update rules with insights
                for rule in rules:
                    if 'issue_description' in insights:
                        rule.description = insights['issue_description']
                    if 'business_impact' in insights:
                        rule.business_impact = insights['business_impact']
                        
            except Exception as e:
                logger.debug(f"LLM enhancement failed for pattern {pattern}: {e}")
    
    def _prioritize_rules(self) -> None:
        """Prioritize rules based on multiple factors"""
        # Sort rules by composite score
        for rule in self.discovered_rules:
            severity_score = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[rule.severity]
            frequency_score = min(rule.frequency / 10, 5)  # Normalize to 0-5
            confidence_score = rule.confidence * 5
            
            rule.priority_score = severity_score + frequency_score + confidence_score
        
        self.discovered_rules.sort(key=lambda r: r.priority_score, reverse=True)
    
    def generate_remediation_plan(self) -> Dict[str, Any]:
        """Generate a prioritized remediation plan"""
        plan = {
            'summary': {
                'total_rules': len(self.discovered_rules),
                'critical_issues': sum(1 for r in self.discovered_rules if r.severity == 'critical'),
                'estimated_effort_days': self._estimate_remediation_effort()
            },
            'phases': [],
            'quick_wins': [],
            'preventive_measures': []
        }
        
        # Phase 1: Critical issues
        critical_rules = [r for r in self.discovered_rules if r.severity == 'critical']
        if critical_rules:
            plan['phases'].append({
                'phase': 1,
                'name': 'Critical Issue Remediation',
                'duration_days': len(critical_rules) * 2,
                'rules': [self._rule_to_dict(r) for r in critical_rules[:10]]
            })
        
        # Phase 2: High frequency issues
        high_freq_rules = [r for r in self.discovered_rules 
                          if r.severity in ['high', 'medium'] and r.frequency > 10]
        if high_freq_rules:
            plan['phases'].append({
                'phase': 2,
                'name': 'High Frequency Issue Resolution',
                'duration_days': len(high_freq_rules) * 1.5,
                'rules': [self._rule_to_dict(r) for r in high_freq_rules[:15]]
            })
        
        # Quick wins - high confidence, low effort fixes
        quick_wins = [r for r in self.discovered_rules 
                     if r.confidence > 0.8 and r.fix_pattern and r.severity in ['medium', 'low']]
        plan['quick_wins'] = [self._rule_to_dict(r) for r in quick_wins[:10]]
        
        # Generate preventive measures using LLM
        plan['preventive_measures'] = self._generate_preventive_measures()
        
        return plan
    
    def _estimate_remediation_effort(self) -> int:
        """Estimate total remediation effort in days"""
        effort_map = {
            'critical': 2,
            'high': 1.5,
            'medium': 1,
            'low': 0.5
        }
        
        total_days = sum(effort_map.get(rule.severity, 1) for rule in self.discovered_rules)
        return int(total_days)
    
    def _rule_to_dict(self, rule: DynamicIntegrityRule) -> Dict[str, Any]:
        """Convert rule to dictionary for reporting"""
        return {
            'rule_id': rule.rule_id,
            'name': rule.rule_name,
            'description': rule.description,
            'severity': rule.severity,
            'tables': rule.affected_tables,
            'frequency': rule.frequency,
            'has_automated_fix': bool(rule.fix_pattern),
            'business_impact': rule.business_impact
        }
    
    def _generate_preventive_measures(self) -> List[str]:
        """Generate preventive measures based on discovered patterns"""
        # Analyze common issue types
        issue_types = Counter(r.check_pattern for r in self.discovered_rules)
        top_issues = issue_types.most_common(5)
        
        prompt = f"""Based on these common data integrity issues in a Fiber Management System:

{chr(10).join(f"- {issue}: {count} occurrences" for issue, count in top_issues)}

Suggest 5 preventive measures that could be implemented to avoid these issues in the future.
Focus on:
1. Database constraints
2. Application validation
3. Process improvements
4. Monitoring recommendations

Provide practical, implementable suggestions.
Return as a JSON array of strings.
"""
        
        try:
            response = answer_from_ollama(prompt, self.llm_model)
            measures = json.loads(response)
            return measures if isinstance(measures, list) else []
        except:
            # Fallback to generic measures
            return [
                "Implement foreign key constraints on all reference columns",
                "Add unique constraints on business key combinations",
                "Create database triggers to validate data on insert/update",
                "Implement regular integrity check jobs with alerting",
                "Add application-level validation before database operations"
            ]

class IntegrityMonitor:
    """Monitors and tracks integrity issues over time"""
    
    def __init__(self, rule_generator: AdaptiveRuleGenerator):
        self.rule_generator = rule_generator
        self.monitoring_data = defaultdict(list)
    
    def generate_monitoring_dashboard(self, output_dir: str) -> Dict[str, Any]:
        """Generate monitoring dashboard data"""
        dashboard = {
            'timestamp': str(datetime.now()),
            'overview': self._generate_overview(),
            'trending_issues': self._analyze_trends(),
            'table_health_scores': self._calculate_table_health(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save dashboard data
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'monitoring_dashboard.json'), 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        return dashboard
    
    def _generate_overview(self) -> Dict[str, Any]:
        """Generate overview statistics"""
        rules = self.rule_generator.discovered_rules
        
        return {
            'total_integrity_rules': len(rules),
            'severity_distribution': dict(Counter(r.severity for r in rules)),
            'tables_with_issues': len(set(t for r in rules for t in r.affected_tables)),
            'automated_fix_available': sum(1 for r in rules if r.fix_pattern),
            'total_issue_occurrences': sum(r.frequency for r in rules)
        }
    
    def _analyze_trends(self) -> List[Dict[str, Any]]:
        """Analyze trending integrity issues"""
        # Group by time periods in the original data
        df = self.rule_generator.analyzer.df
        integrity_df = df[df['purpose'].isin(['integrity_check', 'integrity_fix'])]
        
        if integrity_df.empty:
            return []
        
        # Weekly trends
        integrity_df['week'] = integrity_df['LAST_LOAD_TIME'].dt.to_period('W')
        weekly_counts = integrity_df.groupby(['week', 'sub_type']).size()
        
        trends = []
        for (week, sub_type), count in weekly_counts.items():
            trends.append({
                'period': str(week),
                'issue_type': sub_type,
                'count': int(count),
                'trend': 'increasing' if count > weekly_counts.mean() else 'stable'
            })
        
        return sorted(trends, key=lambda x: x['count'], reverse=True)[:10]
    
    def _calculate_table_health(self) -> Dict[str, float]:
        """Calculate health scores for tables"""
        health_scores = {}
        
        for table_name, profile in self.rule_generator.table_profiles.items():
            # Health factors (0-1, higher is healthier)
            integrity_ratio = 1 - min(
                (profile.integrity_check_frequency + profile.integrity_fix_frequency) / 
                max(sum(profile.primary_operations.values()), 1), 
                1
            )
            
            # Tables with fewer integrity issues are healthier
            issues_count = sum(1 for r in self.rule_generator.discovered_rules 
                             if table_name in r.affected_tables)
            issue_score = 1 - min(issues_count / 10, 1)
            
            # Combined health score
            health_scores[table_name] = round((integrity_ratio + issue_score) / 2, 2)
        
        # Return top 20 tables by importance
        important_tables = sorted(
            health_scores.items(),
            key=lambda x: self.rule_generator.table_profiles[x[0]].estimated_importance,
            reverse=True
        )[:20]
        
        return dict(important_tables)
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze patterns for recommendations
        rules = self.rule_generator.discovered_rules
        
        # Tables with most critical issues
        critical_tables = defaultdict(int)
        for rule in rules:
            if rule.severity == 'critical':
                for table in rule.affected_tables:
                    critical_tables[table] += 1
        
        if critical_tables:
            top_critical = sorted(critical_tables.items(), key=lambda x: x[1], reverse=True)[0]
            recommendations.append({
                'priority': 'high',
                'category': 'critical_tables',
                'recommendation': f"Focus immediate attention on {top_critical[0]} table with {top_critical[1]} critical issues",
                'action': f"Review and execute all critical fixes for {top_critical[0]} table"
            })
        
        # Patterns without automated fixes
        manual_fixes = [r for r in rules if not r.fix_pattern and r.frequency > 5]
        if manual_fixes:
            recommendations.append({
                'priority': 'medium',
                'category': 'automation',
                'recommendation': f"Develop automated fixes for {len(manual_fixes)} recurring manual patterns",
                'action': "Create SQL templates or procedures for common manual fixes"
            })
        
        # High frequency low severity issues (quick wins)
        quick_wins = [r for r in rules if r.severity == 'low' and r.frequency > 10]
        if quick_wins:
            recommendations.append({
                'priority': 'low',
                'category': 'quick_wins',
                'recommendation': f"Address {len(quick_wins)} high-frequency minor issues for quick improvements",
                'action': "Schedule batch execution of low-risk automated fixes"
            })
        
        return recommendations

# Integrated execution function
def analyze_fiber_integrity_complete(csv_path: str,
                                   llm_model: str = "qwen3:30b",
                                   output_dir: str = './integrity_analysis') -> Dict[str, Any]:
    """Complete integrity analysis with adaptive rule generation"""
    
    # Import the main analyzer
    from vsql_integrity_analyzer import analyze_vsql_integrity, VSQLAnalyzer
    
    # Run initial analysis
    initial_results = analyze_vsql_integrity(csv_path, llm_model, output_dir)
    
    # Load analyzer for advanced analysis
    analyzer = VSQLAnalyzer(csv_path, llm_model)
    analyzer.load_vsql_file()
    analyzer.discover_integrity_sequences()
    
    # Generate adaptive rules
    rule_generator = AdaptiveRuleGenerator(analyzer, llm_model)
    discovered_rules = rule_generator.discover_integrity_rules()
    
    # Generate remediation plan
    remediation_plan = rule_generator.generate_remediation_plan()
    
    # Create monitoring dashboard
    monitor = IntegrityMonitor(rule_generator)
    dashboard = monitor.generate_monitoring_dashboard(output_dir)
    
    # Save comprehensive results
    comprehensive_results = {
        'analysis_summary': initial_results['summary'],
        'discovered_rules': [
            {
                'rule_id': r.rule_id,
                'name': r.rule_name,
                'description': r.description,
                'severity': r.severity,
                'frequency': r.frequency,
                'tables': r.affected_tables,
                'automated_fix': bool(r.fix_pattern)
            }
            for r in discovered_rules[:50]  # Top 50 rules
        ],
        'remediation_plan': remediation_plan,
        'monitoring_dashboard': dashboard,
        'table_profiles': {
            name: {
                'importance': profile.estimated_importance,
                'integrity_issues': profile.integrity_check_frequency + profile.integrity_fix_frequency,
                'relationships': len(profile.relationship_tables)
            }
            for name, profile in sorted(
                rule_generator.table_profiles.items(),
                key=lambda x: x[1].estimated_importance,
                reverse=True
            )[:20]  # Top 20 tables
        }
    }
    
    # Save final comprehensive report
    import os
    with open(os.path.join(output_dir, 'comprehensive_integrity_report.json'), 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Generate executive summary
    exec_summary = f"""
=== Fiber Database Integrity Analysis Executive Summary ===

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Total SQL Statements Analyzed: {len(analyzer.df):,}

Key Findings:
- Discovered {len(discovered_rules)} integrity patterns
- {remediation_plan['summary']['critical_issues']} critical issues requiring immediate attention
- {len([r for r in discovered_rules if r.fix_pattern])} issues have automated fixes available
- Estimated {remediation_plan['summary']['estimated_effort_days']} days for complete remediation

Top 3 Critical Issues:
"""
    
    for rule in discovered_rules[:3]:
        if rule.severity == 'critical':
            exec_summary += f"\n- {rule.rule_name}: {rule.description}"
            exec_summary += f"\n  Affects: {', '.join(rule.affected_tables)}"
            exec_summary += f"\n  Frequency: {rule.frequency} occurrences\n"
    
    exec_summary += f"\nFull report available at: {output_dir}"
    
    print(exec_summary)
    
    return comprehensive_results

# Example usage
if __name__ == "__main__":
    results = analyze_fiber_integrity_complete(
        csv_path='path/to/your/vsql_export.csv',
        llm_model='qwen3:30b',
        output_dir='./fiber_integrity_complete'
    )
