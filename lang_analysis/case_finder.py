#!/usr/bin/env python3
"""
Per-instance qualitative comparison of agent trajectories across planning configurations and models.

Performs sensitivity analysis to identify instances where planning choices materially affect outcomes.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import re

import numpy as np
import pandas as pd
from scipy import stats


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('planning_sensitivity_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Fixed configuration lists
PLAN_CONFIGS = [
    "no_plan",
    "no_reproduce_and_verification",
    "no_verification",
    "plan",
    "plan_and_regression"
]

MODELS = [
    "deepseek_r1",
    "deepseek_v3",
    "gpt5_mini"
]


def normalize_step(step: str) -> str:
    """
    Strip numerical suffix from langutory step.
    
    Example: 'L_navigate_3' -> 'L_navigate'
    
    This normalization allows comparison of step types across trajectories
    without being sensitive to repetition counts.
    """
    return re.sub(r'_\d+$', '', step)


def normalize_langutory(sequence: List[str]) -> List[str]:
    """
    Normalize a langutory sequence by stripping numerical suffixes.
    
    Args:
        sequence: Raw langutory sequence with suffixes
    
    Returns:
        Normalized sequence without numerical suffixes
    """
    return [normalize_step(step) for step in sequence]


def compute_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute Levenshtein edit distance between two sequences.
    
    Measures minimum number of insertions, deletions, and substitutions
    needed to transform seq1 into seq2. Higher values indicate more divergent trajectories.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute length of Longest Common Subsequence.
    
    Measures shared structure between trajectories. Higher values indicate
    more similar execution patterns even if exact sequences differ.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def compare_langutories(sequences: List[List[str]]) -> Dict:
    """
    Compare multiple langutory sequences, computing similarity metrics.
    
    Args:
        sequences: List of normalized langutory sequences
    
    Returns:
        Dictionary containing:
        - identical: whether all sequences are identical
        - max_edit_distance: maximum pairwise edit distance
        - min_lcs_length: minimum pairwise LCS length
        - normalized_edit_distances: list of normalized edit distances
    """
    if not sequences:
        return {
            'identical': True,
            'max_edit_distance': 0,
            'min_lcs_length': 0,
            'normalized_edit_distances': []
        }
    
    if len(sequences) == 1:
        return {
            'identical': True,
            'max_edit_distance': 0,
            'min_lcs_length': len(sequences[0]),
            'normalized_edit_distances': [0.0]
        }
    
    # Check exact identity
    identical = all(seq == sequences[0] for seq in sequences[1:])
    
    # Compute pairwise metrics
    edit_distances = []
    lcs_lengths = []
    normalized_edit_distances = []
    
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1, seq2 = sequences[i], sequences[j]
            
            edit_dist = compute_edit_distance(seq1, seq2)
            edit_distances.append(edit_dist)
            
            lcs_len = compute_lcs_length(seq1, seq2)
            lcs_lengths.append(lcs_len)
            
            # Normalize edit distance by maximum possible length
            max_len = max(len(seq1), len(seq2))
            norm_edit = edit_dist / max_len if max_len > 0 else 0.0
            normalized_edit_distances.append(norm_edit)
    
    return {
        'identical': identical,
        'max_edit_distance': max(edit_distances) if edit_distances else 0,
        'min_lcs_length': min(lcs_lengths) if lcs_lengths else 0,
        'normalized_edit_distances': normalized_edit_distances
    }


def load_languatory_file(filepath: Path) -> Optional[List[Dict]]:
    """
    Load and parse a languatory.json file.
    
    Args:
        filepath: Path to languatory.json
    
    Returns:
        List of instance records or None if file cannot be loaded
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} instances from {filepath}")
            return data
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {e}")
        return None


def load_all_languatories(base_dir: Path, max_cases: Optional[int] = None) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Load all languatory.json files across plan configurations and models.
    
    Args:
        base_dir: Base directory (e.g., artifacts/SWE-agent/)
        max_cases: Maximum number of instances to load per file (None for all)
    
    Returns:
        Dictionary mapping (plan, model) tuples to list of instance records
    """
    all_data = {}
    
    for plan in PLAN_CONFIGS:
        for model in MODELS:
            lang_file = base_dir / plan / model / "lang" / "languatory.json"
            
            if not lang_file.parent.exists():
                logger.warning(f"Missing lang/ directory: {lang_file.parent}")
                continue
            
            if not lang_file.exists():
                logger.warning(f"Missing languatory.json: {lang_file}")
                continue
            
            data = load_languatory_file(lang_file)
            if data is not None:
                # Limit to max_cases if specified
                if max_cases is not None and len(data) > max_cases:
                    logger.info(f"Limiting {lang_file} from {len(data)} to {max_cases} instances")
                    data = data[:max_cases]
                all_data[(plan, model)] = data
    
    logger.info(f"Loaded data from {len(all_data)} plan-model combinations")
    return all_data


def group_by_instance_id(records: Dict[Tuple[str, str], List[Dict]], 
                         max_cases: Optional[int] = None) -> Dict[str, Dict[Tuple[str, str], Dict]]:
    """
    Reorganize records by instance_id for cross-configuration comparison.
    
    Args:
        records: Map from (plan, model) to list of instance records
        max_cases: Maximum number of unique instances to include (None for all)
    
    Returns:
        Map from instance_id to map from (plan, model) to instance record
        
    This grouping enables per-instance sensitivity analysis across configurations.
    """
    by_instance = defaultdict(dict)
    
    for (plan, model), instances in records.items():
        for instance in instances:
            instance_id = instance.get('instance_id')
            if instance_id:
                by_instance[instance_id][(plan, model)] = instance
    
    # Limit to max_cases unique instances if specified
    if max_cases is not None and len(by_instance) > max_cases:
        logger.info(f"Limiting analysis from {len(by_instance)} to {max_cases} unique instances")
        # Keep first max_cases instances (deterministic order)
        limited_instances = dict(list(by_instance.items())[:max_cases])
        logger.info(f"Selected instances: {list(limited_instances.keys())}")
        by_instance = limited_instances
    
    logger.info(f"Grouped data into {len(by_instance)} unique instances for analysis")
    return dict(by_instance)


def analyze_resolution_variance(variants: Dict[Tuple[str, str], Dict]) -> Dict:
    """
    Analyze resolution status variance across planning configurations.
    
    Args:
        variants: Map from (plan, model) to instance record
    
    Returns:
        Dictionary containing:
        - resolution_sensitive: whether resolution varies across configs
        - resolved_variants: set of (plan, model) that resolved
        - unresolved_variants: set of (plan, model) that failed
        - resolution_statuses: all unique resolution statuses seen
        - plan_resolutions: dict mapping plan to resolution status
        
    Resolution sensitivity indicates planning choice impacts success/failure,
    which is critical for understanding when planning matters.
    """
    resolution_statuses = set()
    resolved_variants = set()
    unresolved_variants = set()
    plan_resolutions = {}
    
    for (plan, model), record in variants.items():
        status = record.get('resolution_status', 'unknown')
        resolution_statuses.add(status)
        
        if status == 'resolved':
            resolved_variants.add((plan, model))
        else:
            unresolved_variants.add((plan, model))
        
        # Track resolution by plan (aggregate across models for this plan)
        if plan not in plan_resolutions:
            plan_resolutions[plan] = []
        plan_resolutions[plan].append(status)
    
    # Resolution is sensitive if outcomes differ across configurations
    resolution_sensitive = len(resolution_statuses) > 1
    
    return {
        'resolution_sensitive': resolution_sensitive,
        'resolved_variants': resolved_variants,
        'unresolved_variants': unresolved_variants,
        'resolution_statuses': resolution_statuses,
        'plan_resolutions': plan_resolutions
    }


def analyze_instance(instance_id: str, variants: Dict[Tuple[str, str], Dict]) -> Dict:
    """
    Perform comprehensive sensitivity analysis for a single instance.
    
    Args:
        instance_id: Instance identifier
        variants: Map from (plan, model) to instance record
    
    Returns:
        Dictionary of analysis results including trajectory similarity,
        step-count variance, and resolution sensitivity metrics
    """
    # Extract langutories and normalize
    langutories = {}
    raw_langutories = {}
    step_counts = []
    
    for (plan, model), record in variants.items():
        raw_lang = record.get('languatory', [])
        norm_lang = normalize_langutory(raw_lang)
        
        langutories[(plan, model)] = norm_lang
        raw_langutories[(plan, model)] = raw_lang
        step_counts.append(len(norm_lang))
    
    # Trajectory similarity analysis
    sequences = list(langutories.values())
    similarity = compare_langutories(sequences)
    
    # Step-count statistics
    mean_steps = np.mean(step_counts) if step_counts else 0.0
    std_steps = np.std(step_counts) if len(step_counts) > 1 else 0.0
    max_step_diff = max(step_counts) - min(step_counts) if step_counts else 0
    
    # Resolution analysis
    resolution_info = analyze_resolution_variance(variants)
    
    # Debug difficulty (assume consistent across variants)
    debug_difficulty = next(iter(variants.values())).get('debug_difficulty', 'unknown')
    
    # Create per-plan resolution columns
    plan_resolution_columns = {}
    for plan in PLAN_CONFIGS:
        # Check if this instance has any data for this plan across any model
        plan_statuses = []
        for (p, m), record in variants.items():
            if p == plan:
                plan_statuses.append(record.get('resolution_status', 'unknown'))
        
        if plan_statuses:
            # Use majority vote if multiple models, or single value
            resolved_count = sum(1 for s in plan_statuses if s == 'resolved')
            plan_resolution_columns[f'plan_{plan}_resolved'] = resolved_count > 0
            plan_resolution_columns[f'plan_{plan}_status'] = 'resolved' if resolved_count > 0 else plan_statuses[0]
        else:
            plan_resolution_columns[f'plan_{plan}_resolved'] = None
            plan_resolution_columns[f'plan_{plan}_status'] = None
    
    return {
        'instance_id': instance_id,
        'num_plan_model_variants': len(variants),
        'identical_langutories': similarity['identical'],
        'max_edit_distance': similarity['max_edit_distance'],
        'min_lcs_length': similarity['min_lcs_length'],
        'mean_normalized_edit_distance': np.mean(similarity['normalized_edit_distances']) if similarity['normalized_edit_distances'] else 0.0,
        'mean_step_count': mean_steps,
        'std_step_count': std_steps,
        'max_step_diff': max_step_diff,
        'resolution_sensitive': resolution_info['resolution_sensitive'],
        'resolved_variants': len(resolution_info['resolved_variants']),
        'unresolved_variants': len(resolution_info['unresolved_variants']),
        'debug_difficulty': debug_difficulty,
        'variants': variants,  # Keep for detailed inspection if needed
        **plan_resolution_columns  # Add per-plan resolution columns
    }


def compute_plan_predictiveness(instance_analyses: List[Dict], 
                                resolution_rate_by_plan: Dict[str, float]) -> Dict:
    """
    Analyze how well plan-level resolution rates predict instance-level outcomes.
    
    Tests hypothesis: Do instances with high unresolved rates consistently fail 
    with the lowest resolution_rate_by_plan plans?
    
    Args:
        instance_analyses: List of per-instance analysis results
        resolution_rate_by_plan: Overall resolution rate for each plan
    
    Returns:
        Dictionary containing statistical measures of predictiveness
    """
    # Sort plans by resolution rate (lowest to highest)
    sorted_plans = sorted(resolution_rate_by_plan.items(), key=lambda x: x[1])
    
    # For each instance, check if lower-performing plans fail more often
    instance_predictions = []
    
    for analysis in instance_analyses:
        instance_id = analysis['instance_id']
        
        # Collect (plan_resolution_rate, instance_resolved) pairs
        pairs = []
        for plan, plan_rate in resolution_rate_by_plan.items():
            status_key = f'plan_{plan}_status'
            if status_key in analysis and analysis[status_key] is not None:
                is_resolved = analysis[status_key] == 'resolved'
                pairs.append((plan_rate, int(is_resolved), plan))
        
        if len(pairs) >= 2:  # Need at least 2 plans to measure correlation
            instance_predictions.append({
                'instance_id': instance_id,
                'pairs': pairs
            })
    
    # Compute correlations
    all_plan_rates = []
    all_resolutions = []
    
    for pred in instance_predictions:
        for plan_rate, is_resolved, plan in pred['pairs']:
            all_plan_rates.append(plan_rate)
            all_resolutions.append(is_resolved)
    
    correlation_results = {}
    
    if len(all_plan_rates) >= 3:  # Need enough data for meaningful correlation
        # Pearson correlation: measures linear relationship
        pearson_corr, pearson_pval = stats.pearsonr(all_plan_rates, all_resolutions)
        
        # Spearman correlation: measures monotonic relationship (rank-based)
        spearman_corr, spearman_pval = stats.spearmanr(all_plan_rates, all_resolutions)
        
        correlation_results = {
            'pearson_correlation': pearson_corr,
            'pearson_pvalue': pearson_pval,
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_pval,
            'n_datapoints': len(all_plan_rates)
        }
        
        # Interpretation: positive correlation means higher plan resolution rate 
        # predicts higher instance resolution likelihood
        logger.info(f"Plan predictiveness - Pearson r={pearson_corr:.3f} (p={pearson_pval:.4f})")
        logger.info(f"Plan predictiveness - Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.4f})")
    else:
        logger.warning("Insufficient data for correlation analysis")
        correlation_results = {
            'pearson_correlation': None,
            'pearson_pvalue': None,
            'spearman_correlation': None,
            'spearman_pvalue': None,
            'n_datapoints': len(all_plan_rates)
        }
    
    # Analyze per-instance consistency
    per_instance_consistency = []
    
    for pred in instance_predictions:
        pairs = pred['pairs']
        # Sort by plan rate
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        
        # Check if resolution follows plan quality ordering
        resolutions = [p[1] for p in sorted_pairs]
        
        # Simple consistency check: is there monotonic improvement?
        is_monotonic = all(resolutions[i] <= resolutions[i+1] for i in range(len(resolutions)-1))
        
        per_instance_consistency.append({
            'instance_id': pred['instance_id'],
            'is_monotonic': is_monotonic,
            'resolutions': resolutions,
            'plan_rates': [p[0] for p in sorted_pairs]
        })
    
    monotonic_count = sum(1 for p in per_instance_consistency if p['is_monotonic'])
    monotonic_fraction = monotonic_count / len(per_instance_consistency) if per_instance_consistency else 0.0
    
    correlation_results['per_instance_monotonic_fraction'] = monotonic_fraction
    correlation_results['per_instance_monotonic_count'] = monotonic_count
    correlation_results['per_instance_total'] = len(per_instance_consistency)
    
    return correlation_results


def compute_aggregate_statistics(instance_analyses: List[Dict], 
                                 all_records: Dict[Tuple[str, str], List[Dict]]) -> Dict:
    """
    Compute aggregate statistics across all instances and configurations.
    
    Args:
        instance_analyses: List of per-instance analysis results
        all_records: Raw records by (plan, model)
    
    Returns:
        Dictionary of aggregate metrics including resolution rates,
        planning impact frequency, and variance distributions
    """
    # Resolution rate by plan
    resolution_by_plan = defaultdict(lambda: {'resolved': 0, 'total': 0})
    for (plan, model), instances in all_records.items():
        for instance in instances:
            status = instance.get('resolution_status', 'unknown')
            resolution_by_plan[plan]['total'] += 1
            if status == 'resolved':
                resolution_by_plan[plan]['resolved'] += 1
    
    resolution_rate_by_plan = {
        plan: stats['resolved'] / stats['total'] if stats['total'] > 0 else 0.0
        for plan, stats in resolution_by_plan.items()
    }
    
    # Resolution rate by model
    resolution_by_model = defaultdict(lambda: {'resolved': 0, 'total': 0})
    for (plan, model), instances in all_records.items():
        for instance in instances:
            status = instance.get('resolution_status', 'unknown')
            resolution_by_model[model]['total'] += 1
            if status == 'resolved':
                resolution_by_model[model]['resolved'] += 1
    
    resolution_rate_by_model = {
        model: stats['resolved'] / stats['total'] if stats['total'] > 0 else 0.0
        for model, stats in resolution_by_model.items()
    }
    
    # Planning impact metrics
    total_instances = len(instance_analyses)
    resolution_sensitive_count = sum(1 for a in instance_analyses if a['resolution_sensitive'])
    
    # Step-count variance distribution
    step_variances = [a['std_step_count'] for a in instance_analyses if a['std_step_count'] > 0]
    
    # Compute plan predictiveness
    plan_predictiveness = compute_plan_predictiveness(instance_analyses, resolution_rate_by_plan)
    
    return {
        'total_instances': total_instances,
        'resolution_rate_by_plan': resolution_rate_by_plan,
        'resolution_rate_by_model': resolution_rate_by_model,
        'resolution_sensitive_count': resolution_sensitive_count,
        'resolution_sensitive_fraction': resolution_sensitive_count / total_instances if total_instances > 0 else 0.0,
        'step_variance_mean': np.mean(step_variances) if step_variances else 0.0,
        'step_variance_median': np.median(step_variances) if step_variances else 0.0,
        'step_variance_max': np.max(step_variances) if step_variances else 0.0,
        'plan_predictiveness': plan_predictiveness
    }


def write_per_instance_summary(instance_analyses: List[Dict], output_path: Path) -> None:
    """
    Write per-instance summary table to CSV.
    
    Args:
        instance_analyses: List of per-instance analysis results
        output_path: Path to output CSV file
    """
    # Select columns for CSV output
    rows = []
    for analysis in instance_analyses:
        row = {
            'instance_id': analysis['instance_id'],
            'num_plan_model_variants': analysis['num_plan_model_variants'],
            'identical_langutories': analysis['identical_langutories'],
            'mean_step_count': analysis['mean_step_count'],
            'std_step_count': analysis['std_step_count'],
            'max_step_diff': analysis['max_step_diff'],
            'max_edit_distance': analysis['max_edit_distance'],
            'min_lcs_length': analysis['min_lcs_length'],
            'mean_normalized_edit_distance': analysis['mean_normalized_edit_distance'],
            'resolution_sensitive': analysis['resolution_sensitive'],
            'resolved_variants': analysis['resolved_variants'],
            'unresolved_variants': analysis['unresolved_variants'],
            'debug_difficulty': analysis['debug_difficulty']
        }
        
        # Add per-plan resolution columns
        for plan in PLAN_CONFIGS:
            row[f'plan_{plan}_resolved'] = analysis.get(f'plan_{plan}_resolved')
            row[f'plan_{plan}_status'] = analysis.get(f'plan_{plan}_status')
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote per-instance summary to {output_path}")


def write_aggregate_summary(aggregate_stats: Dict, output_path: Path) -> None:
    """
    Write aggregate statistics to JSON file.
    
    Args:
        aggregate_stats: Dictionary of aggregate metrics
        output_path: Path to output JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    logger.info(f"Wrote aggregate summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze planning sensitivity across agent trajectories'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        required=True,
        help='Base directory containing plan/model/lang/languatory.json files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Directory for output files (default: current directory)'
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=3,
        help='Maximum number of instances to analyze (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Validate base directory
    if not args.base_dir.exists():
        logger.error(f"Base directory does not exist: {args.base_dir}")
        return
    
    logger.info(f"Starting analysis with base directory: {args.base_dir}")
    logger.info(f"Maximum cases to analyze: {args.max_cases}")
    
    # Load all languatory files
    all_records = load_all_languatories(args.base_dir, max_cases=args.max_cases)
    
    if not all_records:
        logger.error("No languatory.json files found. Exiting.")
        return
    
    # Group by instance ID with max_cases limit
    by_instance = group_by_instance_id(all_records, max_cases=args.max_cases)
    
    # Analyze each instance
    logger.info("Analyzing per-instance sensitivity...")
    instance_analyses = []
    for instance_id, variants in by_instance.items():
        analysis = analyze_instance(instance_id, variants)
        instance_analyses.append(analysis)
    
    # Compute aggregates
    logger.info("Computing aggregate statistics...")
    aggregate_stats = compute_aggregate_statistics(instance_analyses, all_records)
    
    # Write outputs
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    write_per_instance_summary(
        instance_analyses,
        output_dir / 'per_instance_summary.csv'
    )
    
    write_aggregate_summary(
        aggregate_stats,
        output_dir / 'aggregate_summary.json'
    )
    
    # Log summary statistics
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total instances analyzed: {aggregate_stats['total_instances']}")
    logger.info(f"Resolution-sensitive instances: {aggregate_stats['resolution_sensitive_count']} "
                f"({aggregate_stats['resolution_sensitive_fraction']:.2%})")
    logger.info("")
    logger.info("Resolution rates by plan:")
    for plan, rate in aggregate_stats['resolution_rate_by_plan'].items():
        logger.info(f"  {plan}: {rate:.2%}")
    logger.info("")
    logger.info("Resolution rates by model:")
    for model, rate in aggregate_stats['resolution_rate_by_model'].items():
        logger.info(f"  {model}: {rate:.2%}")
    logger.info("")
    logger.info("Plan predictiveness analysis:")
    pred = aggregate_stats['plan_predictiveness']
    if pred.get('pearson_correlation') is not None:
        logger.info(f"  Pearson correlation: r={pred['pearson_correlation']:.3f} (p={pred['pearson_pvalue']:.4f})")
        logger.info(f"  Spearman correlation: ρ={pred['spearman_correlation']:.3f} (p={pred['spearman_pvalue']:.4f})")
        logger.info(f"  Monotonic instances: {pred['per_instance_monotonic_count']}/{pred['per_instance_total']} "
                   f"({pred['per_instance_monotonic_fraction']:.2%})")
        logger.info(f"  Interpretation: {'Positive' if pred['pearson_correlation'] > 0 else 'Negative'} correlation means "
                   f"{'higher' if pred['pearson_correlation'] > 0 else 'lower'} plan resolution rates predict "
                   f"{'higher' if pred['pearson_correlation'] > 0 else 'lower'} instance success")
    else:
        logger.info("  Insufficient data for correlation analysis")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()