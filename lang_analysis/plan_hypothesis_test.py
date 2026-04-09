"""
Plan Hypothesis Test: Verify agent trajectories follow expected phase order.

CORE LOGIC:
1. Alphabet check: All phases in trajectory must match expected phases (with prefix matching)
2. Presence check: All expected phases must appear in trajectory (with prefix matching)
3. Order check: First appearances must follow strictly increasing order

PREFIX MATCHING (asymmetric):
- "L_reproduce_regression_test" matches expected "L_reproduce" ✓ (actual can be more specific)
- "L_reproduce" does NOT match expected "L_reproduce_regression_test" ✗ (actual insufficient)

USAGE:
  # Single model test with setting
  python lang_analysis/plan_hypothesis_test.py --agent SWE-agent --setting plan --model deepseek_v3

  # Multiple models test with setting
  python plan_hypothesis_test.py --agent SWE-agent --setting plan --models model1 model2 model3

  # Custom expected order (overrides setting default)
  python plan_hypothesis_test.py --agent SWE-agent --setting plan --model deepseek_v3 \
    --expected-order L_navigate L_reproduce P V_newly_generated_test

Expected Order:
- plan: L_navigate → L_reproduce → P → V_newly_generated_test
- no_reproduce_and_verification: L_navigate → P
- no_verification: L_navigate → L_reproduce → P
- plan_and_regression: L_reproduce_regression_test → L_navigate → L_reproduce_newly_generated_test → P → V_newly_generated_test → V_regression_test
- plan_reminded: L_navigate → L_reproduce → P → V_newly_generated_test
- plan_reordered: L_navigate → P → V_newly_generated_test

OUTPUT:
- Single model: artifacts/{agent}/{setting}/{model}/stats/plan_hypothesis_test/{agent}_{setting}_{model}_test.txt
- Multi model: artifacts/{agent}/{setting}/stats/plan_hypothesis_test/{agent}_{setting}_aggregated_test.txt
- Includes: compliance rate, confidence intervals, resolution status correlation
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import stats


@dataclass
class PlanTestConfig:
    """Configuration for plan hypothesis test."""
    agent: str = "SWE-agent"
    models: Optional[List[str]] = None
    setting: str = "plan"
    data_dir: str = "artifacts"
    confidence_level: float = 0.95
    expected_order: Tuple[str, ...] = ("L_navigate", "L_reproduce", "P", "V_newly_generated_test")


# Setting to expected order mapping
SETTING_TO_EXPECTED_ORDER = {
    "plan": ("L_navigate", "L_reproduce", "P", "V_newly_generated_test"),
    "no_reproduce_and_verification": ("L_navigate", "P"),
    "no_verification": ("L_navigate", "L_reproduce", "P"),
    "plan_and_regression": ("L_reproduce_regression_test", "L_navigate", "L_reproduce_newly_generated_test", "P", "V_newly_generated_test", "V_regression_test"),
    "plan_reminded": ("L_navigate", "L_reproduce", "P", "V_newly_generated_test"),
    "plan_reordered": ("L_navigate", "P", "V_newly_generated_test"),
}


def extract_phase_prefix(lang_item: str) -> str:
    """
    Extract phase prefix from language item (e.g., 'L_navigate_2' -> 'L_navigate').

    Args:
        lang_item: Language item from trajectory (e.g., 'L_navigate_2', 'P_1')

    Returns:
        Phase prefix without run-length suffix
    """
    # Split by underscore and remove the last numeric part
    parts = lang_item.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return lang_item


def extract_first_appearances(languatory: List[str]) -> Dict[str, int]:
    """
    Extract first appearance indices for each unique phase, accounting for run lengths.

    Args:
        languatory: List of language items with run-length suffixes

    Returns:
        Dictionary mapping phase prefix to its cumulative index (sum of previous run lengths)

    Example:
        ["L_navigate_2", "L_reproduce_2", "L_navigate_1", "P_1"]
        -> {"L_navigate": 0, "L_reproduce": 2, "P": 5}
    """
    first_appearances = {}
    cumulative_idx = 0

    for item in languatory:
        phase = extract_phase_prefix(item)

        # Extract run length from suffix
        parts = item.rsplit('_', 1)
        run_length = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1

        if phase not in first_appearances:
            first_appearances[phase] = cumulative_idx

        cumulative_idx += run_length

    return first_appearances


def phase_matches_expected(actual_phase: str, expected_phase: str) -> bool:
    """
    Check if an actual phase matches an expected phase using prefix matching.

    Prefix matching rule:
    - Exact match: actual_phase == expected_phase ✓
    - Prefix match: actual_phase starts with expected_phase + "_" ✓
      Examples: "L_reproduce_regression_test" matches "L_reproduce"
                "V_regression_test" matches "V"
    - Reverse does NOT work: "L_reproduce" does NOT match "L_reproduce_regression_test"

    Args:
        actual_phase: Phase from languatory
        expected_phase: Phase from expected_order

    Returns:
        True if actual_phase satisfies the expected_phase requirement
    """
    if actual_phase == expected_phase:
        return True
    # Actual phase can be more specific (longer) than expected phase
    return actual_phase.startswith(expected_phase + "_")


def is_phase_in_expected_alphabet(phase: str, expected_order: Tuple[str, ...]) -> bool:
    """
    Check if a phase belongs to the alphabet of expected phases (with prefix matching).

    Args:
        phase: Phase from languatory
        expected_order: Tuple of expected phases

    Returns:
        True if phase matches any expected phase (including prefix matching)
    """
    for expected_phase in expected_order:
        if phase_matches_expected(phase, expected_phase):
            return True
    return False


def check_trajectory_compliance(
    languatory: List[str],
    expected_order: Tuple[str, ...]
) -> bool:
    """
    Check if a trajectory follows the expected phase order by first appearance.

    The trajectory is compliant if:
    1. All phases in languatory belong to the expected alphabet (with prefix matching)
    2. All expected phases appear in the trajectory (with prefix matching)
    3. The first appearance indices follow a strictly increasing order

    Prefix matching: "L_reproduce_regression_test" satisfies expected "L_reproduce",
    but "L_reproduce" does NOT satisfy expected "L_reproduce_regression_test".

    Args:
        languatory: Trajectory language items
        expected_order: Expected phase order (e.g., ("L_navigate", "L_reproduce", "P", "V_newly_generated_test"))

    Returns:
        True if the trajectory is compliant with the expected plan

    Example 1 (Compliant with prefix matching):
        languatory = ["L_navigate_2", "L_reproduce_regression_test_2", "P_1", "V_newly_generated_test_1"]
        expected_order = ("L_navigate", "L_reproduce", "P", "V_newly_generated_test")

        All phases in alphabet ✓
        L_reproduce_regression_test matches L_reproduce (prefix match)
        First appearances: [0, 2, 4, 5] - strictly increasing ✓
        Returns True

    Example 2 (Non-compliant - unexpected phase):
        languatory = ["L_navigate_2", "L_reproduce_2", "X_unknown_1", "P_1"]
        expected_order = ("L_navigate", "L_reproduce", "P")

        X_unknown not in expected alphabet ✗
        Returns False

    Example 3 (Non-compliant - insufficient prefix):
        languatory = ["L_navigate_2", "L_reproduce_1", "P_1"]
        expected_order = ("L_navigate", "L_reproduce_regression_test", "P")

        L_reproduce does NOT satisfy L_reproduce_regression_test (not specific enough) ✗
        Returns False
    """
    first_appearances = extract_first_appearances(languatory)

    # Step 1: Check that all phases in languatory are in the expected alphabet
    for phase in first_appearances.keys():
        if not is_phase_in_expected_alphabet(phase, expected_order):
            # Found a phase that doesn't match any expected phase
            return False

    # Step 2: Build mapping from expected phases to their actual first appearances
    # using prefix matching
    expected_to_actual = {}

    for expected_phase in expected_order:
        found = False
        for actual_phase, first_idx in first_appearances.items():
            if phase_matches_expected(actual_phase, expected_phase):
                # Take the earliest match for this expected phase
                if expected_phase not in expected_to_actual or first_idx < expected_to_actual[expected_phase]:
                    expected_to_actual[expected_phase] = first_idx
                found = True

        if not found:
            # Required expected phase is missing
            return False

    # Step 3: Check if the first appearances are in strictly increasing order
    indices = [expected_to_actual[phase] for phase in expected_order]
    return all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))


def calculate_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate proportion and Wilson score confidence interval.

    Args:
        successes: Number of compliant trajectories
        total: Total number of trajectories
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (proportion, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p = successes / total

    # Wilson score interval
    # z-score for confidence level (1.96 for 95%, 2.576 for 99%)
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.96)

    denominator = 1 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denominator
    margin = (z * math.sqrt((p * (1 - p) / total) + (z * z) / (4 * total * total))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return p, lower, upper


def compute_resolution_compliance_correlation(
    resolution_stats: Dict[str, Tuple[int, int, List[str], List[str]]]
) -> Dict[str, float]:
    """
    Compute statistical correlation between resolution status and compliance.

    Uses Chi-square test for independence and Phi coefficient for effect size.

    Args:
        resolution_stats: Dict mapping resolution status to (compliant_count, total_count, compliant_ids, non_compliant_ids)

    Returns:
        Dict containing:
            - chi2: Chi-square statistic
            - p_value: P-value for the test
            - phi: Phi coefficient (effect size)
            - cramers_v: Cramer's V (alternative effect size measure)
            - valid: Whether the test is valid (False if insufficient data)
    """
    # Extract counts for 2x2 contingency table
    # Rows: resolved, unresolved
    # Cols: compliant, non-compliant
    resolved_compliant = resolution_stats.get("resolved", [0, 0, [], []])[0]
    resolved_total = resolution_stats.get("resolved", [0, 0, [], []])[1]
    resolved_non_compliant = resolved_total - resolved_compliant

    unresolved_compliant = resolution_stats.get("unresolved", [0, 0, [], []])[0]
    unresolved_total = resolution_stats.get("unresolved", [0, 0, [], []])[1]
    unresolved_non_compliant = unresolved_total - unresolved_compliant

    n = resolved_total + unresolved_total

    # Check if chi-square test is valid
    # Need at least one observation in each row and column
    if (resolved_total == 0 or unresolved_total == 0 or
        (resolved_compliant + unresolved_compliant) == 0 or
        (resolved_non_compliant + unresolved_non_compliant) == 0):
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "phi": 0.0,
            "cramers_v": 0.0,
            "n": n,
            "valid": False
        }

    # Contingency table
    contingency_table = [
        [resolved_compliant, resolved_non_compliant],
        [unresolved_compliant, unresolved_non_compliant]
    ]

    try:
        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Phi coefficient (effect size for 2x2 table)
        phi = math.sqrt(chi2 / n) if n > 0 else 0.0

        # Cramer's V (same as Phi for 2x2 table)
        cramers_v = phi

        return {
            "chi2": chi2,
            "p_value": p_value,
            "phi": phi,
            "cramers_v": cramers_v,
            "n": n,
            "valid": True
        }
    except (ValueError, ZeroDivisionError):
        # Handle cases where chi-square test fails
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "phi": 0.0,
            "cramers_v": 0.0,
            "n": n,
            "valid": False
        }


def test_model(
    agent: str,
    model: str,
    setting: str,
    data_dir: str,
    expected_order: Tuple[str, ...]
) -> Tuple[int, int, List[str], List[str], Dict[str, Tuple[int, int, List[str], List[str]]]]:
    """
    Test a single model's trajectories for plan compliance.

    Args:
        agent: Agent name (e.g., "SWE-agent")
        model: Model name (e.g., "claude-sonnet-4")
        setting: Setting name (e.g., "plan", "no_verification")
        data_dir: Root data directory
        expected_order: Expected phase order

    Returns:
        Tuple of (compliant_count, total_count, compliant_ids, non_compliant_ids, resolution_stats)
        where resolution_stats maps "resolved"/"unresolved" to (compliant_count, total_count, compliant_ids, non_compliant_ids)
    """
    lang_file = Path(data_dir) / agent / setting / model / "lang" / "languatory.json"

    if not lang_file.exists():
        raise FileNotFoundError(f"Languatory file not found: {lang_file}")

    with open(lang_file, 'r') as f:
        data = json.load(f)

    compliant_count = 0
    compliant_ids = []
    non_compliant_ids = []

    # Track stats by resolution_status (resolved/unresolved only)
    resolution_stats = {
        "resolved": [0, 0, [], []],  # [compliant_count, total_count, compliant_ids, non_compliant_ids]
        "unresolved": [0, 0, [], []]
    }

    for entry in data:
        instance_id = entry.get("instance_id", "unknown")
        languatory = entry.get("languatory", [])
        resolution_status = entry.get("resolution_status", "")

        # Only process resolved/unresolved
        if resolution_status not in ["resolved", "unresolved"]:
            continue

        is_compliant = check_trajectory_compliance(languatory, expected_order)

        if is_compliant:
            compliant_count += 1
            compliant_ids.append(instance_id)
            resolution_stats[resolution_status][0] += 1
            resolution_stats[resolution_status][2].append(instance_id)
        else:
            non_compliant_ids.append(instance_id)
            resolution_stats[resolution_status][3].append(instance_id)

        resolution_stats[resolution_status][1] += 1

    total_count = len(data)
    return compliant_count, total_count, compliant_ids, non_compliant_ids, resolution_stats


def format_results(
    model: str,
    compliant: int,
    total: int,
    proportion: float,
    lower: float,
    upper: float,
    confidence_level: float,
    compliant_ids: List[str],
    non_compliant_ids: List[str],
    expected_order: Tuple[str, ...],
    resolution_stats: Dict[str, Tuple[int, int, List[str], List[str]]] = None
) -> str:
    """
    Format test results as a readable string.

    Args:
        model: Model name
        compliant: Number of compliant trajectories
        total: Total trajectories
        proportion: Compliance proportion
        lower: Lower bound of CI
        upper: Upper bound of CI
        confidence_level: Confidence level
        compliant_ids: List of compliant instance IDs
        non_compliant_ids: List of non-compliant instance IDs
        expected_order: Expected phase order
        resolution_stats: Optional dict mapping resolution status to stats

    Returns:
        Formatted result string
    """
    result = []
    result.append("=" * 80)
    result.append(f"Plan Hypothesis Test: {model}")
    result.append("=" * 80)
    result.append("")
    result.append(f"Expected Phase Order (by first appearance): {' → '.join(expected_order)}")
    result.append("")
    # result.append(f"Total Trajectories: {total}")
    # result.append(f"Compliant Trajectories: {compliant}")
    # result.append(f"Non-Compliant Trajectories: {total - compliant}")
    # result.append("")
    # result.append(f"Compliance Rate: {proportion:.2%} ({compliant}/{total})")
    # result.append(f"{int(confidence_level * 100)}% Confidence Interval: [{lower:.2%}, {upper:.2%}]")
    # result.append("")

    # Add breakdown by resolution status
    if resolution_stats:
        result.append("-" * 80)
        result.append("Breakdown by Resolution Status:")
        result.append("-" * 80)
        for status in ["resolved", "unresolved"]:
            if status in resolution_stats:
                res_compliant, res_total, res_compliant_ids, res_non_compliant_ids = resolution_stats[status]
                if res_total > 0:
                    res_proportion, res_lower, res_upper = calculate_confidence_interval(
                        res_compliant, res_total, confidence_level
                    )
                    result.append(f"\n{status.capitalize()}:")
                    result.append(f"  Total: {res_total}")
                    result.append(f"  Compliant: {res_compliant}")
                    result.append(f"  Non-Compliant: {res_total - res_compliant}")
                    result.append(f"  Compliance Rate: {res_proportion:.2%} ({res_compliant}/{res_total})")
                    result.append(f"  {int(confidence_level * 100)}% CI: [{res_lower:.2%}, {res_upper:.2%}]")
        result.append("")

        # Add correlation statistics
        correlation = compute_resolution_compliance_correlation(resolution_stats)
        result.append("-" * 80)
        result.append("Statistical Association (Resolution Status ↔ Compliance):")
        result.append("-" * 80)

        if not correlation.get('valid', True):
            result.append("Chi-square test: Not applicable (insufficient data for analysis)")
            result.append(f"Sample size: n = {correlation['n']}")
            result.append("")
        else:
            result.append(f"Chi-square test: χ² = {correlation['chi2']:.4f}, p = {correlation['p_value']:.4f}")

            # Interpret p-value
            if correlation['p_value'] < 0.001:
                significance = "highly significant (p < 0.001)"
            elif correlation['p_value'] < 0.01:
                significance = "very significant (p < 0.01)"
            elif correlation['p_value'] < 0.05:
                significance = "significant (p < 0.05)"
            else:
                significance = "not significant (p ≥ 0.05)"
            result.append(f"Significance: {significance}")

            result.append(f"Phi coefficient: φ = {correlation['phi']:.4f}")

            # Interpret effect size
            phi_abs = abs(correlation['phi'])
            if phi_abs < 0.1:
                effect_size = "negligible"
            elif phi_abs < 0.3:
                effect_size = "small"
            elif phi_abs < 0.5:
                effect_size = "medium"
            else:
                effect_size = "large"
            result.append(f"Effect size: {effect_size}")

            result.append(f"Sample size: n = {correlation['n']}")
            result.append("")

    result.append("-" * 80)
    result.append(f"Compliant Instances ({len(compliant_ids)}):")
    result.append("-" * 80)
    for instance_id in compliant_ids:
        result.append(f"  - {instance_id}")
    result.append("")
    result.append("-" * 80)
    result.append(f"Non-Compliant Instances ({len(non_compliant_ids)}):")
    result.append("-" * 80)
    for instance_id in non_compliant_ids:
        result.append(f"  - {instance_id}")
    result.append("")
    result.append("=" * 80)

    return "\n".join(result)


def run_single_model_test(config: PlanTestConfig, model: str) -> None:
    """
    Run hypothesis test for a single model and save results.

    Args:
        config: Test configuration
        model: Model name to test
    """
    print(f"Testing model: {model}")

    compliant, total, compliant_ids, non_compliant_ids, resolution_stats = test_model(
        config.agent,
        model,
        config.setting,
        config.data_dir,
        config.expected_order
    )

    proportion, lower, upper = calculate_confidence_interval(
        compliant,
        total,
        config.confidence_level
    )

    # Format results
    results_text = format_results(
        model,
        compliant,
        total,
        proportion,
        lower,
        upper,
        config.confidence_level,
        compliant_ids,
        non_compliant_ids,
        config.expected_order,
        resolution_stats
    )

    # Print to console
    print(results_text)

    # Save to file at artifacts/{agent}/{setting}/{model}/stats/plan_hypothesis_test/
    output_path = Path(config.data_dir) / config.agent / config.setting / model / "stats" / "plan_hypothesis_test" / f"{config.agent}_{config.setting}_{model}_test.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(results_text)

    print(f"\nResults saved to: {output_path}")


def run_multi_model_test(config: PlanTestConfig) -> None:
    """
    Run hypothesis test for multiple models and save aggregated results.

    Args:
        config: Test configuration with models list
    """
    if not config.models:
        raise ValueError("No models specified for multi-model test")

    all_results = []

    for model in config.models:
        print(f"\n{'=' * 80}")
        print(f"Testing model: {model}")
        print(f"{'=' * 80}\n")

        try:
            compliant, total, compliant_ids, non_compliant_ids, resolution_stats = test_model(
                config.agent,
                model,
                config.setting,
                config.data_dir,
                config.expected_order
            )

            proportion, lower, upper = calculate_confidence_interval(
                compliant,
                total,
                config.confidence_level
            )

            results_text = format_results(
                model,
                compliant,
                total,
                proportion,
                lower,
                upper,
                config.confidence_level,
                compliant_ids,
                non_compliant_ids,
                config.expected_order,
                resolution_stats
            )

            all_results.append(results_text)
            print(results_text)

        except Exception as e:
            error_msg = f"Error testing model {model}: {str(e)}"
            print(error_msg)
            all_results.append(error_msg)

    # Save aggregated results at setting level
    output_path = Path(config.data_dir) / config.agent / config.setting / "stats" / "plan_hypothesis_test" / f"{config.agent}_{config.setting}_aggregated_test.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("\n\n".join(all_results))

    print(f"\n\nAggregated results saved to: {output_path}")


def main():
    """Main entry point for plan hypothesis testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test whether agent trajectories follow expected plan structure"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="SWE-agent",
        help="Agent name (default: SWE-agent)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Single model to test (e.g., claude-sonnet-4)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple models to test"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="plan",
        choices=["plan", "no_reproduce_and_verification", "no_verification", "plan_and_regression", "plan_reminded", "plan_reordered"],
        help="Setting name (default: plan)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="artifacts",
        help="Root data directory (default: artifacts)"
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        choices=[0.90, 0.95, 0.99],
        help="Confidence level for interval (default: 0.95)"
    )
    parser.add_argument(
        "--expected-order",
        type=str,
        nargs="+",
        default=None,
        help="Expected phase order (overrides default from setting)"
    )

    args = parser.parse_args()

    # Determine expected order: use provided or default from setting
    if args.expected_order:
        expected_order = tuple(args.expected_order)
    else:
        expected_order = SETTING_TO_EXPECTED_ORDER[args.setting]

    config = PlanTestConfig(
        agent=args.agent,
        models=args.models,
        setting=args.setting,
        data_dir=args.data_dir,
        confidence_level=args.confidence_level,
        expected_order=expected_order
    )

    if args.model:
        # Single model test
        run_single_model_test(config, args.model)
    elif args.models:
        # Multi-model test
        run_multi_model_test(config)
    else:
        # Auto-detect all models in data directory
        setting_dir = Path(args.data_dir) / args.agent / args.setting
        if setting_dir.exists():
            models = [d.name for d in setting_dir.iterdir() if d.is_dir()]
            if models:
                print(f"Auto-detected models: {', '.join(models)}")
                config.models = models
                run_multi_model_test(config)
            else:
                print("No models found. Please specify --model or --models.")
        else:
            print(f"Directory not found: {setting_dir}")
            print("Please specify --model or --models.")


if __name__ == "__main__":
    main()
