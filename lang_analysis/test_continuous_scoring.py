"""Test continuous compliance scoring logic."""

from continuous_plan_hypothesis_test import (
    compute_compliance_score,
    extract_phase_sequence,
)

print("=" * 80)
print("Testing Continuous Compliance Scoring")
print("=" * 80)
print()

expected_order = ("L_navigate", "L_reproduce", "P", "V_newly_generated_test")

# Test Case 1: Perfect compliance
print("Test Case 1: Perfect Compliance")
print("-" * 80)
trajectory1 = ["L_navigate_2", "L_reproduce_2", "P_1", "V_newly_generated_test_1"]
score1 = compute_compliance_score(trajectory1, expected_order)
print(f"Trajectory: {trajectory1}")
print(f"S1 (Vocabulary): {score1.s1_vocabulary:.4f} (expected: 1.0)")
print(f"  Extra phases: {score1.num_extra_phases} (expected: 0)")
print(f"S2 (Coverage):   {score1.s2_coverage:.4f} (expected: 1.0)")
print(f"  Required seen: {score1.num_required_seen}/{score1.num_required_total}")
print(f"S3 (Order):      {score1.s3_order:.4f} (expected: 1.0)")
print(f"  LCS length: {score1.lcs_length}/{score1.expected_length}")
print(f"Final Score:     {score1.final_score:.4f} (expected: 1.0)")
print()

# Test Case 2: Extra phase (relaxed alphabet)
print("Test Case 2: Extra Phase (X_unknown)")
print("-" * 80)
trajectory2 = ["L_navigate_1", "L_reproduce_2", "X_unknown_1", "P_1", "V_newly_generated_test_2"]
score2 = compute_compliance_score(trajectory2, expected_order)
print(f"Trajectory: {trajectory2}")
print(f"S1 (Vocabulary): {score2.s1_vocabulary:.4f}")
print(f"  Formula: m/(m+|X|) = {score2.num_expected_phases}/({score2.num_expected_phases}+{score2.num_extra_phases}) = {score2.s1_vocabulary:.4f}")
print(f"S2 (Coverage):   {score2.s2_coverage:.4f}")
print(f"  All required phases present: {score2.num_required_seen}/{score2.num_required_total}")
print(f"S3 (Order):      {score2.s3_order:.4f}")
print(f"  LCS length: {score2.lcs_length}/{score2.expected_length}")
print(f"Final Score:     {score2.final_score:.4f}")
print(f"  Geometric mean: ({score2.s1_vocabulary:.4f} * {score2.s2_coverage:.4f} * {score2.s3_order:.4f})^(1/3) = {score2.final_score:.4f}")
print()

# Test Case 3: Missing phase
print("Test Case 3: Missing Phase (no V_newly_generated_test)")
print("-" * 80)
trajectory3 = ["L_navigate_2", "L_reproduce_2", "P_1"]
score3 = compute_compliance_score(trajectory3, expected_order)
print(f"Trajectory: {trajectory3}")
print(f"S1 (Vocabulary): {score3.s1_vocabulary:.4f}")
print(f"S2 (Coverage):   {score3.s2_coverage:.4f}")
print(f"  Required seen: {score3.num_required_seen}/{score3.num_required_total} (missing V_newly_generated_test)")
print(f"S3 (Order):      {score3.s3_order:.4f}")
print(f"  LCS length: {score3.lcs_length}/{score3.expected_length}")
print(f"Final Score:     {score3.final_score:.4f}")
print()

# Test Case 4: Wrong order
print("Test Case 4: Wrong Order")
print("-" * 80)
trajectory4 = ["L_reproduce_1", "L_navigate_2", "P_1", "V_newly_generated_test_1"]
score4 = compute_compliance_score(trajectory4, expected_order)
phase_seq4 = extract_phase_sequence(trajectory4)
print(f"Trajectory: {trajectory4}")
print(f"Phase sequence: {phase_seq4}")
print(f"First appearances: L_navigate@1, L_reproduce@0, P@3, V_newly_generated_test@4")
print(f"S1 (Vocabulary): {score4.s1_vocabulary:.4f}")
print(f"S2 (Coverage):   {score4.s2_coverage:.4f}")
print(f"S3 (Order):      {score4.s3_order:.4f}")
print(f"  LIS length: {score4.lcs_length}/{score4.expected_length}")
print(f"  (First appearances [1,0,3,4] → LIS could be [0,3,4] or [1,3,4], length=3)")
print(f"Final Score:     {score4.final_score:.4f}")
print()

# Test Case 4b: Phase revisit (user's example)
print("Test Case 4b: Phase Revisit (User Example)")
print("-" * 80)
trajectory4b = ["L_navigate_1", "L_reproduce_2", "L_navigate_1", "P_1"]
expected_order_short = ("L_navigate", "L_reproduce", "P")
score4b = compute_compliance_score(trajectory4b, expected_order_short)
print(f"Trajectory: {trajectory4b}")
print(f"Expected: {' → '.join(expected_order_short)}")
print(f"First appearances: L_navigate@0, L_reproduce@1, P@4")
print(f"S1 (Vocabulary): {score4b.s1_vocabulary:.4f}")
print(f"S2 (Coverage):   {score4b.s2_coverage:.4f}")
print(f"S3 (Order):      {score4b.s3_order:.4f} (expected: 1.0)")
print(f"  First appearances [0,1,4] are strictly increasing → LIS length = 3")
print(f"Final Score:     {score4b.final_score:.4f} (expected: 1.0)")
print()

# Test Case 5: Prefix matching
print("Test Case 5: Prefix Matching (L_reproduce_regression_test)")
print("-" * 80)
trajectory5 = ["L_navigate_2", "L_reproduce_regression_test_2", "P_1", "V_newly_generated_test_1"]
score5 = compute_compliance_score(trajectory5, expected_order)
print(f"Trajectory: {trajectory5}")
print(f"S1 (Vocabulary): {score5.s1_vocabulary:.4f}")
print(f"  L_reproduce_regression_test matches L_reproduce (prefix match)")
print(f"S2 (Coverage):   {score5.s2_coverage:.4f}")
print(f"S3 (Order):      {score5.s3_order:.4f}")
print(f"Final Score:     {score5.final_score:.4f} (expected: 1.0)")
print()

# Test Case 6: Complex case - extra phases + reordering
print("Test Case 6: Complex Case")
print("-" * 80)
trajectory6 = ["L_navigate_1", "X_extra_1", "L_reproduce_2", "L_navigate_1", "P_1"]
score6 = compute_compliance_score(trajectory6, expected_order)
phase_seq6 = extract_phase_sequence(trajectory6)
print(f"Trajectory: {trajectory6}")
print(f"Phase sequence: {phase_seq6}")
print(f"S1 (Vocabulary): {score6.s1_vocabulary:.4f}")
print(f"  Extra phases: {score6.num_extra_phases} (X_extra)")
print(f"S2 (Coverage):   {score6.s2_coverage:.4f}")
print(f"  Missing: V_newly_generated_test")
print(f"S3 (Order):      {score6.s3_order:.4f}")
print(f"  LCS: {score6.lcs_length}/{score6.expected_length}")
print(f"Final Score:     {score6.final_score:.4f}")
print()

# Test LIS computation directly on first appearances
print("=" * 80)
print("Testing LIS Computation on First Appearances")
print("=" * 80)
print()

from continuous_plan_hypothesis_test import extract_first_appearances, compute_longest_increasing_subsequence_length

test_cases_lis = [
    {
        "trajectory": ["L_navigate_2", "L_reproduce_2", "P_1", "V_newly_generated_test_1"],
        "description": "Perfect order",
        "expected_lis": 4
    },
    {
        "trajectory": ["L_reproduce_1", "L_navigate_2", "P_1", "V_newly_generated_test_1"],
        "description": "L_reproduce before L_navigate",
        "expected_lis": 3
    },
    {
        "trajectory": ["L_navigate_1", "X_unknown_1", "L_reproduce_2", "P_1", "V_newly_generated_test_1"],
        "description": "Extra phase in between",
        "expected_lis": 4
    },
    {
        "trajectory": ["L_navigate_1", "L_reproduce_2", "L_navigate_1", "P_1"],
        "description": "Phase revisit (user example)",
        "expected_order": ("L_navigate", "L_reproduce", "P"),
        "expected_lis": 3
    },
]

for test_case in test_cases_lis:
    traj = test_case["trajectory"]
    test_expected = test_case.get("expected_order", expected_order)
    first_app = extract_first_appearances(traj)
    lis_len = compute_longest_increasing_subsequence_length(first_app, test_expected)
    expected_lis = test_case["expected_lis"]

    print(f"{test_case['description']}:")
    print(f"  Trajectory: {traj}")
    print(f"  First appearances: {first_app}")
    print(f"  LIS length: {lis_len} (expected: {expected_lis}) {'✓' if lis_len == expected_lis else '✗'}")
    print()

print("=" * 80)
print("All tests completed!")
print("=" * 80)
