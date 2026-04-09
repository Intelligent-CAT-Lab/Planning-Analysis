#!/usr/bin/env python3
"""
Graph Generation Script for Agent Trajectories

This script generates trajectory graphs (JSON + HTML) from agent execution traces.
Supports SWE-agent and OpenHands trajectories across multiple models.

Usage:
    python graph_construction/generate_graphs.py --agent sa --model dsk-v3 --trajs path_to_your_trajectory_folder --eval_report path_to_your_report.json --output_dir data/samples
    python graph_construction/generate_graphs.py --agent oh --model dsk-v3 --trajs path_to_your_output.jsonl --eval_report path_to_your_report.json --output_dir data/samples --aggregate_htmls true

Output Structure:
    {output_dir}/SWE-agent/graphs/{model}/{instance_id}/{instance_id}.{json,html}
    {output_dir}/OpenHands/graphs/{model}/{instance_id}/{instance_id}.{json,html}
    {output_dir}/temp/*.html  (if --aggregate_htmls true)
"""

import argparse
import json
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from commandParser import CommandParser
from buildGraph import build_graph_from_sa_trajectory, build_graph_from_oh_trajectory


# ==================== Configuration ====================
SUPPORTED_AGENTS = {"sa", "oh"}
SUPPORTED_MODELS = {"dsk-v3", "dsk-r1", "dev", "cld-4", "gpt-5m", "dsk-v3.2"}

MODEL_NAMES = {
    "dsk-v3": "deepseek-v3",
    "dsk-r1": "deepseek-r1-0528",
    "dev": "devstral-small",
    "cld-4": "claude-sonnet-4",
    "gpt-5m": "gpt5-mini",
    "dsk-v3.2": "deepseek-v3.2"
}

AGENT_NAMES = {
    "sa": "SWE-agent",
    "oh": "OpenHands"
}


# ==================== Data Classes ====================
@dataclass
class ProcessingResult:
    """Result of processing a single trajectory."""
    instance_id: str
    status: str  # "success" or "error"
    json_path: Optional[str] = None
    html_path: Optional[str] = None
    error: Optional[str] = None


# ==================== Path Management ====================
def get_graph_output_dir(base_output_dir: str, agent: str, model: str) -> Path:
    """Construct the graph output directory path.

    Args:
        base_output_dir: Base output directory
        agent: Agent type (sa/oh)
        model: Model type (dsk-v3/dsk-r1/dev/cld-4)

    Returns:
        Path to graph output directory

    Structure:
        {base_output_dir}/{AgentName}/graphs/{model_name}/
    """
    agent_name = AGENT_NAMES[agent]
    model_name = MODEL_NAMES[model]

    return Path(base_output_dir) / agent_name / "graphs" / model_name


# ==================== Trajectory Loaders ====================
class TrajectoryLoader:
    """Base class for loading agent trajectories."""

    @staticmethod
    def load_sa_trajectories(trajs_path: Path) -> List[Dict[str, Any]]:
        """Load SWE-agent trajectories from directory structure.

        Directory structure:
            trajs_path/
                ├── instance-1/
                │   ├── instance-1.traj
                │   └── instance-1.pred
                ├── instance-2/
                │   ├── instance-2.traj
                │   └── instance-2.pred
                └── ...

        Args:
            trajs_path: Path to trajectories directory

        Returns:
            List of trajectory dictionaries
        """
        trajectories = []

        if not trajs_path.is_dir():
            raise ValueError(f"SA trajectories path must be a directory: {trajs_path}")

        for instance_dir in sorted(trajs_path.iterdir()):
            if not instance_dir.is_dir():
                continue

            instance_id = instance_dir.name
            traj_file = instance_dir / f"{instance_id}.traj"

            if not traj_file.exists():
                print(f"[WARN] Missing .traj file for {instance_id}, skipping")
                continue

            try:
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                trajectories.append({"instance_id": instance_id, "traj_data": traj_data})
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse {traj_file}: {e}")
                continue

        return trajectories

    @staticmethod
    def load_oh_trajectories(trajs_path: Path) -> List[Dict[str, Any]]:
        """Load OpenHands trajectories from output.jsonl file.

        File format: JSONL with one trajectory per line, each containing 'instance_id' field

        Args:
            trajs_path: Path to output.jsonl file

        Returns:
            List of trajectory dictionaries
        """
        trajectories = []

        if not trajs_path.is_file():
            raise ValueError(f"OH trajectories path must be a file: {trajs_path}")

        with open(trajs_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    traj_data = json.loads(line)
                    instance_id = traj_data.get("instance_id")

                    if not instance_id:
                        print(f"[WARN] Line {line_num}: Missing instance_id, skipping")
                        continue

                    trajectories.append({"instance_id": instance_id, "traj_data": traj_data})
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_num}: Failed to parse JSON: {e}")
                    continue

        return trajectories


# ==================== Graph Processor ====================
class GraphProcessor:
    """Process trajectories and generate graphs."""

    def __init__(self, agent: str, parser: CommandParser, eval_report_path: str, output_dir: Path, temp_html_dir: Optional[Path] = None):
        self.agent = agent
        self.parser = parser
        self.eval_report_path = eval_report_path
        self.output_dir = output_dir
        self.temp_html_dir = temp_html_dir

    def process_trajectory(self, instance_id: str, traj_data: Dict[str, Any]) -> ProcessingResult:
        """Process a single trajectory and generate graph.

        Args:
            instance_id: Instance identifier
            traj_data: Trajectory data dictionary

        Returns:
            ProcessingResult with status and paths
        """
        try:
            if self.agent == "sa":
                json_path, html_path = build_graph_from_sa_trajectory(
                    traj_data=traj_data,
                    parser=self.parser,
                    instance_id=instance_id,
                    output_dir=str(self.output_dir),
                    eval_report_path=self.eval_report_path
                )
            elif self.agent == "oh":
                json_path, html_path = build_graph_from_oh_trajectory(
                    traj_data=traj_data,
                    parser=self.parser,
                    instance_id=instance_id,
                    output_dir=str(self.output_dir),
                    eval_report_path=self.eval_report_path
                )
            else:
                raise ValueError(f"Unsupported agent: {self.agent}")

            # Copy HTML to temp directory if specified
            if self.temp_html_dir and html_path:
                html_source = Path(html_path)
                if html_source.exists():
                    html_dest = self.temp_html_dir / html_source.name
                    shutil.copy2(html_source, html_dest)

            return ProcessingResult(
                instance_id=instance_id,
                status="success",
                json_path=json_path,
                html_path=html_path
            )

        except Exception as e:
            return ProcessingResult(
                instance_id=instance_id,
                status="error",
                error=str(e)
            )


# ==================== Main Functions ====================
def setup_parser_for_agent(agent: str) -> CommandParser:
    """Setup CommandParser with appropriate tool configurations.

    Args:
        agent: Agent type ("sa" or "oh")

    Returns:
        Configured CommandParser instance
    """
    parser = CommandParser()

    # Load tool configurations based on agent
    tool_configs = []
    if agent == "sa":
        # Add SWE-agent specific tool configs
        tool_configs = [
            "data/SWE-agent/tools/edit_anthropic/config.yaml",
            "data/SWE-agent/tools/review_on_submit_m/config.yaml",
            "data/SWE-agent/tools/registry/config.yaml",
        ]
    elif agent == "oh":
        # Add OpenHands specific tool configs if needed
        pass

    if tool_configs:
        parser.load_tool_yaml_files(tool_configs)

    return parser


def process_batch(
    trajectories: List[Dict[str, Any]],
    processor: GraphProcessor,
    max_workers: int = 8
) -> Dict[str, List]:
    """Process trajectories in parallel.

    Args:
        trajectories: List of trajectory dictionaries
        processor: GraphProcessor instance
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary with 'success' and 'failed' lists
    """
    results = {"success": [], "failed": []}

    total = len(trajectories)
    print(f"\n{'='*70}")
    print(f"Processing {total} trajectories with {max_workers} workers...")
    print(f"{'='*70}\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(
                processor.process_trajectory,
                traj["instance_id"],
                traj["traj_data"]
            ): traj["instance_id"]
            for traj in trajectories
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_instance):
            result = future.result()
            completed += 1

            if result.status == "success":
                results["success"].append(result)
                print(f"[{completed}/{total}] ✓ {result.instance_id}")
            else:
                results["failed"].append(result)
                print(f"[{completed}/{total}] ✗ {result.instance_id}: {result.error}")

    return results


def print_summary(results: Dict[str, List], agent: str, model: str, output_dir: Path, temp_html_dir: Optional[Path] = None):
    """Print processing summary.

    Args:
        results: Processing results dictionary
        agent: Agent type
        model: Model type
        output_dir: Graph output directory path
        temp_html_dir: Optional temp HTML directory path
    """
    success_count = len(results["success"])
    failed_count = len(results["failed"])
    total = success_count + failed_count

    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Agent:        {AGENT_NAMES[agent]}")
    print(f"Model:        {MODEL_NAMES[model]}")
    print(f"Output:       {output_dir}")
    if temp_html_dir:
        print(f"Temp HTMLs:   {temp_html_dir}")
    print(f"Total:        {total}")
    print(f"Success:      {success_count}")
    print(f"Failed:       {failed_count}")
    print(f"{'='*70}\n")

    if failed_count > 0:
        print("Failed instances:")
        for result in results["failed"][:10]:  # Show first 10 failures
            print(f"  - {result.instance_id}: {result.error}")
        if failed_count > 10:
            print(f"  ... and {failed_count - 10} more")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate trajectory graphs for agent executions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SWE-agent with DeepSeek-V3
  python graph_construction/%(prog)s --agent sa --model dsk-v3 --trajs sa_trajectories --eval_report report.json --output_dir output

  # OpenHands with Claude Sonnet 4, with aggregated HTMLs in temp folder
  python graph_construction/%(prog)s --agent oh --model cld-4 --trajs output.jsonl --eval_report report.json --output_dir output --aggregate_htmls true

Output Structure:
  {output_dir}/SWE-agent/graphs/deepseek-v3/{instance_id}/{instance_id}.{json,html}
  {output_dir}/OpenHands/graphs/claude-sonnet-4/{instance_id}/{instance_id}.{json,html}
  {output_dir}/temp/*.html  (if --aggregate_htmls true)

Supported agents: sa (SWE-agent), oh (OpenHands)
Supported models: dsk-v3 (deepseek-v3), dsk-r1 (deepseek-r1-0528), dev (devstral-small), cld-4 (claude-sonnet-4)
        """
    )

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=list(SUPPORTED_AGENTS),
        help="Agent type: sa (SWE-agent) or oh (OpenHands)"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS),
        help="Model type: dsk-v3, dsk-r1, dev, or cld-4"
    )

    parser.add_argument(
        "--trajs",
        type=str,
        required=True,
        help="Path to trajectories (directory for SA, output.jsonl for OH)"
    )

    parser.add_argument(
        "--eval_report",
        type=str,
        required=True,
        help="Path to evaluation report JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory (graphs will be organized by agent and model)"
    )

    parser.add_argument(
        "--aggregate_htmls",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Copy all HTMLs to a 'temp' folder (default: false)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    args = parser.parse_args()

    # Validate paths
    trajs_path = Path(args.trajs)
    eval_report_path = Path(args.eval_report)

    if not trajs_path.exists():
        print(f"[ERROR] Trajectories path does not exist: {trajs_path}")
        sys.exit(1)

    if not eval_report_path.exists():
        print(f"[ERROR] Evaluation report does not exist: {eval_report_path}")
        sys.exit(1)

    # Construct graph output directory
    graph_output_dir = get_graph_output_dir(args.output_dir, args.agent, args.model)
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup temp HTML directory if aggregate_htmls is true
    temp_html_dir = None
    if args.aggregate_htmls == "true":
        temp_html_dir = Path(args.output_dir) / "temp"
        temp_html_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Agent:        {AGENT_NAMES[args.agent]}")
    print(f"Model:        {MODEL_NAMES[args.model]}")
    print(f"Trajectories: {trajs_path}")
    print(f"Eval Report:  {eval_report_path}")
    print(f"Graph Output: {graph_output_dir}")
    if temp_html_dir:
        print(f"Temp HTMLs:   {temp_html_dir}")
    print(f"Workers:      {args.workers}")
    print(f"{'='*70}\n")

    # Load trajectories
    print("Loading trajectories...")
    try:
        if args.agent == "sa":
            trajectories = TrajectoryLoader.load_sa_trajectories(trajs_path)
        elif args.agent == "oh":
            trajectories = TrajectoryLoader.load_oh_trajectories(trajs_path)
        else:
            print(f"[ERROR] Agent '{args.agent}' is not implemented yet")
            print(f"Supported agents: {', '.join(SUPPORTED_AGENTS)}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load trajectories: {e}")
        sys.exit(1)

    if not trajectories:
        print("[ERROR] No trajectories found")
        sys.exit(1)

    print(f"Loaded {len(trajectories)} trajectories\n")

    # Setup parser
    cmd_parser = setup_parser_for_agent(args.agent)

    # Create processor
    processor = GraphProcessor(
        agent=args.agent,
        parser=cmd_parser,
        eval_report_path=str(eval_report_path),
        output_dir=graph_output_dir,
        temp_html_dir=temp_html_dir
    )

    # Process trajectories
    results = process_batch(trajectories, processor, max_workers=args.workers)

    # Print summary
    print_summary(results, args.agent, args.model, graph_output_dir, temp_html_dir)

    # Exit with appropriate code
    if results["failed"]:
        sys.exit(1)
    else:
        print("✓ All trajectories processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()