"""
Graph Construction Module

Builds trajectory graphs from agent execution traces (SWE-agent and OpenHands).
"""

import json
import os
import hashlib
import networkx as nx
from pathlib import Path
from networkx.readwrite import json_graph
from collections import defaultdict
# Import the refactored visualizer
from visualizer import GraphVisualizer

# Optional datasets import for difficulty lookup
try:
    from datasets import load_dataset
    swe_bench_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    difficulty_lookup = {row["instance_id"]: row["difficulty"] for row in swe_bench_ds}
except ImportError:
    # Fallback if datasets is not available
    difficulty_lookup = {}


FONT_FAMILY = os.environ.get("GRAPH_FONT", "DejaVu Sans, Arial, sans-serif")


# -------------------- Helpers --------------------
def hash_node_signature(label, args, flags):
    """Create unique hash for node signature."""
    normalized = json.dumps({"label": label, "args": args, "flags": flags}, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def check_edit_status(tool, subcommand, args, observation):
    """Check if an edit operation succeeded or failed."""
    def check_str_edit_status(obs):
        if not obs:
            return None
        if "has been edited." in obs:
            return "success"
        if "did not appear verbatim" in obs:
            return "failure: not found"
        if "Multiple occurrences of old_str" in obs:
            return "failure: multiple occurrences"
        if "old_str" in obs and "is the same as new_str" in obs:
            return "failure: no change"
        return "failure: unknown"

    if tool == "str_replace_editor" and subcommand in {"str_replace"}:
        return check_str_edit_status(observation)
    return None


def determine_resolution_status(instance_id: str, eval_report_path: str) -> str:
    """Determine resolution status from eval report given an instance ID."""
    with open(eval_report_path, 'r') as f:
        report = json.load(f)
    if instance_id in report.get("resolved_ids", []):
        return "resolved"
    elif instance_id in report.get("unresolved_ids", []):
        return "unresolved"
    return "unsubmitted"


# -------------------- Graph Builder Class --------------------
class GraphBuilder:
    """Utility class for managing graph construction operations.

    This class encapsulates all shared graph construction logic for building
    trajectory graphs from agent execution traces.
    """

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.node_signature_to_key = {}
        self.localization_nodes = []
        self.prev_phases = set()
        self.previous_node = None
        self.thought_history = []  # Track (node_key, thought_text) pairs

    def add_or_update_node(self, node_label, args, flags, phase, step_idx,
                          tool=None, command=None, subcommand=None, thought_length=0):
        """Add a new node or update existing node with a new occurrence.

        Args:
            node_label: Display label for the node
            args: Command arguments dictionary
            flags: Command flags dictionary
            phase: Phase classification (localization/patch/validation/general)
            step_idx: Step index in trajectory
            tool: Tool name (if applicable)
            command: Command name (if applicable)
            subcommand: Subcommand name (if applicable)
            thought_length: Length of thought text for this step

        Returns:
            node_key: The key of the added or updated node
        """
        node_signature = hash_node_signature(node_label, args, flags)

        if node_signature in self.node_signature_to_key:
            # Update existing node
            node_key = self.node_signature_to_key[node_signature]
            self.G.nodes[node_key]["step_indices"].append(step_idx)
            self.G.nodes[node_key]["thought_lengths"].append(thought_length)
            if "phases" not in self.G.nodes[node_key]:
                self.G.nodes[node_key]["phases"] = []
            self.G.nodes[node_key]["phases"].append(phase)
        else:
            # Add new node
            node_key = f"{len(self.G.nodes)}:{node_label}"
            self.G.add_node(
                node_key,
                label=node_label,
                args=args,
                flags=flags,
                phases=[phase],
                step_indices=[step_idx],
                thought_lengths=[thought_length],
                tool=tool,
                command=command,
                subcommand=subcommand
            )
            self.node_signature_to_key[node_signature] = node_key

            # Track localization nodes
            if tool == "str_replace_editor" and subcommand == "view":
                self.localization_nodes.append(node_key)

        return node_key

    def add_execution_edge(self, node_key, step_idx):
        """Add execution edge from previous node to current node.

        Args:
            node_key: Target node key
            step_idx: Step index for edge label
        """
        if self.previous_node:
            self.G.add_edge(self.previous_node, node_key, label=str(step_idx), type="exec")

    def update_previous_node(self, node_key):
        """Update the previous node pointer.

        Args:
            node_key: Node to set as previous
        """
        self.previous_node = node_key

    def add_phase(self, phase):
        """Add phase to the set of previous phases.

        Args:
            phase: Phase to add
        """
        self.prev_phases.add(phase)

    def track_thought(self, node_key, thought_text):
        """Track thought text for a node and detect substring relationships.
        
        Args:
            node_key: The node to associate with this thought
            thought_text: The thought text content
        """
        # Only track non-empty thoughts
        if not thought_text or not thought_text.strip():
            return
        
        # Check if this thought is a substring continuation of the previous thought
        if self.thought_history:
            prev_node_key, prev_thought = self.thought_history[-1]
            
            # Check if previous thought is a substring of current thought
            # (indicating the current thought extends the previous one)
            if prev_thought and thought_text.startswith(prev_thought):
                # Add a "thought" edge to show this relationship
                self.G.add_edge(prev_node_key, node_key, type="thought", label="")
        
        # Add to history
        self.thought_history.append((node_key, thought_text))

    def finalize_and_save(self, output_dir, instance_id, eval_report_path, template_dir=None):
        """Build hierarchical edges, add metadata, and save graph.

        Args:
            output_dir: Base output directory
            instance_id: Instance identifier
            eval_report_path: Path to evaluation report
            template_dir: Optional path to template directory for visualizer

        Returns:
            tuple: (json_path, html_path) paths to saved files
        """
        build_hierarchical_edges(self.G, self.localization_nodes)

        resolution_status = determine_resolution_status(instance_id, eval_report_path)
        self.G.graph["resolution_status"] = resolution_status
        self.G.graph["instance_name"] = instance_id
        self.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")

        # Construct output paths: output_dir/{instance_id}/{instance_id}.{json,html}
        instance_dir = os.path.join(output_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)

        json_path = os.path.join(instance_dir, f"{instance_id}.json")
        html_path = os.path.join(instance_dir, f"{instance_id}.html")

        # Save JSON
        with open(json_path, "w") as f:
            json.dump(json_graph.node_link_data(self.G, edges="edges"), f, indent=2)

        # Save HTML using refactored visualizer
        GraphVisualizer.draw_with_timeout(
            self.G, 
            html_path, 
            timeout_sec=60,
            template_dir=template_dir
        )

        return json_path, html_path


# -------------------- Build graph --------------------
def build_graph_from_sa_trajectory(traj_data, parser, instance_id, output_dir, eval_report_path, template_dir=None):
    """Build graph from SWE-agent trajectory data.

    Args:
        traj_data: SWE-agent trajectory dictionary containing 'trajectory' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'django__django-12345')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file
        template_dir: Optional path to template directory for visualizer

    Returns:
        tuple: (json_path, html_path) paths to the saved graph files

    Output Structure:
        {output_dir}/{instance_id}/{instance_id}.json
        {output_dir}/{instance_id}/{instance_id}.html
    """
    from mapPhase import get_phase
    
    builder = GraphBuilder()
    trajectory = traj_data.get("trajectory", [])

    for step_idx, step in enumerate(trajectory):
        action_str = step.get("action", "")
        thought = step.get("thought", "") or ""
        thought_length = len(thought)

        # Handle explicit "think" steps (blank action)
        if action_str.strip() == "":
            node_key = builder.add_or_update_node(
                node_label="think",
                args={"thought_len": thought_length},
                flags={},
                phase="general",
                step_idx=step_idx,
                tool=None,
                command=None,
                subcommand=None,
                thought_length=thought_length
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase("general")
            builder.track_thought(node_key, thought)
            continue

        # Parse actionable commands
        parsed_commands = parser.parse(action_str)
        if not parsed_commands:
            continue

        for parsed in parsed_commands:
            tool = parsed.get("tool", "").strip() if parsed.get("tool") else ""
            subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
            command = parsed.get("command", "").strip() if parsed.get("command") else ""
            args = parsed.get("args", {})
            flags = parsed.get("flags", {})

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, builder.prev_phases)

            edit_status = check_edit_status(tool, subcommand, args, step.get("observation", ""))
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = builder.add_or_update_node(
                node_label=node_label,
                args=args,
                flags=flags,
                phase=phase,
                step_idx=step_idx,
                tool=tool,
                command=command,
                subcommand=subcommand,
                thought_length=thought_length
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase(phase)
            builder.track_thought(node_key, thought)

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path, template_dir)


def build_graph_from_oh_trajectory(traj_data, parser, instance_id, output_dir, eval_report_path, template_dir=None):
    """Build graph from OpenHands trajectory data.

    Args:
        traj_data: OpenHands trajectory dictionary containing 'history' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'django__django-12345')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file
        template_dir: Optional path to template directory for visualizer

    Returns:
        tuple: (json_path, html_path) paths to the saved graph files

    Output Structure:
        {output_dir}/{instance_id}/{instance_id}.json
        {output_dir}/{instance_id}/{instance_id}.html
    """
    from mapPhase import get_phase
    
    builder = GraphBuilder()
    step_idx = 0

    for step in traj_data.get("history", []):
        action = step.get("observation") if step.get("observation") else None
        if action in ("system", "message") or action is None:
            continue

        # Use action text only as a fallback when command string is empty
        action_str = action or ""
        thought = step.get("content", "") or ""
        thought_length = len(thought)

        tool_calls = step.get("tool_call_metadata", {}).get("model_response", {}).get("choices", [])
        if not tool_calls and "tool_call_metadata" in step:
            tool_calls = [step["tool_call_metadata"]]

        parsed_commands = []
        for call in tool_calls:
            function_call = None
            if isinstance(call, dict):
                if "function" in call:
                    function_call = call["function"]
                elif "message" in call and "tool_calls" in call["message"]:
                    for tc in call["message"]["tool_calls"]:
                        if "function" in tc:
                            function_call = tc["function"]

            if not function_call:
                continue

            tool_name = function_call.get("name")
            args_raw = function_call.get("arguments", "{}")

            try:
                args_loaded = json.loads(args_raw)
            except json.JSONDecodeError:
                args_loaded = {}

            if tool_name == "execute_bash":
                cmd = args_loaded.get("command", "").strip()
                parsed_commands = parser.parse(cmd)
                if not parsed_commands:
                    continue
            else:
                subcommand = args_loaded.pop("command", None)  # remove 'command' key from args
                parsed_commands = [{
                    "tool": tool_name,
                    "subcommand": subcommand,
                    "args": args_loaded,
                }]

        if not parsed_commands:
            continue

        for parsed in parsed_commands:
            tool = parsed.get("tool", "").strip()
            
            # ---- THINK NODES ----
            if tool == "think":
                node_key = builder.add_or_update_node(
                    node_label="think",
                    args={"thought_len": thought_length},
                    flags={},
                    phase="general",
                    step_idx=step_idx,
                    tool=None,
                    command=None,
                    subcommand=None,
                    thought_length=thought_length
                )
                builder.add_execution_edge(node_key, step_idx)
                builder.update_previous_node(node_key)
                builder.add_phase("general")
                builder.track_thought(node_key, thought)
                continue

            subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
            command = parsed.get("command", "").strip() if parsed.get("command") else ""
            args = parsed.get("args", {})
            flags = parsed.get("flags", {})

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, builder.prev_phases)

            edit_status = check_edit_status(tool, subcommand, args, step.get("content", ""))
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = builder.add_or_update_node(
                node_label=node_label,
                args=args,
                flags=flags,
                phase=phase,
                step_idx=step_idx,
                tool=tool,
                command=command,
                subcommand=subcommand,
                thought_length=thought_length
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase(phase)
            builder.track_thought(node_key, thought)

        step_idx += 1

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path, template_dir)


def build_hierarchical_edges(G: nx.MultiDiGraph, localization_nodes):
    """Build hierarchical edges between localization nodes based on file paths and ranges."""
    path_nodes = []  # [(node_id, Path)]
    range_nodes_by_path = defaultdict(list)  # path_str -> [(node_id, [start, end])]

    for node in localization_nodes:
        data = G.nodes[node]
        path = data.get("args", {}).get("path")
        view_range = data.get("args", {}).get("view_range")

        if path:
            path_obj = Path(path)
            if view_range is None:
                path_nodes.append((node, path_obj))
            elif (
                isinstance(view_range, (list, tuple)) and
                len(view_range) == 2 and
                all(isinstance(x, int) for x in view_range)
            ):
                range_nodes_by_path[str(path_obj)].append((node, view_range))
            else:
                print(f"[WARN] Skipping invalid view_range for node {node}: {view_range}")

    # --- 1) Path hierarchy by folder containment ---
    for child_node, child_path in path_nodes:
        best_parent_node = None
        best_parent_path = None
        for parent_node, parent_path in path_nodes:
            if parent_node == child_node:
                continue
            if (len(parent_path.parts) < len(child_path.parts) and
                child_path.parts[:len(parent_path.parts)] == parent_path.parts):
                if best_parent_path is None or len(parent_path.parts) > len(best_parent_path.parts):
                    best_parent_node = parent_node
                    best_parent_path = parent_path
        if best_parent_node:
            G.add_edge(best_parent_node, child_node, type="hier")

    # --- 2) Range nodes: handle nesting + link outermost ---
    path_to_node = {str(p): n for n, p in path_nodes}

    for path_str, range_nodes in range_nodes_by_path.items():
        is_nested = {n: False for n, _ in range_nodes}

        # detect nesting: mark inner ranges
        for i, (node_i, r_i) in enumerate(range_nodes):
            for j, (node_j, r_j) in enumerate(range_nodes):
                if i == j:
                    continue
                try:
                    a1, a2 = r_i
                    b1, b2 = r_j
                    if b1 >= a1 and b2 <= a2:
                        G.add_edge(node_i, node_j, type="hier")
                        is_nested[node_j] = True
                except Exception as e:
                    print(f"[WARN] Failed to unpack ranges for nesting check: {r_i}, {r_j} ({e})")

        # link outermost ranges to:
        #   - exact path node if exists
        #   - else closest parent path node whose path contains this path
        path_node = path_to_node.get(path_str)
        if path_node:
            for node, _ in range_nodes:
                if not is_nested[node]:
                    G.add_edge(path_node, node, type="hier")
        else:
            # No exact path node → find nearest ancestor
            path_parts = Path(path_str).parts
            best_ancestor_node = None
            best_ancestor_depth = -1
            for pn, pp in path_nodes:
                if len(pp.parts) < len(path_parts) and path_parts[:len(pp.parts)] == pp.parts:
                    if len(pp.parts) > best_ancestor_depth:
                        best_ancestor_node = pn
                        best_ancestor_depth = len(pp.parts)
            for node, _ in range_nodes:
                if not is_nested[node] and best_ancestor_node:
                    G.add_edge(best_ancestor_node, node, type="hier")
