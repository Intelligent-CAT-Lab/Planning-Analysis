"""
Word Frequency Analyzer for Trajectory Thoughts and Actions

Analyzes word frequency in thoughts and actions using bag-of-words approach.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


class WordFrequencyAnalyzer:
    """Analyze word frequencies in trajectory thoughts and actions."""
    
    # Common stop words to exclude (can be customized)
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'as', 'if', 'then', 'now', 'also', 'here', 'there'
    }
    
    def __init__(self, use_stop_words: bool = True, min_word_length: int = 2):
        """Initialize analyzer.
        
        Args:
            use_stop_words: Whether to filter out common stop words
            min_word_length: Minimum word length to include
        """
        self.use_stop_words = use_stop_words
        self.min_word_length = min_word_length
        
        # Storage for word counts
        self.thought_words = Counter()
        self.action_words = Counter()
        self.action_type_words = defaultdict(Counter)  # action_type -> word counts
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned, lowercase words
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s_-]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Filter words
        filtered = []
        for word in words:
            # Skip if too short
            if len(word) < self.min_word_length:
                continue
            
            # Skip stop words if enabled
            if self.use_stop_words and word in self.STOP_WORDS:
                continue
            
            filtered.append(word)
        
        return filtered
    
    def _get_action_type(self, tool: str, command: str, subcommand: str) -> str:
        """Determine action type from tool/command/subcommand.
        
        Args:
            tool: Tool name
            command: Command name
            subcommand: Subcommand name
            
        Returns:
            Action type string
        """
        if tool:
            if subcommand:
                return f"{tool}:{subcommand}"
            return tool
        if command:
            return command
        return "unknown"
    
    def analyze_sa_trajectory(self, traj_data: Dict) -> None:
        """Analyze SWE-agent trajectory.
        
        Args:
            traj_data: SWE-agent trajectory dictionary
        """
        try:
            from commandParser import CommandParser
            parser = CommandParser()
            has_parser = True
        except ImportError:
            parser = None
            has_parser = False
        
        trajectory = traj_data.get("trajectory", [])
        
        for step in trajectory:
            # Process thought
            thought = step.get("thought", "") or ""
            if thought.strip():
                thought_words = self._tokenize(thought)
                self.thought_words.update(thought_words)
            
            # Process action
            action_str = step.get("action", "")
            if not action_str.strip():
                continue
            
            # Parse action if parser available
            parsed_commands = []
            if has_parser and parser:
                parsed_commands = parser.parse(action_str)
            
            if not parsed_commands:
                # Fallback: just tokenize the action string directly
                action_words = self._tokenize(action_str)
                self.action_words.update(action_words)
                self.action_type_words["action"].update(action_words)
                continue
            
            for parsed in parsed_commands:
                tool = parsed.get("tool", "").strip() if parsed.get("tool") else ""
                subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
                command = parsed.get("command", "").strip() if parsed.get("command") else ""
                
                # Determine action type
                action_type = self._get_action_type(tool, command, subcommand)
                
                # Tokenize action string
                action_words = self._tokenize(action_str)
                self.action_words.update(action_words)
                self.action_type_words[action_type].update(action_words)
    
    def analyze_oh_trajectory(self, traj_data: Dict) -> None:
        """Analyze OpenHands trajectory.
        
        Args:
            traj_data: OpenHands trajectory dictionary
        """
        try:
            from commandParser import CommandParser
            parser = CommandParser()
            has_parser = True
        except ImportError:
            parser = None
            has_parser = False
        
        for step in traj_data.get("history", []):
            # Skip non-action steps
            action = step.get("observation")
            if action in ("system", "message") or action is None:
                continue
            
            # Process thought (content field)
            thought = step.get("content", "") or ""
            if thought.strip():
                thought_words = self._tokenize(thought)
                self.thought_words.update(thought_words)
            
            # Process tool calls
            tool_calls = step.get("tool_call_metadata", {}).get("model_response", {}).get("choices", [])
            if not tool_calls and "tool_call_metadata" in step:
                tool_calls = [step["tool_call_metadata"]]
            
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
                
                tool_name = function_call.get("name", "")
                args_raw = function_call.get("arguments", "{}")
                
                try:
                    args_loaded = json.loads(args_raw)
                except json.JSONDecodeError:
                    args_loaded = {}
                
                # Get action text
                if tool_name == "execute_bash":
                    action_text = args_loaded.get("command", "")
                else:
                    action_text = f"{tool_name} {json.dumps(args_loaded)}"
                
                # Tokenize action
                action_words = self._tokenize(action_text)
                self.action_words.update(action_words)
                self.action_type_words[tool_name].update(action_words)
    
    def get_top_thought_words(self, n: int = 50) -> List[Tuple[str, int]]:
        """Get top N most common words in thoughts.
        
        Args:
            n: Number of top words to return
            
        Returns:
            List of (word, count) tuples
        """
        return self.thought_words.most_common(n)
    
    def get_top_action_words(self, n: int = 50) -> List[Tuple[str, int]]:
        """Get top N most common words in actions.
        
        Args:
            n: Number of top words to return
            
        Returns:
            List of (word, count) tuples
        """
        return self.action_words.most_common(n)
    
    def get_top_words_by_action_type(self, action_type: str, n: int = 50) -> List[Tuple[str, int]]:
        """Get top N most common words for a specific action type.
        
        Args:
            action_type: Action type to analyze
            n: Number of top words to return
            
        Returns:
            List of (word, count) tuples
        """
        return self.action_type_words[action_type].most_common(n)
    
    def get_all_action_types(self) -> List[str]:
        """Get list of all action types seen.
        
        Returns:
            List of action type strings
        """
        return sorted(self.action_type_words.keys())
    
    def generate_report(self, top_n: int = 30) -> str:
        """Generate a comprehensive word frequency report.
        
        Args:
            top_n: Number of top words to include for each category
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("WORD FREQUENCY ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Overall statistics
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 70)
        lines.append(f"Total unique words in thoughts: {len(self.thought_words)}")
        lines.append(f"Total thought word occurrences: {sum(self.thought_words.values())}")
        lines.append(f"Total unique words in actions: {len(self.action_words)}")
        lines.append(f"Total action word occurrences: {sum(self.action_words.values())}")
        lines.append(f"Number of action types: {len(self.action_type_words)}")
        lines.append("")
        
        # Top thought words
        lines.append(f"TOP {top_n} WORDS IN THOUGHTS")
        lines.append("-" * 70)
        lines.append(f"{'Rank':<6} {'Word':<30} {'Count':<10} {'%':<10}")
        lines.append("-" * 70)
        
        total_thought_words = sum(self.thought_words.values())
        for i, (word, count) in enumerate(self.get_top_thought_words(top_n), 1):
            pct = (count / total_thought_words * 100) if total_thought_words > 0 else 0
            lines.append(f"{i:<6} {word:<30} {count:<10} {pct:.2f}%")
        lines.append("")
        
        # Top action words
        lines.append(f"TOP {top_n} WORDS IN ACTIONS")
        lines.append("-" * 70)
        lines.append(f"{'Rank':<6} {'Word':<30} {'Count':<10} {'%':<10}")
        lines.append("-" * 70)
        
        total_action_words = sum(self.action_words.values())
        for i, (word, count) in enumerate(self.get_top_action_words(top_n), 1):
            pct = (count / total_action_words * 100) if total_action_words > 0 else 0
            lines.append(f"{i:<6} {word:<30} {count:<10} {pct:.2f}%")
        lines.append("")
        
        # Top words by action type
        lines.append("TOP WORDS BY ACTION TYPE")
        lines.append("-" * 70)
        
        action_types = self.get_all_action_types()
        for action_type in action_types:
            top_words = self.get_top_words_by_action_type(action_type, 10)
            if not top_words:
                continue
            
            total = sum(count for _, count in top_words)
            lines.append(f"\n{action_type} (total words: {total})")
            lines.append(f"{'Rank':<6} {'Word':<25} {'Count':<10}")
            lines.append("-" * 45)
            
            for i, (word, count) in enumerate(top_words, 1):
                lines.append(f"{i:<6} {word:<25} {count:<10}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_report(self, output_path: str, top_n: int = 30) -> None:
        """Save report to file.
        
        Args:
            output_path: Path to output file
            top_n: Number of top words to include
        """
        report = self.generate_report(top_n)
        with open(output_path, 'w') as f:
            f.write(report)
    
    def export_json(self, output_path: str, top_n: int = 100) -> None:
        """Export word frequencies as JSON.
        
        Args:
            output_path: Path to output JSON file
            top_n: Number of top words to include for each category
        """
        data = {
            "statistics": {
                "unique_thought_words": len(self.thought_words),
                "total_thought_occurrences": sum(self.thought_words.values()),
                "unique_action_words": len(self.action_words),
                "total_action_occurrences": sum(self.action_words.values()),
                "num_action_types": len(self.action_type_words)
            },
            "top_thought_words": [
                {"word": word, "count": count} 
                for word, count in self.get_top_thought_words(top_n)
            ],
            "top_action_words": [
                {"word": word, "count": count}
                for word, count in self.get_top_action_words(top_n)
            ],
            "by_action_type": {
                action_type: [
                    {"word": word, "count": count}
                    for word, count in self.get_top_words_by_action_type(action_type, top_n)
                ]
                for action_type in self.get_all_action_types()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def analyze_trajectories_batch(
    trajs_dir: Path,
    agent: str,
    output_dir: Path,
    use_stop_words: bool = True,
    min_word_length: int = 2
) -> WordFrequencyAnalyzer:
    """Analyze multiple trajectories in batch.
    
    Args:
        trajs_dir: Directory containing trajectories
        agent: Agent type ('sa' or 'oh')
        output_dir: Directory to save reports
        use_stop_words: Whether to filter stop words
        min_word_length: Minimum word length
        
    Returns:
        WordFrequencyAnalyzer with accumulated results
    """
    analyzer = WordFrequencyAnalyzer(
        use_stop_words=use_stop_words,
        min_word_length=min_word_length
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if agent == 'sa':
        # SWE-agent: directory of subdirectories with .traj files
        for instance_dir in sorted(trajs_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            
            instance_id = instance_dir.name
            traj_file = instance_dir / f"{instance_id}.traj"
            
            if not traj_file.exists():
                continue
            
            try:
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                analyzer.analyze_sa_trajectory(traj_data)
                print(f"✓ Analyzed {instance_id}")
            except Exception as e:
                print(f"✗ Failed to analyze {instance_id}: {e}")
    
    elif agent == 'oh':
        # OpenHands: single JSONL file
        if not trajs_dir.is_file():
            print(f"Error: OpenHands trajectories path must be a file: {trajs_dir}")
            return analyzer
        
        with open(trajs_dir, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    traj_data = json.loads(line)
                    instance_id = traj_data.get("instance_id", f"line_{line_num}")
                    analyzer.analyze_oh_trajectory(traj_data)
                    print(f"✓ Analyzed {instance_id}")
                except Exception as e:
                    print(f"✗ Failed to analyze line {line_num}: {e}")
    
    else:
        print(f"Error: Unsupported agent type: {agent}")
        return analyzer
    
    # Generate reports
    print("\nGenerating reports...")
    
    # Text report
    report_path = output_dir / "word_frequency_report.txt"
    analyzer.save_report(str(report_path))
    print(f"✓ Saved text report: {report_path}")
    
    # JSON export
    json_path = output_dir / "word_frequency_data.json"
    analyzer.export_json(str(json_path))
    print(f"✓ Saved JSON data: {json_path}")
    
    return analyzer


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze word frequencies in trajectory thoughts and actions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze SWE-agent trajectories
  python word_frequency_analyzer.py --agent sa --trajs trajectories/ --output analysis/

  # Analyze OpenHands trajectories
  python word_frequency_analyzer.py --agent oh --trajs output.jsonl --output analysis/

  # Include stop words
  python word_frequency_analyzer.py --agent sa --trajs trajectories/ --output analysis/ --include-stop-words

  # Set minimum word length
  python word_frequency_analyzer.py --agent sa --trajs trajectories/ --output analysis/ --min-length 3
        """
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=['sa', 'oh'],
        help="Agent type: sa (SWE-agent) or oh (OpenHands)"
    )
    
    parser.add_argument(
        "--trajs",
        type=str,
        required=True,
        help="Path to trajectories (directory for SA, .jsonl file for OH)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--include-stop-words",
        action="store_true",
        help="Include common stop words (default: filter them out)"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum word length to include (default: 2)"
    )
    
    args = parser.parse_args()
    
    trajs_path = Path(args.trajs)
    output_dir = Path(args.output)
    
    if not trajs_path.exists():
        print(f"Error: Trajectories path does not exist: {trajs_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("WORD FREQUENCY ANALYZER")
    print("=" * 70)
    print(f"Agent: {args.agent}")
    print(f"Trajectories: {trajs_path}")
    print(f"Output: {output_dir}")
    print(f"Filter stop words: {not args.include_stop_words}")
    print(f"Min word length: {args.min_length}")
    print("=" * 70)
    print()
    
    analyzer = analyze_trajectories_batch(
        trajs_dir=trajs_path,
        agent=args.agent,
        output_dir=output_dir,
        use_stop_words=not args.include_stop_words,
        min_word_length=args.min_length
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Unique words in thoughts: {len(analyzer.thought_words)}")
    print(f"Unique words in actions: {len(analyzer.action_words)}")
    print(f"Action types found: {len(analyzer.action_type_words)}")
    print()
    
    # Show top 10 words
    print("Top 10 thought words:")
    for i, (word, count) in enumerate(analyzer.get_top_thought_words(10), 1):
        print(f"  {i}. {word}: {count}")
    
    print("\nTop 10 action words:")
    for i, (word, count) in enumerate(analyzer.get_top_action_words(10), 1):
        print(f"  {i}. {word}: {count}")
    
    print(f"\n✓ Full reports saved to {output_dir}")


if __name__ == "__main__":
    main()
