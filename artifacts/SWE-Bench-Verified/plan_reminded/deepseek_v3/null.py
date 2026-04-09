import os
import sys
import json

def check_null_patches(path):
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, f"{folder_name}.pred")
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get('model_patch') is None:
                        traj_file = os.path.join(folder_path, f"{folder_name}.traj")
                        has_git_commit = False
                        git_commit_count = False
                        with open(traj_file, 'r') as tf:
                            content = tf.read()
                            has_git_commit = "git commit -m" in content
                            git_commit_count = content.count("git commit -m")
                        print(f"{folder_name}: git_commit={has_git_commit, git_commit_count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        sys.exit(1)
    check_null_patches(sys.argv[1])
