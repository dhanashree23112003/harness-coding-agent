"""Labeled (goal_text, correct_tool_name) pairs for retrieval recall@k eval.

50 pairs covering all 23 tools, with multiple phrasings per tool to test
semantic robustness. 'correct_tool_name' matches the .name on the LangChain
Tool object (the FastMCP function name, without namespace prefix).
"""

PAIRS: list[tuple[str, str]] = [
    # fs.read_file (3 phrasings)
    ("read the contents of main.py", "read_file"),
    ("show me what is inside config.py", "read_file"),
    ("open and display the file README.md", "read_file"),

    # fs.read_file_range (2)
    ("read lines 10 to 50 of server.py", "read_file_range"),
    ("show only lines 100 through 120 of the log file", "read_file_range"),

    # fs.write_file (2)
    ("write the updated content to output.py", "write_file"),
    ("save the new implementation into utils.py", "write_file"),

    # fs.list_dir (2)
    ("list all files in the src directory", "list_dir"),
    ("show the contents of the tests folder", "list_dir"),

    # fs.search_files (2)
    ("find all Python files in the project", "search_files"),
    ("search for files matching the pattern *.toml", "search_files"),

    # fs.grep (3)
    ("search for the string 'import os' across the codebase", "grep"),
    ("find all occurrences of TODO comments in source files", "grep"),
    ("grep for the function name 'build_graph' in Python files", "grep"),

    # fs.file_stat (2)
    ("check the size and last modified time of main.py", "file_stat"),
    ("get metadata for the file config.json", "file_stat"),

    # fs.make_dir (2)
    ("create a new directory called logs", "make_dir"),
    ("make the folder src/utils so it can hold helper modules", "make_dir"),

    # fs.move (2)
    ("move old_utils.py to archive/old_utils.py", "move"),
    ("rename legacy_auth.py to auth.py by moving it to the same folder", "move"),

    # fs.delete (2)
    ("delete the temporary file temp_output.txt", "delete"),
    ("remove the generated build artifact at dist/bundle.js", "delete"),

    # fs.copy (2)
    ("copy config.example.json to config.json", "copy"),
    ("duplicate .env.template into .env for local development", "copy"),

    # git.git_status (3)
    ("check the current git status and see what files are staged", "git_status"),
    ("what files have been modified but not committed", "git_status"),
    ("show staged and unstaged changes in the working tree", "git_status"),

    # git.git_diff (2)
    ("show the diff of all staged changes", "git_diff"),
    ("what are the exact line-level changes in the uncommitted work", "git_diff"),

    # git.git_log (2)
    ("show the git log for the last 10 commits", "git_log"),
    ("display the commit history with authors and dates", "git_log"),

    # git.git_blame (2)
    ("find out who last changed each line in auth.py", "git_blame"),
    ("show git blame for the file router.py to see commit history per line", "git_blame"),

    # git.branch_create (2)
    ("create a new branch called feature/auth", "branch_create"),
    ("make a new git branch named fix/null-pointer from main", "branch_create"),

    # git.branch_list (2)
    ("list all git branches including remote tracking branches", "branch_list"),
    ("which branches exist in this repository", "branch_list"),

    # git.git_checkout (2)
    ("switch to the branch develop", "git_checkout"),
    ("check out the main branch to start working from the latest stable state", "git_checkout"),

    # git.git_commit (2)
    ("commit the staged changes with the message 'add input validation'", "git_commit"),
    ("create a git commit for all staged files", "git_commit"),

    # git.git_stash (2)
    ("stash my current uncommitted changes", "git_stash"),
    ("temporarily shelve my work in progress so I can switch branches cleanly", "git_stash"),

    # git.show_commit (2)
    ("show the full details of commit abc123", "show_commit"),
    ("display the diff and message for a specific commit hash", "show_commit"),

    # git.list_changed_files (2)
    ("which files changed between HEAD and the previous commit", "list_changed_files"),
    ("get a list of files modified in the last commit", "list_changed_files"),

    # git.git_tag (3)
    ("create a git tag v1.0.0 on the current commit", "git_tag"),
    ("annotate this release as v2.0.0-rc1 with a tag", "git_tag"),
    ("mark the current HEAD with a version tag for the release", "git_tag"),
]

assert len(PAIRS) == 50, f"Expected 50 pairs, got {len(PAIRS)}"
