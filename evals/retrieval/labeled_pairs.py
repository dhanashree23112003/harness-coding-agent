"""Labeled (goal_text, correct_tool_name) pairs for retrieval recall@k eval.

115 pairs covering all 54 tools, with multiple phrasings per tool to test
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

    # ── ast namespace (18 pairs) ──────────────────────────────────────────────

    # ast.parse_module (2)
    ("parse the AST of utils.py and check for syntax errors", "parse_module"),
    ("check if models.py is syntactically valid Python", "parse_module"),

    # ast.list_symbols (2)
    ("list all functions and classes defined in auth.py", "list_symbols"),
    ("what symbols are exported from the graph module", "list_symbols"),

    # ast.find_definition (2)
    ("find where the function build_graph is defined in the source", "find_definition"),
    ("locate the class definition for AgentState in the codebase", "find_definition"),

    # ast.find_references (3)
    ("find all call sites of the validate_input function", "find_references"),
    ("where is read_file referenced across the project", "find_references"),
    ("which files call the build_graph function", "find_references"),

    # ast.list_imports (2)
    ("list all imports in the client.py file", "list_imports"),
    ("what modules does nodes.py import", "list_imports"),

    # ast.compute_complexity (2)
    ("compute the cyclomatic complexity of all functions in graph.py", "compute_complexity"),
    ("which functions in retriever.py are too complex and hard to maintain", "compute_complexity"),

    # ast.detect_dead_code (2)
    ("find dead code and unused functions in utils.py", "detect_dead_code"),
    ("which functions in helpers.py are defined but never called", "detect_dead_code"),

    # ast.extract_function_signature (2)
    ("get the signature of the retrieve function including its return type", "extract_function_signature"),
    ("extract the parameter list and return annotation for embed_batch", "extract_function_signature"),

    # ast.find_unused_imports (2)
    ("find all unused imports in main.py that can be removed", "find_unused_imports"),
    ("which imported modules in store.py are never actually used", "find_unused_imports"),

    # ── test namespace (16 pairs) ─────────────────────────────────────────────

    # test.discover_tests (2)
    ("discover all test cases in the tests directory", "discover_tests"),
    ("list all pytest test nodes that will run for the changed files", "discover_tests"),

    # test.run_test_file (2)
    ("run all tests in tests/unit/test_fs_models.py", "run_test_file"),
    ("execute the test file test_registry.py and show pass/fail results", "run_test_file"),

    # test.run_test_node (2)
    ("run the single test test_read_file_output_fields", "run_test_node"),
    ("execute only the specific failing test test_git_status_output_clean", "run_test_node"),

    # test.run_suite (3)
    ("run the full test suite and report overall results", "run_suite"),
    ("execute all unit tests and collect pass/fail counts", "run_suite"),
    ("run the entire pytest suite for the project", "run_suite"),

    # test.coverage_report (2)
    ("generate a coverage report for the src directory", "coverage_report"),
    ("what is the test coverage percentage for the retrieval module", "coverage_report"),

    # test.coverage_diff (2)
    ("show how coverage changed compared to the previous commit", "coverage_diff"),
    ("did the latest changes increase or decrease test coverage", "coverage_diff"),

    # test.last_failures (2)
    ("show the tests that failed in the last pytest run", "last_failures"),
    ("get the list of tests that were failing previously", "last_failures"),

    # test.rerun_failed (2)
    ("rerun only the tests that failed last time to check if they pass now", "rerun_failed"),
    ("run pytest on the previously failing tests", "rerun_failed"),

    # ── deps namespace (14 pairs) ─────────────────────────────────────────────

    # deps.list_dependencies (2)
    ("list all project dependencies from pyproject.toml", "list_dependencies"),
    ("what packages does this project depend on", "list_dependencies"),

    # deps.check_outdated (2)
    ("check which installed packages have newer versions available", "check_outdated"),
    ("list all outdated dependencies that need updating in this project", "check_outdated"),

    # deps.resolve_import (2)
    ("is the psycopg module installed and which package provides it", "resolve_import"),
    ("resolve whether numpy is a stdlib or third-party package", "resolve_import"),

    # deps.find_unused_deps (2)
    ("find dependencies declared in pyproject.toml that are never imported", "find_unused_deps"),
    ("which packages in the dependency list are not actually used in the code", "find_unused_deps"),

    # deps.dependency_graph (2)
    ("show the dependency graph for this project", "dependency_graph"),
    ("what packages does langchain-groq transitively depend on", "dependency_graph"),

    # deps.vulnerability_scan (3)
    ("scan the installed packages for known security vulnerabilities", "vulnerability_scan"),
    ("check if any project dependencies have CVEs", "vulnerability_scan"),
    ("find security advisories affecting this project's dependencies", "vulnerability_scan"),

    # deps.add_dependency (2)
    ("add httpx as a dependency to the project and install it", "add_dependency"),
    ("register and install the tomli-w package as a project dependency", "add_dependency"),

    # ── ci namespace (14 pairs) ───────────────────────────────────────────────

    # ci.run_linter (2)
    ("run ruff linter on the src directory and report violations", "run_linter"),
    ("check src/agent for lint errors and style violations using ruff", "run_linter"),

    # ci.run_formatter (2)
    ("check if all Python files in src are correctly formatted", "run_formatter"),
    ("run ruff format in check mode and report files that need reformatting", "run_formatter"),

    # ci.run_type_check (2)
    ("run mypy type checking on the agent package", "run_type_check"),
    ("check for type errors in src/agent using static type analysis", "run_type_check"),

    # ci.build_check (2)
    ("verify that the package builds cleanly without errors", "build_check"),
    ("run a build check to make sure the wheel can be produced successfully", "build_check"),

    # ci.pre_commit_run (2)
    ("run all pre-commit hooks against the staged files", "pre_commit_run"),
    ("execute the pre-commit checks to ensure code quality before committing", "pre_commit_run"),

    # ci.run_security_scan (2)
    ("run bandit security scan on the server code", "run_security_scan"),
    ("scan src/agent for security vulnerabilities using static analysis", "run_security_scan"),

    # ci.summarize_quality (2)
    ("run all quality checks and give me a summary of lint, types, and security", "summarize_quality"),
    ("what is the overall code quality status of this project", "summarize_quality"),
]

assert len(PAIRS) == 115, f"Expected 115 pairs, got {len(PAIRS)}"
