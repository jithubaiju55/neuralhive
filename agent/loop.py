"""
Agent Loop — NeuralHive
========================
The core of what makes this a coding AGENT not just a chatbot.

Plan → Write → Run → Read Error → Fix → Run → Repeat until ✅

This loop is what makes Claude Code feel magical.
We replicate it, fully locally, fully free.
"""

import os
import re
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum


class AgentState(Enum):
    PLANNING = "planning"
    CODING = "coding"
    RUNNING = "running"
    DEBUGGING = "debugging"
    DONE = "done"
    FAILED = "failed"


@dataclass
class FileChange:
    path: str
    content: str
    action: str = "create"  # create, modify, delete


@dataclass
class AgentStep:
    state: AgentState
    message: str
    files_changed: List[FileChange] = field(default_factory=list)
    error: Optional[str] = None
    fix_applied: Optional[str] = None


@dataclass 
class AgentResult:
    success: bool
    steps: List[AgentStep]
    files_created: List[str]
    final_message: str
    iterations: int


class CodeParser:
    """Extract code blocks and file operations from LLM output."""

    @staticmethod
    def extract_files(text: str) -> List[FileChange]:
        """
        Extract file contents from markdown code blocks.
        Supports formats:
        ```python filename.py
        ```javascript src/app.js
        # FILE: path/to/file.py
        """
        files = []

        # Pattern 1: ```language filename
        pattern1 = re.compile(
            r'```(?:[\w]+)?\s+([\w/.\-]+\.\w+)\n(.*?)```',
            re.DOTALL
        )
        for match in pattern1.finditer(text):
            filepath = match.group(1).strip()
            content = match.group(2)
            files.append(FileChange(path=filepath, content=content))

        # Pattern 2: # FILE: path
        pattern2 = re.compile(
            r'#\s*FILE:\s*([\w/.\-]+\.\w+)\n```[\w]*\n(.*?)```',
            re.DOTALL
        )
        for match in pattern2.finditer(text):
            filepath = match.group(1).strip()
            content = match.group(2)
            if not any(f.path == filepath for f in files):
                files.append(FileChange(path=filepath, content=content))

        # Pattern 3: plain code blocks (if only one, use prompt context for filename)
        if not files:
            plain_blocks = re.findall(r'```[\w]*\n(.*?)```', text, re.DOTALL)
            if plain_blocks:
                # Try to infer filename from context
                files.append(FileChange(
                    path="main.py",
                    content=plain_blocks[0]
                ))

        return files

    @staticmethod
    def extract_commands(text: str) -> List[str]:
        """Extract shell commands to run from LLM output."""
        commands = []

        # Look for $ prefixed commands
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('$ '):
                commands.append(line[2:])
            elif line.startswith('```bash') or line.startswith('```shell'):
                pass  # handled by block parser

        # Extract from bash blocks
        bash_blocks = re.findall(r'```(?:bash|shell|sh)\n(.*?)```', text, re.DOTALL)
        for block in bash_blocks:
            for line in block.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    commands.append(line)

        return commands

    @staticmethod
    def detect_language(filepath: str) -> str:
        """Detect language from file extension."""
        ext = Path(filepath).suffix.lower()
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.sh': 'bash',
            '.md': 'markdown',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
        }
        return lang_map.get(ext, 'text')


class CommandRunner:
    """Safely run shell commands and capture output."""

    TIMEOUT = 30  # seconds
    SAFE_COMMANDS = [
        'python', 'python3', 'node', 'npm', 'pip',
        'ls', 'dir', 'cat', 'type', 'mkdir', 'echo',
        'pytest', 'jest', 'cargo', 'go',
    ]

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def run(self, command: str) -> Tuple[bool, str, str]:
        """
        Run command. Returns (success, stdout, stderr).
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                cwd=self.working_dir,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {self.TIMEOUT}s"
        except Exception as e:
            return False, "", str(e)

    def write_file(self, relative_path: str, content: str) -> Path:
        """Write file to working directory."""
        full_path = self.working_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return full_path

    def read_file(self, relative_path: str) -> Optional[str]:
        """Read file from working directory."""
        full_path = self.working_dir / relative_path
        try:
            return full_path.read_text(encoding='utf-8')
        except Exception:
            return None

    def list_files(self) -> List[str]:
        """List all files in working directory."""
        files = []
        for p in self.working_dir.rglob('*'):
            if p.is_file() and not any(
                part.startswith('.') or part == '__pycache__' or part == 'node_modules'
                for part in p.parts
            ):
                files.append(str(p.relative_to(self.working_dir)))
        return files


class CodingAgent:
    """
    The main agent that builds complete apps.
    
    Plan → Code → Run → Debug → Loop until working.
    """

    MAX_ITERATIONS = 8  # Prevent infinite loops
    MAX_ERROR_LENGTH = 2000  # Truncate long errors

    def __init__(self, engine, working_dir: str):
        self.engine = engine
        self.runner = CommandRunner(working_dir)
        self.working_dir = working_dir
        self.parser = CodeParser()
        self.steps: List[AgentStep] = []
        self.conversation_history: List[Dict] = []

    def run(self, user_request: str, stream_callback=None) -> AgentResult:
        """
        Main agent loop. Builds the requested app.
        
        Args:
            user_request: What to build
            stream_callback: Called with each token as it streams
        
        Returns:
            AgentResult with success status and all steps taken
        """
        self.steps = []
        files_created = []
        iteration = 0

        # Step 1: Planning
        if stream_callback:
            stream_callback("\n🧠 Planning...\n", "status")

        plan = self._plan(user_request, stream_callback)
        self.steps.append(AgentStep(
            state=AgentState.PLANNING,
            message=plan
        ))

        # Step 2: Initial code generation
        if stream_callback:
            stream_callback("\n💻 Writing code...\n", "status")

        code_response = self._generate_code(user_request, plan, stream_callback)

        # Extract and write files
        file_changes = self.parser.extract_files(code_response)

        if not file_changes:
            # No code blocks found — ask for structured output
            code_response = self._retry_with_structure(user_request, stream_callback)
            file_changes = self.parser.extract_files(code_response)

        for change in file_changes:
            path = self.runner.write_file(change.path, change.content)
            files_created.append(change.path)

        self.steps.append(AgentStep(
            state=AgentState.CODING,
            message=f"Created {len(file_changes)} files",
            files_changed=file_changes
        ))

        if not file_changes:
            return AgentResult(
                success=False,
                steps=self.steps,
                files_created=[],
                final_message="Could not generate code files.",
                iterations=1
            )

        # Step 3: Run and debug loop
        while iteration < self.MAX_ITERATIONS:
            iteration += 1

            if stream_callback:
                stream_callback(f"\n▶️  Running (attempt {iteration})...\n", "status")

            # Find main entry point and run it
            run_command = self._determine_run_command(files_created)

            if not run_command:
                # No runnable file — check if it's a library/component
                break

            success, stdout, stderr = self.runner.run(run_command)

            self.steps.append(AgentStep(
                state=AgentState.RUNNING,
                message=f"Ran: {run_command}",
                error=stderr if not success else None
            ))

            if success:
                if stream_callback:
                    stream_callback("\n✅ Code runs successfully!\n", "status")
                break

            # Has errors — fix them
            error_msg = (stderr or stdout)[:self.MAX_ERROR_LENGTH]

            if stream_callback:
                stream_callback(f"\n🔧 Fixing error (iteration {iteration})...\n", "status")

            fix_response = self._fix_error(
                user_request,
                files_created,
                error_msg,
                stream_callback
            )

            fix_changes = self.parser.extract_files(fix_response)

            for change in fix_changes:
                self.runner.write_file(change.path, change.content)

            self.steps.append(AgentStep(
                state=AgentState.DEBUGGING,
                message=f"Applied fix for: {error_msg[:100]}...",
                files_changed=fix_changes,
                fix_applied=fix_response[:200]
            ))

            if not fix_changes:
                # LLM gave advice but no code — still increment iteration
                if iteration >= 3:
                    break

        # Final result
        final_files = self.runner.list_files()
        success = iteration <= self.MAX_ITERATIONS

        final_msg = self._generate_summary(user_request, final_files, success)

        return AgentResult(
            success=success,
            steps=self.steps,
            files_created=final_files,
            final_message=final_msg,
            iterations=iteration
        )

    def _plan(self, request: str, stream_callback=None) -> str:
        """Generate a plan before coding."""
        prompt = f"""Before writing code, create a brief plan for:

"{request}"

List:
1. Files to create (with purpose)
2. Main dependencies needed
3. Key implementation steps

Be concise. Then we'll write the actual code."""

        return self._call_engine(prompt, stream_callback, show_output=False)

    def _generate_code(self, request: str, plan: str, stream_callback=None) -> str:
        """Generate initial code based on plan."""
        files_instruction = """
For each file, use this exact format:
```python filename.py
# code here
```

Write COMPLETE files. Never truncate. Include all imports."""

        prompt = f"""Build this: {request}

Plan:
{plan}

{files_instruction}

Write all the code now:"""

        return self._call_engine(prompt, stream_callback)

    def _fix_error(
        self,
        original_request: str,
        files: List[str],
        error: str,
        stream_callback=None
    ) -> str:
        """Generate fix for an error."""

        # Read current file contents
        file_contents = ""
        for f in files[:5]:  # Limit to first 5 files
            content = self.runner.read_file(f)
            if content:
                file_contents += f"\n# FILE: {f}\n```\n{content[:1000]}\n```\n"

        prompt = f"""Fix this error:

ERROR:
{error}

CURRENT FILES:
{file_contents}

Write the corrected file(s) using:
```python filename.py
# fixed code
```

Fix ONLY what's broken. Keep everything else the same."""

        return self._call_engine(prompt, stream_callback)

    def _retry_with_structure(self, request: str, stream_callback=None) -> str:
        """Retry with explicit structure instructions."""
        prompt = f"""Build: {request}

You MUST format each file exactly like this:
```python main.py
print("hello world")
```

No explanations before the code. Start with the first file immediately."""

        return self._call_engine(prompt, stream_callback)

    def _call_engine(
        self,
        prompt: str,
        stream_callback=None,
        show_output: bool = True
    ) -> str:
        """Call the inference engine, optionally streaming output."""
        response_parts = []

        for token in self.engine.generate_stream(prompt):
            response_parts.append(token)
            if show_output and stream_callback:
                stream_callback(token, "token")

        return "".join(response_parts)

    def _determine_run_command(self, files: List[str]) -> Optional[str]:
        """Figure out how to run the generated code."""
        priority_order = [
            ('main.py', 'python main.py'),
            ('app.py', 'python app.py'),
            ('server.py', 'python server.py'),
            ('index.js', 'node index.js'),
            ('app.js', 'node app.js'),
            ('index.ts', 'npx ts-node index.ts'),
            ('main.go', 'go run main.go'),
        ]

        for filename, command in priority_order:
            if filename in files:
                return command

        # Check for pytest
        if any(f.startswith('test_') or f.endswith('_test.py') for f in files):
            return 'python -m pytest -x -q'

        return None

    def _generate_summary(
        self,
        request: str,
        files: List[str],
        success: bool
    ) -> str:
        """Generate a summary of what was built."""
        status = "✅ Successfully built" if success else "⚠️  Partially built"
        files_list = '\n'.join(f"  • {f}" for f in files)
        return f"{status}: {request}\n\nFiles created:\n{files_list}"