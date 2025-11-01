"""
Utilities for generating training report cards. More messy code than usual, will fix.
"""

import datetime
import os
import platform
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Any, Self

import psutil
import torch

from nanochat.common import FilePath


def run_command(cmd: str) -> str | None:
    """Run a shell command and return output, or None if it fails."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as _e:
        return None


@dataclass
class GitInfo:
    commit: str
    branch: str
    dirty: bool
    message: str

    @classmethod
    def new(cls) -> Self:
        """Get current git commit, branch, and dirty status."""
        commit = run_command("git rev-parse --short HEAD") or "unknown"
        branch = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

        # Check if repo is dirty (has uncommitted changes)
        status = run_command("git status --porcelain")
        dirty = bool(status) if status is not None else False

        # Get commit message
        message = run_command("git log -1 --pretty=%B") or ""
        message = message.split("\n")[0][:80]  # First line, truncated

        return cls(
            commit=commit,
            branch=branch,
            dirty=dirty,
            message=message,
        )


@dataclass
class GpuInfo:
    available: bool
    count: int
    names: list[str] = field(default_factory=list)
    memory_gb: list[float] = field(default_factory=list)
    cuda_version: str = ""

    @classmethod
    def new(cls) -> Self:
        """Get GPU information."""
        if not torch.cuda.is_available():
            return cls(False, 0)

        num_devices = torch.cuda.device_count()
        info = cls(True, num_devices)

        for i in range(num_devices):
            props = torch.cuda.get_device_properties(i)
            info.names.append(props.name)
            info.memory_gb.append(props.total_memory / (1024**3))

        # Get CUDA version
        info.cuda_version = torch.version.cuda or "unknown"
        return info


@dataclass
class SysInfo:
    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cpu_count: int
    cpu_count_logical: int
    memory_gb: float
    user: str
    nanochat_base_dir: str
    working_dir: str

    @classmethod
    def new(cls) -> Self:
        return cls(
            hostname=socket.gethostname(),
            platform=platform.system(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cpu_count=psutil.cpu_count(logical=False) or 1,
            cpu_count_logical=psutil.cpu_count(logical=True) or 1,
            memory_gb=psutil.virtual_memory().total / (1024**3),
            user=os.environ.get("USER", "unknown"),
            nanochat_base_dir=os.environ.get("NANOCHAT_BASE_DIR", "out"),
            working_dir=os.getcwd(),
        )


def estimate_cost(gpu_info: GpuInfo, runtime_hours: float | None = None) -> dict[str, Any]:
    """Estimate training cost based on GPU type and runtime."""

    # Rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.available:
        return {}

    # Try to identify GPU type from name
    hourly_rate = None
    gpu_name = gpu_info.names[0] if gpu_info.names else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info.count
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info.count  # Default estimate

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None,
    }


def generate_header():
    """Generate the header for a training report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = GitInfo.new()
    gpu_info = GpuInfo.new()
    sys_info = SysInfo.new()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info.branch}
- Commit: {git_info.commit} {"(dirty)" if git_info.dirty else "(clean)"}
- Message: {git_info.message}

### Hardware
- Platform: {sys_info.platform}
- CPUs: {sys_info.cpu_count} cores ({sys_info.cpu_count_logical} logical)
- Memory: {sys_info.memory_gb:.1f} GB
"""

    if gpu_info.available:
        gpu_names = ", ".join(set(gpu_info.names))
        total_vram = sum(gpu_info.memory_gb)
        header += f"""- GPUs: {gpu_info.count} x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info.cuda_version}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate: ${cost_info["hourly_rate"]:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info.python_version}
- PyTorch: {sys_info.torch_version}

"""

    # bloat metrics: package all of the source code and assess its weight
    packaged = run_command(
        'files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml'
    )
    if packaged:
        num_chars = len(packaged)
        num_lines = len(packaged.split("\n"))
        num_files = len([x for x in packaged.split("\n") if x.startswith("<source>")])
        num_tokens = num_chars // 4  # assume approximately 4 chars per token

        # count dependencies via uv.lock
        uv_lock_lines = 0
        if os.path.exists("uv.lock"):
            with open("uv.lock", "r") as f:
                uv_lock_lines = len(f.readlines())

        header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header


# -----------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Slugify a text string."""
    return text.lower().replace(" ", "-")


# the expected files and their order
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "midtraining.md",
    "chat-evaluation-mid.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# the metrics we're currently interested in
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]


def extract(section: str, keys: str | list):
    """simple def to extract a single key from a section"""
    if not isinstance(keys, list):
        keys = [keys]  # convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out


def extract_timestamp(content: str, prefix: str) -> datetime.datetime | None:
    """Extract timestamp from content with given prefix."""
    for line in content.split("\n"):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except Exception as _:
                pass
    return None


class Report:
    """Maintains a bunch of logs, generates a final markdown report."""

    def __init__(self, report_dir: FilePath):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section: str, data: list[Any]):
        """Log a section of data to the report."""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # directly write the string
                    f.write(item)
                else:
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """Generate the final report."""
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"Generating report to {report_file}")
        final_metrics = {}  # the most important final metrics we'll add as table at the end
        start_time = None
        end_time = None
        with open(report_file, "w") as out_file:
            # write the header first
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # capture bloat data for summary later (the stuff after Bloat header and until \n\n)
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None  # will cause us to not write the total wall clock time
                bloat_data = "[bloat data missing]"
                print(
                    f"Warning: {header_file} does not exist. Did you forget to run `nanochat reset`?"
                )
            # process all the individual sections
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    continue
                with open(section_file, "r") as in_file:
                    section = in_file.read()
                # Extract timestamp from this section (the last section's timestamp will "stick" as end_time)
                if "rl" not in file_name:
                    # Skip RL sections for end_time calculation because RL is experimental
                    end_time = extract_timestamp(section, "timestamp:")
                # extract the most important metrics from the sections
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-mid.md":
                    final_metrics["mid"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K")  # RL only evals GSM8K
                # append this section of the report
                out_file.write(section)
                out_file.write("\n")
            # add the final metrics table
            out_file.write("## Summary\n\n")
            # Copy over the bloat metrics from the header
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # Collect all unique metric names
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # Custom ordering: CORE first, ChatCORE last, rest in middle
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            # Fixed column widths
            stages = ["base", "mid", "sft", "rl"]
            metric_width = 15
            value_width = 8
            # Write table header
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # Write separator
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # Write table rows
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        # also cp the report.md file to current directory
        print("Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """Reset the report."""
        # Remove section files
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove report.md if it exists
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # Generate and write the header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


class DummyReport:
    def log(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        pass


def get_report():
    # just for convenience, only rank 0 logs to report
    from nanochat.common import get_base_dir, get_dist_info

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate or reset nanochat training reports.")
    parser.add_argument(
        "command",
        nargs="?",
        default="generate",
        choices=["generate", "reset"],
        help="Operation to perform (default: generate)",
    )
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
