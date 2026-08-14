import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
import requests
from tqdm import tqdm

ROOT_DIR = Path(os.getenv("ROOT_DIR", default=Path.cwd()))

# if os.environ.get('DISPLAY', '') == '':
#     print('No display found. Using non-interactive Agg backend.')
#     matplotlib.use('Agg')
# else:
#     try:
#         matplotlib.use('Qt5Agg')
#     except ImportError:
#         print('Qt5Agg not found. Falling back to Agg.')
#         matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# plt.ion()



def read_tbl_into_strings(path: Path, file_line_subsample: int | None = None) -> List[str]:
    """
    Reads a .tbl file (or any text file) and returns a list of strings,
    one per row (line), without the trailing newline.
    """
    lines: List[str] = []
    count = 0

    if file_line_subsample is None:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                # strip only the newline; preserve any other whitespace
                lines.append(line.rstrip('\n'))

    else:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                # strip only the newline; preserve any other whitespace
                lines.append(line.rstrip('\n'))
                count += 1

                if count >= file_line_subsample:
                    break

    return lines


def form_syscall_dict() -> dict:
    file_path = ROOT_DIR / "data/syscall_64.tbl"

    if not file_path.exists() or not file_path.is_file():
        raw_url = "https://raw.githubusercontent.com/torvalds/linux/refs/heads/master/arch/x86/entry/syscalls/syscall_64.tbl"
        dest = ROOT_DIR / "data/syscall_64.tbl"

        resp = requests.get(raw_url)
        resp.raise_for_status()  # will raise HTTPError for bad status
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        print(f"Downloaded {raw_url!r} → {dest}")

    rows_as_strings = read_tbl_into_strings(file_path)

    filtered = [s for s in rows_as_strings
                if s.strip() and not s.lstrip().startswith('#')]
    filtered = [s for s in filtered if "x32" not in s]

    syscall_dict = {}

    for row in filtered:
        parts = row.split()
        if len(parts) >= 4:
            syscall_dict[parts[3]] = int(parts[0])

        elif len(parts) == 3:
            syscall_dict[parts[2]] = int(parts[0])

        else:
            raise ValueError(f"Invalid syscall {row}")

    return syscall_dict


def write_out_syscalls(syscall_dict: dict, syscall_lines: list, output_file_path: str) -> None:
    filtered = [s.replace("_enter_", "_") for s in syscall_lines if "_exit_" not in s]
    filtered = [s for s in filtered if "monitor_syscall" not in s]
    filtered = [s for s in filtered if "trace-cmd" not in s]

    syscall_ints = []
    syscall_time = []

    for row in filtered:
        parts = row.split()

        syscall_text = parts[3][:-1]
        syscall_ts = parts[2][:-1]

        try:
            syscall_ints.append(syscall_dict[syscall_text])
            syscall_time.append(float(syscall_ts))
        except KeyError:
            substr = "sys_"
            index, first = next(((i, s) for i, s in enumerate(parts) if substr in s), (None, None))

            if first:
                first = first[:-1]
                ts = parts[index - 1][:-1]
                # if first == '112882.623291':
                #     print("hi")

                syscall_ints.append(syscall_dict[first])
                syscall_time.append(float(ts))

            else:
                print(f"Invalid syscall {row}")
                continue
                # raise ValueError(f"Invalid syscall: {row}")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(" ".join(map(str, syscall_ints)) + "\n")
        f.write(" ".join(map(str, syscall_time)))

    return


def find_non_txt_files(root: Path = Path.cwd()) -> list[Path]:
    """Return all files under `root` (recursively) that do NOT have a .txt extension."""
    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() != '.txt']


def process_one_file(input_file_path: Path, syscall_dict: dict, file_line_subsample: int | None = None) -> None:
    """
    Read a text file -> parse into a NumPy array -> transform -> write .txt.
    Does not return anything; writes to disk.
    """
    try:
        output_file_path = input_file_path.with_name(input_file_path.name + "_ints.txt")

        syscall_lines = read_tbl_into_strings(input_file_path, file_line_subsample)

        # Skip headers like "cpus=" or "CPU X is empty"
        idx = 0
        for i, line in enumerate(syscall_lines):
            if "sys_enter" in line or "sys_exit" in line:
                idx = i
                break
        
        syscall_lines = syscall_lines[idx:]
        write_out_syscalls(syscall_dict, syscall_lines, output_file_path)

        # TODO START HERE
        #  --------------
        #  --------------

    except Exception as e:
        # Bubble up with file context to see which file failed
        raise RuntimeError(f"Failed on {input_file_path}") from e

    return


def process_files_in_parallel(files, syscall_dict: dict, n_workers: int | None = None,
                              file_line_subsample: int | None = None) -> None:
    """
    Process each file in parallel using up to n_workers processes.
    files: iterable of paths (str or Path) to input .txt files
    out_dir: directory to write outputs
    """
    paths = [Path(p) for p in files]

    n = n_workers or (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n) as ex:
        futures = {ex.submit(process_one_file, p, syscall_dict, file_line_subsample): p for p in paths}
        for fut in as_completed(futures):
            p = futures[fut]  # the input file for this future
            try:
                fut.result()  # raises if the worker failed
                print(f"OK: {p}")
            except Exception as e:
                print(f"FAILED: {p} → {e}")

    return


if __name__ == "__main__":
    TRANSLATE_SYSCALL_FILES = True
    SPECIFY_FILES = False
    DATA_DIR = ROOT_DIR / "data/current_data/idle/syscall_output"
    if len(sys.argv) > 1:
        DATA_DIR = Path(sys.argv[1])
        if not DATA_DIR.is_absolute():
            DATA_DIR = Path.cwd() / DATA_DIR

    if TRANSLATE_SYSCALL_FILES:
        syscall_dict = form_syscall_dict()

        if SPECIFY_FILES:
            file_list = [
                "ftrace/idle_20_trace_system_timed",
                "ftrace/AES_O_exfil_aws1_system_timed",
                "ftrace/AES_O_exfil_aws2_system_timed",
                "ftrace/AES_O_exfil_sftp1_system_timed",
                "ftrace/AES_O_exfil_sftp2_system_timed",
                "ftrace/gzip_system_timed",
            ]
        else:
            file_list = find_non_txt_files(DATA_DIR)

        print(f"Processing files in {DATA_DIR}...")
        paths = [Path(p) for p in file_list]
        for p in tqdm(paths):
            process_one_file(p, syscall_dict)

    






