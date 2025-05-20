import concurrent.futures
import fnmatch
import os
from typing import List, Tuple

from .constant import INTERMEDIATE_DATA_LOC, DIR_BLACKLIST, FILE_BLACKLIST
from .logs import logger


def is_valid_key_in_dict(i: dict, key):
    return key in i and i.get(key) is not None


def is_valid_string(i):
    return i is not None and isinstance(i, str) and len(i) != 0


def save_intermediate_result(filename: str, id: str, content: str):
    if not os.path.exists(INTERMEDIATE_DATA_LOC):
        os.mkdir(INTERMEDIATE_DATA_LOC)

    formatted_filename = f"{INTERMEDIATE_DATA_LOC}/{filename}_{id}.data"
    with open(formatted_filename, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug(f"Successfully saved intermediate result, name: {filename}_{id}.data")


def load_intermediate_result(filename: str, id: str) -> str:
    if not os.path.exists(INTERMEDIATE_DATA_LOC) or not os.path.exists(f"{INTERMEDIATE_DATA_LOC}/{filename}_{id}.data"):
        raise FileNotFoundError("Intermediate result file not found")

    content = open(f"{INTERMEDIATE_DATA_LOC}/{filename}_{id}.data", "r", encoding="utf-8").read()
    logger.debug(f"Successfully loaded intermediate result, name: {filename}_{id}.data")
    return content


def tree_of_dir(
        directory, level=0,
        dir_blacklist=DIR_BLACKLIST,
        file_blacklist=FILE_BLACKLIST,
) -> Tuple[str, List[str]]:
    """Get a tree structure of specified directory."""

    def only_a_subfolder(d: str):
        content = os.listdir(d)
        return len(content) == 1 and os.path.isdir(os.path.join(d, content[0]))

    ret = ""
    ret_files = []

    contents = os.listdir(directory)
    files = [each for each in contents if os.path.isfile(os.path.join(directory, each))]
    dirs = [each for each in contents if os.path.isdir(os.path.join(directory, each))]

    # Print files first
    for f in files:
        if True in [fnmatch.fnmatch(f, e) for e in file_blacklist]:
            continue

        ret += "{}- [FILE] {}\n".format(level * '--', f)
        ret_files.append(os.path.join(directory, f))

    # Then print directories
    for d in dirs:
        if True in [fnmatch.fnmatch(d, e) for e in dir_blacklist]:
            continue

        full_path = os.path.join(directory, d)
        while only_a_subfolder(full_path):
            folder = os.listdir(full_path)[0]
            full_path += f"/{folder}"

        ret += "{}- [DIR] {}/\n".format(
            level * '--',
            full_path.removeprefix(directory + "/" if not directory.endswith("/") else directory))

        ret_str, sub_files = tree_of_dir(full_path, level + 1)
        ret += ret_str
        ret_files.extend(sub_files)

    return ret, ret_files


def tree_with_root_dir_name(directory: str, root: str) -> Tuple[str, List[str]]:
    dir_str, dir_files = tree_of_dir(directory)
    dir_str = f"Tree of directory `{root}`\n{dir_str}"
    return dir_str, dir_files


def relative_path(root: str, path: str) -> str:
    root = root.rstrip("/")
    return "." + path.removeprefix(root)


def absolute_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(os.path.join(root, path))


def multi_thread(func, data, arg_name_of_data: str, thread_cnt: int, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_cnt) as executor:
        future = [
            executor.submit(func, **{arg_name_of_data: item}, **kwargs)
            for item in data
        ]

        _ret = []
        for future in concurrent.futures.as_completed(future):
            result = future.result()
            if result is not None:
                _ret.append(result)
        return _ret
