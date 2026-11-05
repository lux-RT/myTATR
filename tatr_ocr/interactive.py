"""Terminal image selection; pasted paths are data, never shell commands."""
from pathlib import Path
import re
import shlex

from .inputs import validate_stems

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PAGE_SIZE = 20


def parse_selection(text, count):
    indices = []
    for part in re.split(r"[,，、\s]+", text.strip()):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise ValueError("请输入图片编号，例如 1,3 或 2-5。")
        ends = [int(n) for n in part.split("-")]
        start, end = ends[0], ends[-1]
        if not 1 <= start <= end <= count:
            raise ValueError(f"编号范围为 1 到 {count}，区间必须从小到大。")
        indices.extend(range(start - 1, end))
    return list(dict.fromkeys(indices))


def parse_paths(text):
    text = text.strip()
    # Some Windows terminals prefix a dragged, quoted path with '& '.
    if text.startswith("& "):
        text = text[2:].lstrip()
    unquoted = text[1:-1] if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'" else text
    candidate = Path(unquoted).expanduser()
    if candidate.exists():
        return [candidate]
    try:
        parts = shlex.split(text, posix=False)
    except ValueError as error:
        raise ValueError("路径引号不完整，请重新输入。") from error
    paths = []
    for part in parts:
        if len(part) > 1 and part[0] == part[-1] and part[0] in "\"'":
            part = part[1:-1]
        paths.append(Path(part).expanduser())
    return paths


def choose_from_list(paths, read=input, write=print):
    if not paths:
        write("没有找到支持的图片。")
        return []
    page = 0
    while True:
        start = page * PAGE_SIZE
        visible = paths[start:start + PAGE_SIZE]
        write(f"\n共 {len(paths)} 张图片，第 {page + 1}/{(len(paths) - 1) // PAGE_SIZE + 1} 页：")
        for index, path in enumerate(visible, start + 1):
            write(f"  {index}. {path.name}")
        text = read("编号（1,3 或 2-5）；回车选本页；n/p 翻页；all 选全部；q 取消：").strip().lower()
        if text in {"q", "quit"}:
            return []
        if text == "n":
            page = min(page + 1, (len(paths) - 1) // PAGE_SIZE)
            continue
        if text == "p":
            page = max(page - 1, 0)
            continue
        if text == "all":
            return list(paths)
        if not text:
            return visible
        try:
            return [paths[i] for i in parse_selection(text, len(paths))]
        except ValueError as error:
            write(str(error))


def select_images(samples, read=input, write=print):
    """Return selected paths, or [] on q, EOF or Ctrl+C, without loading models."""
    try:
        while True:
            write("\nTATR + OCR 图片测试\n  1. 选择示例图片\n  2. 输入/拖入图片或文件夹路径\n  q. 退出")
            choice = read("请选择 [1]：").strip().lower() or "1"
            if choice in {"q", "quit"}:
                return []
            if choice == "1":
                selected = choose_from_list([p for p in samples if p.is_file()], read, write)
            elif choice == "2":
                while True:
                    text = read("图片或文件夹路径（多个路径用引号分开，q 取消）：").strip()
                    if text.lower() in {"q", "quit"}:
                        return []
                    try:
                        paths = parse_paths(text)
                        if len(paths) == 1 and paths[0].is_dir():
                            candidates = sorted((p for p in paths[0].iterdir()
                                                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                                                key=lambda p: p.name.casefold())
                            selected = choose_from_list(candidates, read, write)
                        else:
                            if not paths or any(not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS for p in paths):
                                raise ValueError("路径不存在或不是支持的图片，请检查后重新输入。")
                            selected = paths
                        break
                    except (ValueError, OSError) as error:
                        write(str(error))
            else:
                write("请输入 1、2 或 q。")
                continue
            if not selected:
                return []
            try:
                validate_stems(selected)
            except ValueError as error:
                write(f"图片名称冲突：{error}，请重新选择。")
                continue
            write(f"\n已选择 {len(selected)} 张图片：")
            for path in selected:
                write(f"  {path}")
            return selected
    except (EOFError, KeyboardInterrupt):
        return []
