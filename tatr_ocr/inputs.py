"""Resolve local inputs without overwriting batch metadata or another image."""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_images(explicit, directory):
    """Explicit files override the default, script-relative sample directory."""
    if explicit is not None:
        return [path.resolve() for path in explicit]
    if not directory.is_dir():
        raise ValueError(f"Missing image directory: {directory}")
    paths = sorted((p.resolve() for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=lambda path: path.name.casefold())
    if not paths:
        raise ValueError(f"No supported images found in directory: {directory}")
    return paths


def validate_stems(paths):
    stems = [p.stem.casefold() for p in paths]
    if any(s in {"summary", "ocr"} for s in stems):
        raise ValueError("Input filenames summary and ocr are reserved for batch metadata")
    if len(set(stems)) != len(stems):
        raise ValueError("Input image filenames must have unique stems, ignoring case")


def default_model_dir(repo):
    local = repo / "models/table-transformer-structure-recognition"
    sibling = repo.parent / "PubTables-1M复现/models/table-transformer-structure-recognition"
    required = ("model.safetensors", "config.json", "preprocessor_config.json")
    for candidate in (local, sibling):
        if all((candidate / name).is_file() for name in required):
            return candidate
    return local
