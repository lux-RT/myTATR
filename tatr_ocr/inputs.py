"""Resolve local inputs without overwriting batch metadata or another image."""


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
