"""Download only the pinned Transformers TATR structure checkpoint."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "models/table-transformer-structure-recognition")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    revision = "f4d4bdc85c3fe4b1fa49658882a5d38bbdd0f343"
    expected = {"config.json": "cc00bc06445308c84b9be145174a0eadc6b44523d721d910e94efaf52436096e",
                "preprocessor_config.json": "ea8e4375cbaff4fdafe42b22590e72ff7eb89b6d235f59cdeb746d5de7987f3e",
                "model.safetensors": "6a3494f8d341fc16700d0dfd7e321d3c7344e5496ba5d7b4b0dd3e95e89d1506"}
    for name, digest in expected.items():
        path = Path(hf_hub_download("microsoft/table-transformer-structure-recognition", name,
                                  revision=revision, local_dir=args.output))
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise RuntimeError(f"Checksum mismatch for {name}")
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
