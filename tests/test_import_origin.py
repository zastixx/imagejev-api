from pathlib import Path

import imagejev


def test_import_comes_from_canonical_source_tree():
    package_path = Path(imagejev.__file__).resolve()
    expected_root = Path(__file__).resolve().parents[1] / "src" / "imagejev"
    assert package_path.parent == expected_root
