from pathlib import Path


def test_public_release_has_no_private_markers():
    root = Path(__file__).parents[1]
    markers = (
        "/" + "Users/", "/" + "home/", "/" + "root/", "10." + "158.", "192." + "168.",
        "codehub" + ".devcloud", "huaweicloud", "chenyinle" + "@", "/" + "cache/",
        "BEGIN " + "OPENSSH PRIVATE KEY", "BEGIN " + "RSA PRIVATE KEY",
    )
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".zip"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path == Path(__file__):
            continue
        for marker in markers:
            assert marker not in text, f"private marker {marker!r} in {path}"
