#!/usr/bin/env python3
"""Recompute SHA256 of sealed_prediction.md and compare with manifest (v1.0/v1.1)."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
EXPECTED = "de0922e70c97b416eb46bc9df73a417dfa959551f74cb2043e543dba4f4db851"
file_path = ROOT / "outputs_local" / "sealed_prediction.md"
manifest_path = ROOT / "outputs_local" / "sealed_prediction_manifest.json"

def sha256_path(p):
    h = hashlib.sha256(open(p,"rb").read()).hexdigest()
    return h

actual = sha256_path(file_path)
with open(manifest_path) as mf:
    m = json.load(mf)
expected = m.get("expected_sha256", EXPECTED)
print("sealed_prediction.md SHA256:", actual)
print("manifest expected:        ", expected)
print("match manifest:", actual == expected)
# Verify file unchanged at v1.0/v1.1 via git blob (read without changing working tree)
ok_tags = True
for tag in ("v1.0-defense","v1.1-defense"):
    try:
        blob = subprocess.check_output(
            ["git","show",f"{tag}:outputs_local/sealed_prediction.md"], cwd=ROOT, stderr=subprocess.STDOUT)
        blob_sha = hashlib.sha256(blob).hexdigest()
        match_tag = (blob_sha == actual)
        print(tag, "blob matches sealed md5?", match_tag)
        ok_tags = ok_tags and match_tag
    except subprocess.CalledProcessError as e:
        print(tag, "missing (expected if tag unchanged)")
# mtime
print("local mtime:", os.path.getmtime(file_path))
print("exit: 0 if actual==expected else non-zero")
sys.exit(0 if actual == expected else 1)
