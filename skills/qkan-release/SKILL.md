---
name: "qkan-release"
title: "QKAN Release Runbook"
description: "Maintainer note for releasing qkan: hash-stamped dev versions, the branch-then-tag release flow via the release-prep workflow, and publish behavior. For the repo owner — not end-user onboarding."
version: "1.0.0"
author: "Jiun-Cheng Jiang <jcjiang@phys.ntu.edu.tw>"
tags: [qkan, release, versioning, maintainer]
tools: [Read, Glob, Grep]
license: "Apache-2.0"
---

## QKAN Release Runbook

Maintainer note. End users never need this — route them to `qkan-guide`.

## Versioning scheme

`src/qkan/__init__.py` carries `__version__` (e.g. `"0.2.3dev"`); `setup.py`
derives the build version from it:

| Build context | Version |
|---|---|
| Dev clone | `0.2.3.dev0+g<short-hash>` (source commit of the build) |
| Dev CUDA wheel (`QKAN_LOCAL_VERSION` set) | `0.2.3.dev0+cu130torch2.11.g<hash>` |
| Dev sdist / tarball without `.git` / `QKAN_NO_GIT_VERSION=TRUE` | `0.2.3.dev0` |
| Release (marker stripped) | `0.2.3`, wheels `0.2.3+cuXX...` |

Rules baked into `setup.py::get_git_revision`:

- The hash applies only while the base version ends in `dev`, and only when
  this tree has its own `.git` (git discovery would otherwise walk up to an
  enclosing repo and stamp the wrong hash).
- sdists never get the hash — the tarball rebuilds without `.git`, and a
  hash in sdist metadata would mismatch the wheel pip builds from it (pip
  rejects the inconsistency).
- Failsafe: PyPI rejects local versions (`+...`), so tagging a commit that
  still carries the dev marker fails loudly at upload instead of silently
  publishing a dev build.

## Release method: branch-then-tag

`master` never leaves dev state. Releases live on disposable
`release/vX.Y.Z` branches; the tag — not any branch merge — triggers
`publish.yml`.

```text
master:            ... B (0.2.3dev+g...) ─── next: 0.2.4dev
                        \
release/v0.2.3:          C (__version__ = "0.2.3")   <- tag v0.2.3 here
```

Runbook:

1. Dispatch the workflow from master:
   `gh workflow run release-prep -f version=0.2.3 --ref master`
   (Actions -> release-prep -> Run workflow). It refuses to run unless
   dispatched from master AND `__version__` is exactly `"0.2.3dev"`, then
   pushes `release/v0.2.3` with the marker stripped.
2. Review the branch — it must differ from master by exactly one line:
   `git diff master..origin/release/v0.2.3`
3. Tag and push the tag YOURSELF (the exact commands are printed in the
   workflow run summary):
   `git tag v0.2.3 origin/release/v0.2.3 && git push origin v0.2.3`
   This is deliberately manual — a tag pushed by the workflow's
   GITHUB_TOKEN would NOT trigger `publish.yml`.
4. `publish.yml` builds the clean `0.2.3` sdist for PyPI and the CUDA
   wheel matrix onto the GitHub Release, and rebuilds the gh-pages wheel
   index.
5. Reopen development on master with an ordinary commit:
   `__version__ = "0.2.4dev"`. Do NOT merge the release branch back.
6. Optional: delete `release/v0.2.3` — the tag keeps the commit reachable
   forever.

## Release description

`publish.yml` creates the GitHub Release with `--generate-notes`
(auto-generated from merged PRs). To author the description yourself,
edit the release body in the GitHub UI after creation — or switch the
create step to `--draft` so nothing is public until you write the notes
and click Publish (wheel-index links go live at that moment). For
commit-derived notes reviewed inside the release branch, wire git-cliff
into release-prep (conventional-commit history supports it; not set up).

## Cautions

- `git describe` on master never sees release tags (they are not in
  master's ancestry) — do not build tooling on tag distance.
- The version input of release-prep reaches scripts via `env` binding;
  keep it that way — inline `${{ }}` interpolation in `run:` blocks is
  command-injectable before any validation executes.
- Calibration for the wheel matrix (torch/CUDA versions) lives in
  `publish.yml`; update it alongside torch releases.
