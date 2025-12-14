set shell := ["bash", "-cu"]

default: build

install:
    git lfs install --skip-smudge
    uv sync
    uv pip install -e .

install-linux:
    sudo apt update
    sudo apt install -y libgtk2.0-dev pkg-config libcanberra-gtk-module libcanberra-gtk3-module
    just install

install-mac:
    brew install git-lfs
    just install

add package:
    uv add {{package}} --index-strategy unsafe-best-match

build:
    just format
    just lint-fix
    just test

test:
    uv run bash -c "PYTHONPATH=src pytest -q"

format:
    uv run black src tests

check-format:
    uv run black --check src tests

lint:
    uv run ruff check --select F,I --fix --unsafe-fixes src tests

lint-fix:
    uv run -- ruff check --fix src tests

run mode experiment:
    uv run --env-file .env python -m run.run --mode {{mode}} --experiment {{experiment}}
