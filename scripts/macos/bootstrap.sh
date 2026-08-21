#!/usr/bin/env bash

bootstrap_check() {
  require_cmd python3 "Install Python 3.10+ via Homebrew: brew install python"
  node_check
}

bootstrap_steps() {
  python_venv_steps python3 "bin/activate"
  node_steps
}
