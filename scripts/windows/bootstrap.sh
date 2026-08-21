#!/usr/bin/env bash

bootstrap_check() {
  require_cmd py "Install Python 3.10+ from https://www.python.org (include the py launcher)."
  node_check
}

bootstrap_steps() {
  python_venv_steps py "Scripts/activate"
  node_steps
}
