# Kotha GPT CLI

Command-line interface for the Kotha GPT platform.

## Install

```bash
pip install -e packages/cli
```

## Usage

```bash
export KOTHAGPT_API_URL=http://localhost:8000

# Models
kothagpt models

# Chat (with streaming)
kothagpt chat "বাংলায় হ্যালো বলো"
kothagpt chat --stream --model kothagpt-small "একটি গল্প বলো"

# Embeddings
kothagpt embed "বাংলা ভাষা"
kothagpt embed --json "বাংলা ভাষা" "বাংলাদেশ"

# Rerank
kothagpt rerank --query "বাংলা ভাষা" --document "বাংলা শেখার বই" --document "রান্নার রেসিপি" --top-n 1

# Tools
kothagpt tools list
kothagpt tools invoke calculator --arg expression="(2 + 3) * 4"

# Agents
kothagpt agents create --name assistant --tool calculator
kothagpt agents list
kothagpt agents run <agent_id> "দুই এবং তিনের যোগফল কত?"
kothagpt agents run --stream <agent_id> "দুই এবং তিনের যোগফল কত?"
```