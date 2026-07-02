# nanobanana

Package-managed Gemini Image CLI. Also runnable as a single-file script.

## Quickstart

```bash
# 1. Set your API key
export GEMINI_API_KEY="..."

# 2. Run directly with uv
uv run --script ./nanobanana generate "A brutalist library in the Andes mountains at sunset"

# 3. Or install it
chmod +x nanobanana
cp nanobanana ~/.local/bin/
nanobanana generate "A brutalist library in the Andes mountains at sunset"
```

## Install as Package

```bash
git clone https://github.com/rmax-ai/nanobanana.git
cd nanobanana
uv sync
uv run nanobanana generate "test"
```

Or install globally:

```bash
uv tool install git+https://github.com/rmax-ai/nanobanana.git
```

## What it does

nanobanana wraps Google's Gemini image generation models behind task-oriented commands:

| Command | What it does |
|---|---|
| `generate` | Generate an image from text |
| `edit` | Modify an existing image |
| `compose` | Combine multiple reference images |
| `diagram` | Architecture diagrams, infographics, flowcharts |
| `product` | Product shots, mockups, packaging |
| `grounded` | Generate visuals using current web information |
| `variations` | Produce prompt or image variations |
| `batch` | Execute jobs from a JSONL file |
| `inspect` | Inspect local image properties and run metadata |
| `models` | Display supported model capabilities |
| `config` | Read or update non-secret defaults |

Model selection is automatic — `auto` routes to the right model based on your task, resolution, references, and quality needs.

## Example Commands

```bash
# Text-to-image with automatic model selection
nanobanana generate "A modular agent control plane as a technical cutaway diagram" \
  --aspect 16:9 --size 2K

# Edit an existing image
nanobanana edit portrait.png "Replace the background with a concrete dance studio"

# Compose multiple references
nanobanana compose \
  --character person.jpg \
  --style reference-art.png \
  --background location.jpg \
  "Place the character in the location, matching the style"

# Architecture diagram (auto-selects Pro, 2K, high thinking)
nanobanana diagram "Sequence: user → agent → gateway → API" --type sequence

# Product mockup
nanobanana product --subject shoe.png --logo brand.png \
  "Premium studio shot on dark volcanic stone" --view three-quarter

# Grounded visual with web research
nanobanana grounded "Visualize Amsterdam's next 5 days of weather for cyclists"

# Batch processing
nanobanana batch jobs.jsonl --parallel 4 --continue-on-error

# JSON output for scripting
nanobanana generate "A test" --json
```

## Configuration

Optional `~/.config/nanobanana/config.toml`:

```toml
default_model = "auto"
default_output_dir = "./generated"
default_image_size = "1K"
default_aspect_ratio = "1:1"
save_manifest = true
```

## Model Reference

| CLI alias | API model | Best for |
|---|---|---|
| `lite` | gemini-3.1-flash-lite-image | Cheapest, fastest — thumbnails, icons, drafts |
| `flash` | gemini-3.1-flash-image | Default — general generation, editing, multi-reference |
| `pro` | gemini-3-pro-image | Complex composition, precise text, professional assets |
| `legacy` | gemini-2.5-flash-image | Compatibility only (migrate to flash) |
| `auto` | Selected by policy | Default — the CLI picks the right model |

## Output

Every run produces:
- `output.png` — The generated image
- `output.manifest.json` — Reproducibility manifest with SHA-256, model decision, normalized prompt

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (for dependency management)
- A [Gemini API key](https://aistudio.google.com/apikey)

Package-managed with uv. Also available as a single-file script.

## Documentation

- [SPEC.md](SPEC.md) — Full specification
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture and design decisions
- [ROADMAP.md](ROADMAP.md) — Versioned milestones
- [AGENTS.md](AGENTS.md) — Contributor and agent conventions
- [PYTHON_DEVELOPMENT.md](PYTHON_DEVELOPMENT.md) — Python engineering standards
- [PYTHON_CLI_DESIGN.md](PYTHON_CLI_DESIGN.md) — CLI architecture and output contracts
- [PYTHON_ARCHITECTURE.md](PYTHON_ARCHITECTURE.md) — Internal architecture patterns

## License

MIT
