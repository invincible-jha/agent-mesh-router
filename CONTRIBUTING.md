# Contributing to agent-mesh-router

Thank you for contributing to agent-mesh-router, part of the AumOS portfolio.

## Organisation-Wide Guidelines

Please read the AumOS organisation contributing guidelines first:
https://github.com/aumos-ai/.github/blob/main/CONTRIBUTING.md

The following sections cover project-specific conventions.

## Development Setup

```bash
git clone https://github.com/aumos-ai/agent-mesh-router.git
cd agent-mesh-router
pip install -e ".[dev]"
```

## Running the Test Suite

```bash
make test          # run all tests with coverage
make lint          # ruff lint + format check
make typecheck     # mypy strict
make ci            # full CI suite locally
```

## Branch Naming

- Features: `feature/<short-description>`
- Bug fixes: `fix/<short-description>`
- Documentation: `docs/<short-description>`

Branch from `main`. Squash-merge PRs to keep history linear.

## Commit Messages

Use conventional commits:

```
feat: add plugin hot-reload support
fix: handle missing registry key gracefully
refactor: extract validation to separate module
docs: update architecture diagram
test: cover registry edge cases
chore: bump ruff to 0.4
```

Commit messages explain WHY, not WHAT.

## Pull Request Checklist

- [ ] Tests added or updated for all changed behaviour
- [ ] `make ci` passes locally
- [ ] Type hints present on all new function signatures
- [ ] Docstrings on all public symbols
- [ ] CHANGELOG.md updated under `[Unreleased]`

## Code Style

This project enforces ruff (lint + format) and mypy strict. Run
`make format` to auto-fix formatting before committing.

## Security Issues

Do not open a public issue for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for the responsible disclosure process.

## License

By contributing you agree that your contributions will be licensed
under the Apache 2.0 License.
