# Contributing

Use feature branches and conventional, focused commits. Before opening a pull request:

```bash
pip install -e ".[dev,docs]"
pre-commit run --all-files
pytest
python -m build
python -m twine check dist/*
mkdocs build --strict
```

Every physics model should document assumptions and units, include analytical small-system tests, and preserve the backend-neutral boundary. Public API changes require release notes.
