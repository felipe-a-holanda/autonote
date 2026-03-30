# Autonote Test Suite

This directory contains the test suite for the Autonote project using pytest.

## Structure

```
tests/
├── unit/           # Unit tests for individual modules
├── integration/    # Integration tests for workflows
├── conftest.py     # Shared fixtures and configuration
└── README.md       # This file
```

## Running Tests

### Install test dependencies

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest
```

### Run specific test categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Run with coverage report
pytest --cov=autonote --cov-report=html

# Run specific test file
pytest tests/unit/test_config.py

# Run specific test
pytest tests/unit/test_config.py::TestGetConfig::test_default_config_values
```

### Test markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Use fixtures from `conftest.py` for common setup
- Mark tests appropriately: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

## Coverage

Aim for >80% code coverage on core modules. Check coverage with:

```bash
pytest --cov=autonote --cov-report=term-missing
```
