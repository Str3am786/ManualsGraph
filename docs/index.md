# Welcome to our documentation

This is the main page of our documentation. 

## Code Style and Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code style with Black, isort, and flake8.

### Setup pre-commit

1. Install pre-commit and style tools:
    ```bash
    uv pip install -r requirements.txt
    ```
2. Install the pre-commit hooks:
    ```bash
    pre-commit install
    ```

### Usage
- Hooks will run automatically on `git commit`.
- To run checks manually on all files:
    ```bash
    pre-commit run --all-files
    ``` 