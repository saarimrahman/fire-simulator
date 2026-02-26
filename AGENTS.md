# AGENTS.md

## Cursor Cloud specific instructions

### Overview
This is a Python/Streamlit FIRE (Financial Independence, Retire Early) Monte Carlo Simulator. It has two source files (`fire.py` for the simulation engine, `fire_dashboard.py` for the Streamlit UI) and no external service dependencies.

### Running the app
```bash
streamlit run fire_dashboard.py --server.headless true --server.port 8501
```
The dashboard will be available at `http://localhost:8501`. The `--server.headless true` flag is required in headless/CI environments.

### Linting
No linter is configured in the project. Use `flake8` (installed as a dev dependency) for basic checks:
```bash
flake8 fire.py fire_dashboard.py --max-line-length=120
```
Note: the codebase has pre-existing style warnings (E501, E702, etc.) that are not treated as errors.

### Testing
There are no automated tests in the repository. Verify the simulation engine with:
```bash
python3 -c "import numpy as np; from fire import run_vectorized; r = run_vectorized(200000, 'Sacramento', 100, np.random.default_rng(42)); print('OK')"
```

### Key notes
- `streamlit` installs to `~/.local/bin` when using `pip install --user`; ensure it is on `PATH`.
- The app runs 10,000 Monte Carlo simulations by default, which takes a few seconds on first load.
- No database, Docker, environment variables, or secrets are required.
