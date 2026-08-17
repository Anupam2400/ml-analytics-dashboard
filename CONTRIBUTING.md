# Contributing to ML Analytics Dashboard

Thank you for your interest in contributing! Here's how to get started.

## 🛠️ Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/ml-analytics-dashboard.git
cd ml-analytics-dashboard

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies (including dev tools)
pip install -r requirements.txt
```

## 🧪 Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## 🏃 Running the Project

```bash
# Train the ML models (required before launching the dashboard)
python src/models/train.py

# Launch the Streamlit dashboard
streamlit run src/dashboard/app.py
```

## 📐 Code Style

- Use **type hints** on all function signatures.
- Add **docstrings** (Google style) to every public function and class.
- Use `logging` instead of `print()` for status messages.
- Keep imports organized: stdlib → third-party → local.

## 🔀 Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with tests.
3. Run the test suite and ensure all tests pass.
4. Push and open a PR against `main`.
5. Describe your changes clearly in the PR description.

## 💡 Ideas for Contributions

- Add more ML models (XGBoost, LightGBM)
- Implement cross-validation in training
- Add SHAP explanations to the dashboard
- Add data drift monitoring
- Create a REST API for predictions (FastAPI)
