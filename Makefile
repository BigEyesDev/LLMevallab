.PHONY: setup run test

setup:
	uv sync
	python main.py

run:
	uv run streamlit run app/dashboard.py

test:
	uv run pytest
