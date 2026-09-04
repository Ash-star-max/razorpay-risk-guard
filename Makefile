.PHONY: setup data train evaluate run dashboard test clean all

setup:
	pip install -r requirements.txt

data:
	python -m src.data_generator.generator

train:
	python -m src.models.ensemble --train

evaluate:
	python -m src.models.ensemble --train --evaluate

run:
	uvicorn src.api.server:app --reload --port 8000

dashboard:
	streamlit run src/dashboard/app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/ models/ docs/pr_curve.png docs/threshold_optimization.png
	find . -type d -name __pycache__ -exec rm -rf {} +

all: setup data evaluate
	@echo "✅ Full pipeline complete. Run 'make dashboard' to visualize."
