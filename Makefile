PYTHON   ?= python
MODELS   := models
DATA     := data/merged

.PHONY: help all merge preprocess train train-rf train-tune train-ablation \
        shap serve clean

help:                                    ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

all: merge preprocess train              ## Full pipeline: merge → preprocess → train

merge:                                   ## Merge raw CSV files into one
	$(PYTHON) training/merge_data.py

preprocess:                              ## Clean & label-group the merged CSV
	$(PYTHON) training/preprocess.py

train:                                   ## Train XGBoost (per_file split, class weights)
	$(PYTHON) training/train_xgb.py \
	  --split-strategy per_file --save-plots

train-rf:                                ## Train Random Forest baseline
	$(PYTHON) training/train_xgb.py \
	  --model-type rf --split-strategy per_file --save-plots \
	  --model $(MODELS)/rf_ids_model.pkl

train-tune:                              ## Train XGBoost with Optuna tuning
	$(PYTHON) training/train_xgb.py \
	  --tune --tune-method optuna --split-strategy per_file --save-plots

train-ablation:                          ## Train XGBoost without Destination Port
	$(PYTHON) training/train_xgb.py \
	  --drop-port --split-strategy per_file --save-plots \
	  --model $(MODELS)/xgb_ids_model_no_port.pkl

shap:                                    ## Train + SHAP interpretability
	$(PYTHON) training/train_xgb.py \
	  --shap --split-strategy per_file --save-plots

serve:                                   ## Start FastAPI inference server (port 8000)
	uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

clean:                                   ## Remove generated artifacts
	rm -f $(DATA)/*.csv
	rm -f $(MODELS)/*.pkl $(MODELS)/*.csv $(MODELS)/*.json $(MODELS)/*.png
