PYTHON   ?= python
MODELS   := models
DATA     := data/merged
IFACE    ?= eth0

.PHONY: help all merge preprocess train train-rf train-tune train-ablation \
        train-mixed train-cross-2017to2018 train-cross-2018to2017 \
        train-cross-2017to2018-hardened train-cross-2018to2017-hardened \
        train-fused-2018 train-fused-2017 \
        shap serve live-validate live-capture clean

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

train-mixed:                             ## Train on 2017+2018 with mixed file-holdout split
	$(PYTHON) training/train_xgb.py \
	  --split-strategy mixed_holdout --save-plots \
	  --model $(MODELS)/xgb_ids_mixed.pkl

train-cross-2017to2018:                  ## Zero-shot: train on 2017, test on 2018
	$(PYTHON) training/train_xgb.py \
	  --split-strategy cross_dataset_2017to2018 --save-plots \
	  --model $(MODELS)/xgb_ids_cross_2017to2018.pkl

train-cross-2018to2017:                  ## Zero-shot: train on 2018, test on 2017
	$(PYTHON) training/train_xgb.py \
	  --split-strategy cross_dataset_2018to2017 --save-plots \
	  --model $(MODELS)/xgb_ids_cross_2018to2017.pkl

train-cross-2017to2018-hardened:         ## Zero-shot 2017→2018 + env drop + hardened HP
	$(PYTHON) training/train_xgb.py \
	  --split-strategy cross_dataset_2017to2018 \
	  --drop-env-features --hardened-hp --save-plots \
	  --model $(MODELS)/xgb_ids_cross_2017to2018_hardened.pkl

train-cross-2018to2017-hardened:         ## Zero-shot 2018→2017 + env drop + hardened HP
	$(PYTHON) training/train_xgb.py \
	  --split-strategy cross_dataset_2018to2017 \
	  --drop-env-features --hardened-hp --save-plots \
	  --model $(MODELS)/xgb_ids_cross_2018to2017_hardened.pkl

train-fused-2018:                        ## Fused: all 2017 + 80% 2018 → 20% 2018 holdout
	$(PYTHON) training/train_xgb.py \
	  --split-strategy fused_2018holdout --save-plots \
	  --model $(MODELS)/xgb_ids_fused_2018holdout.pkl

train-fused-2017:                        ## Fused: all 2018 + 80% 2017 → 20% 2017 holdout
	$(PYTHON) training/train_xgb.py \
	  --split-strategy fused_2017holdout --save-plots \
	  --model $(MODELS)/xgb_ids_fused_2017holdout.pkl

shap:                                    ## Train + SHAP interpretability
	$(PYTHON) training/train_xgb.py \
	  --shap --split-strategy per_file --save-plots

serve:                                   ## Start FastAPI inference server (port 8000)
	uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

live-validate:                           ## Self-check live_ids feature mapping vs model
	$(PYTHON) -m live_ids validate

live-capture:                            ## Run live IDS on $(IFACE) (requires CAP_NET_RAW)
	$(PYTHON) -m live_ids capture -i $(IFACE) --only-non-benign

clean:                                   ## Remove generated artifacts
	rm -f $(DATA)/*.csv
	rm -f $(MODELS)/*.pkl $(MODELS)/*.csv $(MODELS)/*.json $(MODELS)/*.png
