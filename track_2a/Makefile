.PHONY: run build smoke selftest shell

IMAGE ?= tsemin-swissgov-rsd
SPLIT ?= test
WORKERS ?= 2
SYSTEM ?= tsemin

# Endpoint variables of the Hack Apertus template; LLM_API_KEY must be set in the environment.
LLM_NAME ?= swiss-ai/Apertus-v1.5-8B
LLM_BASE_URL ?= https://api.inference.cscs.ch/v1

build:
	docker build -t $(IMAGE) .

# Reproduce the submission: m2 + d4 on $(SPLIT) (default: the 56-page test split, ~25 min), fused into the final
# prediction files, audited, copied to ./output/$(SPLIT)/. ./runs keeps every raw request/response and the cache,
# so an interrupted run continues where it stopped.
run: build
	mkdir -p runs output
	docker run --rm -e LLM_NAME="$(LLM_NAME)" -e LLM_BASE_URL="$(LLM_BASE_URL)" -e LLM_API_KEY="$(LLM_API_KEY)" \
	  -e SPLIT="$(SPLIT)" -e WORKERS="$(WORKERS)" -e SYSTEM="$(SYSTEM)" \
	  -v "$(CURDIR)/runs:/work/runs" -v "$(CURDIR)/output:/work/output" $(IMAGE)

# Three-minute end-to-end check on 12 development documents (scored).
smoke:
	$(MAKE) run SPLIT=smoke

# Offline tests only (no endpoint needed).
selftest: build
	docker run --rm --entrypoint python $(IMAGE) run.py selftest

shell: build
	docker run --rm -it --entrypoint sh -e LLM_API_KEY="$(LLM_API_KEY)" -v "$(CURDIR)/runs:/work/runs" $(IMAGE)
