# Benchmark README

This directory contains scripts and configurations for benchmarking
the throughput of FlagScale with and without the FlagGems operator library.

## 1. Requirements

Install Python dependencies with:
    pip install -r requirements.txt

## 2. Running Base Benchmark (FlagScale(vLLM) only)

Use the provided script `online.sh`.

1. Edit the script and change the `MODEL` variable:
       MODEL="/Change/To/Your/Real/Path/Here/Qwen/Qwen3-8B"
   Replace with the actual path to your model.

2. Run the script:
       bash online.sh

3. Results will be written under a timestamped folder, e.g.:
       online-benchmark-Qwen3-8B/2025_09_01_12_00/result.txt

Each run includes multiple configurations of:
- Input length: 128, 512, 1024, 2048, 6144, 14336, 30720
- Output length: 128, 512, 1024, 2048
- Number of prompts: 1, 100, 1000, 2000

The script automatically starts a vLLM server, waits until it is ready,
and then launches the benchmark.

## 3. Running Benchmark with FlagGems

⚠️ Before running the FlagGems benchmark, you **must integrate FlagGems into FlagScale**.  
Follow these steps carefully.

### 3.1 Integration with FlagScale

1. **Baseline Verification**  
   First, confirm that your model can run normally with plain FlagScale.  
   For example, `Qwen3-8B` typically requires at least one A100 GPUs and may take up to 1 minutes to load.  
   Do not proceed until this step works.

2. **Patch vLLM to Load FlagGems**  
   Locate the model runner file depending on your vLLM version:
   - vLLM v1 (≥ 0.8): `vllm/v1/worker/gpu_model_runner.py`
   - vLLM v0 (legacy): `vllm/worker/model_runner.py`

   After the last `import` statement, insert:

   ```python
   import os
   if os.getenv("USE_FLAGGEMS", "false").lower() in ("1", "true", "yes"):
        try:
            import flag_gems
            flag_gems.enable(record=True, path="./gems_operators.log")
            # flag_gems.apply_gems_patches_to_vllm(verbose=True)
            logger.info("Successfully enabled flag_gems as default ops implementation.")
        except ImportError:
            logger.warning("Failed to import 'flag_gems'. Falling back to default implementation.")
        except Exception as e:
            logger.warning(f"Failed to enable 'flag_gems': {e}. Falling back to default implementation.")
   ```  


2. ** Confirm Successful Injection**
When the service starts, check logs for operator override messages like:

Overriding a previously registered kernel for the same operator...
operator: aten::add.Tensor(...)


This indicates that FlagGems has been injected correctly.


Use the provided script `online_with_gems.sh`.

1. Edit the script and change the `MODEL` variable to your real model path.

2. Export the environment variable to enable FlagGems integration:
       export USE_FLAGGEMS=1

3. Run the script:
       bash online_with_gems.sh

4. Results will be saved in the same format as the base benchmark.

## 4. Comparing Results

After running both scripts, you will have two sets of results:
- Base vLLM results (from `online.sh`)
- vLLM + FlagGems results (from `online_with_gems.sh`)

You can compare the `result.txt` files directly or use the `summary.py`
utility if provided.