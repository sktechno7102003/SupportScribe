import os
import json

def test_adapter_files_exist():
    adapter_dir = "models/support_scribe_lora_final"
    assert os.path.exists(adapter_dir), f"Directory {adapter_dir} does not exist!"
    
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    
    assert os.path.exists(config_path), "adapter_config.json is missing!"
    assert os.path.exists(weights_path), "adapter_model.safetensors is missing!"
    
    # Weight file size sanity check
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    assert file_size_mb > 10.0, f"Adapter file unexpectedly small: {file_size_mb:.2f} MB"

def test_training_loss_convergence():
    log_path = "reports/training_loss_log.json"
    assert os.path.exists(log_path), "reports/training_loss_log.json is missing!"
    
    with open(log_path, "r") as f:
        log_data = json.load(f)
        
    required_keys = ["initial_loss", "final_loss", "loss_reduction_pct", "peak_vram_mb", "training_time_sec"]
    for key in required_keys:
        assert key in log_data, f"Missing key in log: {key}"
        
    assert log_data["final_loss"] < log_data["initial_loss"], "Training loss did not decrease!"
    assert log_data["loss_reduction_pct"] >= 40.0, f"Loss reduction too low: {log_data['loss_reduction_pct']}%"
    assert log_data["peak_vram_mb"] < 12000.0, f"Peak VRAM exceeded 12 GB: {log_data['peak_vram_mb']} MB"

print("All tests passed! LoRA training and artifacts successfully verified.")