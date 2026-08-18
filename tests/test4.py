import os
import json

def test_merged_artifacts():
    merged_dir = "models/support_scribe_merged"
    assert os.path.exists(merged_dir), "Merged directory missing!"
    assert os.path.exists(os.path.join(merged_dir, "config.json")), "Base config.json missing!"
    
    # Ensure it's not a PEFT directory
    assert not os.path.exists(os.path.join(merged_dir, "adapter_config.json")), "Error: Directory contains adapter_config.json. It was not merged properly!"

def test_roi_and_improvements():
    base_path = "reports/baseline_report.json"
    ft_path = "reports/finetuned_report.json"
    roi_path = "reports/impact_report.json"
    
    assert os.path.exists(base_path), "Baseline report missing!"
    assert os.path.exists(ft_path), "Finetuned report missing!"
    assert os.path.exists(roi_path), "Impact report missing!"
    
    with open(base_path, "r") as f: base_data = json.load(f)
    with open(ft_path, "r") as f: ft_data = json.load(f)
    with open(roi_path, "r") as f: roi_data = json.load(f)
        
    assert ft_data["schema_compliance_rate"] > base_data["schema_compliance_rate"], "Fine-tuning did not improve the Schema Compliance Rate!"
    assert roi_data["monthly_savings_usd"] > 0, "ROI calculator shows negative or zero savings!"
    assert "scr_improvement" in roi_data, "Missing SCR improvement metric in ROI report"

print("All tests passed! Adapter successfully merged and ROI verified.")