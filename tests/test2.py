import os
import json

def test_baseline_report_exists():
    assert os.path.exists("reports/baseline_report.json"), "baseline_report.json is missing!"

def test_report_schema():
    with open("reports/baseline_report.json", "r") as f:
        report = json.load(f)
        
    required_keys = [
        "total_samples", 
        "valid_json_count", 
        "valid_schema_count", 
        "syntax_validity_rate", 
        "schema_compliance_rate",
        "category_macro_f1"
    ]
    
    for key in required_keys:
        assert key in report, f"Missing required metric: {key}"
        
    assert 0.0 <= report["syntax_validity_rate"] <= 1.0, "Rate must be between 0 and 1"
    assert 0.0 <= report["schema_compliance_rate"] <= 1.0, "Rate must be between 0 and 1"

print("All tests passed! Baseline benchmark is correctly formatted.")