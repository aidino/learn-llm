"""
Comet ML Configuration for Gemma3N Math Tutor Fine-tuning

Cách setup:
1. Tạo tài khoản tại https://www.comet.com/
2. Tạo workspace và project
3. Lấy API key từ https://www.comet.com/api/my/settings/
4. Cập nhật thông tin dưới đây
"""

import os

# =============================================================================
# COMET ML CONFIGURATION
# =============================================================================

# Comet ML Credentials
COMET_CONFIG = {
    # API Key - REQUIRED
    # Cách 1: Set environment variable
    # export COMET_API_KEY="your-api-key-here"
    
    # Cách 2: Set trực tiếp (không khuyến nghị cho production)
    "api_key": os.getenv("COMET_API_KEY"),  # Hoặc thay bằng API key của bạn
    
    # Workspace - REQUIRED
    # Tên workspace trên Comet ML
    "workspace": "your-workspace-name",  # Thay bằng workspace của bạn
    
    # Project Name - REQUIRED  
    # Tên project trên Comet ML
    "project": "gemma3n-math-tutor",  # Có thể thay đổi tên project
    
    # Experiment Name - OPTIONAL
    # Tên experiment cụ thể (tự động generate nếu không set)
    "experiment_name": None,  # Hoặc đặt tên custom như "exp-001"
    
    # Tags - OPTIONAL
    # Tags để phân loại experiments
    "tags": [
        "gemma3n",
        "vision-language", 
        "math-tutor",
        "vietnamese",
        "sixth-grade",
        "fine-tuning"
    ],
    
    # Additional Settings
    "auto_metric_logging": True,     # Tự động log metrics
    "auto_param_logging": True,      # Tự động log parameters
    "auto_histogram_weight_logging": True,   # Log weight histograms
    "auto_histogram_gradient_logging": True, # Log gradient histograms
    "auto_histogram_activation_logging": False,  # Tắt để tiết kiệm memory
    "auto_output_logging": "default",  # Log output (stdout/stderr)
    
    # Model Logging
    "log_model": True,              # Upload model artifacts
    "log_graph": False,             # Log model graph (có thể chậm)
    "log_code": True,               # Log source code
    "log_git_metadata": True,       # Log git information
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_comet_config():
    """Validate Comet ML configuration."""
    
    if not COMET_CONFIG["api_key"]:
        print("❌ COMET_API_KEY is required!")
        print("Please set environment variable: export COMET_API_KEY='your-key'")
        print("Or get your API key from: https://www.comet.com/api/my/settings/")
        return False
    
    if not COMET_CONFIG["workspace"]:
        print("❌ Workspace name is required!")
        print("Please update COMET_CONFIG['workspace'] in comet_config.py")
        return False
    
    if not COMET_CONFIG["project"]:
        print("❌ Project name is required!")
        print("Please update COMET_CONFIG['project'] in comet_config.py")
        return False
    
    return True

def setup_comet_environment():
    """Setup environment variables for Comet ML."""
    
    if COMET_CONFIG["api_key"]:
        os.environ["COMET_API_KEY"] = COMET_CONFIG["api_key"]
    
    if COMET_CONFIG["workspace"]:
        os.environ["COMET_WORKSPACE"] = COMET_CONFIG["workspace"]
    
    if COMET_CONFIG["project"]:
        os.environ["COMET_PROJECT_NAME"] = COMET_CONFIG["project"]

def print_comet_info():
    """Print Comet ML configuration info."""
    
    print("🔧 Comet ML Configuration:")
    print(f"   Workspace: {COMET_CONFIG['workspace']}")
    print(f"   Project: {COMET_CONFIG['project']}")
    print(f"   API Key: {'✅ Set' if COMET_CONFIG['api_key'] else '❌ Not set'}")
    print(f"   Tags: {', '.join(COMET_CONFIG['tags'])}")

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Test configuration
    print("Testing Comet ML configuration...")
    
    if validate_comet_config():
        print("✅ Configuration is valid!")
        setup_comet_environment()
        print_comet_info()
        
        # Test connection
        try:
            import comet_ml
            experiment = comet_ml.Experiment(
                workspace=COMET_CONFIG["workspace"],
                project_name=COMET_CONFIG["project"],
            )
            print(f"✅ Successfully connected to Comet ML!")
            print(f"🔗 Experiment URL: {experiment.url}")
            experiment.end()
            
        except ImportError:
            print("❌ comet_ml not installed. Run: pip install comet-ml")
        except Exception as e:
            print(f"❌ Failed to connect to Comet ML: {e}")
            
    else:
        print("❌ Configuration is invalid!")