#!/usr/bin/env python3
"""
Test script để kiểm tra kết nối Comet ML

Usage:
    python test_comet_connection.py
"""

import os
import sys

def test_comet_connection():
    """Test Comet ML connection và configuration."""
    
    print("🧪 Testing Comet ML Connection")
    print("=" * 50)
    
    # 1. Check if comet_ml is installed
    try:
        import comet_ml
        print("✅ comet_ml package installed")
        print(f"   Version: {comet_ml.__version__}")
    except ImportError:
        print("❌ comet_ml not installed")
        print("   Please install: pip install comet-ml")
        return False
    
    # 2. Check API key
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        print("❌ COMET_API_KEY not found")
        print("   Please set: export COMET_API_KEY='your-api-key'")
        print("   Get your key from: https://www.comet.com/api/my/settings/")
        return False
    else:
        print("✅ COMET_API_KEY found")
        print(f"   Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    
    # 3. Check configuration file
    try:
        from comet_config import COMET_CONFIG, validate_comet_config
        print("✅ comet_config.py found")
        
        if validate_comet_config():
            print("✅ Configuration is valid")
            print(f"   Workspace: {COMET_CONFIG['workspace']}")
            print(f"   Project: {COMET_CONFIG['project']}")
        else:
            print("❌ Configuration is invalid")
            return False
            
    except ImportError:
        print("⚠️  comet_config.py not found, using default settings")
        workspace = os.getenv("COMET_WORKSPACE")
        project = os.getenv("COMET_PROJECT_NAME", "test-project")
        
        if not workspace:
            print("❌ COMET_WORKSPACE not set")
            print("   Please set: export COMET_WORKSPACE='your-workspace'")
            return False
    
    # 4. Test actual connection
    try:
        print("\n🔄 Testing connection to Comet ML...")
        
        # Import config if available
        try:
            from comet_config import COMET_CONFIG
            workspace = COMET_CONFIG['workspace']
            project = COMET_CONFIG['project']
        except ImportError:
            workspace = os.getenv("COMET_WORKSPACE")
            project = os.getenv("COMET_PROJECT_NAME", "test-project")
        
        # Create test experiment
        experiment = comet_ml.Experiment(
            workspace=workspace,
            project_name=project,
            auto_metric_logging=False,
            auto_param_logging=False,
        )
        
        # Log test data
        experiment.log_metric("test_metric", 1.0)
        experiment.log_parameter("test_param", "test_value")
        experiment.add_tag("test")
        
        print("✅ Successfully connected to Comet ML!")
        print(f"🔗 Test experiment URL: {experiment.url}")
        
        # End experiment
        experiment.end()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Comet ML: {e}")
        print("\nPossible solutions:")
        print("1. Check your API key is correct")
        print("2. Verify workspace and project names exist")
        print("3. Check your internet connection")
        print("4. Ensure you have permissions for the workspace")
        return False

def print_setup_instructions():
    """Print detailed setup instructions."""
    
    print("\n📋 Comet ML Setup Instructions")
    print("=" * 50)
    
    print("\n1. 🌐 Create Comet ML Account:")
    print("   → Go to: https://www.comet.com/")
    print("   → Sign up for free account")
    
    print("\n2. 🏗️  Create Workspace & Project:")
    print("   → Login to Comet ML dashboard")
    print("   → Create a new workspace (if needed)")
    print("   → Create a new project: 'gemma3n-math-tutor'")
    
    print("\n3. 🔑 Get API Key:")
    print("   → Go to: https://www.comet.com/api/my/settings/")
    print("   → Copy your API key")
    
    print("\n4. ⚙️  Set Environment Variables:")
    print("   → export COMET_API_KEY='your-api-key-here'")
    print("   → export COMET_WORKSPACE='your-workspace-name'")
    
    print("\n5. 📝 Configure comet_config.py:")
    print("   → Edit COMET_CONFIG['workspace']")
    print("   → Edit COMET_CONFIG['project']")
    
    print("\n6. 🧪 Test Connection:")
    print("   → python test_comet_connection.py")

if __name__ == "__main__":
    success = test_comet_connection()
    
    if success:
        print("\n🎉 All tests passed! Comet ML is ready to use.")
        print("You can now run the training script with Comet ML logging.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        print_setup_instructions()
    
    sys.exit(0 if success else 1)