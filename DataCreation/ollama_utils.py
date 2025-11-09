import subprocess
import json

def list_ollama_models():
    """Get list of downloaded Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error listing Ollama models:", result.stderr)
            return []
        
        # Parse output - skip header line and empty lines
        lines = result.stdout.strip().split('\n')[1:]
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except Exception as e:
        print(f"Error running ollama list: {e}")
        return []

def select_models():
    """Let user select which Ollama models to use."""
    models = list_ollama_models()
    if not models:
        print("No Ollama models found. Please download models using 'ollama pull <model>'")
        return []
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    print("\nEnter numbers of models to use (comma-separated), or 'all' for all models:")
    choice = input("> ").strip().lower()
    
    if choice == 'all':
        return models
    
    try:
        indices = [int(idx.strip()) - 1 for idx in choice.split(',')]
        selected = [models[i] for i in indices if 0 <= i < len(models)]
        if not selected:
            print("No valid models selected. Using default model 'llama2'")
            return ['llama2']
        return selected
    except (ValueError, IndexError):
        print("Invalid selection. Using default model 'llama2'")
        return ['llama2']