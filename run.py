 #!/usr/bin/env python3
"""
AI Waste Classification System Runner
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please install them manually.")
        return False
    return True

def check_model():
    """Check if the trained model exists"""
    if os.path.exists('waste_classifier.h5'):
        print("Trained model found: waste_classifier.h5")
        return True
    else:
        print("No trained model found. Please run 'python train.py' to train the model first.")
        return False

def run_app():
    """Run the Flask application"""
    print("Starting AI Waste Classification System...")
    print("Open your browser and go to http://localhost:5000")
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\nApplication stopped.")

def main():
    print("AI Waste Classification System")
    print("=" * 40)

    # Install dependencies if needed
    if not install_dependencies():
        return

    # Check for trained model
    if not check_model():
        train_choice = input("Would you like to train the model now? (y/n): ").lower().strip()
        if train_choice == 'y':
            print("Training model...")
            try:
                subprocess.run([sys.executable, "train.py"])
            except subprocess.CalledProcessError:
                print("Training failed. Please check your dataset and try again.")
                return
        else:
            print("Please prepare your dataset and run 'python train.py' to train the model.")
            return

    # Run the application
    run_app()

if __name__ == "__main__":
    main()
