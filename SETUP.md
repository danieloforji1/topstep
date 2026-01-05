# Python Setup Guide

## ✅ Python 3.11 Installed

Python 3.11.14 is now installed on your system at:
- `/opt/homebrew/bin/python3.11`

## Using Python 3.11

### Option 1: Use python3.11 directly
```bash
python3.11 --version
python3.11 -m pip install -r requirements.txt
python3.11 main.py
```

### Option 2: Create a virtual environment (Recommended)

Virtual environments keep project dependencies isolated:

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Verify you're using Python 3.11
python --version  # Should show 3.11.14

# Install project dependencies
pip install -r requirements.txt
```

### Option 3: Make python3.11 the default (Optional)

If you want `python3` to point to Python 3.11:

```bash
# Add to your ~/.zshrc (since you're using zsh)
echo 'alias python3="/opt/homebrew/bin/python3.11"' >> ~/.zshrc
source ~/.zshrc
```

## Quick Start

1. **Navigate to project directory:**
   ```bash
   cd "/Users/danieloforji/Library/Mobile Documents/com~apple~CloudDocs/Documents/topstep"
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your TopstepX credentials
   ```

5. **Run the strategy:**
   ```bash
   python main.py
   ```

## Troubleshooting

### "python3.11: command not found"
- Make sure Homebrew installed correctly
- Try: `/opt/homebrew/bin/python3.11 --version`

### "pip: command not found"
- Use: `python3.11 -m pip` instead of just `pip`
- Or activate your virtual environment first

### Multiple Python versions
- Check which Python is active: `which python3`
- Use `python3.11` explicitly to ensure correct version

