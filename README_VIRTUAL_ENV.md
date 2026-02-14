# Virtual Environment Setup Instructions

## 🚨 Important: Use the Correct Virtual Environment

Your project has **TWO** virtual environments:
1. `env/` - ✅ **CORRECT** - Contains all ML packages (pandas, scikit-learn, etc.)
2. `.venv/` - ❌ **WRONG** - Missing ML packages

## 🔧 How to Use the Correct Virtual Environment

### Method 1: Command Line
```bash
# Navigate to project directory
cd C:\Users\Lenovo\OneDrive\Desktop\cc\collegepredection

# Activate the CORRECT virtual environment
env\Scripts\activate

# Verify packages are installed
python -c "import pandas; print('Pandas version:', pandas.__version__)"

# Run Django server
python manage.py runserver
```

### Method 2: Easy Activation Script
```bash
# Double-click on activate_env.bat
# This will open a command prompt with the correct environment
```

## 📦 Installed Packages in `env/` Virtual Environment

- Django 5.2.7
- pandas 2.3.3
- scikit-learn 1.7.2
- numpy 2.3.3
- openpyxl 3.1.5
- mysqlclient 2.2.7
- scipy 1.16.2
- joblib 1.5.2

## ⚠️ Common Issues

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution:** You're using the wrong virtual environment (`.venv` instead of `env`)

### Issue: Server won't start
**Solution:** Make sure you're in the correct virtual environment:
```bash
# Check which environment you're in
echo %VIRTUAL_ENV%

# Should show: C:\Users\Lenovo\OneDrive\Desktop\cc\collegepredection\env
```

## 🎯 Quick Start Commands

```bash
# 1. Navigate to project
cd C:\Users\Lenovo\OneDrive\Desktop\cc\collegepredection

# 2. Activate correct environment
env\Scripts\activate

# 3. Check Django setup
python manage.py check

# 4. Start server
python manage.py runserver

# 5. Open browser to http://127.0.0.1:8000
```

## 🔍 Troubleshooting

If you still get import errors:
1. Make sure you're in the `env` virtual environment (not `.venv`)
2. Check that packages are installed: `python -m pip list`
3. Reinstall packages if needed: `python -m pip install -r requirements.txt`

