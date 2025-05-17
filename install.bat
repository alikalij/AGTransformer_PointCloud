@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing torch torchvision torchaudio...
pip install torch==2.7.0+cpu torchvision==0.18.0+cpu torchaudio==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu

python -c "import torch; print(torch.__version__)"

echo Installing PyG dependencies...
pip install -r requirements.txt

echo Done! You can now run your project.
pause