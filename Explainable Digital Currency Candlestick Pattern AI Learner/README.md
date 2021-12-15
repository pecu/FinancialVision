# Step 1. Create virtual environment
conda create -n py37 python=3.7
# Step 2. Active virtual environment
activate py37
# Step 3. Install dependencies
pip install -r requirements.txt
# Step 4. Run multi-threads code (more illustrations are in `perturb_multi.py`)
python perturb_multi.py