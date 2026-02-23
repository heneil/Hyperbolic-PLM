python3.11 -m venv plm_env
source plm_env/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch_geometric
pip install -r hypercore_requirements.txt
pip install scipy==1.13.1 numpy==1.26.4
pip install llm-foundry==0.18.0
pip install -e .