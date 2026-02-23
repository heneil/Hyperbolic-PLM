# Hyperbolic-PLM

## Installation
To install the dependencies run 
```
bash setup_env.sh
```

The modified hyperbolic model and modules are in ***hyp_plm/esm/model/esm2.py*** and ******hyp_plm/esm/module.py***. To test the model and installation is correct, run
```
python download_test_data.py
bash test_data.sh
bash test_train.sh
```
***Reminder to change the root_path variable in download_test_data.py, test_data.py, and esm/data.py***