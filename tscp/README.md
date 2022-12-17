# $TS-CP^2$ replication

## Quickstart to reproduce results from table 2 of the article (Deldari, Shohreh, et al., (2021)) for $TS-CP^2$ model
Open and run all in Google Colab notebooks:

# for HASC dataset with corresponding parameters listed in name of the file
- TSCP2_dataset_name_HASC_win_60_code_size_10_quantile_0_4_train_share_bs_64_step_5_epochs_40.ipynb
- TSCP2_dataset_name_HASC_win_100_code_size_10_quantile_0_4_train_share_bs_64_step_5_epochs_40.ipynb
- TSCP2_dataset_name_HASC_win_200_code_size_10_quantile_0_4_train_share_bs_64_step_5_epochs_40.ipynb



Replicated results:

| Dataset     | F1 with margin        |     F1 with margin   |     F1 with margin   |
| ----------- | ----------------------|----------------------|----------------------|
| HASC        | win:60,bs:64    0.4426|win:100,bs:64   0.4507|win:200,bs:64   0.4768|
| USC         | win:100,bs:8          |win:200,bs:8          |win:300,bs:8          |

Additional results: ROC AUC (computed without margin)

| Dataset     | ROC AUC  w/o margin   |ROC AUC  w/o margin   |ROC AUC  w/o margin   |
| ----------- | ----------------------|----------------------|----------------------|
| HASC        | win:60,bs:64    0.5962|win:100,bs:64   0.5663|win:200,bs:64   0.6375|
| USC         | win:100,bs:8          |win:200,bs:8          |win:300,bs:8          |



notes:
- It is recommended to use GPU to obtain results faster





[1] Deldari, Shohreh, et al. "Time series change point detection with self-supervised contrastive predictive coding." Proceedings of the Web Conference 2021. 2021.
