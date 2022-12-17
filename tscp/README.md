# $TS-CP^2$ replication

## Quickstart to reproduce results from table 2 of the article (Deldari, Shohreh, et al., (2021)) for $TS-CP^2$ model
Open and run all in Google Colab notebooks:

For HASC dataset with corresponding parameters listed in name of the file
- TSCP2_dataset_name_HASC_win_60_code_size_10_quantile_0_4_train_share_bs_64_step_5_epochs_40.ipynb
- TSCP2_dataset_name_HASC_win_100_code_size_10_quantile_0_4_train_share_bs_64_step_5_epochs_40.ipynb
- TSCP2_dataset_name_HASC_win_200_code_size_10_quantile_0_4_train_share_bs_64_step_5_epochs_40.ipynb

For USC dataset with corresponding parameters listed in name of the file
- TSCP2_dataset_name_USC_win_100_step_20_bs_8_epochs_5_margin_1.ipynb
- TSCP2_dataset_name_USC_win_200_step_20_bs_8_epochs_5_margin_1.ipynb
- TSCP2_dataset_name_USC_win_300_step_20_bs_8_epochs_5_margin_1.ipynb



Replicated results:

| Dataset     | F1 with margin        |     F1 with margin   |     F1 with margin   |
| ----------- | ----------------------|----------------------|----------------------|
| HASC        | win:60,bs:64    0.4426|win:100,bs:64   0.4507|win:200,bs:64   0.4768|
| USC         | win:100,bs:8    0.7304|win:200,bs:8          |win:300,bs:8          |

Additional results: ROC AUC (computed without margin)

| Dataset     | ROC AUC  w/o margin   |ROC AUC  w/o margin   |ROC AUC  w/o margin   |
| ----------- | ----------------------|----------------------|----------------------|
| HASC        | win:60,bs:64    0.5962|win:100,bs:64   0.5663|win:200,bs:64   0.6375|
| USC         | win:100,bs:8    0.8015|win:200,bs:8          |win:300,bs:8          |


For HASC dataset we used 40% quantile to detect change point in terms of cosine distance between embeddings, this value gave us quite close values to those that authors demonstrate in the original paper.
For USC dataset we used 5% quantile to detect change point in terms of cosine distance between embeddings.

notes:
- For some reasons performance on the USC is very poor with high margin value, that's why for USC we used very law margin values (1)
- It is recommended to use GPU to obtain results faster






[1] Deldari, Shohreh, et al. "Time series change point detection with self-supervised contrastive predictive coding." Proceedings of the Web Conference 2021. 2021.
