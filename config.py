import argparse


argparser = argparse.ArgumentParser()
# https://github.com/BigRoddy/CMIN-Dataset/tree/main/CMIN-US
argparser.add_argument('--data_dir', type=str, default='data/CMIN-US/price/preprocessed/')
argparser.add_argument('--news_dir', type=str,default='data/CMIN-US/news/preprocessed/')
argparser.add_argument('--dataset_save_dir', type=str,default='dataset/CMIN_FinBERT/')

argparser.add_argument('--learning_rate', type=float, default=1e-5)
argparser.add_argument('--weight_decay', type=float, default=1e-5)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--dr', type=float, default=0.3)  
argparser.add_argument('--hidden_size', type=int, default=50)
argparser.add_argument('--train_epochs', type=int, default=50)
argparser.add_argument('--check_interval', type=int, default=1)
argparser.add_argument('--device', type=str, default='cuda')
argparser.add_argument('--usewb', type=bool, default=False)
argparser.add_argument('--save_dir', type=str, default='wandb')
argparser.add_argument('--num_workers', type=int, default=0)
argparser.add_argument('--CHECKPOINT_PATH', type=str, default='checkpoint/')
argparser.add_argument('--optimizer', type=str, default='adamw')
argparser.add_argument('--freeze', type=bool, default=True)
argparser.add_argument('--num_class', type=int, default=2)
argparser.add_argument('--modelpath', type=str, default='yiyanghkust/finbert-pretrain')

argparser.add_argument('--days', type=int, default=5)  
argparser.add_argument('--max_num_tweets_len', type=int, default=20)  
argparser.add_argument('--max_num_tokens_len', type=int, default=30) 
argparser.add_argument('--seed', type=int, default=2024)


argparser.add_argument('--train_start_date', type=str, default='2018-01-01')
argparser.add_argument('--train_end_date', type=str, default='2021-04-30')
argparser.add_argument('--dev_start_date', type=str, default='2021-05-01')
argparser.add_argument('--dev_end_date', type=str, default='2021-08-31')
argparser.add_argument('--test_start_date', type=str, default='2021-09-01')
argparser.add_argument('--test_end_date', type=str, default='2021-12-31')

argparser.add_argument('--train_x_path', type=str, default='dataset/CMIN_FinBERT/train_x.csv')
argparser.add_argument('--train_y_path', type=str, default='dataset/CMIN_FinBERT/train_y.csv')
argparser.add_argument('--dev_x_path', type=str, default='dataset/CMIN_FinBERT/dev_x.csv')
argparser.add_argument('--dev_y_path', type=str, default='dataset/CMIN_FinBERT/dev_y.csv')
argparser.add_argument('--test_x_path', type=str, default='dataset/CMIN_FinBERT/test_x.csv')
argparser.add_argument('--test_y_path', type=str, default='dataset/CMIN_FinBERT/test_y.csv')



args = argparser.parse_args()
