from datetime import date, datetime, timedelta
import json
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import config
args = config.args


filenames = os.listdir(args.data_dir)
stock_name_price = set([filename.split('.')[0] for filename in filenames])
stock_name_news = set(os.listdir(args.news_dir))
stock_names = set.intersection(stock_name_news, stock_name_price)

start = datetime.strptime(args.train_start_date, '%Y-%m-%d')
end = datetime.strptime(args.test_end_date, '%Y-%m-%d')

date_list = [start + timedelta(days=i) for i in range((end-start).days+1)]
y = pd.DataFrame(index=date_list, columns=list(stock_names))

for filename in filenames:
    stock_name = filename.split(".")[0]
    if stock_name not in stock_names:
        continue
    
    filepath = args.data_dir + filename
    df = pd.read_csv(filepath, header=None, index_col=0, parse_dates=True, sep='\t')
    for index, move_per  in zip(df.index, df[1]):
        y[stock_name][index] = move_per
        
        
y[(-0.005 <= y) & (y <= 0.0055)] = float('nan')
y[y > 0.0055] = 1
y[y < -0.005] = 0


BERT_tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
news_data = dict() #(key: stock_name + date, value: tokenized_news)
for stock_name in stock_names:
    print(stock_name + ' token')
    file_names = os.listdir(args.news_dir + stock_name)
    for file_name in file_names:
        file_path = args.news_dir + stock_name + '/' + file_name
        key = stock_name + ' + ' + file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            jsons = [json.loads(line) for line in lines]
            
            text_data = [' '.join(jsons[i]['text']) if i < len(jsons) \
                else '' for i in range(args.max_num_tweets_len)]
            tokens = BERT_tokenizer(text_data, 
                        max_length=args.max_num_tokens_len,
                        truncation=True,
                        padding='max_length',
                    ) # input_ids(20, 30), token_type_ids(20, 30), attension_mask(20, 30)
            news_data[key] = tokens
    
            
train_x = pd.DataFrame()
train_y = pd.DataFrame()
dev_x = pd.DataFrame()
dev_y = pd.DataFrame()
test_x = pd.DataFrame()
test_y = pd.DataFrame()

train_start_date = datetime.strptime(args.train_start_date, '%Y-%m-%d')
train_end_date = datetime.strptime(args.train_end_date, '%Y-%m-%d')
dev_start_date = datetime.strptime(args.dev_start_date, '%Y-%m-%d')
dev_end_date = datetime.strptime(args.dev_end_date, '%Y-%m-%d')
test_start_date = datetime.strptime(args.test_start_date, '%Y-%m-%d')
test_end_date = datetime.strptime(args.test_end_date, '%Y-%m-%d')

train_idx = 0
dev_idx = 0
test_idx = 0
num_filtered_samples = 0

for stock_name in stock_names:
    sample = np.zeros((args.days, args.max_num_tweets_len, \
                    3, args.max_num_tokens_len))
    print(stock_name)
    for target_date in date_list:
        if y[stock_name][target_date] not in (0,1):
            continue
        num_no_news_days = 0
        for lag in range(args.days + 1, 1, -1):
            news_date = target_date - timedelta(days=lag)
            key = stock_name + ' + ' + str(news_date.date())
            if key in news_data:
                news_ids = news_data[key]
                sample[args.days - lag, :, 0, :] = np.array(news_ids['input_ids'])
                sample[args.days - lag, :, 1, :] = np.array(news_ids['token_type_ids'])
                sample[args.days - lag, :, 2, :] = np.array(news_ids['attention_mask']) #(20, 30)
            else:
                num_no_news_days += 1
                if num_no_news_days > 1:
                    break
        
        if num_no_news_days > 1:
            num_filtered_samples += 1
            continue
        
        label = y[stock_name][target_date]
        
        if train_start_date <= target_date <= train_end_date:
            train_x = pd.concat([train_x, pd.DataFrame(np.expand_dims(np.ravel(sample), axis=0), index=[target_date])])
            train_y = pd.concat([train_y, pd.DataFrame([label], index = [target_date])])
        elif dev_start_date <= target_date <= dev_end_date:
            dev_x = pd.concat([dev_x, pd.DataFrame(np.expand_dims(np.ravel(sample), axis=0), index = [target_date])])
            dev_y = pd.concat([dev_y, pd.DataFrame([label], index = [target_date])])
        elif test_start_date <= target_date <= test_end_date:
            test_x = pd.concat([test_x, pd.DataFrame(np.expand_dims(np.ravel(sample), axis=0), index = [target_date])])
            test_y = pd.concat([test_y, pd.DataFrame([label], index = [target_date])])

save_path = args.dataset_save_dir
dir = Path(save_path)
dir.mkdir(parents=True, exist_ok=True)

train_x.to_csv(save_path + 'train_x.csv', index=False, header=False)
train_y.to_csv(save_path + 'train_y.csv', index=False, header=False)
dev_x.to_csv(save_path + 'dev_x.csv', index=False, header=False)
dev_y.to_csv(save_path + 'dev_y.csv', index=False, header=False)
test_x.to_csv(save_path + 'test_x.csv', index=False, header=False)
test_y.to_csv(save_path + 'test_y.csv', index=False, header=False)


        
            