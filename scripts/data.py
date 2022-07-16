#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import json
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from store import createRecords
from utils import LOG_INFO, create_dir
from multiprocessing import Process
#------------------------
# fixed
#-------------------------
SPLIT=10240
#------------------------
# data
#-------------------------
def convert_txt2df(data_txt):
    '''
        converts data_txt to a dataframe
    '''
    image_paths=[]
    labels=[]
    with open(data_txt,"r") as f:
        lines=f.readlines()
    for line in tqdm(lines):
        if line.strip():
            if len(line.split("\t"))==2:
                line=line.replace("\n","")
                _path,_label=line.split("\t")
                image_paths.append(_path)
                labels.append(_label)
    df=pd.DataFrame({"filepath":image_paths,"word":labels})
    return df

def get_vocab(vocab_txt):
    vocab=[]
    with open(vocab_txt,"r") as f:
        lines=f.readlines()
    for line in lines:
        if line.strip():
            vocab.append(line.strip())
    vocab=["blank"]+vocab+["sep","pad"]
    return vocab

def check_text(x,vocab):
    for i in x:
        if i not in vocab:
            return None
    return x


def main(args):
    data_txt    =   args.data_txt
    vocab_txt   =   args.vocab_txt
    save_path   =   args.save_path
    num_proc    =   int(args.num_process)
    
    
    save_path   =   create_dir(save_path,"tfrecords")
    temp_path   =   create_dir(save_path,"temp")
    
    class cfg:
        img_height = int(args.img_height)
        img_width  = int(args.img_width)
        pos_max    = int(args.label_max_len)
        vocab      = get_vocab(vocab_txt)

    df=convert_txt2df(data_txt)
    print("total data:",len(df))
    df["word"]=df["word"].progress_apply(lambda x: x if len(x)<cfg.pos_max-2 else None)
    df.dropna(inplace=True)
    df["word"]=df["word"].progress_apply(lambda x: check_text(x,cfg.vocab))
    df.dropna(inplace=True)
    print("filtered number of data:",len(df))
    dfs=[df[idx:idx+SPLIT] for idx in range(0,len(df),SPLIT)]
    max_end=len(dfs)
    

    def run(idx):
        if idx <len(dfs):
            tf_path=create_dir(save_path,str(idx))
            createRecords(dfs[idx],tf_path,idx,temp_path,cfg)


    def execute(start,end):
        process_list=[]
        for idx in range(start,end):
            p =  Process(target= run, args = [idx])
            p.start()
            process_list.append(p)
        for process in process_list:
            process.join()


    if max_end==1:
        dfs=[df]
        run(0)
    elif max_end<=num_proc:
        for i in range(0,max_end):
            start=i
            end=start+max_end-1
            execute(start,end) 
    else:
        for i in range(0,max_end,num_proc):
            start=i
            end=start+num_proc
            if end>max_end:end=max_end-1
            execute(start,end) 

    # save config
    config={"vocab":cfg.vocab,
            "pos_max":cfg.pos_max,
            "img_height":cfg.img_height,
            "img_width" :cfg.img_width}

    config_json="../config.json"
    with open(config_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)

    LOG_INFO("use the config.json for training on the dataset.",mcolor="red")


if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Robust Scanner tfrecord Dataset Creation Script")
    parser.add_argument("data_txt", help="Path of the data.txt file holding absolute image path and word by tab separation")
    parser.add_argument("vocab_txt", help="Path of the vocab.txt file holding the unicodes to use")
    parser.add_argument("save_path", help="Path of the directory to save the tfrecord dataset")
    
    parser.add_argument("--img_height",required=False,default=32,help ="the desired height of the image:default=32")
    parser.add_argument("--img_width",required=False,default=256,help ="the desired width of the image:default=256")
    parser.add_argument("--label_max_len",required=False,default=40,help ="maximum length for the text label:default=40")
    
    parser.add_argument("--num_process",required=False,default=16,help ="number of processes to be used:default=16")
    
    args = parser.parse_args()
    main(args)
