import pandas as pd
import re
import glob
import nltk
import nltk.tokenize as tokenize
import tqdm

nltk.download('punkt_tab')

def raw2tabular() :
    word_cnt = 0

    raw_Data = glob.glob("raw_data\*.txt")
    print(raw_Data)
    sent_n = 3
    df = pd.DataFrame({})
    for i, url in enumerate(raw_Data) :
        print(i, url)
        sent_l = []
        temp = []
        cnt = 0
        with open(url, "r", encoding="utf-8") as f :
            raw_text = f.read().replace("\n", " ")
            sent_l = tokenize.sent_tokenize(raw_text)
            sent_l = sent_l[:len(sent_l)-(len(sent_l)%sent_n)]
            print(sent_l[:5])
            print(len(sent_l))
            for s in tqdm.tqdm(sent_l) :
                word_cnt += len(list(sent_l[cnt]))
                temp.append(sent_l[cnt])
                cnt += 1
                if cnt % sent_n == 0 :
                    temp_df = pd.DataFrame({"label" : [i], "text" : [(' '.join(temp)).lower()]})
                    df = pd.concat([df, temp_df], ignore_index = True)
                    temp = []
    df.to_csv("data\\data.csv", index=False)
    print(word_cnt)

if __name__ == "__main__" :
    raw2tabular()