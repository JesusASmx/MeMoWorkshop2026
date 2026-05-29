from datasets import Dataset, load_dataset, concatenate_datasets, DatasetDict, Value
import pandas as pd
import numpy as np
import ast

def reduce_style(df, to_delete):
    return df[~df['style'].isin(to_delete)]

def simpletext(multilabel=False, ablated=False, limit=5000):
    ds = load_dataset("SimpleStories/SimpleStories")
    to_delete = [ ### ONLY STYLE RELATED WITH TOPIC AND WRITING STRUCTURE. Also, redundant ones were removed.
        ## COMMENTED = KEEP

        #"tragic", 
        #"romantic",
        #"humorous",
        
        "heartwarming",
        "suspenseful",
        "minimalist",
        "whimsical",
        "playful",
        "mythological",
        "fable-like",
        "surreal",
        "modern",
        "epic",
        "lyric",
        "noir",
        "classic",
        "mystical",
        "fairy tale-like",
        "action-packed",
        "lighthearted",
        "adventurous",
        "philosophical",
        "melancholic",
    ]

    train_small = pd.DataFrame(ds["train"][:250000])
    test_small = pd.DataFrame(ds["test"])

    train_small = reduce_style(train_small, to_delete)
    test_small = reduce_style(test_small, to_delete)

    train_small = train_small[['story', 'style']].groupby('style').head(limit)
    test_small = test_small[['story', 'style']].groupby('style').head(limit//5) #0.2 "TEST SPLIT"

    train_small = train_small.rename(columns={'story': 'text'})
    test_small = test_small.rename(columns={'story': 'text'})

    numeric_labels = {
        "tragic":0, 
        "romantic":1,
        "suspenseful":2,
        "humorous":3,
        "heartwarming":4,
    }       

    if not multilabel:
        train_small["labels"] = train_small['style'].apply(lambda x: [1 if i == numeric_labels[x] else 0 for i in range(len(numeric_labels))])
        test_small["labels"] = test_small['style'].apply(lambda x: [1 if i == numeric_labels[x] else 0 for i in range(len(numeric_labels))])
    
        train_small["labels"] = train_small["labels"].apply(lambda x:np.array(x, dtype=np.float32))
        test_small["labels"] = test_small["labels"].apply(lambda x:np.array(x, dtype=np.float32))

    else:
        train_small["labels"] = train_small['style'].apply(lambda x:numeric_labels[x])
        test_small["labels"] = test_small['style'].apply(lambda x:numeric_labels[x])




    if ablated:
        abreviated = {
            "tragic":"T", 
            "romantic":"R",
            "suspenseful":"S",
            "humorous":"H",
            "heartwarming":"HW",
        }       
        
        test_small["labels"] = test_small['style']
        test_small["labels"] = test_small["labels"].apply(lambda x:abreviated[x])

        test_small = test_small.groupby('labels').head(ablated)

        test_small = test_small.drop(columns=['style'])
        return test_small


    
    train_small = train_small.drop(columns=['style'])
    test_small = test_small.drop(columns=['style'])

    #train_small.to_csv("simplestory_train.csv")
    #test_small.to_csv("simplestory_test.csv")

    raw_dataset = DatasetDict({
            "train": Dataset.from_pandas(train_small),
            "test": Dataset.from_pandas(test_small),
        })

    if multilabel:
        return raw_dataset.cast_column("label", Value("int64")) #

    else:
        return raw_dataset