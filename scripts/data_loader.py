from torch.utils.data import Dataset
from transformers import T5Tokenizer, AutoTokenizer
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor


class DataClass(Dataset):
    def __init__(self, args, filename):
        self.args = args
        self.filename = filename
        self.max_length = int(args['--max-length'])
        self.data, self.labels = self.load_dataset()
        # Arabic
                # self.bert_tokeniser = AutoTokenizer.from_pretrained("AraT5-base")
        self.bert_tokeniser = AutoTokenizer.from_pretrained("MARBERT")
        self.inputs, self.lengths, self.label_indices = self.process_data()


    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        df = pd.read_csv(self.filename, sep='\t')
        x_train = 'multilabel classification: ' + df.Tweet.values
        y_train = df.iloc[:, 2:].values

        # labels_li = [' '.join(x.lower().split()) for x in df.columns.to_list()[2:]]
        # labels_matrix = np.array([labels_li] * len(df))
        # print(labels_li)

        # mask = df.iloc[:, 2:].values.astype(bool)
        # y_train = []
        # for l, m in zip(labels_matrix, mask):
        #     x = l[m]
        #     if len(x) > 0:
        #         y_train.append(' , '.join(x.tolist()) + ' </s>')
        #     else:
        #         y_train.append('none </s>')
        print(x_train[0])
        print(y_train[0])
        return x_train, y_train

    def process_data(self):
        desc = "PreProcessing dataset {}...".format('')
        preprocessor = twitter_preprocessor()
        segment_a = "غضب توقع قرف خوف سعادة حب تفأول اليأس حزن اندهاش أو ثقة؟"
        # label_names = ['غضب', 'توقع', 'قر', 'خوف', 'سعادة', 'حب', 'تف', 'الياس', 'حزن', 'اند', 'ثقة']
        label_names = ['غضب', 'توقع', 'قرف', 'خوف', 'سعادة', 'حب', 'تفاول', 'الياس', 'حزن', 'انده', 'ثقة'] # MARBERT TOKENS
        inputs, lengths, label_indices = [], [], []
        for x in tqdm(self.data, desc=desc):
            x = ' '.join(preprocessor(x))
            x = self.t5_tokenizer.encode_plus(segment_a,
                                                x,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                truncation=True,
                                                return_attention_mask=True,
                                            return_token_type_ids=False,
                                                return_tensors='pt')
            input_id = x['input_ids']
            input_length = len([i for i in x['attention_mask'] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)
            # DEBUGGER
            # print(x)
            # print(self.t5_tokenizer.convert_ids_to_tokens(input_id))
            #label indices
            label_idxs = [self.t5_tokenizer.convert_ids_to_tokens(input_id).index(label_names[idx])
                             for idx, _ in enumerate(label_names)]
            label_indices.append(label_idxs)

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        return inputs, data_length, label_indices

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        return inputs, labels, length, label_idxs

    def __len__(self):
        return len(self.inputs)
