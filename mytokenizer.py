from transformers import BertTokenizerFast

class MyTokenizer:
    def __init__(self, vocab_path : str) :
        self.tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")

        self.vocab_size = self.tokenizer.vocab_size * 2
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
    
    # 转化为数组
    def encode(self, utterance : str, character : bool, add_special_tokens : bool):
        encode_vector = []
        encode_vector.extend(self.tokenizer.encode(utterance, add_special_tokens=False))

        if character == True:
            for encode_vector_index in range(len(encode_vector)):
                encode_vector[encode_vector_index] += self.tokenizer.vocab_size
        if add_special_tokens == True:
            encode_vector.insert(0, self.cls_token_id)
            encode_vector.append(self.sep_token_id)
        return encode_vector

    
    # 支持字符串数组输入
    # [CLS] [my char] [SEP] [her char + vocab_size] [SEP] ... [her char + vocab_size] [SEP] [my char] [SEP] [her char + vocab_size] [SEP]
    def encode_utterances(self, utterances : list):
        encode_allarray = [self.cls_token_id]
        
        for utterance_index, utterance in enumerate(utterances):
            encode_vector = self.tokenizer.encode(utterance, add_special_tokens=False)
            
            if utterance_index % 2 == 0:
                encode_allarray.extend(encode_vector)
            else:
                for encode_vector_index in range(len(encode_vector)):
                    encode_vector[encode_vector_index] += self.tokenizer.vocab_size
                encode_allarray.extend(encode_vector)

            encode_allarray.append(self.sep_token_id)

        return encode_allarray

    # 返回索引
    def convert_tokens_to_ids(self, token : str):
        return self.tokenizer.convert_tokens_to_ids(token)
    
    # 返回字符串数组
    def convert_ids_to_tokens(self, ids : list, character : bool):
        if character == False:
            return self.tokenizer.convert_ids_to_tokens(ids)
        else:
            true_ids = []
            for id in ids:
                true_ids.append(id - self.tokenizer.vocab_size)
            return self.tokenizer.convert_ids_to_tokens(true_ids)