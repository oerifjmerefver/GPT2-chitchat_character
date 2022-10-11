from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
import torch.nn.functional as F
import torch
import os
import argparse

from mytokenizer import MyTokenizer

PAD = '[PAD]'
pad_id = 0


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='model/epoch40', type=str, required=False, help='对话模型路径')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=15, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def gpt(text: str, history : list, model, tokenizer, args, device):
    history.append(text)
    input_ids = tokenizer.encode_utterances(history[-args.max_history_len:])

    input_ids = torch.tensor(input_ids).long().to(device)
    input_ids = input_ids.unsqueeze(0)

    response = []
    for _ in range(args.max_len):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        for id in set(response):
            next_token_logits[id] /= args.repetition_penalty
        next_token_logits = next_token_logits / args.temperature
        next_token_logits[tokenizer.convert_tokens_to_ids(
            '[UNK]')] = -float('Inf')
        next_token_logits[tokenizer.convert_tokens_to_ids(
            '[UNK]') + tokenizer.tokenizer.vocab_size] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits,
                                                top_k=args.topk,
                                                top_p=args.topp)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1),
                                       num_samples=1)
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        if next_token < tokenizer.tokenizer.vocab_size:
            history.pop()
            return gpt(text, model, history)
        response.append(next_token.item())
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
    
    text = tokenizer.convert_ids_to_tokens(response, True)
    reply = "".join(text)
    reply = reply.replace('#', '')
    history.append(reply)
    return reply

def main():
    args = set_args()
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = MyTokenizer(args.vocab_path)
    # tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        text = input("user:")
        
        reply = gpt(text, history, model, tokenizer, args, device)
        print("chatbot:" + reply)


if __name__ == '__main__':
    main()