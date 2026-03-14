import torch


def extract_attention_from_model(model, tokenizer, input_text, max_length=512):
    """
    对给定输入文本，运行模型并提取所有层的注意力权重。
    Returns:
        tokens: 解码后的token列表
        attentions: 一个元组，包含每层的注意力权重 [num_layers, num_heads, seq_len, seq_len]
    """
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions 是一个元组，每个元素是 [batch_size, num_heads, seq_len, seq_len]
    # 我们取 batch_size=0
    raw_attentions = tuple(att[0].cpu() for att in outputs.attentions)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    return tokens, raw_attentions