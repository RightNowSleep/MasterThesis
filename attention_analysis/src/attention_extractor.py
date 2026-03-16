import torch


def extract_attention_from_model(model, tokenizer, input_text, max_length=512):
    """
    Run the model on given input text and extract attention weights from all layers.
    Returns:
        tokens: list of decoded tokens
        attentions: a tuple containing attention weights for each layer [num_layers, num_heads, seq_len, seq_len]
    """
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple, each element is [batch_size, num_heads, seq_len, seq_len]
    # We take batch_size=0
    raw_attentions = tuple(att[0].cpu() for att in outputs.attentions)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return tokens, raw_attentions
