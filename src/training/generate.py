import tensorflow as tf

def generate(model, tokenizer, prompt, max_tokens=100):
    ids = tokenizer.encode(prompt).ids
    ids=tf.constant([ids], dtype=tf.int32)
    for _ in range(max_tokens):
        logits = model(ids, training=False)
        next_token = tf.argmax(logits[:,-1,:], axis=-1)
        ids=tf.concat([ids, next_token[:,None]], axis=1)
    return tokenizer.decode(ids[0].numpy())
