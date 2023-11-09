
def estimate_price(num_of_episodes, in_context_len,tokens_per_step, n_steps, last_n, model, feed_episode_history):
    max_context_size = 10e10
    price_per_1k_big_context = 0

    if model == "text-ada-001":
        price_per_1k = 0.0004
    elif model == "text-curie-001":
        price_per_1k = 0.003
    elif model == "text-davinci-003":
        price_per_1k = 0.02
    elif model == "gpt-3.5-turbo-0301":
        price_per_1k = 0.0015
        max_context_size = 4000
        price_per_1k_big_context = 0.003
    elif model == "gpt-3.5-turbo-instruct-0914":
        price_per_1k = 0.0015
        max_context_size = 4000
        price_per_1k_big_context = 0.003
    elif model == "gpt-3.5-turbo-0613":
        price_per_1k = 0.0015
        max_context_size = 4000
        price_per_1k_big_context = 0.003
    elif model == "gpt-4-0314":
        price_per_1k = 0.03
        max_context_size = 8000
        price_per_1k_big_context = 0.06
    elif model == "gpt-4-0613":
        price_per_1k = 0.03
        max_context_size = 8000
        price_per_1k_big_context = 0.06

    else:
        print(f"Price for model {model} not found.")
        price_per_1k = 0

    # check if the maximum context size if bigger than default (4k gpt-3.5;8k gpt-4) and update the price accordingly
    if (
            feed_episode_history and in_context_len+n_steps*tokens_per_step
    ) or (
            not feed_episode_history and in_context_len+last_n * tokens_per_step > max_context_size
    ):
        # context is bigger, update the price
        assert "gpt-4" in model or "gpt-3.5" in model
        price_per_1k = price_per_1k_big_context

    if feed_episode_history:
        total_tokens = num_of_episodes*(in_context_len + tokens_per_step*sum(range(n_steps)))

    else:
        total_tokens = num_of_episodes*n_steps*(in_context_len + last_n*tokens_per_step)

    price = (total_tokens/1000)*price_per_1k
    return total_tokens, price


if __name__ == "__main__":
    total_tokens, price = estimate_price()
    print("tokens:", total_tokens)
    print("price:", price)