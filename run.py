from fuzzylm.logic import NGram


def main() -> None:
    train_text = [
        "a computer can help you",
        "he wants to help you",
        "he wants a computer",
        "he can swim",
    ]
    train_data = [s.split() for s in train_text]

    model = NGram(n=2, alpha=1.0, fuzzy=True)
    model.fit(train_data)

    sample = "he can hope you".split()
    score = model.log_text_prob(sample)

    print("Training samples:")
    for line in train_text:
        print(f"- {line}")

    print("\nModel settings:")
    print(f"- n: {model.n}")
    print(f"- alpha: {model.alpha}")
    print(f"- fuzzy: {model.fuzzy}")

    print("\nScoring sample text:")
    print(f"- text: {' '.join(sample)}")
    print(f"- log probability: {score}")


if __name__ == "__main__":
    main()
