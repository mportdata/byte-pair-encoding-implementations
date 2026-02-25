def create_types(corpus: str) -> list[tuple[int, ...]]:
    words: list[SyntaxWarning] = corpus.split()   # split on any whitespace
    print("Total words in corpus:", len(words))
    print("First 10 words:", words[:10])
    if not words:
        return []
    
    types: list[str] = [words[0]] + [f"Ġ{word}" for word in words[1:]]
    print("Total types in corpus:", len(types))
    print("First 10 types:", types[:10])

    encoded_types: list[tuple[int, ...]] = [tuple(t.encode("utf-8")) for t in types]
    print("Converted text to types list (%d types)", len(encoded_types))
    print("First 10 byte encoded types:", encoded_types[:10])
    return encoded_types