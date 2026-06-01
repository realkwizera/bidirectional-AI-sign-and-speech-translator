def split_into_words(text):

    words = []
    current = ""

    for ch in text:

        if ch == " ":
            if current:
                words.append(current)
                current = ""
        else:
            current += ch

    if current:
        words.append(current)

    return words