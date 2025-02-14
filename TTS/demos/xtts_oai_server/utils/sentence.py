
def split_sentence(sentence, delimiters=",;-!?"):
    """
    Splits a sentence into two halves, prioritizing the delimiter closest to the middle.
    If no delimiter is found, it ensures words are not split in the middle.

    Args:
        sentence (str): The input sentence to split.
        delimiters (str): A string of delimiters to prioritize for splitting (default: ",;!?").

    Returns:
        tuple: A tuple containing the two halves of the sentence.
    """
    # Find all delimiter indices in the sentence
    delimiter_indices = [i for i, char in enumerate(sentence) if char in delimiters]

    if delimiter_indices:
        # Calculate the midpoint of the sentence
        midpoint = len(sentence) // 2

        # Find the delimiter closest to the midpoint
        closest_delimiter = min(delimiter_indices, key=lambda x: abs(x - midpoint))

        # Split at the closest delimiter
        first_half = sentence[:closest_delimiter].strip()
        second_half = sentence[closest_delimiter + 1:].strip()
    else:
        # If no delimiter, split at the nearest space (word boundary)
        midpoint = len(sentence) // 2

        # Find the nearest space (word boundary) around the midpoint
        left_space = sentence.rfind(" ", 0, midpoint)
        right_space = sentence.find(" ", midpoint)

        # Choose the closest space to the midpoint
        if left_space == -1 and right_space == -1:
            # No spaces found (single word), split at midpoint
            split_index = midpoint
        elif left_space == -1:
            # Only right space found
            split_index = right_space
        elif right_space == -1:
            # Only left space found
            split_index = left_space
        else:
            # Choose the closest space to the midpoint
            split_index = left_space if (midpoint - left_space) <= (right_space - midpoint) else right_space

        # Split the sentence into two parts
        first_half = sentence[:split_index].strip()
        second_half = sentence[split_index:].strip()

    return first_half, second_half


def merge_sentences(sentences):
    """ handling short sentences by merging them to next/prev ones """
    merged_sentences = []
    i = 0
    while i < len(sentences): 
        s = sentences[i]
        word_count = len(s.split())
        j = 1
        # merge the short sentence to the next one until long enough
        while word_count <= 6 and i+j < len(sentences):
            s += ' ' + sentences[i+j]
            word_count = len(s.split())
            j += 1
        merged_sentences.append(s)
        i += j
    # merge the last one to the prev one until long enough
    while len(merged_sentences) > 1 and len(merged_sentences[len(merged_sentences) - 1].split()) < 6:
        merged_sentences[len(merged_sentences) - 2] += ' ' + merged_sentences[len(merged_sentences) - 1]
        merged_sentences.pop()
    return merged_sentences