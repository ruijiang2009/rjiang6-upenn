
def compute_pair_freqs(word_freqs, splits):
  """
  Compute the frequency of pairs of tokens

  @param1 word_freqs: the dictionary we get in 2.3.1
  @param2 splits: the dictionary we get in 2.3.2
  @return: a dictionary in which keys are pairs of tokens,
    values are frequencies
  """
  pair_freqs = {}

  for word in splits:
    split_len = len(splits[word])
    for i in range(split_len -1):
      pair = (splits[word][i], splits[word][ i +1])
      if pair not in pair_freqs:
        pair_freqs[pair] = 0
      pair_freqs[word] = pair_freqs[pair] + word_freqs[word]

  return pair_freqs


def merge_pair(a, b, splits):
    """
    Merge the a and b into ab and return the updated splits dict
    This function will be used repeatedly

    @param1 a: the first token to be merged
    @param2 b: the second token to be merged
    @param3 splits: a dictionary in which keys are pairs of tokens,
      values are frequencies
    @return: the updated splits
    """
    updated_splits = {}
    target_pair = (a, b)
    for word in splits:
      split_len = len(splits[word])
      new_splits = []
      i = 0
      while i < split_len:
        if i == split_len - 1:
          new_splits.append(splits[word][i])
          i += 1
        else:
          pair = (splits[word][i], splits[word][i + 1])
          if pair == target_pair:
            new_splits.append(a + b)
            i += 2
          else:
            new_splits.append(splits[word][i])
            i += 1
      updated_splits[word] = new_splits
    return updated_splits

word_freqs = {'april': 2, 'proud': 3}


def test1():
  splits = {'april': ['a', 'p', 'r', 'i', 'l'], 'proud': ['p', 'r', 'o', 'u', 'd']}
  a = 'p'
  b = 'r'
  updated_splits = merge_pair(a, b, splits)
  print(updated_splits)


if __name__ == '__main__':
  test1()