import re
from collections import defaultdict
from collections import Counter


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_word(token):
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token


def split_sentence(sentence, n):
    words = defaultdict(int)
    # tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


def calculate_f1score(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    
    word_set = set(candidate_words.keys()).union(reference_words.keys())
    
    tp = 0  # True Positive：重叠部分
    fp = 0  # False Positive：候选中超出的部分
    fn = 0  # False Negative：参考中缺失的部分
    
    for word in word_set:
        cand_count = candidate_words.get(word, 0)
        ref_count = reference_words.get(word, 0)
        common = min(cand_count, ref_count)  # 重叠部分按照最小计数统计
        tp += common
        fp += (cand_count - common)
        fn += (ref_count - common)
    
    if sum(candidate_words.values()) == 0 or sum(reference_words.values()) == 0 or tp == 0:
        return 0, 0, 0  # 返回 (F1, precision, recall)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def calculate_f1_single_ref(candidate: str, reference: str) -> tuple[float, float, float]:
    """
    使用 collections.Counter 计算候选答案与单个参考答案之间的F1分数。
    这个版本在功能上与您提供的版本等价，但代码更简洁。
    """
    # 1. 标准化文本，这部分完全复用您强大的 normalize_word 函数
    candidate_norm = normalize_word(candidate)
    reference_norm = normalize_word(reference)

    # 2. 使用 Counter 直接进行分词和计数，替代 split_sentence(sentence, 1)
    candidate_tokens = Counter(candidate_norm.split())
    reference_tokens = Counter(reference_norm.split())

    # 如果任一答案为空，直接处理边缘情况
    if not candidate_tokens or not reference_tokens:
        return (1.0, 1.0, 1.0) if candidate_tokens == reference_tokens else (0.0, 0.0, 0.0)

    # 3. 使用 Counter 的交集(&)操作，替代手动循环计算 tp
    # common_tokens 是一个 Counter，包含了两个答案共有的词及其最小频次
    common_tokens = candidate_tokens & reference_tokens
    num_common = sum(common_tokens.values())

    # 如果没有共同的词，直接返回0
    if num_common == 0:
        return 0.0, 0.0, 0.0

    # 4. 计算 precision 和 recall
    # candidate_tokens 的总词数就是 tp + fp
    # reference_tokens 的总词数就是 tp + fn
    precision = num_common / sum(candidate_tokens.values())
    recall = num_common / sum(reference_tokens.values())

    # 5. 计算 F1 分数
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall

# # --- 使用示例 ---
# # 假设 normalize_word 和其依赖项已定义
# candidate_answer = "There are signs of cardiomegaly and a little pleural effusion."
# reference_answer = "cardiomegaly and pleural effusion"

# # 使用您原来的函数
# f1_orig, _, _ = calculate_f1score(candidate_answer, reference_answer)

# # 使用重构后的函数
# f1_new, precision_new, recall_new = calculate_f1_single_ref(candidate_answer, reference_answer)

# print(f"Original F1: {f1_orig:.4f}")
# print(f"Refactored F1: {f1_new:.4f}")
# print(f"Refactored Precision: {precision_new:.4f}")
# print(f"Refactored Recall: {recall_new:.4f}")