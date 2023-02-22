import re
from thefuzz import fuzz


def no_accent_vietnamese(s):
    s = re.sub("[áàảãạăắằẳẵặâấầẩẫậ]", "a", s)
    s = re.sub("[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]", "A", s)
    s = re.sub("[éèẻẽẹêếềểễệ]", "e", s)
    s = re.sub("[ÉÈẺẼẸÊẾỀỂỄỆ]", "E", s)
    s = re.sub("[óòỏõọôốồổỗộơớờởỡợ]", "o", s)
    s = re.sub("[ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ]", "O", s)
    s = re.sub("[íìỉĩị]", "i", s)
    s = re.sub("[ÍÌỈĨỊ]", "I", s)
    s = re.sub("[úùủũụưứừửữự]", "u", s)
    s = re.sub("[ÚÙỦŨỤƯỨỪỬỮỰ]", "U", s)
    s = re.sub("[ýỳỷỹỵ]", "y", s)
    s = re.sub("[ÝỲỶỸỴ]", "Y", s)
    s = re.sub("đ", "d", s)
    s = re.sub("Đ", "D", s)
    return s


def compare_vietnamese(s1, s2, thresh=90):
    s1 = no_accent_vietnamese(s1).lower()
    s1 = "".join(ch for ch in s1 if ch.isalnum() or ch == " ")
    s2 = no_accent_vietnamese(s2).lower()
    s2 = "".join(ch for ch in s2 if ch.isalnum() or ch == " ")
    if fuzz.ratio(s1, s2) > thresh:
        return True
    else:
        return False


def check_text_in_list_vietnamese(text, list_text):
    for t in list_text:
        if compare_vietnamese(text, t):
            return True
    return False


def check_start_with_vietnamese(s1, s2):
    """
    check if s1 start with s2
    """
    s1 = no_accent_vietnamese(s1).lower()
    s2 = no_accent_vietnamese(s2).lower()
    if " " not in s2:
        s2 += " "
    len_s2 = len(s2.split())
    if fuzz.ratio(" ".join(s1.split()[:len_s2]), s2) > 80:
        return True
    else:
        return False


def check_text_contain_digit(text):
    for character in text:
        if character.isdigit():
            return True
    return False


def replace_special_char_by_space(
    text, remove_char_list='!"#$%&' "()*+,-./:;<=>?@[\]^_`{|}~"
):
    for remove_char in remove_char_list:
        text = text.replace(remove_char, " ")
    return text


def clean_text(text):
    """
    remove all special char but ':', '/', '-', ','
    """
    new_text = ""
    for word in text.split():
        if check_text_contain_digit(word):
            # if it contains digit, do not remove '.'
            remove_char_list = '!"#$%&' "()*+:;<=>?@[\]^_`{|}~"
        else:
            remove_char_list = '!"#$%&' "()*+.:;<=>?@[\]^_`{|}~"
        for remove_char in remove_char_list:
            word = word.replace(remove_char, " ")
        new_text += " " + word
    new_text = " ".join(new_text.split())
    return new_text


def clean_text_list(texts):
    new_texts = []
    for text in texts:
        new_text = clean_text(text)
        new_texts.append(new_text)
    return new_texts


def split_with_idx(s):
    list_s = []
    list_idx = []

    new_word = ""
    for i, c in enumerate(s + " "):
        if c != " ":
            new_word += c
        elif new_word != "":
            list_s.append(new_word)
            list_idx.append(i - len(new_word))
            new_word = ""

    return list_s, list_idx


def find_increase_sub_array(arr, min=-1):
    rs = []
    for i, start_v in enumerate(arr):
        if start_v > min:
            if i < len(arr) - 1:
                for j in find_increase_sub_array(arr[i + 1 :], start_v):
                    if j[0] < 2:
                        rs.append([i, i + j[1] + 1])
            rs.append([i, i])
    new_rs = []
    for i in rs:
        if i not in new_rs:
            new_rs.append(i)
    return new_rs


def find_similar_substring_idx(
    text, search_key, word_thresh=50, string_thresh=80, use_re=True
):
    """
    Return:
    - list of [start_idx, end_idx] which text[start_idx, end_idx] is similar with search_key
    """
    rs = []
    text = no_accent_vietnamese(text).lower()
    search_key = no_accent_vietnamese(search_key).lower()

    # find first substring with 100% similarity by re
    if use_re:
        first_100_match = re.search(search_key, text)
        if first_100_match is not None:
            rs = find_similar_substring_idx(
                text[: first_100_match.start()],
                search_key,
                word_thresh,
                string_thresh,
                use_re=False,
            )
            rs.append([first_100_match.start(), first_100_match.end()])
            for (start, end) in find_similar_substring_idx(
                text[first_100_match.end() :], search_key, word_thresh, string_thresh
            ):
                rs.append([start + first_100_match.end(), end + first_100_match.end()])
            return rs

    text = replace_special_char_by_space(text)
    search_key = replace_special_char_by_space(search_key)

    list_text_word, list_text_word_idx = split_with_idx(text)
    list_search_key_word = search_key.split()

    similar_word_key_idx = [-1 for _ in list_text_word]
    for i, word in enumerate(list_text_word):
        for j, key in enumerate(list_search_key_word):
            similar_score = fuzz.ratio(word, key)
            if similar_score > word_thresh:
                if (
                    similar_word_key_idx[i] > -1
                    and fuzz.ratio(word, list_search_key_word[similar_word_key_idx[i]])
                    > similar_score
                ):
                    continue
                similar_word_key_idx[i] = j

    for (start, end) in find_increase_sub_array(similar_word_key_idx):
        if (
            fuzz.ratio(" ".join(list_text_word[start : end + 1]), search_key)
            > string_thresh
        ):
            rs.append(
                [
                    list_text_word_idx[start],
                    list_text_word_idx[end] + len(list_text_word[end]),
                ]
            )

    return rs


def find_first_similar_substring_idx(
    text, search_key, word_thresh=50, string_thresh=95
):
    if isinstance(search_key, list):
        final_rs = (None, None)
        for key in search_key:
            start, end = find_first_similar_substring_idx(
                text, key, word_thresh, string_thresh
            )
            if final_rs[0] is None:
                final_rs = (start, end)
            elif start is not None and start < final_rs[0]:
                final_rs = (start, end)
        return final_rs

    rs = find_similar_substring_idx(
        text, search_key, word_thresh=word_thresh, string_thresh=string_thresh
    )
    if len(rs) > 0:
        rs = sorted(rs, key=lambda x: x[0])
        return rs[0]
    return None, None
