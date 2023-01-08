from sklearn.feature_extraction.text import CountVectorizer




def split_sentence_into_words(sentence: str):
    sentence = sentence.lower()
    vectorizer = CountVectorizer()
    tokener = vectorizer.build_tokenizer()
    words = tokener(sentence)
    return words


# seps = ['.', ',', '!', '?', ':', '/', '@', '&', '*', '$', '#', '(', ')', '<', '>', '[', ']', '{', '}']
# def split_sentence_into_words_old(sentence: str):
#     sentence = sentence.lower()
#     for item in seps:
#         sentence = sentence.replace(item, ' ')
#     words = sentence.split()
#     return words




def words_hit_keywords(words_in_sentence, keywords, return_labels, return_origin_index):
    hit_words = []
    origin_index = []
    labels = []
    for index, word in enumerate(words_in_sentence):
        if (word in keywords.keywords_to_label):
            hit_words.append(word)
            origin_index.append(index)
            labels.append(keywords.keywords_to_label[word])

    ans = []
    ans.append(hit_words)
    if(return_labels):
        ans.append(labels)
    if(return_origin_index):
        ans.append(origin_index)

    if(len(ans)>1):
        return tuple(ans)
    else:
        return ans[0]


def get_sentence_hit_keywords(sentence, keywords, return_labels = False, return_origin_index = False):
    words = split_sentence_into_words(sentence)
    ans = words_hit_keywords(words, keywords, return_labels, return_origin_index)
    return ans


if __name__ == '__main__':
    s = 'goldviking (29/m) is inviting you to be his friend. http://www.baidu.com reply yes-762 or no-762 see him: www.sms.ac/u/goldviking stop? send stop frnd to 62468'

    words = split_sentence_into_words(s)
    print(words)
