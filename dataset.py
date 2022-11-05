import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def get_tag_keys(tag):
    if not tag:
        return tag

    tag = tag.replace('><', ',')
    tag = tag.replace('<', '')
    tag = tag.replace('>', '')

    return tag


def code_available(content):
    if '<' in content:
        return 1

    return 0


def remove_short_words(content):
    new_content_list = []
    for item in content:

        if len(item) > 3:
            new_content_list.append(item)

    return " ".join(new_content_list)


def clean_text(content):
    content = content.lower()

    content = re.sub(r"(@[A-Za-z0-9]+)|^rt|http.+?", "", content)
    content = re.sub(r"(\w+:\/\/\S+)", "", content)
    content = re.sub(r"([^0-9A-Za-z \t])", " ", content)
    content = re.sub(r"^rt|http.+?", "", content)
    content = re.sub(" +", " ", content)
    content = re.sub(r"\d+", "", content)

    return content


def remove_stopword(words):
    stop_words = set(stopwords.words('english'))
    list_clean = [w for w in words if not w in stop_words]
    return list_clean


def lemmatizer_words(content):
    lemmer = WordNetLemmatizer()
    new_content_list = []

    for item in content:
        new_content_list.append(lemmer.lemmatize(item))

    return new_content_list


def to_str(content):
    new_content_list = []
    for item in content:
        new_content_list.append(item)

    return " ".join(new_content_list)


def get_data():
    train = pd.read_parquet('files/train.parquet', engine='pyarrow')
    test = pd.read_parquet('files/test.parquet', engine='pyarrow')

    train['TagsKeys'] = train.Tags.apply(get_tag_keys)
    test['TagsKeys'] = test.Tags.apply(get_tag_keys)

    train['code_available'] = train['Body'].apply(code_available)
    test['code_available'] = test['Body'].apply(code_available)

    train['Body'] = train['Title'] + " " + train['Body'] + " " + train['TagsKeys']
    test['Body'] = test['Title'] + " " + test['Body'] + " " + test['TagsKeys']

    train['Body'] = train['Body'].apply(lambda x: str(x).split())
    train['Body'] = train['Body'].apply(remove_short_words)

    test['Body'] = test['Body'].apply(lambda x: str(x).split())
    test['Body'] = test['Body'].apply(remove_short_words)

    train['Body'] = train['Body'].apply(clean_text)
    test['Body'] = test['Body'].apply(clean_text)

    train['Body'] = train['Body'].apply(lambda x: str(x).split())
    train['Body'] = train['Body'].apply(remove_stopword)

    test['Body'] = test['Body'].apply(lambda x: str(x).split())
    test['Body'] = test['Body'].apply(remove_stopword)

    train_data = train
    test_data = test
    target = train.target
    train = train.drop(['Tags', 'Title', 'TagsKeys', 'target'], axis=1)
    test = test.drop(['Tags', 'Title', 'TagsKeys', ], axis=1)

    train['str'] = train['Body'].apply(to_str)
    test['str'] = test['Body'].apply(to_str)

    return train, test, target
