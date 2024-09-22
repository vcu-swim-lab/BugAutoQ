import string

printable = set(string.printable)


def clear_text(text, keep_punctuation=False):
    text = to_unicode(text)
    tokenized_text = list()
    for token in text.split(' '):
        word = u''
        code_token = u''
        for char in token:
            if char.isupper() and all(map(lambda x: x.isupper(), word)):
                # keep building if word is currently all uppercase
                word += char
                code_token += char
            elif char.islower() and all(map(lambda x: x.isupper(), word)):
                # stop building if word is currently all uppercase,
                # but be sure to take the first letter back
                # new version: emit splitted camel case but also preserve code token
                if len(word) > 1:
                    tokenized_text.append(word[:-1])
                    word = word[-1]
                word += char
                code_token += char
            elif char.islower() and any(map(lambda x: x.islower(), word)):
                # keep building if the word is has any lowercase
                # (word came from above case)
                word += char
                code_token += char
            elif char.isdigit() and all(map(lambda x: x.isdigit(), word)):
                # keep building if all of the word is a digit so far
                word += char
                code_token += char
            elif char in string.punctuation:
                if len(word) > 0 and not all(map(lambda x: x.isdigit(), word)):
                    tokenized_text.append(word)
                    if code_token != word:
                        tokenized_text.append(code_token)
                    code_token = u''
                    word = u''

                if keep_punctuation is True:
                    tokenized_text.append(char)

                # dont yield punctuation
                # yield char
            elif char == ' ':
                if len(word) > 0 and not all(map(lambda x: x.isdigit(), word)):
                    tokenized_text.append(word)
                    if code_token != word:
                        tokenized_text.append(code_token)

                word = u''
                code_token = u''
            else:
                if len(word) > 0 and not all(map(lambda x: x.isdigit(), word)):
                    tokenized_text.append(word)

                # to make sure we have only unicode characters
                word = u''
                word += char
                code_token += char

        tokenized_text.append(word)
        if code_token != word:
            tokenized_text.append(code_token)

    # we preserve spaces as tokens so we need to remove double-spaces after joining
    return ' '.join(tokenized_text).replace('  ', ' ').lower()


def to_unicode(document):
    printable = set(string.printable)
    # document = document.encode('ascii', errors='ignore').decode()
    # normal way above didn't work; we are doing it less smart way then... (below)
    document = ''.join(filter(lambda x: x in printable, document))
    document = document.replace('\x00', ' ')  # remove nulls
    document = document.replace('\r', '\n')
    document = document.replace('#', ' ')
    document = document.strip()
    return document
