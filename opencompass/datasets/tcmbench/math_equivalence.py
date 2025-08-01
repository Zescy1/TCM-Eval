# flake8: noqa


# code from https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace('\n', '')
    # print(string)

    # remove inverse spaces
    string = string.replace('\\!', '')
    # print(string)

    # replace \\ with \
    string = string.replace('\\\\', '\\')
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    # print(string)

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    # print(string)

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False, strict=False, ignore_order=True):
    """
    判断两个答案是否相等，支持 str/list/tuple/None 类型组合
    
    Args:
        str1 (Union[str, list, tuple, None]): 预测结果
        str2 (Union[str, list, tuple, None]): 正确答案
        verbose (bool): 是否打印详细信息
        strict (bool): 是否开启严格模式（完全匹配），默认宽松匹配
        ignore_order (bool): 是否忽略顺序，默认忽略
        
    Returns:
        bool: 是否等价
    """
    # 处理 None 情况
    if str1 is None and str2 is None:
        print('WARNING: Both None')
        return True
    if str1 is None or str2 is None:
        if verbose:
            print(f"❌ 一个为 None: {str1}, {str2}")
        return False

    # 统一类型处理：str -> list，其他保持原样
    def _convert(x):
        if isinstance(x, str):
            if not strict:
                x = _strip_string(x)
            return list(x)
        elif isinstance(x, (list, tuple)):
            return list(x)
        else:
            return [x]

    try:
        a = _convert(str1)
        b = _convert(str2)

        if verbose:
            print(f"🧠 转换后比较: {a} vs {b}")

        if ignore_order:
            result = sorted(a) == sorted(b)
        else:
            result = a == b

        if verbose and result:
            print("✅ 匹配成功")
        elif verbose:
            print("❌ 匹配失败")

        return result

    except Exception as e:
        if verbose:
            print(f"⚠️ 异常发生: {e}")
            print(f"原始值比较: {str1} vs {str2}")
        return str1 == str2