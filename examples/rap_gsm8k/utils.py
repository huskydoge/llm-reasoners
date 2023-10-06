import re
from typing import Optional, Union

def is_number_or_symbol(s: str) -> bool:
    match = re.match(r'[0-9$.,\- ]', s)
    if match is None:
        return False
    return True
def retrieve_answer(output: Union[list, str]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, list):
        output = output[-1].sub_question
    match = re.match(r'.*[Tt]he answer is .*?([ $.0-9,\-]+).*\.', output)

        
    if match is None:
        return None
    dot_idx = match[0].rfind('.')
    end = 0
    start = 100
    ans = ''
    for i in reversed(range(dot_idx)):
        if is_number_or_symbol(match[0][i]):
            if end == 0:
                end = i
        else:
            if start == 100 and end != 0:
                start = i+1
                if match[0][start:end+1] != ' ':
                    ans = match[0][start:end+1]
                    break
            else:
                end = 0
                start = 100
    answer = ans.replace(',', '').replace('$', '').replace(' ', '')
    # answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer

if __name__ == '__main__':
    print(retrieve_answer("The answer is 3 * 365 = 1095 pages a year."))