from collections import Counter, defaultdict


def uniquify(model_names):
    names = []
    count_state = defaultdict(lambda: 1)
    counter = Counter(model_names)
    for model_name in model_names:
        if counter[model_name] == 1:
            names.append(model_name)
        else:
            names.append(model_name + f'({count_state[model_name]})')
            count_state[model_name] += 1
    return names

    # for name, count in counter.items():
    #     if count == 1:
    #         names.append(name)
    #     else:
    #         for i in range(count):
    #             names.append(name + f'({i+1})')
    # return names


models = ['mn', 'rn']
print(uniquify(models))
