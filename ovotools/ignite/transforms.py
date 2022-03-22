def from_dict(key):
    '''
    Creates transform function extracting value from dict if input is dict
    :param key: string
    :return: functions: x -> x[key] if x is dict, x otherwise
    '''

    def call(data):
        if isinstance(data, dict):
            return data[key]
        elif isinstance(data, tuple):
            return tuple(call(di) for di in data)
        else:
            return data

    return call