def str_to_bool(src : str):
    if src.lower() == 'true':
        return True
    elif src.lower() == 'false':
        return False
    else:
        return None

def str_to_int(src : str):
    if src.isdigit() == True:
        return int(src)
    else:
        return None

def str_to_float(src : str):
    if src.isdigit() == True:
        return float(src)
    else:
        return None

def str_to_value(src : str):
    value = str_to_bool(src)
    if value == None:
        value = str_to_int(src)
        if value == None:
            value == str_to_float(src)
            if value == None:
                value = src
    return value

def readconfig():
    with open('config.conf', mode='r', encoding='utf-8') as f:
        config_file = f.read()

        if '\r\n' in config_file:
            config_datas = config_file.split('\r\n')
        else:
            config_datas = config_file.split('\n')
        
    configs = {}
    for config_data in config_datas:
        if len(config_data) > 2 and config_data[0] != '#':
            config_data = config_data.replace(' ', '').replace('\t', '')
            config = config_data.split('=')

            if len(config) == 2:
                configs[config[0]] = str_to_value(config[1])

    return configs