def load_model_opt(path, opt=None):
    if not opt:
        opt = type('Opt', (object,), {})()
    load_list = {'arch': str, 'fc_n': int, 'input_nc': int,
                 'mode': str, 'ncf': list, 'neigbs': int,
                 'ninput_edges': int, 'nclasses': int,
                 'aug_method': str, 'num_aug': int,
                 'scale_verts': bool, 'flip_edges': float,
                 'slide_verts': float}
    with open(path) as f:
        lines = f.readlines()
        for line in lines[1:-1]:
            name, value = line.strip('\n').split(': ')
            if name in load_list.keys():
                if load_list[name] == list:
                    value = value.strip('[] ').split(',')
                    setattr(opt, name, [int(x) for x in value])
                else:
                    value = load_list[name](value)
                    setattr(opt, name, value)
    return opt
