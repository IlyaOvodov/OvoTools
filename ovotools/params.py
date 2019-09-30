import hashlib
import json
import ast
import os

class AttrDict(dict):
    '''
    dictionary allowing to access by dic[xxx] as well as dic.xxx syntax, including nested dictionaries:
        m = param.AttrDict(a = 1, b = {'b1': 2, 'b2':3}, c = 4)
        m.c = 5
        m.d = 6
        print(m.a, m.b.b1, m.b.b2, m.c, m.d)
    Example:
    from params import AttrDict
    params = AttrDict(
        data_root = local_config.data_path,
        model_name = 'model/inner_halo_types_m{inner_halo_params.margin}_w{inner_halo_params.loss_weights}',
        fold_test_substrs = ['/cam_7_7/', '/cam_7_8/', '/cam_7_9/'],
        fold_no = 0,
        model_params = AttrDict(output_channels=3, enc_type='se_resnet50',
                                dec_type='unet_scse',
                                num_filters=16, pretrained=True),
        mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
    )
    ...
    params.save()

    parameters 'data_root' and 'model_name' are required for save() and base_filename() functions.
    parameter 'data_root' is not stored and does not influence on hash
    '''
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for k,v in self.items():
            assert '.' not in k, "AttrDict: attribute '" + k + "' is invalid ('.' char is not allowed)"
            if isinstance(v, dict):
                self[k] = AttrDict(v)

    def __repr__(self):
        return ('{\n' + 
                '\n'.join([repr(x[0]) + ' : ' + repr(x[1]) + ',' 
                           for x in vars(self).items() if not x[0].startswith('__') and x[0] != 'data_root']) +
                ' \n}')

    def has(self, name):
        '''
        checks if self contains attribute with some name, including recursive, i.e. 'b.b1' etc.
        '''
        names = name.split('.')
        dic = self
        for n in names:
            if not hasattr(dic, n):
                return False
            dic = dic[n]
        return True
                
    def hash(self, shrink_to = 6):
        '''
        hash of dict values, invariant to values order
        '''
        hash_dict = self.copy()
        hash_dict.pop('data_root', None)
        return hashlib.sha1(json.dumps(hash_dict, sort_keys=True).encode()).hexdigest()[:shrink_to]

    def get_model_name(self):
        assert self.has('model_name')
        return self.model_name.format(**self) + '_' + self.hash()

    def get_base_filename(self):
        assert self.has('data_root')
        return os.path.join(self.data_root, self.get_model_name())

    def save(self, base_fn = None, verbose = 1, can_overwrite = False, create_dirs = False):
        '''
        save to file adding '.param.txt' to name
        '''
        if base_fn is None:
            base_fn = self.get_base_filename()
        params_fn = base_fn + '.param.txt'
        if not can_overwrite:
            assert not os.path.exists(params_fn), "Can't save parameters to {}: File exists".format(params_fn)
        if create_dirs:
            dir_name = os.path.dirname(params_fn)
            os.makedirs(dir_name, exist_ok=True)
        with open(params_fn, 'w+') as f:
            s = repr(self)
            s = s + '\nhash: ' + self.hash()
            f.write(s)
            if verbose >= 2:
                print('params: '+ s)
            if verbose >= 1:
                print('saved to ' + params_fn)
                
    def load_from_str(s, data_root):
        s = ''.join(s)
        s = s.replace('\n', '')
        assert len(s) >= 2
        assert s[0][0] == '{'
        assert s[-1][-1] == '}'
        params = ast.literal_eval(s)
        if data_root:
            params.data_root = data_root
        return AttrDict(params)
        
    def load(params_fn, data_root = None, verbose = 1):
        '''
        loads from file, adding '.param.txt' to name
        '''
        import ast
        with open(params_fn) as f:
            s = f.readlines()
            assert s[-1].startswith('hash:')
            params = AttrDict.load_from_str(s[:-1], data_root)
        if verbose >= 2:
            print('params: '+ repr(params) + '\nhash: ' + params.hash())
        if verbose >= 1:
            print('loaded from ' + params_fn)
        return params

        
if __name__=='__main__':        
    m = AttrDict(b = AttrDict(b1="b1v", b2 = "qwe"), a=1, c = "qweqweqwe")
    fn = 'test_' + m.hash()
    m.save(fn)
    mm = AttrDict.Load(fn)
    import os
    os.remove(fn + '.param.txt')
    print(m)
    print(mm)
    assert str(m)==str(mm)