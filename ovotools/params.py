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
        fold_test_substrs = ['/cam_7_7/', '/cam_7_8/', '/cam_7_9/'],
        fold_no = 0,
        model_name = 'model/inner_halo_types_m{inner_halo_params.margin}_w{inner_halo_params.loss_weights}',
        model_params = AttrDict(output_channels=3, enc_type='se_resnet50',
                                dec_type='unet_scse',
                                num_filters=16, pretrained=True),
        mean = (0.4138001444901419, 0.4156750182887099, 0.3766904444889663),
        std = (0.2965651186330059, 0.2801510185680299, 0.2719146471588908),
    )
    ...
    base_filename = my_models.ModelFileName(params)
    params.save(base_filename)
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
                           for x in vars(self).items() if not x[0].startswith('__')]) +
                ' \n}')
    
    def has(self, name):
        '''
        checks if self containes attribute with some name, including recursive, i.e. 'b.b1' etc.
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
        return hashlib.sha1(json.dumps(self, sort_keys=True).encode()).hexdigest()[:shrink_to]
        
    def save(self, base_fn, verbose = False):
        '''
        save to file adding '.param.txt' to name
        '''
        params_fn = base_fn + '.param.txt'
        assert not os.path.exists(params_fn), "Can't save parameters to {}: File exists".format(params_fn)
        with open(params_fn, 'w+') as f:
            s = repr(self)
            s = s + '\nhash: ' + self.hash()
            f.write(s)
            if verbose:
                print('params: '+ s + '\nsaved to ' + params_fn)
                
    def LoadFromStr(s):
        s = ''.join(s)
        s = s.replace('\n', '')
        assert len(s) >= 2
        assert s[0][0] == '{'
        assert s[-1][-1] == '}'
        params = ast.literal_eval(s)
        return AttrDict(params)
        
    def Load(params_fn, verbose = False):
        '''
        loads from file, adding '.param.txt' to name
        '''
        import ast
        with open(params_fn + '.param.txt') as f:
            s = f.readlines()
            assert s[-1].startswith('hash:')
            params = AttrDict.LoadFromStr(s[:-1])
        if verbose:
            print('params: '+ repr(params) + '\nhash: ' + params.hash() + '\nloaded from ' + params_fn)
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