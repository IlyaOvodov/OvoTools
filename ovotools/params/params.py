import hashlib
import json
import ast
import os
from collections import OrderedDict


class AttrDict(OrderedDict):
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
            elif isinstance(v, list):
                self[k] = [AttrDict(item) if isinstance(item, dict) else item for item in v]

    def __repr__(self):
        def write_item(item, margin='\n'):
            if isinstance(item, dict):
                s = '{'
                margin2 = margin + '    '
                for k, v in item.items():
                    if not k.startswith('__') and k != 'data_root':
                        s += margin2 + "'{0}': ".format(k) + write_item(v, margin=margin2) + ","
                if item.items():
                    s += margin
                s += '}'
            elif isinstance(item, (list, tuple)):
                s = '[' if isinstance(item, list) else '('
                for v in item:
                    if isinstance(v, dict):
                        s += margin + '    '
                    else:
                        s += ' '
                    s += write_item(v, margin=margin + '    ')  + ","
                s += ' ' + (']' if isinstance(item, list) else ')')
            else:
                s = repr(item)
            return s
        return write_item(self)

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
            s = str(self)
            s = s + '\nhash: ' + self.hash()
            f.write(s)
            if verbose >= 2:
                print('params: '+ s)
            if verbose >= 1:
                print('saved to ' + params_fn)
                
    def load_from_str(s, data_root):
        assert len(s) >= 2
        assert s[0][0] == '{' and s[-1][-2:] == '}\n'
        s = ''.join(s)
        s = s.replace('\n', '')
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
            print('params: '+ str(params) + '\nhash: ' + params.hash())
        if verbose >= 1:
            print('loaded from ' + str(params_fn))
        return params

        
if __name__=='__main__':
    m = AttrDict(
        data_root='abc',
        model_name='NN_results/segmentation/unet',
        data=AttrDict(
            val_folds=(4,),
            batch_size=100,
            resize=(128, 800),  # H, W
            crop=(128, 256),  # H, W
            train_augmentations=AttrDict(
                # HorizontalFlip = AttrDict(),
                # RandomBrightnessContrast = AttrDict(),
            ),
            crop_for_val=False,
            add_coordconv=False,
        ),
        model=AttrDict(
            type='segmentation_models_pytorch.Unet',
            params=AttrDict(
                encoder_name='resnet34',  # 'se_resnext50_32x4d' 'resnet34'
                encoder_weights='imagenet',
            ),
            # load_from = 'NN_results/segmentation/unet_66fa48/models/best.t7',
        ),
        dann=AttrDict(
            use_dann=False,
            lambda_max=1.,
            epochs=40,
            weight=0.1,
        ),
        loss=[
            AttrDict(
                type='torch.nn.BCEWithLogitsLoss',
                params=AttrDict(),
            ),
             AttrDict(
                type='pytorch_toolbelt.losses.dice.DiceLoss',
                params = AttrDict(
                    mode='multilabel',
                    log_loss=True,
                    smooth=1,
                ),
                weight = 0.5,
             ),
        ],
        optimizer=AttrDict(
            type='torch.optim.SGD',
            params=AttrDict(
                lr=0.2,
                momentum=0.9,
                weight_decay = 5e-4,  # 0.001,
                # nesterov = False,
            ),
        ),
        lr_finder = AttrDict(
            iters_num=200,
            log_lr_start=-4,
            log_lr_end=-0,
        ),
        lr_scheduler = AttrDict(
            type = 'torch.optim.lr_scheduler.ReduceLROnPlateau',
            clr=AttrDict(
                warmup_epochs=1,
                min_lr=0.0002,
                max_lr=1e-1,
                period_epochs=40,
                scale_max_lr=0.95,
                scale_min_lr=0.95,
            ),
            params=AttrDict( # ReduceLROnPlateau
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=2.e-4,
            ),
            StepLR_params=AttrDict(
                step_size=20,
                gamma=0.5,
            ),
            MultiStepLR_params=AttrDict(
                milestones=[25, 50, 75, 100, 125, 150, 175, 200, 215, 230, 245, 260, 275, 290, 300],
                gamma=0.5,
            ),
        ),
    )
    print(m)

    fn = 'test_' + m.hash()
    m.save(fn, can_overwrite=True)
    m.save(fn+'0', can_overwrite=True)
    mm = AttrDict.load(fn+'.param.txt')
    import os
    mm.save(fn, can_overwrite=True)
    print(m)
    print(mm)
    assert repr(m)==repr(mm)
    os.remove(fn + '.param.txt')
    os.remove(fn + '0.param.txt')
