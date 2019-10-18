import random

def glue_augmentation(type='x', var_range=(0.1, 0.9), p=0.5, check_f = lambda i1,i2: True, **tensor_args):
    '''
    :param type: 'x' - glues 2 vertical slices
    :param range: cut by x in this range
    :param p: with probability p
    :param check_f: function gets 2 data ingexes and returns if they can be glued
    :param tensor_args: arguments with tensors (BxCxHxW) to be glued
    :return: None. modifies  tensor_args inplace.
    '''
    assert type == 'x'
    b = None
    for _, t in tensor_args.items():
        if b is None:
            b = t.shape[0]
        else:
            assert b == t.shape[0]
    assert b > 1, "Batch must be > 1 for glue_augmentation"
    if random.random() < p:
        for i1 in range(0, b-1, 2):
            i2 = i1+1
            if check_f(i1, i2):
                x = int(random.uniform(var_range[0], var_range[1])*t.shape[3])
                tmp = t[i1, :,:, :x].clone()
                t[i1, :, :, :x] = t[i2, :,:, :x].clone()
                t[i2, :, :, :x] = tmp


if __name__=='__main__':
    import torch
    t = torch.stack([torch.zeros(1,2,4), torch.ones(1,2,4), torch.zeros(1,2,4), torch.ones(1,2,4)])
    glue_augmentation(type='x', var_range=(0.5,0.5), p=1, t=t)
    print(t)
    assert( (t == torch.Tensor([[[[1., 1., 0., 0.],
          [1., 1., 0., 0.]]],


        [[[0., 0., 1., 1.],
          [0., 0., 1., 1.]]],


        [[[1., 1., 0., 0.],
          [1., 1., 0., 0.]]],


        [[[0., 0., 1., 1.],
          [0., 0., 1., 1.]]]]) ).all() )
