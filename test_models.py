import os
import time
# for d in ['cifar10', 'cifar100', 'caltech101', 'caltech256'][2:3]:
#     for model in ['alexnet', 'vgg16']:
#         if os.path.exists('models/baseline/{1}_{0}.pt'.format(d, model)):
#             cmd = 'python training.py --dataset {0} --model_path models/baseline/{1}_{0}.pt --test_only --logfile ./test --outfile ./test --batch_size 64 --cuda'.format(d, model)
#             print('executing:',cmd)
#             os.system(cmd)
#         else:
#             print('models/baseline/{1}_{0}.pt does not exist.'.format(d, model))

models = [
    # ('cifar10', 'models/baseline/alexnet_cifar10.pt'),
    # ('cifar10', 'models/distilled/alexnet_cifar10_distilled_tuned.pt'),
    # ('cifar10', 'models/baseline/vgg16_cifar10.pt'),
    # ('cifar10', 'models/distilled/vgg16_cifar10_distilled_tuned.pt'),
    # ('cifar100', 'models/baseline/alexnet_cifar100.pt'),
    # ('cifar100', 'models/distilled/alexnet_cifar100_distilled_tuned.pt'),
    # ('cifar100', 'models/baseline/vgg16_cifar100.pt'),
    # ('cifar100', 'models/distilled/vgg16_cifar100_distilled_tuned.pt'),
    # ('caltech101', 'models/baseline/alexnet_caltech101_1C.pt'),
    # ('caltech101', 'models/distilled/alexnet_caltech101_1C_distilled_tuned.pt'),
    ('caltech101', 'models/distilled/alexnet_caltech101_allLayers_reverse_distilled_tuned.pt'),
    ('caltech101', 'models/distilled/alexnet_caltech101_allLayers_reverse_lw75_distilled_tuned.pt'),
    ('caltech101', 'models/distilled/alexnet_caltech101_allLayers_reverse_lw25_distilled_tuned.pt'),
    # ('caltech101', 'models/baseline/vgg16_caltech101_1C.pt'),
    # ('caltech101', 'models/distilled/vgg16_caltech101_1C_distilled_tuned.pt'),
    # ('caltech256', 'models/baseline/alexnet_caltech256_1C.pt'),
    # ('caltech256', 'models/distilled/alexnet_caltech256_1C_distilled_tuned.pt'),
    # ('caltech256', 'models/baseline/vgg16_caltech256_1C.pt'),
    # ('caltech256', 'models/distilled/vgg16_caltech256_1C_distilled_tuned.pt'),
]
for dataset, model_path in models:
    cmd = "CUDA_VISIBLE_DEVICE='1' python distillation.py --dataset %s --teacher_model_file %s --test_only --cuda --batch_size 8" % (dataset, model_path)
    print('executing:',cmd)
    os.system(cmd)
    time.sleep(5)