import json
import os
from collections import OrderedDict
from copy import deepcopy
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation # resize_softmax_output
from skimage.transform import resize
from torch.optim import lr_scheduler
from torch import nn
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes

'''
This code is not intended to be looked at by anyone. It is messy. It is undocumented.
And the entire training pipeline is missing.
'''

max_num_filters_3d = 320
max_num_filters_2d = 480
join = os.path.join


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def resize_image(image, old_spacing, new_spacing, order=3, cval=0):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    if any([i != j for i, j in zip(image.shape, new_shape)]):
        res = resize(image, new_shape, order=order, mode='edge', cval=cval)
    else:
        res = image
    return res


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = nn.LeakyReLU(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

def pad_nd_image(image, new_shape=None, mode="edge", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    if kwargs is None:
        kwargs = {}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])
    res = np.pad(image, pad_list, mode, **kwargs)
    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        self.input_shape_must_be_divisible_by = None
        self.conv_op = None
        super(NeuralNetwork, self).__init__()
        self.inference_apply_nonlin = lambda x:x

    def predict_3D(self, x, do_mirroring, num_repeats=1, use_train_mode=False, batch_size=1, mirror_axes=(2, 3, 4),
                   tiled=False, tile_in_z=True, step=2, patch_size=None, regions_class_order=None, use_gaussian=False,
                   pad_border_mode="edge", pad_kwargs=None):
        """
        :param x: (c, x, y , z)
        :param do_mirroring:
        :param num_repeats:
        :param use_train_mode:
        :param batch_size:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param patch_size:
        :param regions_class_order:
        :param use_gaussian:
        :return:
        """
        current_mode = self.training
        if use_train_mode is not None and use_train_mode:
            self.train()
        elif use_train_mode is not None and not use_train_mode:
            self.eval()
        else:
            pass
        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"
        if self.conv_op == nn.Conv3d:
            if tiled:
                res = self._internal_predict_3D_3Dconv_tiled(x, num_repeats, batch_size, tile_in_z, step, do_mirroring,
                                                             mirror_axes, patch_size, regions_class_order, use_gaussian,
                                                             pad_border_mode, pad_kwargs=pad_kwargs)
            else:
                res = self._internal_predict_3D_3Dconv(x, do_mirroring, num_repeats, patch_size, batch_size,
                                                       mirror_axes, regions_class_order, pad_border_mode, pad_kwargs=pad_kwargs)
        elif self.conv_op == nn.Conv2d:
            if tiled:
                res = self._internal_predict_3D_2Dconv_tiled(x, do_mirroring, num_repeats, batch_size, mirror_axes,
                                                             step, patch_size, regions_class_order, use_gaussian,
                                                             pad_border_mode, pad_kwargs=pad_kwargs)
            else:
                res = self._internal_predict_3D_2Dconv(x, do_mirroring, num_repeats, patch_size, batch_size,
                                                       mirror_axes, regions_class_order, pad_border_mode, pad_kwargs=pad_kwargs)
        else:
            raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")
        if use_train_mode is not None:
            self.train(current_mode)
        return res

    def _internal_maybe_mirror_and_pred_3D(self, x, num_repeats, mirror_axes, do_mirroring=True):
        with torch.no_grad():
            a = torch.zeros(x.shape).float()
            if self.get_device() == "cpu":
                a = a.cpu()
            else:
                a = a.cuda(self.get_device())

            if do_mirroring:
                mirror_idx = 8
            else:
                mirror_idx = 1
            all_preds = []
            for i in range(num_repeats):
                for m in range(mirror_idx):
                    data_for_net = np.array(x)
                    do_stuff = False
                    if m == 0:
                        do_stuff = True
                        pass
                    if m == 1 and (4 in mirror_axes):
                        do_stuff = True
                        data_for_net = data_for_net[:, :, :, :, ::-1]
                    if m == 2 and (3 in mirror_axes):
                        do_stuff = True
                        data_for_net = data_for_net[:, :, :, ::-1, :]
                    if m == 3 and (4 in mirror_axes) and (3 in mirror_axes):
                        do_stuff = True
                        data_for_net = data_for_net[:, :, :, ::-1, ::-1]
                    if m == 4 and (2 in mirror_axes):
                        do_stuff = True
                        data_for_net = data_for_net[:, :, ::-1, :, :]
                    if m == 5 and (2 in mirror_axes) and (4 in mirror_axes):
                        do_stuff = True
                        data_for_net = data_for_net[:, :, ::-1, :, ::-1]
                    if m == 6 and (2 in mirror_axes) and (3 in mirror_axes):
                        do_stuff = True
                        data_for_net = data_for_net[:, :, ::-1, ::-1, :]
                    if m == 7 and (2 in mirror_axes) and (3 in mirror_axes) and (4 in mirror_axes):
                        do_stuff = True
                        data_for_net = data_for_net[:, :, ::-1, ::-1, ::-1]

                    if do_stuff:
                        _ = a.data.copy_(torch.from_numpy(np.copy(data_for_net)))
                        p = self.inference_apply_nonlin(self(a))
                        p = p.data.cpu().numpy()

                        if m == 0:
                            pass
                        if m == 1 and (4 in mirror_axes):
                            p = p[:, :, :, :, ::-1]
                        if m == 2 and (3 in mirror_axes):
                            p = p[:, :, :, ::-1, :]
                        if m == 3 and (4 in mirror_axes) and (3 in mirror_axes):
                            p = p[:, :, :, ::-1, ::-1]
                        if m == 4 and (2 in mirror_axes):
                            p = p[:, :, ::-1, :, :]
                        if m == 5 and (2 in mirror_axes) and (4 in mirror_axes):
                            p = p[:, :, ::-1, :, ::-1]
                        if m == 6 and (2 in mirror_axes) and (3 in mirror_axes):
                            p = p[:, :, ::-1, ::-1, :]
                        if m == 7 and (2 in mirror_axes) and (3 in mirror_axes) and (4 in mirror_axes):
                            p = p[:, :, ::-1, ::-1, ::-1]
                        all_preds.append(p)
        return np.vstack(all_preds)


    def _internal_predict_3D_3Dconv(self, x, do_mirroring, num_repeats, min_size=None, BATCH_SIZE=None,
                                    mirror_axes=(2, 3, 4), regions_class_order=None, pad_border_mode="edge",
                                    pad_kwargs=None):
        with torch.no_grad():
            x, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True, self.input_shape_must_be_divisible_by)
            #x, old_shape = pad_patient_3D_incl_c(x, self.input_shape_must_be_divisible_by, min_size)

            new_shp = x.shape

            data = np.zeros(tuple([1] + list(new_shp)), dtype=np.float32)

            data[0] = x

            if BATCH_SIZE is not None:
                data = np.vstack([data] * BATCH_SIZE)

            stacked = self._internal_maybe_mirror_and_pred_3D(data, num_repeats, mirror_axes, do_mirroring)

            slicer = [slice(0, stacked.shape[i]) for i in range(len(stacked.shape) - (len(slicer) - 1))] + slicer[1:]
            stacked = stacked[slicer]
            uncertainty = stacked.var(0)
            bayesian_predictions = stacked
            softmax_pred = stacked.mean(0)

            if regions_class_order is None:
                predicted_segmentation = softmax_pred.argmax(0)
            else:
                predicted_segmentation_shp = softmax_pred[0].shape
                predicted_segmentation = np.zeros(predicted_segmentation_shp)
                for i, c in enumerate(regions_class_order):
                    predicted_segmentation[softmax_pred[i] > 0.5] = c
        return predicted_segmentation, bayesian_predictions, softmax_pred, uncertainty

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None):
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([ConvDropoutNormNonlin(input_feature_channels, output_feature_channels, self.conv_op,
                                     self.conv_kwargs_first_conv,
                                     self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                     self.nonlin, self.nonlin_kwargs)] +
              [ConvDropoutNormNonlin(output_feature_channels, output_feature_channels, self.conv_op,
                                     self.conv_kwargs,
                                     self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                     self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1.):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()
    return result

def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input


class Generic_UNet_Cotraining(SegmentationNetwork):
    def __init__(self, input_channels, base_num_features, num_classes, num_conv_per_stage=2, num_downscale=4,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False):
        """
        Have fun lookint at that one. This is my go-to model. I crammed the cotraining code in there somehow, so yeah.
        What a mess.
        You know what's the best part? No documentation. What a great piece of code.

        :param input_channels:
        :param base_num_features:
        :param num_classes:
        :param num_conv_per_stage:
        :param num_downscale:
        :param feat_map_mul_on_downscale:
        :param conv_op:
        :param conv_kwargs:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param deep_supervision:
        :param dropout_in_localization:
        :param final_nonlin:
        :param weightInitializer:
        :param pool_op_kernel_sizes:
        :param upscale_logits:
        :param convolutional_pooling:
        :param convolutional_upsampling:
        """
        super(Generic_UNet_Cotraining, self).__init__()
        assert isinstance(num_classes, (list, tuple)), "for cotraining, num_classes must be list or tuple of int"
        self.num_classes = num_classes
        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes

        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
             nonlin_kwargs = {'negative_slope':1e-2, 'inplace':True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p':0.5, 'inplace':True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps':1e-5, 'affine':True, 'momentum':0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size':3, 'stride':1, 'padding':1, 'dilation':1, 'bias':True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op

        if pool_op_kernel_sizes is None:
            if conv_op == nn.Conv2d:
                pool_op_kernel_sizes = [(2, 2)] * num_downscale
            elif conv_op == nn.Conv3d:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_downscale
            else:
                raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.pool_op_kernel_sizes = pool_op_kernel_sizes

        self.final_nonlin = final_nonlin
        assert num_conv_per_stage > 1, "this implementation does not support only one conv per stage"
        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.do_ds = deep_supervision
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels
        for d in range(num_downscale):
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d-1]
            else:
                first_stride = None
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            if self.conv_op == nn.Conv3d:
                output_features = min(output_features, max_num_filters_3d)
            else:
                output_features = min(output_features, max_num_filters_2d)

        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs)))

        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        for u in range(num_downscale):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_downscale-1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(nn.Upsample(scale_factor=pool_op_kernel_sizes[-(u+1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u+1)], pool_op_kernel_sizes[-(u+1)], bias=False))
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                                  self.nonlin_kwargs),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                                  self.nonlin_kwargs)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.ModuleList([conv_op(self.conv_blocks_localization[ds][-1].output_channels, i, 1, 1, 0, 1, 1, False) for i in num_classes]))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_downscale - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(nn.Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl+1]]), mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops) # lambda x:x is not a Module so we need to distinguish here

        self.apply(self.weightInitializer)

        self.test_return_output = 0
        self.inference = False

    def train(self, mode=True):
        super(Generic_UNet_Cotraining, self).train(mode)

    def eval(self):
        super(Generic_UNet_Cotraining, self).eval()

    def infer(self, infer):
        self.train(False)
        self.inference = infer

    def forward(self, x):
        #input_var = x
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            if not self.inference:
                seg_outputs.append([self.final_nonlin(self.seg_outputs[u][i](x[(x.shape[0]//len(self.num_classes) * i): (x.shape[0]//len(self.num_classes) * (i+1))])) for i in range(len(self.num_classes))])
            else:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u][self.test_return_output](x)))

        if self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]


class NetworkTrainerBraTS2018Baseline2RegionsCotrainingBraTSDecSDCE(object):
    def __init__(self):
        self.preprocessed_data_directory = None
        # set through arguments from init
        self.experiment_name = "baseline_inspired_by_decathlon 2_regions_cotraining brats dec sd ce"
        self.experiment_description = "NetworkTrainerBraTS2018Baseline 2_regions_cotraining brats dec sd ce"
        self.output_folder = 'model/params'
        self.dataset_directory = None
        self.device = 0
        self.fold = 0

        self.preprocessed_data_directory = None
        self.gt_niftis_folder = None

        # set in self.initialize()
        self.network = None

        self.num_input_channels = self.num_classes = self.net_pool_per_axis = self.patch_size = self.batch_size = \
            self.threeD = self.base_num_features = self.intensity_properties = self.normalization_schemes = None # loaded automatically from plans_file
        self.basic_generator_patch_size = self.data_aug_params = self.plans = None

        self.was_initialized = False
        self.also_val_in_tr_mode = False
        self.dataset = None

        self.inference_apply_nonlin = nn.Sigmoid()

    def initialize(self, training=True):
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        self.output_folder = os.path.join(self.output_folder, "fold%d" % self.fold)
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        self.process_plans_file()
        if training:
            raise NotImplementedError
        self.initialize_network_optimizer_and_scheduler()
        self.network.inference_apply_nonlin = self.inference_apply_nonlin
        self.was_initialized = True

    def initialize_network_optimizer_and_scheduler(self):
        net_numpool = max(self.net_pool_per_axis)
        net_pool_kernel_sizes = []
        for s in range(1, net_numpool+1):
            this_pool_kernel_sizes = [1, 1, 1]
            if self.net_pool_per_axis[0] >= s:
                this_pool_kernel_sizes[0] = 2
            if self.net_pool_per_axis[1] >= s:
                this_pool_kernel_sizes[1] = 2
            if len(self.patch_size)>2:
                if self.net_pool_per_axis[2] >= s:
                    this_pool_kernel_sizes[2] = 2
            else:
                this_pool_kernel_sizes = this_pool_kernel_sizes[:-1]
            net_pool_kernel_sizes.append(tuple(this_pool_kernel_sizes))

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
        conv_kwargs = {'kernel_size':3, 'stride':1, 'padding':1, 'dilation':1, 'bias':True}
        norm_op_kwargs = {'eps':1e-5, 'affine':True, 'momentum':0.02, 'track_running_stats':False}
        dropout_op_kwargs = {'p':0, 'inplace':True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope':1e-2, 'inplace':True}
        self.network = Generic_UNet_Cotraining(self.num_input_channels, self.base_num_features, self.num_classes, 2, net_numpool, 2,
                                    conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x:x, InitWeights_He(1e-2),
                                    net_pool_kernel_sizes, True, False, False)
        self.optimizer = None
        self.lr_scheduler = None
        self.network.set_device(self.device)

    def process_plans_file(self):
        self.batch_size = 2
        self.net_pool_per_axis = [4, 4, 4]
        self.patch_size = (128, 128, 128)
        self.intensity_properties = None
        self.normalization_schemes = ["nonCT"] * 4
        self.base_num_features = 30
        self.num_input_channels = 4
        self.do_dummy_2D_aug = False
        self.use_mask_for_norm = True
        self.only_keep_largest_connected_component = {(0, ): False}

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        self.regions = ((1, 2, 3, 4), (2, 3, 4), (2,))
        self.regions_class_order = (1, 3, 2)
        self.batch_size = 2
        self.base_num_features = 30
        self.num_classes = (3, 3)

    def predict_preprocessed_data_return_softmax(self, data, do_mirroring, num_repeats, use_train_mode, batch_size, mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian):
        return self.network.predict_3D(data, do_mirroring, num_repeats, use_train_mode, batch_size, mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian=use_gaussian)[2]

    def load_best_checkpoint(self, train=True):
        self.load_checkpoint(os.path.join(self.output_folder, "model_best.model"), train=train)

    def load_checkpoint(self, fname, train=True):
        print("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize()
        saved_model = torch.load(fname)
        new_state_dict = OrderedDict()
        for k, value in saved_model['state_dict'].items():
            key = k
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)
        self.epoch = saved_model['epoch']
        if train:
            optimizer_state_dict = saved_model['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.load_state_dict(saved_model['lr_scheduler_state_dict'])
        if len(saved_model['plot_stuff']) < 9:
            self.all_tr_losses_x, self.all_tr_losses, self.all_tr_eval_metrics, self.all_val_losses_x, \
            self.all_val_losses, self.all_val_eval_metrics_dc_per_sample, self.all_val_losses_tr_mode, \
            self.all_val_eval_metrics_dc_glob = saved_model['plot_stuff']
            self.all_val_eval_metrics_dc_per_sample_std = []
        else:
            self.all_tr_losses_x, self.all_tr_losses, self.all_tr_eval_metrics, self.all_val_losses_x, \
            self.all_val_losses, self.all_val_eval_metrics_dc_per_sample, self.all_val_losses_tr_mode, \
            self.all_val_eval_metrics_dc_glob, self.all_val_eval_metrics_dc_per_sample_std = saved_model['plot_stuff']
        self.network.set_device(self.device)

def resize_softmax_output(softmax_output, new_shape, order=3):
    '''
    Resizes softmax output. Resizes each channel in c separately and fuses results back together

    :param softmax_output: c x x x y x z
    :param new_shape: x x y x z
    :param order:
    :return:
    '''
    tpe = softmax_output.dtype
    new_shp = [softmax_output.shape[0]] + list(new_shape)
    result = np.zeros(new_shp, dtype=softmax_output.dtype)
    for i in range(softmax_output.shape[0]):
        result[i] = resize(softmax_output[i].astype(float), new_shape, order, "constant", 0, True)
    return result.astype(tpe)


def save_segmentation_nifti_softmax(softmax_output, dct, out_fname, order=3, region_class_order=None):
    '''
    segmentation must have the same spacing as the original nifti (for now). segmentation may have been cropped out
    of the original image
    :param segmentation:
    :param dct:
    :param out_fname:
    :return:
    '''
    old_size = dct.get('size_before_cropping')
    bbox = dct.get('brain_bbox')

    if bbox is not None:
        seg_old_size = np.zeros([softmax_output.shape[0]] + list(old_size))
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + softmax_output.shape[c+1], old_size[c]))
        seg_old_size[:, bbox[0][0]:bbox[0][1],
                     bbox[1][0]:bbox[1][1],
                     bbox[2][0]:bbox[2][1]] = softmax_output
    else:
        seg_old_size = softmax_output

    segmentation = resize_softmax_output(seg_old_size, np.array(dct['size'])[[2, 1, 0]], order=order)

    if region_class_order is None:
        segmentation = segmentation.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(segmentation.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[segmentation[i] > 0.5] = c
        segmentation = seg_old_spacing_final

    return segmentation.astype(np.uint8)


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            os.mkdir(os.path.join("/", *splits[:i+1]))


def convert_labels_back(seg):
    new_seg = np.zeros(seg.shape, dtype=seg.dtype)
    new_seg[seg == 1] = 2
    new_seg[seg == 2] = 4
    new_seg[seg == 3] = 1
    return new_seg


def preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.5, 0.5), brain_mask=None, cval=0):
    """
    brain mask must be a numpy array that has the same shape as itk_image's pixel array. This function is not ideal but
    gets the job done
    :param itk_image:
    :param is_seg:
    :param spacing_target:
    :param brain_mask:
    :return:
    """
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    image = sitk.GetArrayFromImage(itk_image).astype(float)
    if not is_seg:
        if brain_mask is None:
            brain_mask = (image!=image[0,0,0]).astype(float)
        if np.any([[i!=j] for i, j in zip(spacing, spacing_target)]):
            image = resize_image(image, spacing, spacing_target, 3, cval).astype(np.float32)
            brain_mask = resize_image(brain_mask.astype(float), spacing, spacing_target, order=0).astype(int)
        image[brain_mask==0] = 0
        #subtract mean, divide by std. use heuristic masking
        image[brain_mask!=0] -= image[brain_mask!=0].mean()
        image[brain_mask!=0] /= image[brain_mask!=0].std()
    else:
        new_shape = (int(np.round(spacing[0] / spacing_target[0] * float(image.shape[0]))),
                     int(np.round(spacing[1] / spacing_target[1] * float(image.shape[1]))),
                     int(np.round(spacing[2] / spacing_target[2] * float(image.shape[2]))))
        image = resize_segmentation(image, new_shape, 1, cval)
    return image


def create_brain_masks(data):
    """
    data must be (b, c, x, y, z), brain mask is hole filled binary mask where all sequences are 0 (this is a heuristic
    to recover a brain mask form brain extracted mri sequences, not an actual brain ectraction)
    :param data:
    :return:
    """
    shp = list(data.shape)
    brain_mask = np.zeros(shp, dtype=np.float32)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            this_mask = data[b, c] != 0
            this_mask = binary_fill_holes(this_mask)
            brain_mask[b, c] = this_mask
    return brain_mask


def extract_brain_region(image, segmentation, outside_value=0):
    brain_voxels = np.where(segmentation != outside_value)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))
    return image[resizer], [[minZidx, maxZidx], [minXidx, maxXidx], [minYidx, maxYidx]]


def load_and_preprocess(t1_file, t1km_file, t2_file, flair_file, seg_file=None, bet_file=None, encode_bet_mask_in_seg=False, label_conversion_fn=None):
    images = {}
    # t1
    images["T1"] = sitk.ReadImage(t1_file)
    # t1km
    images["T1KM"] = sitk.ReadImage(t1km_file)

    properties_dict = {
        "spacing": images["T1"].GetSpacing(),
        "direction": images["T1"].GetDirection(),
        "size": images["T1"].GetSize(),
        "origin": images["T1"].GetOrigin()
    }

    # t2
    images["T2"] = sitk.ReadImage(t2_file)

    # flair
    images["FLAIR"] = sitk.ReadImage(flair_file)

    if seg_file is not None:
        images['seg'] = sitk.ReadImage(seg_file)

    if bet_file is not None:
        images['bet_mask'] = sitk.ReadImage(bet_file)
    else:
        t1_npy = sitk.GetArrayFromImage(images["T1"])
        mask = create_brain_masks(t1_npy[None])[0].astype(int)
        mask = sitk.GetImageFromArray(mask)
        mask.CopyInformation(images["T1"])
        images['bet_mask'] = mask

    try:
        images["t1km_sub"] = images["T1KM"] - images["T1"]
    except RuntimeError:
        tmp1 = sitk.GetArrayFromImage(images["T1KM"])
        tmp2 = sitk.GetArrayFromImage(images["T1"])
        res = tmp1 - tmp2
        res_itk = sitk.GetImageFromArray(res)
        res_itk.CopyInformation(images["T1"])
        images["t1km_sub"] = res_itk

    for k in ['T1', 'T1KM', 'T2', 'FLAIR', "t1km_sub"]:
        images[k] = sitk.Mask(images[k], images['bet_mask'], 0)

    bet_numpy = sitk.GetArrayFromImage(images['bet_mask'])
    for k in images.keys():
        is_seg = (k == "seg") | (k == "bet_mask")
        if is_seg:
            cval = -1
        else:
            cval = 0
        images[k] = preprocess_image(images[k], is_seg=is_seg,
                                     spacing_target=(1., 1., 1.), brain_mask=np.copy(bet_numpy), cval=cval)

    properties_dict['size_before_cropping'] = images["T1"].shape

    mask = np.copy(images['bet_mask'])
    for k in images.keys():
        images[k], bbox = extract_brain_region(images[k], mask, False)

    properties_dict['brain_bbox'] = bbox

    if (label_conversion_fn is not None) and ("seg" in images.keys()):
        images["seg"] = label_conversion_fn(images["seg"])

    use_these = ['T1', 'T1KM', 'T2', 'FLAIR', "t1km_sub", 'seg']
    if (not encode_bet_mask_in_seg) or ("seg" not in images.keys()):
        use_these.append("bet_mask")
    else:
        images["seg"][images["bet_mask"] <= 0] = -1
    imgs = []
    for seq in use_these:
        if seq not in images.keys():
            imgs.append(np.zeros(images["T1"].shape)[None])
        else:
            imgs.append(images[seq][None])
    all_data = np.vstack(imgs)
    return all_data, properties_dict

def segment(t1_file, t1ce_file, t2_file, flair_file, netLoc):
    """
    Segments the passed files
    """

    trainer = NetworkTrainerBraTS2018Baseline2RegionsCotrainingBraTSDecSDCE()

    trainer.initialize(False)

    all_data, dct = load_and_preprocess(t1_file, t1ce_file, t2_file, flair_file, None, None,
                                        True, None)
    all_softmax = []
    for fold in range(5):
        trainer.output_folder = join(netLoc, "%d" % fold)
        trainer.load_best_checkpoint(False)
        trainer.network.infer(True)
        trainer.network.test_return_output = 0
        softmax = trainer.predict_preprocessed_data_return_softmax(all_data[:4], True, 1, False, 1, (2, 3, 4), False,
                                                                   None, None, trainer.patch_size, True)
        all_softmax.append(softmax[None])
    softmax_consolidated = np.vstack(all_softmax).mean(0)

    output = save_segmentation_nifti_softmax(softmax_consolidated, dct,
                                        "tumor_isen2018_class.nii.gz", 1,
                                        trainer.regions_class_order)
    return output
