from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
from torch.cuda import amp
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import torch.onnx


def convert_model_to_onnx(model, opt, model_path, suffix=''):
  """
  在验证阶段将模型转换为ONNX格式
  """
  if not hasattr(opt, 'export_onnx') or not opt.export_onnx:
    return

  print(f'Converting model to ONNX format...')

  # 设置ONNX输出路径
  base_name = os.path.splitext(os.path.basename(model_path))[0]
  onnx_path = os.path.join(opt.save_dir, f"{base_name}_{opt.arch}{suffix}.onnx")

  # 创建输入张量
  dummy_input = torch.randn(1, 3, opt.input_h, opt.input_w)
  if opt.device != torch.device('cpu'):
    dummy_input = dummy_input.to(opt.device)

  # 获取模型用于转换（去除DataParallel包装）
  export_model = model
  if hasattr(model, 'module'):
    export_model = model.module
  export_model.eval()

  try:
    # 转换为ONNX
    torch.onnx.export(
      export_model,
      dummy_input,
      onnx_path,
      export_params=True,
      opset_version=11,
      do_constant_folding=True,
      input_names=['input'],
      output_names=list(opt.heads.keys()),
      dynamic_axes={
        'input': {0: 'batch_size'},
        **{head: {0: 'batch_size'} for head in opt.heads.keys()}
      } if getattr(opt, 'dynamic_batch', False) else None
    )

    print(f'Model successfully converted to ONNX: {onnx_path}')

    # 如果启用验证，验证ONNX模型
    if getattr(opt, 'verify_onnx', False):
      verify_onnx_model(export_model, onnx_path, dummy_input, opt)

  except Exception as e:
    print(f'ONNX conversion failed: {str(e)}')


def verify_onnx_model(pytorch_model, onnx_path, dummy_input, opt):
  """
  验证ONNX模型输出与PyTorch模型是否一致
  """
  try:
    import onnx
    import onnxruntime as ort

    print('Verifying ONNX model...')

    # 检查ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX model check passed!')

    # 比较输出
    print('Comparing PyTorch and ONNX outputs...')
    ort_session = ort.InferenceSession(onnx_path)

    # PyTorch推理
    with torch.no_grad():
      torch_out = pytorch_model(dummy_input)

    # ONNX推理
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # 比较输出
    if isinstance(torch_out, list) and len(torch_out) > 0:
      torch_out_dict = torch_out[0]
      for i, head_name in enumerate(opt.heads.keys()):
        if head_name in torch_out_dict:
          torch_output = torch_out_dict[head_name].cpu().numpy()
          onnx_output = ort_outs[i]
          max_diff = abs(torch_output - onnx_output).max()
          print(f'{head_name}: max difference = {max_diff:.6f}')

    print('ONNX verification completed!')

  except ImportError:
    print('Warning: onnx or onnxruntime not installed, skipping verification')
  except Exception as e:
    print(f'ONNX verification failed: {str(e)}')


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_dataset = Dataset(opt, 'val')
  val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_dataset = Dataset(opt, 'train')
  drop_last = len(train_dataset) >= opt.batch_size
  if not drop_last:
    print('Keeping last train batch (dataset has {} samples < batch_size {}).'.format(len(train_dataset), opt.batch_size))
  train_loader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=drop_last
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        best_model_path = os.path.join(opt.save_dir, 'model_best.pth')
        save_model(best_model_path, epoch, model)

        # 转换最佳模型为ONNX格式
        convert_model_to_onnx(model, opt, best_model_path, suffix='_best')

      # 验证阶段可视化调试 - 已经在BaseTrainer.run_epoch中通过save_debug_images实现
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  # 训练完成后转换最终模型为ONNX格式
  if hasattr(opt, 'export_onnx') and opt.export_onnx:
    final_model_path = os.path.join(opt.save_dir, 'model_last.pth')
    convert_model_to_onnx(model, opt, final_model_path, suffix='_final')

  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)