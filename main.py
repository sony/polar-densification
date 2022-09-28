import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time
import random
import numpy as np
from torchvision import datasets
from torch.utils.data.dataset import Subset

from dataloaders.polar_loader import PolarCG
from metrics import AverageMeter, Result
import criteria
from args import parser
import util.helper as helper
from model.model_sna import SNA

args = parser()
print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    gpuid = 0 if args.gpu<0 else args.gpu
    device = torch.device("cuda:{}".format(gpuid))
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

# For reproducibility
def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if not args.seed == -1:
    torch_fix_seed(args.seed)


# Loss fuction for polar compensation
if args.criterion == 'l2':
    polar_criterion = criteria.MSELoss()
elif args.criterion == 'l1':
    polar_criterion = criteria.L1Loss()
elif args.criterion == 'l2_s12':
    polar_criterion = criteria.MSES12Loss()
elif args.criterion == 'l1_s12':
    polar_criterion = criteria.L1S12Loss()

# Loss fuction for RGB refinement
if args.rgb_criterion == 'l2':
    rgb_criterion = criteria.MSELoss()
elif args.rgb_criterion == 'l1':
    rgb_criterion = criteria.L1Loss()

multi_batch_size = 1

# Iterate processing for each Epoch
def iterate(mode, args, loader, model, optimizer, logger, epoch):
    actual_epoch = epoch

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    assert mode in ["train", "val", "eval"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        # Set the learning rate according to the number of epochs
        lr = helper.adjust_learning_rate(args.lr, optimizer, actual_epoch, args)
    else:
        model.eval() # BN and DropOut behavior changes
        lr = 0
    
    torch.cuda.empty_cache() # Releases all free cache memory currently held
    for i, batch_data in enumerate(loader):
        if(args.evaluate and cuda):
            torch.cuda.synchronize()
        dstart = time.time()

        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        s0gt = batch_data['s0gt']
        s1gt = batch_data['s1gt']
        s2gt = batch_data['s2gt']
        s0gt_gray = (s0gt[:,0,:,:] + s0gt[:,1,:,:] + s0gt[:,2,:,:])/3.0
        s012gt = torch.stack([s0gt_gray, s1gt, s2gt], dim=1)

        if args.evaluate and cuda:
            torch.cuda.synchronize()
        data_time = time.time() - dstart

        pred, s012spspred, s012spsgt, mask = None, None, None, None
        start = None
        gpu_time = 0

        if args.evaluate and cuda:
            torch.cuda.synchronize()
        start = time.time()

        st1_pred, st2_pred, pred, s0pred = model(batch_data, epoch)

        if args.evaluate and cuda:
            torch.cuda.synchronize()
        gpu_time = time.time() - start

        s0_gray = (s0pred[:,0,:,:] + s0pred[:,1,:,:] + s0pred[:,2,:,:])/3.0

        pred = torch.stack([s0_gray, pred[:,0,:,:], pred[:,1,:,:]], dim=1)
        st1_pred = torch.stack([s0_gray, st1_pred[:,0,:,:], st1_pred[:,1,:,:]], dim=1)
        st2_pred = torch.stack([s0_gray, st2_pred[:,0,:,:], st2_pred[:,1,:,:]], dim=1)
        
        polar_loss = 0

        st1_loss, st2_loss, loss = 0, 0, 0
        w_st1, w_st2 = 0, 0
        round1, round2, round3 = 1, 3, None
        if(actual_epoch <= round1):
            w_st1, w_st2 = 0.2, 0.2
        elif(actual_epoch <= round2):
            w_st1, w_st2 = 0.05, 0.05
        else:
            w_st1, w_st2 = 0, 0

        if mode == 'train':
            polar_loss = polar_criterion(pred, s012gt)

            st1_loss = polar_criterion(st1_pred, s012gt)
            st2_loss = polar_criterion(st2_pred, s012gt)
            rgb_loss = rgb_criterion(s0pred, s0gt)
            rgb_lambda = 1.0
            loss = (1 - w_st1 - w_st2) * polar_loss + w_st1 * st1_loss + w_st2 * st2_loss + rgb_lambda * rgb_loss

            if i % multi_batch_size == 0:
                optimizer.zero_grad()
            loss.backward()

            if i % multi_batch_size == (multi_batch_size-1) or i==(len(loader)-1):
                optimizer.step()
            print("loss:", loss, " epoch:", epoch, " ", i, "/", len(loader))

        if(not args.evaluate):
            gpu_time = time.time() - start

        if mode != 'train':
            vispred = torch.cat((pred, s0pred), dim=1)

        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()

            if mode != 'train' or (mode=='train' and args.train_eval):
                result.evaluate(pred.data, s012gt.data,
                output_rgb=s0pred, target_rgb=s0gt)
            [
                m.update(result, gpu_time, data_time, mini_batch_size)
                for m in meters
            ]

            if mode != 'train':
                if args.eval_each:
                    logger.conditional_save_info(mode, block_average_meter, i)
                    block_average_meter.reset(False)
                else:
                    logger.conditional_print(mode, i, epoch, lr, len(loader),
                                    block_average_meter, average_meter)
                skip = 100
                if args.small: skip = 5
                elif args.evaluate: skip = 1
                if args.vis_skip!=0: skip=args.vis_skip
                logger.conditional_save_img_comparison(mode, i, batch_data, vispred,
                                                epoch, skip=skip)

                if args.disp_all:
                    logger.one_save_img_comparison(mode, i, batch_data, vispred, epoch, i)
                else:
                    logger.one_save_img_comparison(mode, i, batch_data, vispred, epoch, args.evalcomp_num)


    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    if mode == 'eval':
        logger.save_img_comparison_eval(mode, epoch)

    return avg, is_best

def select_backbone(args, spsconv=None):
    model = SNA(args).to(device)
    return model

def main():
    global args
    checkpoint = None
    is_eval = False

    logger = helper.logger(args)

    if args.resume:
        # In Resume mode, reads Checkpoint from args.resume and automatically sets start_epoch
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    if args.evaluate:
        # Load checkpoint from args.evaluate
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            is_eval = True

            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))


    print("=> creating model and optimizer ... ", end='')
    model = None
    penet_accelerated = False

    model = select_backbone(args)
    
    model_named_params = None
    model_bone_params = None
    model_new_params = None
    optimizer = None

    # Load model if checkpoint exists
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        print("=> checkpoint state loaded.")

    # Generating logger
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    logger.save_args_txt()
    print("=> logger created.")

    val_dataset = PolarCG('val', args) # batchsize=1, shuffle=False

    if args.small: # For training on small data sets (for debugging)
        small_val_random = False
        n_samples = len(val_dataset)
        small_size = int(n_samples * args.small_rate)
        if not small_val_random:
            subset_indices = list(range(0, small_size)) 
            val_dataset   = Subset(val_dataset, subset_indices)
        else:
            val_dataset, _ = torch.utils.data.random_split(val_dataset, [small_size, n_samples - small_size])

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    if is_eval == True:
        for p in model.parameters():
            p.requires_grad = False

        result, is_best = iterate("eval", args, val_loader, model, None, logger,
                              args.start_epoch - 1)
        return

    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

    if checkpoint is not None and args.optimizer_load:            
            optimizer.load_state_dict(checkpoint['optimizer'])

    print("completed.")

    # Parallelization
    if args.gpu < 0:
        model = torch.nn.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = PolarCG('train', args)
        if args.small:
            n_samples = len(train_dataset)
            small_size = int(n_samples * args.small_rate)
            train_dataset, _ = torch.utils.data.random_split(train_dataset, [small_size, n_samples - small_size])
        elif args.train_num > 0:
            n_samples = len(train_dataset)
            train_random = False
            if args.train_random:
                train_dataset, _ = torch.utils.data.random_split(train_dataset, [args.train_num, n_samples - args.train_num])
            else:
                subset_indices = list(range(0, args.train_num)) 
                train_dataset   = Subset(train_dataset, subset_indices)

        if args.seed==-1: # When using random seed
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True,
                                                    sampler=None,
                                                    )
        else: # When using fixed seed
            g = torch.Generator()
            g.manual_seed(args.seed) 
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True,
                                                    sampler=None,
                                                    worker_init_fn=seed_worker,
                                                    generator=g,
                                                    )
        print("\t==> train_loader size:{}".format(len(train_loader)))

    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch)  # train for one epoch

        # validation memory reset
        for p in model.parameters():
            p.requires_grad = False

        if epoch % args.val_interval==0:
            result, is_best = iterate("val", args, val_loader, model, None, logger, epoch)  # evaluate on validation set

        # Enable gradient calculation
        for p in model.parameters():
            p.requires_grad = True

        # Save checkpoint
        if epoch % args.save_interval == 0:
            if args.gpu<0:
                helper.save_checkpoint({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'best_result': logger.best_result,
                    'optimizer' : optimizer.state_dict(),
                    'args' : args,
                }, is_best, epoch, logger.output_directory, args.save_interval)
            else:
                helper.save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_result': logger.best_result,
                    'optimizer' : optimizer.state_dict(),
                    'args' : args,
                }, is_best, epoch, logger.output_directory, args.save_interval)


if __name__ == '__main__':
    main()