import math
import os, time
import shutil
import torch
import csv
import glob
import util.vis_utils as vis_utils
from metrics import Result
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=+9), 'JST')

fieldnames = [
    'epoch',
    'rmse_s12', 'mae_s12', 'psnr_dolp', 'diff_angle_admap',
    'diff_angle_nml',
    'psnr_s12', 
    'rmse_s0', 'mae_s0', 'psnr_s0', 
    'rmse_rgb', 'mae_rgb', 'psnr_rgb', 'ssim_rgb',
    'rmse_s012', 'mae_s012', 'psnr_s012', 
    'rmse_4polar', 'mae_4polar', 'psnr_4polar', 
    'rmse_dolp', 'mae_dolp', 'delta1_dolp', 'delta2_dolp', 'delta3_dolp', 
    'diff_angle_aolp', 'cos_admap',
    'data_time', 'gpu_time', 'datetime'
]

def search_checkpoint_latest(dir):
    pths = sorted(glob.glob(dir + '/checkpoint-*.pth.tar'))
    if len(pths)==0:
        pth = ''
    else:
        pth = pths[-1] # Get latest checkpoint
    return pth

def search_checkpoint_best(dir):
    return dir + '/model_best.pth.tar'


class logger:
    def __init__(self, args, prepare=True):
        self.args = args
        output_directory = get_folder_name(args)
        self.output_directory = output_directory
        self.best_result = Result()
        self.best_result.set_to_worst()

        if not prepare:
            return
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.train_csv = os.path.join(output_directory, 'train.csv')
        self.val_csv = os.path.join(output_directory, 'val.csv')
        self.eval_csv = os.path.join(output_directory, 'eval.csv')
        self.best_txt = os.path.join(output_directory, 'best.txt')
        self.eval_txt = os.path.join(output_directory, 'eval.txt')
        self.args_txt = os.path.join(output_directory, 'args.txt')

        # Auto resume
        if args.autoresume:
            args.resume = search_checkpoint_latest(output_directory)
            print('Resume from:', args.resume)

        # Resume for eval
        if args.bestresume:
            args.resume = search_checkpoint_best(output_directory)
            print('Resume from:', args.resume)
            with open(self.eval_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # backup the source code
        if args.resume == '':
            print("=> creating source code backup ...")
            backup_directory = os.path.join(output_directory, "code_backup")
            self.backup_directory = backup_directory
            backup_source_code(args.source_directory, backup_directory)
            # create new csv files with only header
            with open(self.train_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(self.val_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print("=> finished creating source code backup.")

    def conditional_print(self, split, i, epoch, lr, n_set, blk_avg_meter,
                          avg_meter):
        if (i + 1) % self.args.print_freq == 0:
            avg = avg_meter.average()
            blk_avg = blk_avg_meter.average()
            print('=> output: {}'.format(self.output_directory))
            print(
                '{split} Epoch: {0} [{1}/{2}]\tlr={lr} '
                't_Data={blk_avg.data_time:.3f}({average.data_time:.3f}) '
                't_GPU={blk_avg.gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                'RMSE_S12={blk_avg.rmse[s12]:.5f}({average.rmse[s12]:.5f}) '
                'MAE_S12={blk_avg.mae[s12]:.5f}({average.mae[s12]:.5f}) '
                'PSNR_DOLP={blk_avg.psnr[dolp]:.3f}({average.psnr[dolp]:.3f}) '
                'ANGLE_ADMAP={blk_avg.diff_angle[admap]:.3f}({average.diff_angle[admap]:.3f}) '
                'ANGLE_NML={blk_avg.diff_angle[nml]:.3f}({average.diff_angle[nml]:.3f}) '
                'PSNR_S0={blk_avg.psnr[s0]:.3f}({average.psnr[s0]:.3f}) '
                'PSNR_RGB={blk_avg.psnr[rgb]:.3f}({average.psnr[rgb]:.3f}) '
                'SSIM_RGB={blk_avg.ssim[rgb]:.3f}({average.ssim[rgb]:.3f}) '
                'RMSE_S1={blk_avg.rmse[s1]:.5f}({average.rmse[s1]:.5f}) '
                'RMSE_S2={blk_avg.rmse[s2]:.5f}({average.rmse[s2]:.5f}) '
                'PSNR_S12={blk_avg.psnr[s12]:.3f}({average.psnr[s12]:.3f}) '
                'RMSE_4POLAR={blk_avg.rmse[4polar]:.5f}({average.rmse[4polar]:.5f}) '
                'MAE_4POLAR={blk_avg.mae[4polar]:.5f}({average.mae[4polar]:.5f}) '
                'PSNR_4POLAR={blk_avg.psnr[4polar]:.3f}({average.psnr[4polar]:.3f}) '
                'Delta1_4POLAR={blk_avg.delta1[4polar]:.3f}({average.delta1[4polar]:.3f}) '
                'RMSE_DOLP={blk_avg.rmse[dolp]:.5f}({average.rmse[dolp]:.5f}) '
                'MAE_DOLP={blk_avg.mae[dolp]:.5f}({average.mae[dolp]:.5f}) '
                'Delta1_DOLP={blk_avg.delta1[dolp]:.3f}({average.delta1[dolp]:.3f}) '
                'ANGLE_AOLP={blk_avg.diff_angle[aolp]:.3f}({average.diff_angle[aolp]:.3f}) '
                'COS_ADMAP={blk_avg.cos[admap]:.3f}({average.cos[admap]:.5f}) '

                .format(epoch,
                        i + 1,
                        n_set,
                        lr=lr,
                        blk_avg=blk_avg,
                        average=avg,
                        split=split.capitalize()))
            blk_avg_meter.reset(False)

    def conditional_save_info(self, split, average_meter, epoch):
        avg = average_meter.average()
        if split == "train":
            csvfile_name = self.train_csv
        elif split == "val":
            csvfile_name = self.val_csv
        elif split == "eval":
            self.save_single_txt(self.eval_txt, avg, epoch)
            csvfile_name = self.eval_csv
            # return avg
        else:
            raise ValueError("wrong split provided to logger")
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                # 'rmse_s1': avg.rmse['s1'],
                # 'rmse_s2': avg.rmse['s2'],
                'rmse_s12': avg.rmse['s12'],
                'mae_s12': avg.mae['s12'],
                'psnr_dolp': avg.psnr['dolp'],
                'diff_angle_admap': avg.diff_angle['admap'],
                'diff_angle_nml': avg.diff_angle['nml'],
                # 'mse_s12': avg.mse['s12'],
                'psnr_s12': avg.psnr['s12'],
                'rmse_s0': avg.rmse['s0'],
                'mae_s0': avg.mae['s0'],
                'psnr_s0': avg.psnr['s0'],
                'rmse_rgb': avg.rmse['rgb'],
                'mae_rgb': avg.mae['rgb'],
                'psnr_rgb': avg.psnr['rgb'],
                'ssim_rgb': avg.ssim['rgb'],
                'rmse_s012': avg.rmse['s012'],
                'mae_s012': avg.mae['s012'],
                'psnr_s012': avg.psnr['s012'],
                'rmse_4polar': avg.rmse['4polar'],
                'mae_4polar': avg.mae['4polar'],
                # 'mse_4polar': avg.mse['4polar'],
                'psnr_4polar': avg.psnr['4polar'],
                # 'delta1_4polar': avg.delta1['4polar'],
                # 'delta2_4polar': avg.delta2['4polar'],
                # 'delta3_4polar': avg.delta3['4polar'],
                'rmse_dolp': avg.rmse['dolp'],
                'mae_dolp': avg.mae['dolp'],
                # 'mse_dolp': avg.mse['dolp'],
                'delta1_dolp': avg.delta1['dolp'],
                'delta2_dolp': avg.delta2['dolp'],
                'delta3_dolp': avg.delta3['dolp'],
                'diff_angle_aolp': avg.diff_angle['aolp'],
                'cos_admap': avg.cos['admap'],
                'gpu_time': avg.gpu_time,
                'data_time': avg.data_time,
                'datetime': datetime.now(JST).strftime('%Y%m%d %H:%M:%S')
            })
        return avg

    def save_single_txt(self, filename, result, epoch):
        with open(filename, 'w') as txtfile:
            txtfile.write(
                ("rank_metric={}\n" + "epoch={}\n" +
                 "rmse_s12={:.5f}\n" + "mae_s12={:.5f}\n" + "psnr_dolp={:.3f}\n" + "diff_angle_admap={:.3f}\n" +
                 "psnr_s12={:.3f}\n" + 
                 "rmse_s0={:.5f}\n" + "mae_s0={:.5f}\n" + "psnr_s0={:.3f}\n" + 
                 "rmse_rgb={:.5f}\n" + "mae_rgb={:.5f}\n" + "psnr_rgb={:.3f}\n" + "ssim_rgb={:.3f}\n" + 
                 "rmse_s012={:.5f}\n" + "mae_s012={:.5f}\n" + "psnr_s012={:.3f}\n" + 
                 "mae_4polar={:.5f}\n" + "psnr_4polar={:.3f}\n" + "delta1_4polar={:.3f}\n" +
                 "mae_dolp={:.5f}\n" + "delta1_dolp={:.3f}\n" +
                 "diff_angle_aolp={:.3f}\n" + "cos_admap={:.5f}\n" +
                 "diff_angle_nml={:.3f}\n" +
                 "t_gpu={:.4f}").format(self.args.rank_metric, epoch,
                                        result.rmse['s12'], result.mae['s12'], result.psnr['dolp'], result.diff_angle['admap'],
                                        result.psnr['s12'],
                                        result.rmse['s0'], result.mae['s0'], result.psnr['s0'],
                                        result.rmse['rgb'], result.mae['rgb'], result.psnr['rgb'], result.ssim['rgb'],
                                        result.rmse['s012'], result.mae['s012'], result.psnr['s012'],
                                        result.mae['4polar'], result.psnr['4polar'], result.delta1['4polar'],
                                        result.mae['dolp'], result.delta1['dolp'],
                                        result.diff_angle['aolp'], result.cos['admap'], 
                                        result.diff_angle['nml'],
                                        result.gpu_time))

    def save_best_txt(self, result, epoch):
        self.save_single_txt(self.best_txt, result, epoch)

    def save_args_txt(self):
        with open(self.args_txt, 'w') as txtfile:
            txtfile.write(str(self.args))


    def _get_img_comparison_name(self, mode, epoch, is_best=False):
        if mode == 'eval':
            return self.output_directory + '/comparison_eval.png'
        if mode == 'val':
            if is_best:
                return self.output_directory + '/comparison_best.png'
            else:
                return self.output_directory + '/comparison_' + str(epoch) + '.png'

    def conditional_save_img_comparison(self, mode, i, ele, pred, epoch, skip=5):
        # save 8 images for visualization
        if mode == 'val' or mode == 'eval':
            if i == 0:
                self.img_merge = vis_utils.merge_into_row(self.args, ele, pred)
            elif i % skip == 0 and i <= 7 * skip:
                row = vis_utils.merge_into_row(self.args, ele, pred)
                self.img_merge = vis_utils.add_row(self.img_merge, row)
                if i == 7 * skip and self.args.save_img_comp:
                    filename = self._get_img_comparison_name(mode, epoch)
                    vis_utils.save_image(self.img_merge, filename)

    def _get_img_comparison_name_one(self, mode, num):
        if mode == 'eval':
            return self.output_directory + '/comparison_n={}.png'.format(str(num).zfill(3))

    def one_save_img_comparison(self, mode, i, ele, pred, epoch, num=0):
        # save 1 image for visualization
        if mode == 'eval':
            if i == num:
                img_eval = vis_utils.merge_into_row(self.args, ele, pred)
                filename = self._get_img_comparison_name_one(mode, i)
                vis_utils.save_image(img_eval, filename)

    def save_img_comparison_as_best(self, mode, epoch):
        if mode == 'val' and epoch % self.args.save_interval==0:
            filename = self._get_img_comparison_name(mode, epoch, is_best=True)
            vis_utils.save_image(self.img_merge, filename)

    def save_img_comparison_eval(self, mode, epoch):
        if mode == 'eval':
            filename = self._get_img_comparison_name(mode, epoch, is_best=True)
            vis_utils.save_image(self.img_merge, filename)


    def get_ranking_error(self, result):
        attr = getattr(result, self.args.rank_metric)
        attr = attr[self.args.rank_metric_domain]
        return attr

    def rank_conditional_save_best(self, mode, result, epoch):
        error = self.get_ranking_error(result)
        best_error = self.get_ranking_error(self.best_result)
        is_best = error < best_error
        if is_best and mode == "val" and not self.args.evaluate:
            self.old_best_result = self.best_result
            self.best_result = result
            self.save_best_txt(result, epoch)
        return is_best

    def conditional_save_pred(self, mode, i, pred, epoch):
        if ("test" in mode or mode == "eval") and self.args.save_pred:

            # save images for visualization/ testing
            image_folder = os.path.join(self.output_directory,
                                        mode + "_output")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred.data.cpu()).numpy()
            filename = os.path.join(image_folder, '{0:010d}.png'.format(i))
            vis_utils.save_depth_as_uint16png(img, filename)

    def conditional_summarize(self, mode, avg, is_best):
        print("\n*\nSummary of ", mode, "round")
        print(''
              'RMSE_S12={average.rmse[s12]:.5f}\n'
              'MAE_S12={average.mae[s12]:.5f}\n'
              'PSNR_DOLP={average.psnr[dolp]:.3f}\n'
              'ANGLE_ADMAP={average.diff_angle[admap]:.3f}\n'
              'ANGLE_NML={average.diff_angle[nml]:.3f}\n'
              'PSNR_S12={average.psnr[s12]:.3f}\n'
              'RMSE_S0={average.rmse[s0]:.5f}\n'
              'MAE_S0={average.mae[s0]:.5f}\n'
              'PSNR_S0={average.psnr[s0]:.3f}\n'
              'RMSE_RGB={average.rmse[rgb]:.5f}\n'
              'MAE_RGB={average.mae[rgb]:.5f}\n'
              'PSNR_RGB={average.psnr[rgb]:.3f}\n'
              'SSIM_RGB={average.ssim[rgb]:.3f}\n'
              'RMSE_S012={average.rmse[s012]:.5f}\n'
              'MAE_S012={average.mae[s012]:.5f}\n'
              'PSNR_S012={average.psnr[s012]:.3f}\n'
              'RMSE_4POLAR={average.rmse[4polar]:.5f}\n'
              'MAE_4POLAR={average.mae[4polar]:.5f}\n'
              'PSNR_4POLAR={average.psnr[4polar]:.3f}\n'
              'Delta1_4POLAR={average.delta1[4polar]:.3f}\n'
              'RMSE_DOLP={average.rmse[dolp]:.5f}\n'
              'MAE_DOLP={average.mae[dolp]:.5f}\n'
              'Delta1_DOLP={average.delta1[dolp]:.3f}\n'
              'ANGLE_AOLP={average.diff_angle[aolp]:.3f}\n'
              'COS_ADMAP={average.cos[admap]:.5f}\n'
              't_GPU={time:.3f}'.format(average=avg, time=avg.gpu_time))
        if is_best and mode == "val":
            print("New best model by %s %s (was %.5f)" %
                  (self.args.rank_metric,
                  self.args.rank_metric_domain,
                   self.get_ranking_error(self.old_best_result)))
        elif mode == "val":
            print("(best %s %s is %.5f)" %
                  (self.args.rank_metric,
                   self.args.rank_metric_domain,
                   self.get_ranking_error(self.best_result)))
        print("*\n")


ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*",
                                       "*build", "*.fuse*", "*_drive_*")

def backup_source_code(source_directory, backup_directory):
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree(source_directory, backup_directory, ignore=ignore_hidden)


def adjust_learning_rate(lr_init, optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init
    if (epoch >= 10):
        lr = lr_init * 0.5
    if (epoch >= 15):
        lr = lr_init * 0.1
    if (epoch >= 25):
        lr = lr_init * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, epoch, output_directory, interval):
    checkpoint_filename = os.path.join(output_directory,
                                       'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, 'checkpoint-' + str(epoch - interval) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def get_folder_name(args):
    current_time = datetime.now(JST).strftime('%Y-%m-%d@%H-%M-%S-%f')
    if args.not_random_crop:
        he = args.val_h
        w = args.val_w
    else:
        he = args.random_crop_height
        w = args.random_crop_width

    if not args.suffix == '':
        return os.path.join(args.result,
            'se={}.rp={}.c={}.bs={}.cr={}x{}.tn={}.{}'.
            format(args.seed, args.raw_pattern, args.criterion, \
                args.batch_size, he, w, args.train_num, args.suffix
                ))
    else:
        return os.path.join(args.result,
            'se={}.rp={}.c={}.bs={}.cr={}x{}.tn={}.time={}'.
            format(args.seed, args.raw_pattern, args.criterion, \
                args.batch_size, he, w, args.train_num, current_time
                ))



avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2).cuda()
