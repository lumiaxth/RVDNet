#!/usr/bin/env bash

cd ..
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 
python test.py --data_root /home/xth/dataset/SnowScape \
               --eval_file /home/xth/RVDNet/data/r10.json \
               --backbone resnet18 \
               --refinenet R_CLSTM_5 \
               --use_bilstm \
               --input_residue \
               --val_mode mid \
               --F_dvd False \
               --loadckpt \
               --loadckpt_C /home/xth/RVDNet/code/best_checkpoints/ntu/C/checkpoints_best_221.pth.tar \
               --loadckpt_F /home/xth/RVDNet/code/best_checkpoints/ntu/F/checkpoints_best_221.pth.tar \
               --out_dir /home/xth/RVDNet/out/snowscape/esti_r10
#               --loadckpt_C /home/dxli/workspace/derain/proj/code/best_checkpoints/ntu/C/checkpoints_best_101.pth.tar\
#               --loadckpt_F /home/dxli/workspace/derain/proj/code/best_checkpoints/ntu/F/checkpoints_best_101.pth.tar \
#               --out_dir /home/dxli/workspace/derain/proj/out/ntu/101
            #    --val_mode all \



#def _get_test_opt():
#    parser = argparse.ArgumentParser(description='Evaluate performance on test set')
#
#    # network
#    parser.add_argument('--backbone', type=str, default='resnet18')
#    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
#    parser.add_argument('--use_bilstm', action='store_true')
#    parser.add_argument('--use_bn', action='store_true', default=False)
#    parser.add_argument('--input_residue', action='store_true')
#    parser.add_argument('--compress_channels', type=int, default=8, help='number of channels in R_CLSTM.')
#
#    # validation
#    parser.add_argument('--val_mode', type=str, default='mid', help='validation on the frame in the middle (mid) or '
#                                                                    'all the frames (all) of coarse network.')
#    # paths
#    parser.add_argument('--eval_file', type=str, required=True, help='the path of indexfile',
#                        default='/home/dxli/workspace/derain/proj/data/Dataset_Testing_Synthetic.json')
#    parser.add_argument('--checkpoint_dir_C', required=True, help="the directory to save the checkpoints of coarse net",
#                        default='./checkpoint/C')
#    parser.add_argument('--checkpoint_dir_F', required=True, help="the directory to save the checkpoints of finenet",
#                        default='./checkpoint/F')
#    parser.add_argument('--data_root', type=str, required=True, help="the root path of image data.",
#                        default='/media/hdd/derain/NTU-derain')
#
#    # misc
#    parser.add_argument('--loadckpt', action='store true')
#    parser.add_argument('--loadckpt_C', type=str, help='pretrain weights of coarse net.')
#    parser.add_argument('--loadckpt_F', type=str, help='pretrain weights of fine net.')
#    parser.add_argument('--use_cuda', type=bool, default=True)
#
#    return parser.parse_args()
