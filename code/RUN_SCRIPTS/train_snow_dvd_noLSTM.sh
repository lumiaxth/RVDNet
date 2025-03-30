#!/usr/bin/env bash

cd ..
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
python train_dvd_stgrl_noLSTM_cin.py --epochs 120 \
               --data_root /home/xth/dataset/SnowScape \
               --train_file /home/xth/RVDNet/data/Dataset_Training_Synthetic.json \
               --eval_file /home/xth/RVDNet/data/Dataset_Testing_Synthetic_C.json \
               --checkpoint_dir_best /home/xth/RVDNet/code/checkpoint/best/t7 \
               --batch_size 8 \
               --input_residue \
	           --F_dvd \
	           --val_mode all \
               --backbone resnet18 \
               --refinenet R_CLSTM_5 \
               --checkpoint_dir_C ./checkpoint/C/noLSTM/trial7 \
               --checkpoint_dir_F ./checkpoint/F/noLSTM/trial7 \
               --logdir ./log/ \
               --eval_on_the_fly False \
               --eval_interval 2 \
               --use_bilstm \
               --use_grl \
               --resume \
               --lr_C 0.000001 \
               --lr_F 0.000001 \
               


#def _get_train_opt():
#    parser = argparse.ArgumentParser(description = 'Monocular Depth Estimation')
#    parser.add_argument('--train_file', required=True, help='the path of indexfile', default='/home/dxli/workspace/derain/proj/data/Dataset_Training_Synthetic.json')
#    parser.add_argument('--data_root', required=True, help="the root path of image data.", default='/media/hdd/derain/NTU-derain')
#    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
#    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
#    parser.add_argument('--backbone', type=str, default='resnet18')
#    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
#    parser.add_argument('--logdir', required=True, help="the directory to save logs and checkpoints", default='./checkpoint')
#    parser.add_argument('--checkpoint_dir', required=True, help="the directory to save the checkpoints", default='./log_224')
#    parser.add_argument('--loadckpt', type=str)
#    parser.add_argument('--overlap', type=int, default=0)
#    parser.add_argument('--use_cuda', type=bool, default=True)
#    parser.add_argument('--devices', type=str, default='0')
#    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
#    parser.add_argument('--resume', action='store_true',default=False, help='continue training the model')
#    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum parameter used in the Optimizer.')
#    parser.add_argument('--epsilon', default=0.001, type=float, help='epsilon')
#    parser.add_argument('--optimizer_name', default="adam", type=str, help="Optimizer selection")
#    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
#    parser.add_argument('--do_summary', action='store_true', default=False, help='whether do summary or not')
#    parser.add_argument('--pretrained_dir', required=False,type=str, help="the path of pretrained models")
#    return parser.parse_args()
