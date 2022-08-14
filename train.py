from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from load_data import SparseDataset
import time
import torch.multiprocessing

from models.matcher import Matcher

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Pose Graph Matching',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by Matcher')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='Match threshold')
parser.add_argument(
    '--learning_rate', type=int, default=0.0001,
    help='Learning rate')
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')
parser.add_argument(
    '--data_path', type=str, default='path/to/freiburg/data', # MSCOCO2014_yingxin
    help='Path to the directory of training imgs.')
parser.add_argument(
    '--model_save_path', type=str, default='/ckpt/',
    help='Path to save model checkpoint')
parser.add_argument(
    '--epoch', type=int, default=20,
    help='Number of epoches')



if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    print('Will save model to',
        'directory \"{}\"'.format(opt.model_save_path))
    config = {
        'matcher': {
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    train_set = SparseDataset(opt.data_path, opt.max_keypoints)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)

    matcher = Matcher(config.get('matcher', {}))
    if torch.cuda.is_available():
        matcher.cuda()
    else:
        print("### CUDA not available ###")
    optimizer = torch.optim.Adam(matcher.parameters(), lr=opt.learning_rate)

    
    for epoch in range(1, opt.epoch+1):
        epoch_loss = 0
        matcher.double().train()
        
        for i, pred in enumerate(train_loader):
            start = time.time()
            for k in pred:
                if k != 'file_name' and k!='image0' and k!='image1':
                    if type(pred[k]) == torch.Tensor:
                        if torch.cuda.is_available():
                            pred[k] = Variable(pred[k].cuda())
                        else:
                            pred[k] = Variable(pred[k])
                    else:
                        if torch.cuda.is_available():
                            pred[k] = Variable(torch.stack(pred[k]).cuda())
                        else:
                            pred[k] = Variable(torch.stack(pred[k])) # No cuda
                
            data = matcher(pred)

            for k, v in pred.items():
                pred[k] = v[0]
            pred = {**pred, **data}

            if pred['skip_train'] == True: # Posegraph empty
                continue

            matcher.zero_grad()
            Loss = pred['loss']
            epoch_loss += Loss.item()
            Loss.backward()
            optimizer.step()

            end = time.time()
            print("Iteration time: ",(end-start))
        epoch_loss /= len(train_loader)
        model_out_path = "path/to/modelckpt_epoch_{}.pth".format(epoch)
        torch.save(matcher, model_out_path)
        print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {} for lambda 0.05"
            .format(epoch, opt.epoch, epoch_loss, model_out_path))
        

