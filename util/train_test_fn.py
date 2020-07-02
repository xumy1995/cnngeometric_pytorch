from __future__ import print_function, division
from tensorboardX import SummaryWriter
from geotnf.transformation import GeometricTnf
from image.normalization import NormalizeImageDict, normalize_image
import numpy as np
affTnf = GeometricTnf(geometric_model='affine', use_cuda=True)

tb_logger = SummaryWriter('events')

def train(epoch,model,loss_fn,optimizer,dataloader,pair_generation_tnf,use_cuda=True,log_interval=50, logger=None):
    model.train()
    train_loss = 0
    B = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy().item()
        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
               epoch, batch_idx , len(dataloader),
               100. * batch_idx / len(dataloader), loss.item()))
            src_img = tnf_batch['source_image'][0].unsqueeze(0)
            tgt_img = tnf_batch['target_image'][0].unsqueeze(0)
            #resizeTgt = GeometricTnf(out_h=tgt_img.shape[2], out_w=tgt_img.shape[3], use_cuda = True)
            warped_image_aff = affTnf(src_img, theta[0].view(-1,2,3))
            warped_image_aff_np = normalize_image(warped_image_aff,forward=False)
            
            src_img = normalize_image(src_img,forward=False).detach().cpu().numpy()
            tgt_img = normalize_image(tgt_img,forward=False).detach().cpu().numpy()
            warped_image_aff_np = warped_image_aff_np.detach().cpu().numpy()
            
            img_cat = np.concatenate((src_img, tgt_img, warped_image_aff_np), axis=3)
            info = {
                'img': img_cat
            }
            for tag, images in info.items():
                tb_logger.add_images(tag, images, epoch*B + batch_idx)
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            #   epoch, batch_idx , len(dataloader),
            #   100. * batch_idx / len(dataloader), loss.item()))
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

def test(model,loss_fn,dataloader,pair_generation_tnf,use_cuda=True, logger=None):
    model.eval()
    test_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'])
        test_loss += loss.data.cpu().numpy().item()

    test_loss /= len(dataloader)
    #print('Test set: Average loss: {:.4f}'.format(test_loss))
    logger.info('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss
