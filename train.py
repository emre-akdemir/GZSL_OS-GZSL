from __future__ import print_function
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import openpyxl as xl 
import model as model
import util as util
import classifier as classifier
import cluster as cluster
#import clusters.cluster_dbscan as cluster2
from config import opt

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
netF = model.Feedback(opt)
netDec = model.AttDec(opt,opt.attSize)

print(netE)
print(netG)
print(netD)
print(netF)
print(netDec)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize) 
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1, dtype=torch.float) 
mone = one * -1
##########
# Cuda
if opt.cuda:
    netD.cpu()
    netE.cpu()
    netF.cpu()
    netG.cpu()
    netDec.cpu()
    input_res = input_res.cpu()
    noise, input_att = noise.cpu(), input_att.cpu()
    one = one.cpu()
    mone = mone.cpu()

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction = 'sum')  #size_average=False) uyarı
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
    
def generate_syn_feature(generator,classes, attribute,num,netF=None,netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cpu()  #cuda
        syn_noise = syn_noise.cpu()  #cuda
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        ### mixup function for open set images
        if(opt.mixup):
            if i >= int(opt.nclass_openset):           
                lam = opt.lam
                batch_size = iclass_att.size()[0]
                index = torch.randperm(batch_size)
                index = index.cpu()               
                mixed_iclass_att = lam * iclass_att + (1 - lam) * iclass_att[index]
                iclass_att = mixed_iclass_att                                             
        ###
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise) 
        with torch.no_grad():
            syn_attv = Variable(syn_att) 
        fake = generator(syn_noisev,c= syn_attv) 
        if netF is not None:
            dec_out = netDec(fake)
            dec_hidden_feat = netDec.getLayersOutDet()
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


optimizer          = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerF         = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec       = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# excel file for result
book = xl.Workbook()
sheet = book.active
sheet['A1'] = 'Epoch'
sheet['B1'] = 'T_Epoch'
sheet['C1'] = 'Loss_D'
sheet['D1'] = 'Loss_G'
sheet['E1'] = 'Wasserstein_dist'
sheet['F1'] = 'vae_loss_seen'
if opt.os_gzsl:
    sheet['G1'] = 's'
    sheet['H1'] = 'u'
    sheet['I1'] = 'o'
    sheet['J1'] = 'h'
    sheet['K1'] = 'S'
    sheet['L1'] = 'U'
    sheet['M1'] = 'O'
    sheet['N1'] = 'H'
else:
    sheet['G1'] = 'GZSL_seen'
    sheet['H1'] = 'GZSL_unseen'
    sheet['I1'] = 'GZSL_h'
    sheet['J1'] = 'ZSL_unseen'
    sheet['K1'] = 'S'
    sheet['L1'] = 'U'
    sheet['M1'] = 'H'
    sheet['N1'] = 'T1'
book.save(opt.excel_filename)

#K Means clustering for Visaul Features
'''
if opt.k_means_visual:
    opt.k = 150
    kmeans = cluster.cluster_kmeans(opt)
    filter_train_feature, filter_train_label = kmeans.filters(data.train_feature, data.train_label)
    opt.k = 50
'''  
best_gzsl_acc = 0
best_zsl_acc = 0
for epoch in range(0,opt.nepoch):
    for loop in range(0,opt.feedback_loop):
        for i in range(0, data.ntrain, opt.batch_size):
            #########Discriminator training ##############
            for p in netD.parameters(): 
                p.requires_grad = True

            for p in netDec.parameters(): 
                p.requires_grad = True
            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()          
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                netDec.zero_grad()
                recons = netDec(input_resv)
                R_cost = opt.recons_weight*WeightedL1(recons, input_attv) 
                R_cost.backward()
                optimizerDec.step()
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD*criticD_real.mean()
                criticD_real.backward(mone)
                if opt.encoded_noise:        
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cpu())  #cuda
                    z = eps * std + means
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)
                    
                if loop == 1:
                    fake = netG(z, c= input_attv )
                    dec_out = netDec(fake)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)                  
                    fake = netG(z, a1=opt.a1, c=input_attv , feedback_layers=feedback_out)
                else:
                    fake = netG(z, c= input_attv ) # input_attv)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD*criticD_fake.mean()
                criticD_fake.backward(one)
                gradient_penalty = opt.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty 
                optimizerD.step()

            gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            #############Generator training ##############
            # Train Generator and Decoder
            for p in netD.parameters(): #freeze discrimator
                p.requires_grad = False
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec.parameters(): #freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            input_resv = Variable(input_res)
            input_attv =  Variable(input_att)
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = Variable(eps.cpu())
            z = eps * std + means
            if loop == 1:
                recon_x = netG(z, c=input_attv)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_attv)

            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
            errG = vae_loss_seen
            
            if opt.encoded_noise:
                criticG_fake = netD(recon_x,input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if loop == 1:
                    fake = netG(noisev, c=input_attv) 
                    dec_out = netDec(recon_x) 
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(noisev, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)  
                else:
                    fake = netG(noisev, c=input_attv) 
                criticG_fake = netD(fake,input_attv).mean()
                
            G_cost = -criticG_fake
            errG += opt.gammaG*G_cost
            netDec.zero_grad()
            recons_fake = netDec(fake)
            R_cost = WeightedL1(recons_fake, input_attv)
            errG += opt.recons_weight * R_cost
            errG.backward()
            # write a condition here
            optimizer.step()
            optimizerG.step()
            if loop == 1:
                optimizerF.step()
            if opt.recons_weight > 0 and not opt.freeze_dec: # not train decoder at feedback time
                optimizerDec.step() 
        
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f'% (epoch, opt.nepoch, D_cost.data, G_cost.data, Wasserstein_D.data,vae_loss_seen.data),end=" ")
    netG.eval()
    netDec.eval()
    netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, opt.syn_num,netF=netF,netDec=netDec)
    

    #DBSCAN clustering
    if opt.dbscan_:
        dbscan = cluster.cluster_dbscan(opt)
        filter_syn_feature, filter_syn_label = dbscan.filters(syn_feature, syn_label)
        
    #K Means clustering
    if opt.k_means:
        kmeans = cluster.cluster_kmeans(opt)
        filter_syn_feature, filter_syn_label = kmeans.filters(syn_feature, syn_label)
    
    # Generalized zero-shot learning
    if opt.gzsl:   
        # Concatenate real seen features with synthesized unseen features
        if opt.k_means or opt.dbscan_:
            train_X = torch.cat((data.train_feature, filter_syn_feature), 0)
            train_Y = torch.cat((data.train_label, filter_syn_label), 0)
        else:
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        
        # Train OS GZSL classifier                                           
        if opt.os_gzsl:
            gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                    25, opt.syn_num, generalized=False, openset_gzsl=True, _nclass_openset = opt.nclass_openset, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
            if best_gzsl_acc < gzsl_cls.H:
                best_acc_seen, best_acc_unseen, best_acc_openset, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.acc_openset, gzsl_cls.H
            print('GZSL: seen=%.4f, unseen=%.4f, openset=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.acc_openset, gzsl_cls.H),end=" ")
        # Train GZSL classifier    
        else:
            gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                    25, opt.syn_num, generalized=True, openset_gzsl=False, _nclass_openset = opt.nclass_openset, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
            if best_gzsl_acc < gzsl_cls.H:
                best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
            print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")
        

    # Zero-shot learning
    # Train ZSL classifier
    
    if opt.zsl:
        '''
        if opt.k_means:
            train_syn_feature =  filter_syn_feature
            train_syn_label = filter_syn_label
        else:
            train_syn_feature = syn_feature
            train_syn_label = syn_label
        '''  
        
        zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                        data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, \
                        generalized=False, openset_gzsl=opt.openset, _nclass_openset = opt.nclass_openset, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        acc = zsl_cls.acc
        if best_zsl_acc < acc:
            best_zsl_acc = acc
        print('ZSL: unseen accuracy=%.4f, the best ZSL: T1=%.4f' % (acc, best_zsl_acc),end=" ")
    
    if opt.os_gzsl:
        print('the best: S=%.4f, U=%.4f, O=%.4f, H=%.4f' % (best_acc_seen, best_acc_unseen, best_acc_openset, best_gzsl_acc))
    else:
        print('the best: S=%.4f, U=%.4f, H=%.4f' % (best_acc_seen, best_acc_unseen, best_gzsl_acc))
    
    no = int(epoch) + 2    
    #excel file for results   
    sheet['A' + str(no)] = str(epoch)
    sheet['B' + str(no)] = str(opt.nepoch)
    sheet['C' + str(no)] = str(D_cost.data.numpy())
    sheet['D' + str(no)] = str(G_cost.data.numpy())
    sheet['E' + str(no)] = str(Wasserstein_D.data.numpy())
    sheet['F' + str(no)] = str(vae_loss_seen.data.numpy())
    if opt.os_gzsl:
        sheet['G' + str(no)] = str(gzsl_cls.acc_seen.numpy())
        sheet['H' + str(no)] = str(gzsl_cls.acc_unseen.numpy())
        sheet['I' + str(no)] = str(gzsl_cls.acc_openset.numpy())
        sheet['J' + str(no)] = str(gzsl_cls.H.numpy())
        sheet['K' + str(no)] = str(best_acc_seen.numpy())
        sheet['L' + str(no)] = str(best_acc_unseen.numpy())
        sheet['M' + str(no)] = str(best_acc_openset.numpy())
        sheet['N' + str(no)] = str(best_gzsl_acc.numpy())
    else:
        sheet['G' + str(no)] = str(gzsl_cls.acc_seen.numpy())
        sheet['H' + str(no)] = str(gzsl_cls.acc_unseen.numpy())
        sheet['I' + str(no)] = str(gzsl_cls.H.numpy())
        if opt.zsl:
            sheet['J' + str(no)] = str(acc.numpy())
        sheet['K' + str(no)] = str(best_acc_seen.numpy())
        sheet['L' + str(no)] = str(best_acc_unseen.numpy())
        sheet['M' + str(no)] = str(best_gzsl_acc.numpy())
        if opt.zsl:
            sheet['N' + str(no)] = str(best_zsl_acc.numpy())        
    book.save(opt.excel_filename)
    
    # reset G to training mode
    netG.train()
    netDec.train()    
    netF.train()
    
    
print('Dataset', opt.dataset)
if opt.zsl:
    print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl:
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    if opt.os_gzsl:
        print('the best GZSL openset accuracy is', best_acc_openset)
    print('the best GZSL H is', best_gzsl_acc)
